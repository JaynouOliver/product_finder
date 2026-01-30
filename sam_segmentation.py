"""
SAM3 segmentation module using Roboflow inference and Modal SAM3.
Supports both local file paths and image URLs.
"""

import os
import tempfile
import requests
import numpy as np
import cv2
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from inference_sdk import InferenceHTTPClient

from config import (
    ROBOFLOW_API_URL,
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_WORKFLOW_ID,
    MODAL_SAM3_URL,
    OUTPUT_DIR
)
from image_processing import (
    load_image,
    decode_rle_mask,
    process_mask_to_cropped_white,
    image_to_base64,
    save_image
)


def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ('http', 'https')
    except Exception:
        return False


def download_image(url: str, timeout: int = 30) -> str:
    """
    Download an image from URL to a temporary file.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
    
    Returns:
        Path to temporary file
    
    Raises:
        Exception: If download fails
    """
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    
    # Determine file extension from content-type or URL
    content_type = response.headers.get('content-type', '')
    if 'jpeg' in content_type or 'jpg' in content_type:
        ext = '.jpg'
    elif 'png' in content_type:
        ext = '.png'
    elif 'webp' in content_type:
        ext = '.webp'
    else:
        # Try to get from URL
        path = urlparse(url).path
        ext = os.path.splitext(path)[1] or '.jpg'
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception:
        os.unlink(temp_path)
        raise
    
    return temp_path


@dataclass
class SegmentedObject:
    """Represents a segmented object from SAM3."""
    class_name: str
    confidence: float
    cropped_image: Any  # numpy array
    base64_image: str
    bbox: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'base64_image': self.base64_image,
            'bbox': self.bbox
        }


class SAM3Segmenter:
    """SAM3 segmentation using Roboflow workflow."""
    
    def __init__(
        self,
        api_key: str = ROBOFLOW_API_KEY,
        api_url: str = ROBOFLOW_API_URL,
        workspace: str = ROBOFLOW_WORKSPACE,
        workflow_id: str = ROBOFLOW_WORKFLOW_ID
    ):
        """
        Initialize SAM3 segmenter.
        
        Args:
            api_key: Roboflow API key
            api_url: Roboflow API URL
            workspace: Roboflow workspace name
            workflow_id: Roboflow workflow ID
        """
        self.client = InferenceHTTPClient(
            api_url=api_url,
            api_key=api_key
        )
        self.workspace = workspace
        self.workflow_id = workflow_id
    
    def segment_image(
        self,
        image_source: str,
        prompts: List[str],
        use_cache: bool = True,
        save_outputs: bool = False,
        output_dir: str = OUTPUT_DIR
    ) -> List[SegmentedObject]:
        """
        Segment an image using SAM3 with text prompts.
        
        Args:
            image_source: Path to input image OR image URL (http/https)
            prompts: List of object names to segment (e.g., ["sofa", "carpet"])
            use_cache: Whether to use Roboflow cache
            save_outputs: Whether to save cropped images to disk
            output_dir: Directory to save outputs
        
        Returns:
            List of SegmentedObject instances
        """
        # Handle URL input - download to temp file
        temp_file = None
        if is_url(image_source):
            temp_file = download_image(image_source)
            image_path = temp_file
        else:
            image_path = image_source
        
        try:
            # Load original image
            original_image = load_image(image_path)
            img_h, img_w = original_image.shape[:2]
            
            # Run SAM3 workflow
            result = self.client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": image_path},
                parameters={"prompts": prompts},
                use_cache=use_cache
            )
            
            # Process results
            segmented_objects = []
            
            if save_outputs:
                os.makedirs(output_dir, exist_ok=True)
            
            for item in result:
                if 'sam' not in item:
                    continue
                
                sam_data = item['sam']
                if 'predictions' not in sam_data:
                    continue
                
                img_info = sam_data.get('image', {})
                h = img_info.get('height', img_h)
                w = img_info.get('width', img_w)
                
                for i, pred in enumerate(sam_data['predictions']):
                    class_name = pred.get('class', f'object_{i}')
                    confidence = pred.get('confidence', 0.0)
                    rle_mask = pred.get('rle_mask')
                    
                    if not rle_mask:
                        continue
                    
                    # Decode mask
                    mask = decode_rle_mask(rle_mask, h, w)
                    
                    # Create cropped white background image
                    cropped_image = process_mask_to_cropped_white(original_image, mask)
                    
                    # Convert to base64
                    base64_image = image_to_base64(cropped_image)
                    
                    # Extract bbox info
                    bbox = {
                        'x': pred.get('x', 0),
                        'y': pred.get('y', 0),
                        'width': pred.get('width', 0),
                        'height': pred.get('height', 0)
                    }
                    
                    # Create segmented object
                    seg_obj = SegmentedObject(
                        class_name=class_name,
                        confidence=confidence,
                        cropped_image=cropped_image,
                        base64_image=base64_image,
                        bbox=bbox
                    )
                    segmented_objects.append(seg_obj)
                    
                    # Optionally save to disk
                    if save_outputs:
                        filename = f"{i+1:02d}_{class_name}_cropped_white.jpg"
                        save_image(cropped_image, os.path.join(output_dir, filename))
            
            return segmented_objects
        
        finally:
            # Clean up temp file if we downloaded from URL
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def segment_and_filter(
        self,
        image_source: str,
        prompts: List[str],
        min_confidence: float = 0.5,
        **kwargs
    ) -> List[SegmentedObject]:
        """
        Segment image and filter by confidence threshold.
        
        Args:
            image_source: Path to input image OR image URL
            prompts: List of object names to segment
            min_confidence: Minimum confidence threshold
            **kwargs: Additional arguments for segment_image
        
        Returns:
            Filtered list of SegmentedObject instances
        """
        objects = self.segment_image(image_source, prompts, **kwargs)
        return [obj for obj in objects if obj.confidence >= min_confidence]


def segment_image(
    image_source: str,
    prompts: List[str],
    save_outputs: bool = False
) -> List[SegmentedObject]:
    """
    Convenience function to segment an image.
    
    Args:
        image_source: Path to input image OR image URL
        prompts: List of object names to segment
        save_outputs: Whether to save cropped images
    
    Returns:
        List of SegmentedObject instances
    """
    segmenter = SAM3Segmenter()
    return segmenter.segment_image(image_source, prompts, save_outputs=save_outputs)


class ModalSAM3Segmenter:
    """
    SAM3 segmentation using Modal-hosted FastAPI endpoint.
    
    Key differences from Roboflow:
    - No prompts required (auto-detects all segments)
    - Uses polygon coordinates instead of RLE masks
    - Faster (~3-5s vs ~8-10s)
    """
    
    def __init__(self, api_url: str = MODAL_SAM3_URL):
        """
        Initialize Modal SAM3 segmenter.
        
        Args:
            api_url: Modal SAM3 FastAPI endpoint URL
        """
        self.api_url = api_url.rstrip('/')
    
    def _polygon_to_mask(
        self, 
        polygon_flat: List[int], 
        width: int, 
        height: int
    ) -> np.ndarray:
        """
        Convert flat polygon coordinates to binary mask.
        
        Args:
            polygon_flat: Flat list of [x0, y0, x1, y1, ...] coordinates
            width: Image width
            height: Image height
        
        Returns:
            Binary mask as numpy array (0 or 255)
        """
        import cv2
        import numpy as np
        
        # Reshape flat list to [(x, y), ...] pairs
        points = np.array(polygon_flat).reshape(-1, 2).astype(np.int32)
        
        # Create empty mask and fill polygon
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def segment_image(
        self,
        image_source: str,
        top_k: int = 20,
        vertices: int = 100,
        save_outputs: bool = False,
        output_dir: str = OUTPUT_DIR
    ) -> List[SegmentedObject]:
        """
        Segment an image using Modal SAM3 endpoint.
        
        Args:
            image_source: Path to input image OR image URL (http/https)
            top_k: Maximum number of segments to return (default: 20)
            vertices: Polygon simplification level (default: 100)
            save_outputs: Whether to save cropped images to disk
            output_dir: Directory to save outputs
        
        Returns:
            List of SegmentedObject instances
        """
        import cv2
        import numpy as np
        
        # Handle URL input - download to temp file
        temp_file = None
        if is_url(image_source):
            temp_file = download_image(image_source)
            image_path = temp_file
        else:
            image_path = image_source
        
        try:
            # Load original image
            original_image = load_image(image_path)
            img_h, img_w = original_image.shape[:2]
            
            # Call Modal SAM3 API
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {'top_k': top_k, 'vertices': vertices}
                
                response = requests.post(self.api_url, files=files, data=data, timeout=60)
            
            if response.status_code != 200:
                raise Exception(f"Modal SAM3 API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Process results
            segmented_objects = []
            
            if save_outputs:
                os.makedirs(output_dir, exist_ok=True)
            
            api_width = result.get('image_width', img_w)
            api_height = result.get('image_height', img_h)
            
            for i, segment in enumerate(result.get('segments', [])):
                label = segment.get('label', f'object_{i}')
                score = segment.get('score', 0.0)
                polygon_flat = segment.get('polygon_flat', [])
                bbox = segment.get('bbox', [0, 0, img_w, img_h])
                
                if not polygon_flat:
                    continue
                
                # Convert polygon to mask
                mask = self._polygon_to_mask(polygon_flat, api_width, api_height)
                
                # Resize mask if API dimensions differ from actual image
                if api_width != img_w or api_height != img_h:
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Create cropped white background image
                cropped_image = process_mask_to_cropped_white(original_image, mask)
                
                # Convert to base64
                base64_image = image_to_base64(cropped_image)
                
                # Extract bbox info (Modal format: [x_min, y_min, x_max, y_max])
                bbox_dict = {
                    'x': (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0,
                    'y': (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0,
                    'width': (bbox[2] - bbox[0]) if len(bbox) >= 4 else 0,
                    'height': (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0
                }
                
                # Create segmented object
                seg_obj = SegmentedObject(
                    class_name=label,
                    confidence=score,
                    cropped_image=cropped_image,
                    base64_image=base64_image,
                    bbox=bbox_dict
                )
                segmented_objects.append(seg_obj)
                
                # Optionally save to disk
                if save_outputs:
                    filename = f"{i+1:02d}_{label}_cropped_white.jpg"
                    save_image(cropped_image, os.path.join(output_dir, filename))
            
            return segmented_objects
        
        finally:
            # Clean up temp file if we downloaded from URL
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def segment_and_filter(
        self,
        image_source: str,
        min_confidence: float = 0.5,
        **kwargs
    ) -> List[SegmentedObject]:
        """
        Segment image and filter by confidence threshold.
        
        Args:
            image_source: Path to input image OR image URL
            min_confidence: Minimum confidence threshold
            **kwargs: Additional arguments for segment_image
        
        Returns:
            Filtered list of SegmentedObject instances
        """
        objects = self.segment_image(image_source, **kwargs)
        return [obj for obj in objects if obj.confidence >= min_confidence]

