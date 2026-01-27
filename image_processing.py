"""
Image processing utilities for mask application and cropping.
"""

import base64
import cv2
import numpy as np
from typing import Tuple, Optional

from config import DEFAULT_JPEG_QUALITY, CROP_PADDING


def decode_rle_mask(rle_data: dict, height: int, width: int) -> np.ndarray:
    """
    Decode Run-Length Encoded mask to binary mask.
    Handles COCO compressed string format.
    
    Args:
        rle_data: RLE data dict with 'counts' and 'size' keys
        height: Image height
        width: Image width
    
    Returns:
        Binary mask as numpy array (0 or 255)
    """
    if isinstance(rle_data, dict):
        rle_counts = rle_data.get('counts', [])
        rle_size = rle_data.get('size', [height, width])
        h, w = rle_size if len(rle_size) == 2 else (height, width)
    else:
        rle_counts = rle_data
        h, w = height, width
    
    if isinstance(rle_counts, str):
        # COCO compressed RLE format
        try:
            from pycocotools import mask as mask_util
            rle = {'size': [h, w], 'counts': rle_counts.encode('utf-8')}
            mask = mask_util.decode(rle)
            return (mask * 255).astype(np.uint8)
        except ImportError:
            raise ImportError("pycocotools required for RLE decoding. Install with: pip install pycocotools")
    else:
        # List of counts format
        mask = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        for i, count in enumerate(rle_counts):
            count = int(count)
            if i % 2 == 1:
                mask[pos:pos + count] = 255
            pos += count
        return mask.reshape((h, w), order='F')


def apply_mask_white_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image with white background.
    Subject visible, background is pure white.
    
    Args:
        image: Original BGR image
        mask: Binary mask (255 = foreground)
    
    Returns:
        Image with white background where mask is 0
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create white background
    result = np.ones_like(image) * 255
    
    # Copy subject pixels where mask is active
    mask_bool = mask > 128
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    result = np.where(mask_3ch, image, result)
    
    return result


def crop_to_mask_bbox(
    image: np.ndarray, 
    mask: np.ndarray, 
    padding: int = CROP_PADDING
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop image and mask to bounding box of the mask with padding.
    
    Args:
        image: Original image
        mask: Binary mask
        padding: Pixels to add around bounding box
    
    Returns:
        Tuple of (cropped_image, cropped_mask)
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Find bounding box
    rows = np.any(mask > 128, axis=1)
    cols = np.any(mask > 128, axis=0)
    
    if not rows.any() or not cols.any():
        return image, mask
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add padding
    h, w = image.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding)
    
    return image[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax]


def process_mask_to_cropped_white(
    original_image: np.ndarray, 
    mask: np.ndarray
) -> np.ndarray:
    """
    Process mask to create cropped image with white background.
    This is the recommended format for embedding generation.
    
    Args:
        original_image: Original BGR image
        mask: Binary mask
    
    Returns:
        Cropped image with white background
    """
    cropped_img, cropped_mask = crop_to_mask_bbox(original_image, mask)
    return apply_mask_white_background(cropped_img, cropped_mask)


def image_to_base64(image: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """
    Convert OpenCV image to base64 string with data URL prefix.
    
    Args:
        image: OpenCV BGR image
        quality: JPEG quality (0-100)
    
    Returns:
        Base64 encoded string with data URL prefix
    """
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        OpenCV BGR image
    
    Raises:
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def save_image(image: np.ndarray, path: str, quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """
    Save image to path.
    
    Args:
        image: OpenCV image
        path: Output path
        quality: JPEG quality
    
    Returns:
        Path where image was saved
    """
    if path.endswith('.jpg') or path.endswith('.jpeg'):
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(path, image)
    return path
