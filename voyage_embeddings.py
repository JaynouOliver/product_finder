"""
Voyage AI multimodal embeddings module.
"""

import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import VOYAGE_API_KEY, VOYAGE_API_URL, VOYAGE_MODEL
from sam_segmentation import SegmentedObject


@dataclass
class EmbeddingResult:
    """Represents an embedding result from Voyage AI."""
    class_name: str
    confidence: float
    embedding: List[float]
    text_tokens: int
    image_pixels: int
    total_tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'embedding': self.embedding,
            'usage': {
                'text_tokens': self.text_tokens,
                'image_pixels': self.image_pixels,
                'total_tokens': self.total_tokens
            }
        }


class VoyageEmbeddings:
    """Voyage AI multimodal embeddings client."""
    
    def __init__(
        self,
        api_key: str = None,
        api_url: str = VOYAGE_API_URL,
        model: str = VOYAGE_MODEL
    ):
        """
        Initialize Voyage embeddings client.
        
        Args:
            api_key: Voyage AI API key (defaults to env var)
            api_url: Voyage AI API URL
            model: Model name (voyage-multimodal-3 or voyage-multimodal-3.5)
        """
        self.api_key = api_key or VOYAGE_API_KEY
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not set. Set it in .env or pass explicitly.")
        
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _build_content(
        self,
        image_base64: str,
        text: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build content array for Voyage API.
        
        Args:
            image_base64: Base64 encoded image with data URL prefix
            text: Optional text description
        
        Returns:
            Content array for API request
        """
        content = []
        
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        content.append({
            "type": "image_base64",
            "image_base64": image_base64
        })
        
        return content
    
    def get_embedding(
        self,
        image_base64: str,
        text: Optional[str] = None,
        input_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get embedding for a single image.
        
        Args:
            image_base64: Base64 encoded image with data URL prefix
            text: Optional text description
            input_type: Optional input type ('query', 'document', or None)
        
        Returns:
            API response with embedding
        """
        payload = {
            "inputs": [
                {"content": self._build_content(image_base64, text)}
            ],
            "model": self.model
        }
        
        if input_type:
            payload["input_type"] = input_type
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Voyage API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_embeddings_batch(
        self,
        items: List[Dict[str, str]],
        input_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get embeddings for multiple images in a single request.
        
        Args:
            items: List of dicts with 'image_base64' and optional 'text' keys
            input_type: Optional input type ('query', 'document', or None)
        
        Returns:
            API response with embeddings
        
        Note:
            Maximum 1000 inputs per request.
            Total tokens across all inputs must not exceed 320,000.
        """
        inputs = []
        for item in items:
            content = self._build_content(
                item['image_base64'],
                item.get('text')
            )
            inputs.append({"content": content})
        
        payload = {
            "inputs": inputs,
            "model": self.model
        }
        
        if input_type:
            payload["input_type"] = input_type
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Voyage API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def embed_segmented_objects(
        self,
        objects: List[SegmentedObject],
        include_class_text: bool = True,
        input_type: Optional[str] = "document"
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for segmented objects.
        
        Args:
            objects: List of SegmentedObject from SAM3 segmentation
            include_class_text: Whether to include class name as text
            input_type: Input type for retrieval optimization
        
        Returns:
            List of EmbeddingResult instances
        """
        if not objects:
            return []
        
        # Prepare batch request
        items = []
        for obj in objects:
            item = {'image_base64': obj.base64_image}
            if include_class_text:
                item['text'] = f"A {obj.class_name}"
            items.append(item)
        
        # Get embeddings
        response = self.get_embeddings_batch(items, input_type=input_type)
        
        # Process results
        results = []
        usage = response.get('usage', {})
        
        for i, (obj, data) in enumerate(zip(objects, response.get('data', []))):
            result = EmbeddingResult(
                class_name=obj.class_name,
                confidence=obj.confidence,
                embedding=data.get('embedding', []),
                text_tokens=usage.get('text_tokens', 0),
                image_pixels=usage.get('image_pixels', 0),
                total_tokens=usage.get('total_tokens', 0)
            )
            results.append(result)
        
        return results


def get_embeddings_for_objects(
    objects: List[SegmentedObject],
    include_class_text: bool = True
) -> List[EmbeddingResult]:
    """
    Convenience function to get embeddings for segmented objects.
    
    Args:
        objects: List of SegmentedObject instances
        include_class_text: Whether to include class name as text
    
    Returns:
        List of EmbeddingResult instances
    """
    client = VoyageEmbeddings()
    return client.embed_segmented_objects(objects, include_class_text=include_class_text)
