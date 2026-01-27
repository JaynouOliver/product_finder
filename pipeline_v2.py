"""
Product Finder V2 Pipeline.

================================================================================
ALGORITHM OVERVIEW
================================================================================

Goal: Find real materials from our product library based on a concept room image.

Input:  f(image_url, refine_params) where refine_params = {slots, region, price}
Output: { "Floor": [3 products], "Wall": [3 products], ... } (~12 total results)

================================================================================
PIPELINE STEPS & LATENCY
================================================================================

Step 1: IMAGE DOWNLOAD (if URL provided)
        - Download image from URL to temp file
        - Latency: ~0.5-1s (depends on image size/network)

Step 2: SAM3 SEGMENTATION (Roboflow Serverless API)
        - Input: Image + text prompts (e.g., ["Floor", "Wall", "Worktop"])
        - Process: Segment-Anything-Model v3 with text-prompted segmentation
        - Output: RLE masks + confidence scores for each detected surface
        - Post-process: Decode masks, crop segments, convert to base64
        - Latency: ~8-10s (external API, biggest bottleneck)

Step 3: VOYAGE EMBEDDINGS (Voyage AI API)
        - Input: Cropped segment images (base64) + slot name text
        - Model: voyage-multimodal-3.5 (1024-dim vectors)
        - Process: Batch request for all segments
        - Output: Embedding vectors per segment
        - Latency: ~1-2s

Step 4: VECTOR SIMILARITY SEARCH (Supabase pgvector)
        - Input: Embedding vectors + metadata filters (region, price)
        - Process: 
            a. HNSW index search (over-fetch 10x for filtering headroom)
            b. Apply metadata filters (application type, region, price)
            c. Return top N matches per slot
        - Output: Product matches with similarity scores
        - Latency: ~0.3-0.5s

Step 5: RESPONSE ASSEMBLY
        - Deduplicate slots (keep highest confidence per slot type)
        - Format response: { slot_name: { confidence, results[] } }
        - Latency: <0.1s

================================================================================
TOTAL LATENCY: ~10-13s 
================================================================================

FALLBACK LOGIC:
    - If avg segment confidence < 0.5:
        1. First try: Relaxed metadata search (no region/price filter)
        2. If insufficient: Whole-image similarity search (12 mixed results)
    - Low confidence slots (< 0.3) are hidden from response

OPTIMIZATION NOTES:
    - SAM3 uses Roboflow cache (use_cache=True) for repeated images
    - Voyage embeddings sent as batch (not individual requests)
    - Supabase search runs in parallel across slots (ThreadPoolExecutor)
    - To achieve <5s latency, self-hosted SAM3 on GPU is required

================================================================================

Usage:
    python pipeline_v2.py --image image.jpg --slots Floor Wall Worktop
    python pipeline_v2.py --image image.jpg --region US EU --results-per-slot 3
"""

import argparse
import json
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from sam_segmentation import SAM3Segmenter, SegmentedObject
from voyage_embeddings import VoyageEmbeddings, EmbeddingResult
from supabase_search import SupabaseSearch, SlotResult, SearchResult
from config import OUTPUT_DIR


@dataclass
class PipelineConfig:
    """Configuration for V2 pipeline."""
    # Slot configuration
    default_slots: List[str] = None
    results_per_slot: int = 3
    min_confidence: float = 0.3
    
    # Metadata filters
    region: Optional[List[str]] = None
    price: Optional[int] = None
    
    # Fallback settings
    enable_fallback: bool = True
    fallback_threshold: float = 0.5  # If avg confidence < this, use fallback
    
    def __post_init__(self):
        if self.default_slots is None:
            self.default_slots = ["Floor", "Wall", "Worktop", "Backsplash"]


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    image_source: str
    slots: Dict[str, Any]
    fallback_used: bool
    timing: Dict[str, float]
    total_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_source': self.image_source,
            'slots': {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in self.slots.items()},
            'fallback_used': self.fallback_used,
            'timing': self.timing,
            'total_time': self.total_time
        }


class ProductFinderV2:
    """
    V2 Product Finder Pipeline.
    
    Optimized for:
    - Stateless operation (no LLM descriptions)
    - Parallel processing where possible
    - < 5 second latency target (excluding SAM3)
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components (lazy load for faster startup)
        self._segmenter = None
        self._voyage = None
        self._search = None
    
    @property
    def segmenter(self) -> SAM3Segmenter:
        if self._segmenter is None:
            self._segmenter = SAM3Segmenter()
        return self._segmenter
    
    @property
    def voyage(self) -> VoyageEmbeddings:
        if self._voyage is None:
            self._voyage = VoyageEmbeddings()
        return self._voyage
    
    @property
    def search(self) -> SupabaseSearch:
        if self._search is None:
            self._search = SupabaseSearch()
        return self._search
    
    def run(
        self,
        image_source: str,
        slots: List[str] = None,
        region: List[str] = None,
        price: int = None
    ) -> PipelineResult:
        """
        Run the full V2 pipeline.
        
        Args:
            image_source: Path to input image OR image URL (http/https)
            slots: List of slot names to detect (e.g., ["Floor", "Wall"])
            region: Region filter (e.g., ["US", "EU"])
            price: Price tier filter (1-5)
        
        Returns:
            PipelineResult with all slot matches
        """
        total_start = time.time()
        timing = {}
        
        slots = slots or self.config.default_slots
        region = region or self.config.region
        price = price or self.config.price
        
        print(f"[V2 Pipeline] Processing: {image_source}")
        print(f"[V2 Pipeline] Slots: {slots}")
        print("-" * 50)
        
        # Step 1: SAM3 Segmentation (supports both file paths and URLs)
        t0 = time.time()
        print("[Step 1] Running SAM3 segmentation...")
        segmented_objects = self.segmenter.segment_image(
            image_source=image_source,
            prompts=slots,
            use_cache=True,
            save_outputs=False
        )
        timing['sam3_segmentation'] = time.time() - t0
        print(f"  Segmented {len(segmented_objects)} objects in {timing['sam3_segmentation']:.2f}s")
        
        # Check if we need fallback
        avg_confidence = sum(obj.confidence for obj in segmented_objects) / max(len(segmented_objects), 1)
        use_fallback = (
            self.config.enable_fallback and 
            (len(segmented_objects) == 0 or avg_confidence < self.config.fallback_threshold)
        )
        
        if use_fallback:
            print(f"  Low confidence ({avg_confidence:.2f}), will use fallback")
        
        # Step 2: Generate embeddings (batch for speed)
        t0 = time.time()
        print("[Step 2] Generating Voyage embeddings...")
        
        if segmented_objects:
            embeddings = self.voyage.embed_segmented_objects(
                segmented_objects,
                include_class_text=True,
                input_type="query"  # Query type for retrieval
            )
        else:
            embeddings = []
        
        timing['voyage_embeddings'] = time.time() - t0
        print(f"  Generated {len(embeddings)} embeddings in {timing['voyage_embeddings']:.2f}s")
        
        # Step 3: Similarity search
        t0 = time.time()
        print("[Step 3] Running similarity search...")
        
        if use_fallback or len(embeddings) == 0:
            # Fallback: Use whole-image search or first available embedding
            slot_results = self._run_fallback_search(
                segmented_objects, embeddings, region, price
            )
        else:
            # Normal: Search each slot in parallel
            slot_results = self._run_slot_search(
                segmented_objects, embeddings, region, price
            )
        
        timing['supabase_search'] = time.time() - t0
        print(f"  Search completed in {timing['supabase_search']:.2f}s")
        
        # Compile results
        total_time = time.time() - total_start
        timing['total'] = total_time
        
        result = PipelineResult(
            image_source=image_source,
            slots=slot_results,
            fallback_used=use_fallback,
            timing=timing,
            total_time=total_time
        )
        
        # Summary
        print("-" * 50)
        print(f"[V2 Pipeline] Complete in {total_time:.2f}s")
        print(f"  - SAM3: {timing['sam3_segmentation']:.2f}s")
        print(f"  - Voyage: {timing['voyage_embeddings']:.2f}s")
        print(f"  - Supabase: {timing['supabase_search']:.2f}s")
        print(f"  - Slots found: {len(slot_results)}")
        
        for slot_name, slot_result in slot_results.items():
            if isinstance(slot_result, SlotResult):
                print(f"    - {slot_name}: {len(slot_result.results)} results (conf: {slot_result.confidence:.2f})")
        
        return result
    
    def _run_slot_search(
        self,
        objects: List[SegmentedObject],
        embeddings: List[EmbeddingResult],
        region: List[str],
        price: int
    ) -> Dict[str, SlotResult]:
        """Run parallel search for each slot."""
        # Prepare slots data
        slots_data = []
        for obj, emb in zip(objects, embeddings):
            slots_data.append({
                'name': obj.class_name,
                'embedding': emb.embedding,
                'confidence': obj.confidence
            })
        
        # Deduplicate by slot name (keep highest confidence)
        unique_slots = {}
        for slot in slots_data:
            name = slot['name'].lower()
            if name not in unique_slots or slot['confidence'] > unique_slots[name]['confidence']:
                unique_slots[name] = slot
        
        # Run parallel search
        return self.search.search_all_slots_parallel(
            slots=list(unique_slots.values()),
            region=region,
            price=price,
            match_count=self.config.results_per_slot,
            min_confidence=self.config.min_confidence
        )
    
    def _run_fallback_search(
        self,
        objects: List[SegmentedObject],
        embeddings: List[EmbeddingResult],
        region: List[str],
        price: int
    ) -> Dict[str, SlotResult]:
        """
        Fallback search when confidence is low.
        
        Strategy (per V2 spec):
        1. First try with REDUCED metadata restrictions (no region/price filter)
        2. If still insufficient, switch to pure whole-image similarity search
        """
        if not embeddings:
            return {}
        
        # Use highest confidence embedding (not just first)
        best_idx = max(range(len(objects)), key=lambda i: objects[i].confidence)
        best_embedding = embeddings[best_idx].embedding
        best_slot_name = objects[best_idx].class_name
        
        # Step 1: Try slot-based search with RELAXED metadata (no region/price)
        print("  Fallback Step 1: Trying relaxed metadata filters...")
        relaxed_results = self.search.search_by_embedding(
            embedding=best_embedding,
            application=None,  # No application filter
            region=None,       # No region filter  
            price=None,        # No price filter
            match_count=12,
            similarity_threshold=0.3
        )
        
        if len(relaxed_results) >= 6:
            # Got enough results with relaxed filters
            print(f"  Fallback Step 1: Found {len(relaxed_results)} results with relaxed filters")
            return {
                'fallback': SlotResult(
                    slot_name='fallback_relaxed',
                    confidence=objects[best_idx].confidence,
                    results=relaxed_results
                )
            }
        
        # Step 2: Pure whole-image similarity search (lowest threshold)
        print("  Fallback Step 2: Switching to whole-image similarity search...")
        results = self.search.search_fallback(
            embedding=best_embedding,
            match_count=12
        )
        
        return {
            'fallback': SlotResult(
                slot_name='fallback_whole_image',
                confidence=0.0,
                results=results
            )
        }


def run_pipeline(
    image_source: str,
    slots: List[str] = None,
    region: List[str] = None,
    price: int = None,
    results_per_slot: int = 3
) -> PipelineResult:
    """
    Convenience function to run the V2 pipeline.
    
    Args:
        image_source: Path to input image OR image URL
        slots: Slot names to detect
        region: Region filter
        price: Price tier filter
        results_per_slot: Number of results per slot
    
    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        default_slots=slots,
        results_per_slot=results_per_slot,
        region=region,
        price=price
    )
    finder = ProductFinderV2(config)
    return finder.run(image_source, slots, region, price)


def main():
    parser = argparse.ArgumentParser(
        description="Product Finder V2: Find materials from concept images"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input image OR image URL (http/https)"
    )
    parser.add_argument(
        "--slots", "-s",
        nargs="+",
        default=["Floor", "Wall", "Worktop", "Backsplash"],
        help="Slot names to detect (default: Floor Wall Worktop Backsplash)"
    )
    parser.add_argument(
        "--region", "-r",
        nargs="+",
        default=None,
        help="Region filter (e.g., US EU UK)"
    )
    parser.add_argument(
        "--price", "-p",
        type=int,
        default=None,
        help="Price tier filter (1-5)"
    )
    parser.add_argument(
        "--results-per-slot", "-n",
        type=int,
        default=3,
        help="Number of results per slot (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON to stdout"
    )
    
    args = parser.parse_args()
    
    # Validate image - check if it's a URL or file path
    is_url = args.image.startswith('http://') or args.image.startswith('https://')
    if not is_url and not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1
    
    # Run pipeline
    result = run_pipeline(
        image_source=args.image,
        slots=args.slots,
        region=args.region,
        price=args.price,
        results_per_slot=args.results_per_slot
    )
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    if args.json:
        print("\n" + json.dumps(result.to_dict(), indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())
