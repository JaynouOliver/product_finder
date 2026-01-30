"""
Product Finder V2 - Streamlit UI

Upload an image -> See detected segments -> View similar products

Deployment: Streamlit Cloud
    1. Push to GitHub
    2. Connect to share.streamlit.io
    3. Set secrets in Streamlit Cloud dashboard

Usage (local):
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
import io
import time
import os
import tempfile
from PIL import Image
from pathlib import Path


# Page config
st.set_page_config(
    page_title="Product Finder V2",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Make drag-drop area more prominent */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        background: #f8fff8 !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2E7D32 !important;
        background: #e8f5e9 !important;
    }
    
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background: white;
    }
    .slot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        margin: 16px 0 12px 0;
        font-weight: 600;
    }
    .confidence-badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)





def run_pipeline_local(image_path: str, region: list = None, price: int = None):
    """
    Run the pipeline locally. Returns result and segment images.
    
    Modal SAM3 auto-detects segments - no prompts/slots needed!
    
    Includes fallback logic:
    - If avg confidence < 0.4 OR no segments found → use fallback
    - Fallback Step 1: Search with relaxed metadata (no region/price filter)
    - Fallback Step 2: Whole-image similarity search (lowest threshold)
    """
    import time
    from pipeline_v2 import ProductFinderV2, PipelineConfig, PipelineResult
    from supabase_search import SlotResult
    
    FALLBACK_THRESHOLD = 0.4  # Trigger fallback if avg confidence below this
    
    config = PipelineConfig(
        results_per_slot=10,
        region=region,
        price=price
    )
    finder = ProductFinderV2(config)
    
    total_start = time.time()
    timing = {}
    
    # Step 1: Run Modal SAM3 segmentation (auto-detects segments, no prompts needed)
    t0 = time.time()
    segmented_objects = finder.segmenter.segment_image(
        image_source=image_path,
        top_k=20,
        save_outputs=False
    )
    timing['sam3_segmentation'] = time.time() - t0
    
    # Store segment images for display (base64)
    segment_images = {}
    for obj in segmented_objects:
        slot_name = obj.class_name.lower()
        if slot_name not in segment_images or obj.confidence > segment_images[slot_name]['confidence']:
            segment_images[slot_name] = {
                'base64': obj.base64_image,
                'confidence': obj.confidence,
                'class_name': obj.class_name
            }
    
    # Check if we need fallback
    avg_confidence = sum(obj.confidence for obj in segmented_objects) / max(len(segmented_objects), 1)
    use_fallback = (
        config.enable_fallback and 
        (len(segmented_objects) == 0 or avg_confidence < FALLBACK_THRESHOLD)
    )
    
    # Step 2: Generate embeddings
    t0 = time.time()
    if segmented_objects:
        embeddings = finder.voyage.embed_segmented_objects(
            segmented_objects,
            include_class_text=True,
            input_type="query"
        )
    else:
        embeddings = []
    timing['voyage_embeddings'] = time.time() - t0
    
    # Step 3: Run similarity search
    t0 = time.time()
    
    if use_fallback or len(embeddings) == 0:
        # Fallback search strategy
        slot_results = _run_fallback_search(finder, segmented_objects, embeddings, region, price)
    else:
        # Normal: Search each slot in parallel
        # Prepare slots data
        slots_data = []
        for obj, emb in zip(segmented_objects, embeddings):
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
        if unique_slots:
            slot_results = finder.search.search_all_slots_parallel(
                slots=list(unique_slots.values()),
                region=region,
                price=price,
                match_count=config.results_per_slot,
                min_confidence=config.min_confidence
            )
        else:
            slot_results = {}
    
    timing['supabase_search'] = time.time() - t0
    
    # Build result
    total_time = time.time() - total_start
    timing['total'] = total_time
    
    result = PipelineResult(
        image_source=image_path,
        slots=slot_results,
        fallback_used=use_fallback,
        timing=timing,
        total_time=total_time
    )
    
    return result, segment_images


def _run_fallback_search(finder, objects, embeddings, region, price):
    """
    Fallback search when confidence is low.
    
    Strategy (per V2 spec):
    1. First try with REDUCED metadata restrictions (no region/price filter)
    2. If still insufficient, switch to pure whole-image similarity search
    """
    from supabase_search import SlotResult
    
    if not embeddings:
        return {}
    
    # Use highest confidence embedding (not just first)
    best_idx = max(range(len(objects)), key=lambda i: objects[i].confidence)
    best_embedding = embeddings[best_idx].embedding
    
    # Step 1: Try search with RELAXED metadata (no region/price)
    relaxed_results = finder.search.search_by_embedding(
        embedding=best_embedding,
        application=None,  # No application filter
        region=None,       # No region filter  
        price=None,        # No price filter
        match_count=12,
        similarity_threshold=0.3
    )
    
    if len(relaxed_results) >= 6:
        # Got enough results with relaxed filters
        return {
            'fallback': SlotResult(
                slot_name='Similar Products',
                confidence=objects[best_idx].confidence,
                results=relaxed_results
            )
        }
    
    # Step 2: Pure whole-image similarity search (lowest threshold)
    results = finder.search.search_fallback(
        embedding=best_embedding,
        match_count=12
    )
    
    return {
        'fallback': SlotResult(
            slot_name='Similar Products',
            confidence=0.0,
            results=results
        )
    }


def main():
    st.title("🏠 Product Finder V2")
    st.markdown("*Upload a room image and discover matching products automatically*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Filters
        st.subheader("🔍 Filters")
        
        region_options = st.multiselect(
            "🌍 Region",
            ["US", "EU", "UK", "APAC"],
            default=[]
        )
        region = region_options if region_options else None
        
        price = st.selectbox(
            "💰 Price Tier",
            [None, 1, 2, 3, 4, 5],
            format_func=lambda x: "💰 Any Price" if x is None else f"💰 Tier {x}"
        )
        
        st.divider()
        
        # Info
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **Product Finder V2** uses AI to automatically:
            1. 🎯 **Detect segments** in your image (Modal SAM3)
            2. 🧠 **Generate embeddings** for each segment (Voyage AI)
            3. 🔎 **Find matching products** from our library (pgvector)
            
            ⚡ Processing takes ~5-8 seconds.
            """)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Image")
        
        image = None
        image_source = None
        
        # Drag and drop file uploader
        uploaded_file = st.file_uploader(
            "Drag and drop an image here",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a room or interior design image",
            key="image_upload"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp, format="JPEG", quality=90)
                image_source = tmp.name
        
        # OR use URL
        with st.expander("Or paste image URL"):
            image_url = st.text_input(
                "Image URL",
                placeholder="https://example.com/room.jpg",
                label_visibility="collapsed"
            )
            if image_url and not uploaded_file:
                try:
                    response = requests.get(image_url, timeout=15)
                    image = Image.open(io.BytesIO(response.content))
                    image_source = image_url
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
        
        if image:
            st.image(image, caption="Input Image", width="stretch")
            
            # Process button
            if st.button("🚀 Find Products", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing... AI is analyzing your image (~5-8s)"):
                    try:
                        start_time = time.time()
                        
                        result, segment_images = run_pipeline_local(
                            image_path=image_source,
                            region=region,
                            price=price
                        )
                        result_dict = result.to_dict()
                        
                        elapsed = time.time() - start_time
                        
                        # Store in session state
                        st.session_state['result'] = result_dict
                        st.session_state['elapsed'] = elapsed
                        st.session_state['segments'] = segment_images
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.subheader("Results")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            elapsed = st.session_state.get('elapsed', 0)
            
            # Timing info
            timing = result.get('timing', {})
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            with col_t1:
                st.metric("Total Time", f"{elapsed:.1f}s")
            with col_t2:
                st.metric("SAM3", f"{timing.get('sam3_segmentation', 0):.1f}s")
            with col_t3:
                st.metric("Embeddings", f"{timing.get('voyage_embeddings', 0):.1f}s")
            with col_t4:
                st.metric("Search", f"{timing.get('supabase_search', 0):.1f}s")
            
            if result.get('fallback_used'):
                st.warning("Low confidence - showing fallback results")
            
            st.divider()
            
            # Get segment images
            segments = st.session_state.get('segments', {})
            
            # Display slots
            slots_data = result.get('slots', {})
            
            if not slots_data:
                st.info("No products found. Try different applications or a clearer image.")
            else:
                for slot_name, slot_data in slots_data.items():
                    if isinstance(slot_data, dict):
                        confidence = slot_data.get('confidence', 0)
                        results = slot_data.get('results', [])
                        
                        # Slot header
                        st.markdown(f"""
                        <div class="slot-header">
                            <strong>{slot_name.upper()}</strong>
                            <span class="confidence-badge">Confidence: {confidence:.0%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show segment image and products side by side
                        seg_col, prod_col = st.columns([1, 3])
                        
                        with seg_col:
                            st.caption("Detected Segment")
                            # Find segment image for this slot
                            segment_key = slot_name.lower()
                            if segment_key in segments:
                                seg_data = segments[segment_key]
                                # Display base64 image
                                st.image(seg_data['base64'], width="stretch")
                            else:
                                st.info("No segment")
                        
                        with prod_col:
                            st.caption("Similar Products")
                            if results:
                                # Display products in grid (5 per row)
                                COLS_PER_ROW = 5
                                for row_start in range(0, len(results), COLS_PER_ROW):
                                    row_items = results[row_start:row_start + COLS_PER_ROW]
                                    cols = st.columns(COLS_PER_ROW)
                                    for i, product in enumerate(row_items):
                                        with cols[i]:
                                            with st.container():
                                                # Thumbnail
                                                thumb_url = product.get('thumbnail_url')
                                                if thumb_url:
                                                    try:
                                                        st.image(thumb_url, width="stretch")
                                                    except:
                                                        st.image("https://via.placeholder.com/150?text=No+Image", width="stretch")
                                                else:
                                                    st.image("https://via.placeholder.com/150?text=No+Image", width="stretch")
                                                
                                                # Product info
                                                st.markdown(f"**{product.get('name', 'Unknown')}**")
                                                st.caption(f"Supplier: {product.get('supplier', 'N/A')}")
                                                st.caption(f"Type: {product.get('product_type', 'N/A')}")
                                                
                                                similarity = product.get('similarity', 0)
                                                st.progress(similarity, text=f"Match: {similarity:.0%}")
                            else:
                                st.info(f"No products found for {slot_name}")
                        
                        st.divider()
            
            # Raw JSON expander
            with st.expander("View Raw JSON"):
                st.json(result)
        else:
            st.info("Upload an image and click 'Find Products' to see results")


if __name__ == "__main__":
    main()
