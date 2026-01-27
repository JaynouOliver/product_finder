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


def load_applications():
    """Load application types from JSON config."""
    config_path = Path(__file__).parent / "applications.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # Fallback defaults
    return [
        {"app": "Floors", "default": True},
        {"app": "Walls", "default": True},
        {"app": "Worktop / Surface", "default": True},
        {"app": "Ceilings", "default": False},
        {"app": "Upholstery", "default": False},
    ]


def run_pipeline_local(image_path: str, slots: list, region: list = None, price: int = None):
    """Run the pipeline locally. Returns result and segment images."""
    from pipeline_v2 import ProductFinderV2, PipelineConfig
    
    config = PipelineConfig(
        default_slots=slots,
        results_per_slot=3,
        region=region,
        price=price
    )
    finder = ProductFinderV2(config)
    
    # Run segmentation first to capture segment images
    segmented_objects = finder.segmenter.segment_image(
        image_source=image_path,
        prompts=slots,
        use_cache=True,
        save_outputs=False
    )
    
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
    
    # Run full pipeline
    result = finder.run(image_path, slots, region, price)
    
    return result, segment_images


def main():
    st.title("Product Finder V2")
    st.markdown("Find real materials from concept room images")
    
    # Load application types
    applications = load_applications()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.divider()
        
        # Application/Slot selection with checkboxes
        st.subheader("Applications to Detect")
        st.caption("Select surface types or add custom ones")
        
        selected_apps = []
        
        # Create checkboxes for each application
        for app_config in applications:
            app_name = app_config["app"].strip()
            is_default = app_config.get("default", False)
            
            if st.checkbox(app_name, value=is_default, key=f"app_{app_name}"):
                selected_apps.append(app_name)
        
        # Custom applications input
        st.divider()
        custom_apps = st.text_input(
            "Custom Applications",
            placeholder="e.g., Sofa, Carpet, Rug",
            help="Add custom surface types (comma-separated)"
        )
        if custom_apps:
            custom_list = [app.strip() for app in custom_apps.split(",") if app.strip()]
            selected_apps.extend(custom_list)
        
        if selected_apps:
            st.caption(f"Selected: {', '.join(selected_apps)}")
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        
        region_options = st.multiselect(
            "Region",
            ["US", "EU", "UK", "APAC"],
            default=[]
        )
        region = region_options if region_options else None
        
        price = st.selectbox(
            "Price Tier",
            [None, 1, 2, 3, 4, 5],
            format_func=lambda x: "Any" if x is None else f"Tier {x}"
        )
        
        st.divider()
        
        # Info
        with st.expander("About"):
            st.markdown("""
            **Product Finder V2** uses AI to:
            1. Segment your room image (SAM3)
            2. Generate visual embeddings (Voyage AI)
            3. Find matching products (pgvector)
            
            Processing takes ~10-15 seconds.
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
            st.image(image, caption="Input Image", use_container_width=True)
            
            # Process button
            if st.button("Find Products", type="primary", use_container_width=True):
                if not selected_apps:
                    st.error("Please select at least one application type")
                else:
                    with st.spinner("Processing... This may take 10-15 seconds"):
                        try:
                            start_time = time.time()
                            
                            result, segment_images = run_pipeline_local(
                                image_path=image_source,
                                slots=selected_apps,
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
                                st.image(seg_data['base64'], use_container_width=True)
                            else:
                                st.info("No segment")
                        
                        with prod_col:
                            st.caption("Similar Products")
                            if results:
                                # Display products in columns
                                cols = st.columns(min(len(results), 3))
                                for i, product in enumerate(results[:3]):
                                    with cols[i]:
                                        with st.container():
                                            # Thumbnail
                                            thumb_url = product.get('thumbnail_url')
                                            if thumb_url:
                                                try:
                                                    st.image(thumb_url, use_container_width=True)
                                                except:
                                                    st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
                                            else:
                                                st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
                                            
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
