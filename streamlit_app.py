"""
Product Finder -- Streamlit Test UI

Calls the deployed Modal API and displays results with:
- Segment polygon overlays on the original image
- Latency breakdown
- Product matches per segment
"""

import streamlit as st
import requests
import json
import base64
import time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# --------------- Config ---------------

API_ENDPOINTS = {
    "Modal v3 (app-aware)": "https://mattoboard--product-finder-v3-fastapi-app.modal.run",
    "Modal v2 (deployed)": "https://mattoboard--product-finder-v2-fastapi-app.modal.run",
    "Modal v1 (legacy)": "https://mattoboard--product-finder-api-fastapi-app.modal.run",
    "Local (localhost:8000)": "http://localhost:8000",
}

# Distinct colors for segment overlays (up to 20)
SEGMENT_COLORS = [
    (59, 130, 246),   # blue
    (139, 92, 246),   # purple
    (236, 72, 153),   # pink
    (245, 158, 11),   # amber
    (16, 185, 129),   # emerald
    (6, 182, 212),    # cyan
    (244, 63, 94),    # rose
    (168, 85, 247),   # violet
    (34, 197, 94),    # green
    (251, 146, 60),   # orange
    (99, 102, 241),   # indigo
    (20, 184, 166),   # teal
    (248, 113, 113),  # red-light
    (96, 165, 250),   # blue-light
    (74, 222, 128),   # green-light
    (253, 186, 116),  # orange-light
    (192, 132, 252),  # purple-light
    (103, 232, 249),  # cyan-light
    (252, 165, 165),  # red-pale
    (147, 197, 253),  # blue-pale
]


def draw_segments_overlay(img: Image.Image, segments: list, highlight_idx: int = None) -> Image.Image:
    """
    Draw polygon overlays on the image for each segment.

    Args:
        img: PIL Image (original)
        segments: List of segment dicts with polygon_flat, bbox, label, score
        highlight_idx: Index of segment to highlight (brighter fill). None = all equal.

    Returns:
        New PIL Image with overlays drawn.
    """
    overlay = img.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_sm = font

    for i, seg in enumerate(segments):
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        pf = seg.get("polygon_flat", [])
        if len(pf) < 6:
            continue

        polygon = [(pf[j], pf[j + 1]) for j in range(0, len(pf), 2)]
        is_highlighted = highlight_idx is not None and i == highlight_idx
        fill_alpha = 80 if is_highlighted else 35
        outline_width = 3 if is_highlighted else 2

        # Fill polygon
        fill_color = (*color, fill_alpha)
        draw.polygon(polygon, fill=fill_color)

        # Outline
        outline_color = (*color, 220)
        draw.polygon(polygon, outline=outline_color)
        # Draw thicker outline by offsetting
        if outline_width > 1:
            draw.line(polygon + [polygon[0]], fill=outline_color, width=outline_width)

        # Label badge at top of bbox
        bbox = seg.get("bbox", [0, 0, 0, 0])
        x, y = bbox[0], bbox[1]
        label = seg.get("label", "")
        score = seg.get("score", 0)
        badge_text = f" {label} ({score:.0%}) "

        # Measure text
        text_bbox = draw.textbbox((0, 0), badge_text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        # Badge background
        badge_x = max(0, x)
        badge_y = max(0, y - th - 8)
        draw.rectangle(
            [badge_x, badge_y, badge_x + tw + 8, badge_y + th + 6],
            fill=(*color, 200),
        )
        draw.text((badge_x + 4, badge_y + 2), badge_text, fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(overlay, draw_layer)
    return result.convert("RGB")


# --------------- Page Setup ---------------

st.set_page_config(
    page_title="Product Finder",
    page_icon="::mag::",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stExpander"] { border: 1px solid #262730; border-radius: 8px; }
    .seg-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        color: #fff;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --------------- Sidebar ---------------

with st.sidebar:
    st.title("Product Finder")
    st.caption("SAM3 + Voyage + HNSW Pipeline")

    st.divider()

    endpoint_name = st.selectbox("API Endpoint", list(API_ENDPOINTS.keys()))
    api_base = API_ENDPOINTS[endpoint_name]

    st.divider()

    similarity_score = st.slider("Min Similarity", 0.0, 1.0, 0.3, 0.05)
    limit = st.slider("Results per Segment", 1, 20, 5)

    region_filter = st.selectbox("Region Filter", ["None", "USCA", "EU", "APAC"])
    filters = None
    if region_filter != "None":
        filters = {"regionServed": [region_filter]}

    st.divider()
    st.markdown("**Input**")
    input_mode = st.radio("Input Type", ["Image URL", "Upload Image", "Base64"], horizontal=True)


# --------------- Main Area ---------------

st.markdown("## Product Finder Pipeline")

image_url = None
image_base64 = None
preview_image = None
source_pil_image = None

if input_mode == "Image URL":
    image_url = st.text_input(
        "Image URL",
        placeholder="https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=800",
    )
    if image_url:
        preview_image = image_url

elif input_mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        image_bytes = uploaded.read()
        image_base64 = base64.b64encode(image_bytes).decode()
        preview_image = image_bytes
        source_pil_image = Image.open(BytesIO(image_bytes))

elif input_mode == "Base64":
    raw_b64 = st.text_area("Paste base64 string", height=120, placeholder="iVBORw0KGgo...")
    if raw_b64:
        cleaned = raw_b64.strip()
        if cleaned.startswith("data:"):
            cleaned = cleaned.split(",", 1)[1]
        image_base64 = cleaned
        try:
            img_bytes = base64.b64decode(cleaned)
            preview_image = img_bytes
            source_pil_image = Image.open(BytesIO(img_bytes))
        except Exception:
            st.error("Invalid base64 string")
            preview_image = None


# --------------- Preview + Run ---------------

col_preview, col_run = st.columns([2, 1])

with col_preview:
    if preview_image:
        st.image(preview_image, caption="Input Image", width=600)

with col_run:
    can_run = image_url or image_base64
    run_button = st.button(
        "Run Pipeline",
        type="primary",
        disabled=not can_run,
    )
    if not can_run:
        st.caption("Provide an image URL, upload, or base64 to start.")


# --------------- API Call + Results ---------------

if run_button and can_run:
    payload = {
        "similarity_score": similarity_score,
        "limit": limit,
    }
    if image_url:
        payload["image_url"] = image_url
    elif image_base64:
        payload["image_base64"] = image_base64
    if filters:
        payload["filters"] = filters

    with st.spinner("Running pipeline... SAM3 -> Extract -> Embed -> Search"):
        wall_start = time.perf_counter()
        try:
            resp = requests.post(
                f"{api_base}/process_image",
                json=payload,
                timeout=300,
            )
            wall_elapsed = round((time.perf_counter() - wall_start) * 1000)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            st.error("Request timed out (300s). The API may be cold-starting.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to {api_base}. Is the server running?")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            if 'resp' in dir() and hasattr(resp, 'text'):
                st.code(resp.text, language="json")
            st.stop()

    # ---- Latency Breakdown ----
    latency = data.get("latency_ms", {})
    segments = data.get("segments", [])

    st.divider()
    st.markdown("### Latency Breakdown")

    total_api = latency.get("total_ms", 0)
    latency_stages = [
        ("Download", latency.get("download_ms"), "#3b82f6"),
        ("SAM3 Segmentation", latency.get("sam3_ms"), "#8b5cf6"),
        ("Segment Extraction", latency.get("extract_ms"), "#06b6d4"),
        ("Voyage Embeddings", latency.get("embedding_ms"), "#f59e0b"),
        ("DB Search", latency.get("db_ms"), "#10b981"),
    ]
    active_stages = [(name, ms, color) for name, ms, color in latency_stages if ms is not None]

    metric_cols = st.columns(len(active_stages) + 2)
    metric_cols[0].metric("Total (API)", f"{total_api:,}ms")
    metric_cols[1].metric("Wall Clock", f"{wall_elapsed:,}ms")
    for i, (name, ms, _) in enumerate(active_stages):
        pct = round(ms / total_api * 100) if total_api > 0 else 0
        metric_cols[i + 2].metric(name, f"{ms:,}ms", delta=f"{pct}%", delta_color="off")

    if total_api > 0:
        bar_parts = []
        for name, ms, color in active_stages:
            pct = max(ms / total_api * 100, 2)
            bar_parts.append(
                f'<div style="width:{pct}%;background:{color};height:10px;border-radius:3px;display:inline-block;" '
                f'title="{name}: {ms:,}ms"></div>'
            )
        st.markdown(
            f'<div style="display:flex;gap:2px;margin:8px 0 4px 0;">{"".join(bar_parts)}</div>',
            unsafe_allow_html=True,
        )
        legend_parts = [
            f'<span style="color:{color};font-size:0.75rem;">&#9632; {name}</span>'
            for name, _, color in active_stages
        ]
        st.markdown(
            f'<div style="display:flex;gap:16px;margin-bottom:12px;">{"  ".join(legend_parts)}</div>',
            unsafe_allow_html=True,
        )

    # ---- Segment Overlay on Original Image ----
    st.divider()
    st.markdown(f"### Segments ({len(segments)} detected)")

    if segments:
        # Load source image for overlay drawing
        if source_pil_image is None and image_url:
            try:
                img_resp = requests.get(image_url, timeout=30)
                img_resp.raise_for_status()
                source_pil_image = Image.open(BytesIO(img_resp.content))
            except Exception:
                source_pil_image = None

        if source_pil_image is not None:
            annotated = draw_segments_overlay(source_pil_image, segments)
            st.image(annotated, caption="Detected Segments", width=700)

            # Segment legend
            legend_html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 16px 0;">'
            for i, seg in enumerate(segments):
                color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
                r, g, b = color
                label = seg.get("label", "")
                score = seg.get("score", 0)
                n_matches = len(seg.get("matches", []))
                legend_html += (
                    f'<span class="seg-badge" style="background:rgba({r},{g},{b},0.85);">'
                    f'{label} ({score:.0%}) -- {n_matches} matches</span>'
                )
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.warning("Could not load source image for overlay.")

    if not segments:
        st.warning("No segments passed the score/area filter. Try lowering the similarity threshold.")
    else:
        # ---- Per-Segment Matches ----
        for seg_idx, seg in enumerate(segments):
            seg_id = seg["segment_id"]
            label = seg["label"]
            score = seg.get("score", 0)
            area = seg.get("area", 0)
            matches = seg.get("matches", [])
            top_score = matches[0]["match_score"] if matches else 0
            color = SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]
            r, g, b = color

            with st.expander(
                f"{label}  --  {len(matches)} matches  --  top: {top_score:.2%}  --  area: {area:,.0f}px",
                expanded=False,
            ):
                # Show individual segment crop overlay
                if source_pil_image is not None:
                    highlighted = draw_segments_overlay(source_pil_image, segments, highlight_idx=seg_idx)
                    # Crop to segment bbox with padding for context
                    bbox = seg.get("bbox", [0, 0, 0, 0])
                    pad = 60
                    cx1 = max(0, bbox[0] - pad)
                    cy1 = max(0, bbox[1] - pad)
                    cx2 = min(highlighted.width, bbox[0] + bbox[2] + pad)
                    cy2 = min(highlighted.height, bbox[1] + bbox[3] + pad)
                    cropped_view = highlighted.crop((cx1, cy1, cx2, cy2))
                    st.image(cropped_view, caption=f"Segment: {label}", width=500)

                matched_app = seg.get("matched_application")
                app_str = ", ".join(matched_app) if matched_app else "none (similarity only)"
                st.caption(
                    f"Segment ID: `{seg_id}` | Score: {score:.2%} | Area: {area:,.0f}px | "
                    f"App filter: **{app_str}**"
                )

                if not matches:
                    st.info("No products matched above the similarity threshold.")
                    continue

                for idx, m in enumerate(matches):
                    mscore = m["match_score"]
                    score_color = "#00d26a" if mscore >= 0.55 else ("#f5a623" if mscore >= 0.4 else "#ff4757")

                    c1, c2, c3 = st.columns([1, 3, 1])

                    with c1:
                        img_url = m.get("image_url", "")
                        is_valid_url = img_url and img_url.startswith(("http://", "https://"))
                        if is_valid_url:
                            try:
                                st.image(img_url, width=80)
                            except Exception:
                                st.markdown(
                                    '<div style="width:80px;height:80px;background:#1a1a2e;border-radius:6px;'
                                    'display:flex;align-items:center;justify-content:center;color:#808495;'
                                    'font-size:0.7rem;">Load err</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                '<div style="width:80px;height:80px;background:#1a1a2e;border-radius:6px;'
                                'display:flex;align-items:center;justify-content:center;color:#808495;'
                                'font-size:0.7rem;">No image</div>',
                                unsafe_allow_html=True,
                            )

                    with c2:
                        st.markdown(f"**{m['name']}**")
                        supplier_display = m.get('supplier') or 'N/A'
                        if supplier_display == 'No Supplier':
                            supplier_display = 'N/A'
                        st.caption(
                            f"{m['product_type']}  |  {supplier_display}  |  "
                            f"`{m['product_id']}`"
                        )

                    with c3:
                        st.markdown(
                            f'<div style="text-align:center;padding-top:8px;">'
                            f'<span style="color:{score_color};font-size:1.5rem;font-weight:700;">'
                            f'{mscore:.1%}</span></div>',
                            unsafe_allow_html=True,
                        )

                    if idx < len(matches) - 1:
                        st.markdown(
                            '<hr style="margin:4px 0;border-color:#262730;">',
                            unsafe_allow_html=True,
                        )

    # ---- Raw JSON ----
    st.divider()
    with st.expander("Raw JSON Response", expanded=False):
        st.json(data)

    with st.expander("Request Payload", expanded=False):
        display_payload = {k: v for k, v in payload.items() if k != "image_base64"}
        if "image_base64" in payload:
            display_payload["image_base64"] = f"<{len(payload['image_base64'])} chars>"
        st.json(display_payload)
