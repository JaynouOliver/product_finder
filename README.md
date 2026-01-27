# Product Finder V2 API

Find real materials and products from concept room images using AI-powered visual search.

## Algorithm Overview

**Input:** `f(image_url, refine_params)` where `refine_params = {slots, region, price}`  
**Output:** `{ "Floor": [3 products], "Wall": [3 products], ... }` (~12 total results)

### Pipeline Steps & Latency

| Step | Component | Process | Latency |
|------|-----------|---------|---------|
| 1 | **Image Download** | Download URL to temp file | ~0.5-1s |
| 2 | **SAM3 Segmentation** | Text-prompted segmentation via Roboflow | ~8-10s |
| 3 | **Voyage Embeddings** | Multimodal embeddings (voyage-multimodal-3.5) | ~1-2s |
| 4 | **Vector Search** | pgvector HNSW + metadata filters | ~0.3-0.5s |
| 5 | **Response Assembly** | Deduplicate, format response | <0.1s |
| | **Total** | | **~10-13s** |

### Flow Diagram

```
                        ┌─────────────────────────────────────┐
                        │           INPUT IMAGE               │
                        │        (URL or file path)           │
                        └─────────────────┬───────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: SAM3 SEGMENTATION (Roboflow API) - ~8-10s                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Input:  Image + prompts ["Floor", "Wall", "Worktop", "Backsplash"]         │
│  Output: RLE masks + confidence scores per detected surface                 │
│  Post:   Decode masks → Crop segments → Base64 encode                       │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              ┌──────────┐          ┌──────────┐          ┌──────────┐
              │  Floor   │          │   Wall   │          │ Worktop  │
              │ conf:0.96│          │ conf:0.89│          │ conf:0.74│
              └────┬─────┘          └────┬─────┘          └────┬─────┘
                   │                     │                     │
                   └─────────────────────┼─────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: VOYAGE EMBEDDINGS (Batch API) - ~1-2s                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Model:  voyage-multimodal-3.5 (1024 dimensions)                            │
│  Input:  Cropped images (base64) + slot name text ("A Floor")               │
│  Output: Embedding vector per segment                                       │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: VECTOR SEARCH (Supabase pgvector) - ~0.3-0.5s                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Index:   HNSW (approximate nearest neighbor)                               │
│  Query:   Over-fetch 10x → Filter metadata → Return top N                   │
│  Filters: application (Floors/Walls), region (US/EU), price (1-5)           │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  {                                                                          │
│    "floor": { "confidence": 0.96, "results": [3 products] },                │
│    "wall":  { "confidence": 0.89, "results": [3 products] },                │
│    "worktop": { "confidence": 0.74, "results": [3 products] }               │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fallback Logic

1. **Low confidence slot** (< 0.3): Hidden from response
2. **All slots low confidence** (avg < 0.5):
   - First try: Relaxed search (no metadata filters)
   - If insufficient: Whole-image similarity search (12 mixed results)

### Latency Optimization Notes

- SAM3 serverless is the bottleneck (~80% of latency)
- To achieve <5s: requires self-hosted SAM3 on GPU (Replicate/Modal)
- Current optimizations: Roboflow cache, batch embeddings, parallel search

## Quick Start

### 1. Setup

```bash
cd apps/product_finder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `VOYAGE_API_KEY` - Voyage AI API key for embeddings
- `DB_HOST`, `DB_PASSWORD` - Supabase PostgreSQL credentials

### 3. Run the Server

```bash
uvicorn server:app --reload --port 8000
```

Server will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```bash
GET /health
```

### Find Products

```bash
POST /find-products
Content-Type: application/json

{
  "image_url": "https://example.com/room-image.jpg",
  "slots": ["Floor", "Wall", "Worktop", "Backsplash"],
  "region": ["US", "EU"],
  "price": null,
  "results_per_slot": 3
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_url` | string | Yes | URL of the room image |
| `slots` | string[] | No | Surfaces to detect (default: Floor, Wall, Worktop, Backsplash) |
| `region` | string[] | No | Filter by region (e.g., US, EU, UK) |
| `price` | int | No | Price tier filter (1-5) |
| `results_per_slot` | int | No | Results per slot (default: 3, max: 10) |

**Response:**

```json
{
  "success": true,
  "image_source": "https://example.com/room-image.jpg",
  "slots": {
    "floor": {
      "slot_name": "floor",
      "confidence": 0.96,
      "results": [
        {
          "product_id": "abc123",
          "name": "Natural Oak Flooring",
          "supplier": "Supplier Co",
          "product_type": "wood",
          "thumbnail_url": "https://...",
          "similarity": 0.85
        }
      ]
    },
    "wall": { ... }
  },
  "fallback_used": false,
  "timing": {
    "sam3_segmentation": 8.5,
    "voyage_embeddings": 1.2,
    "supabase_search": 0.3,
    "total": 10.1
  },
  "total_time": 10.1
}
```

## Example Usage

### cURL

```bash
curl -X POST http://localhost:8000/find-products \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800",
    "slots": ["Floor", "Wall"],
    "results_per_slot": 3
  }'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/find-products",
    json={
        "image_url": "https://example.com/room.jpg",
        "slots": ["Floor", "Wall", "Worktop"],
        "results_per_slot": 3
    }
)
print(response.json())
```

## API Documentation

Interactive docs available when server is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

```
server.py           # FastAPI server
pipeline_v2.py      # Main pipeline orchestration
sam_segmentation.py # SAM3 image segmentation (Roboflow)
voyage_embeddings.py # Multimodal embeddings (Voyage AI)
supabase_search.py  # pgvector similarity search
image_processing.py # Image utilities (mask processing, cropping)
config.py           # Configuration and env vars
```

## Performance

Target latency breakdown:
- SAM3 Segmentation: ~8-10s (external API)
- Voyage Embeddings: ~1-2s
- Supabase Search: <0.5s
- **Total: ~10-12s**

## Fallback Behavior

If segmentation confidence is low:
1. Reduces metadata restrictions
2. Falls back to whole-image similarity search
3. Returns 12 mixed results instead of slot-based results
