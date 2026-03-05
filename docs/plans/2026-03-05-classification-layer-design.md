# Classification Layer Design — Proto C

## Goal

Add product identification to the existing detection pipeline. After detecting where products are on a shelf, classify *what* each product is by matching against a reference database of known SKUs.

## Approach

Embedding model + nearest-neighbor lookup (Approach A from `docs/knowledge/classification-approaches.md`). No retraining needed when products are added or removed.

## Architecture

```
Detection output (bounding boxes)
    → Crop each box from original image
    → Resize crop to 224×224
    → Run CLIP ViT-B/32 embedding model → feature vector
    → Cosine similarity against reference embeddings
    → Return top match + confidence per detection
```

## Server Components

### Embedding Model
- **Model**: CLIP ViT-B/32 (ONNX)
- **Runtime**: Same `ort` crate used for detection
- **Input**: 224×224 RGB image crop, normalized with CLIP preprocessing
- **Output**: 512-dim feature vector (L2-normalized)

### Reference Store
- JSON file on disk (`products.json`) loaded into memory at startup
- Maps SKU name → averaged embedding vector(s)
- Can migrate to SQLite later if catalog grows large

### Matching
- Cosine similarity between crop embedding and all reference embeddings
- Top-1 match returned
- If best similarity < 0.6 threshold → `"sku": null` (unknown product)

## API Changes

### POST /analyze — Updated Response

Each detection gains optional classification fields:

```json
{
  "label": "object",
  "confidence": 0.92,
  "bbox": { "x": 0.15, "y": 0.10, "width": 0.08, "height": 0.15 },
  "sku": "apu-milk-1l",
  "sku_confidence": 0.87
}
```

- `sku`: matched product name, or `null` if unknown
- `sku_confidence`: cosine similarity score (0–1)

### POST /products — New Endpoint

Register a new product with reference images.

```
POST /products (multipart)
  name: "APU Milk 1L"
  images: [img1.jpg, img2.jpg, ...]
```

Server embeds each image, averages the vectors, stores in `products.json`.

### DELETE /products/:name — New Endpoint

Remove a product from the reference database.

## Error Handling

- **Unknown product**: Best similarity below threshold → `"sku": null`
- **Empty reference store**: Classification skipped, detection-only results returned (backwards compatible)
- **No embedding model configured**: Server works exactly as today — detection only
- **Classification is optional**: Existing detection pipeline is untouched

## Data Requirements

- 1–5 reference images per product
- Images can be catalog photos or phone photos of individual products
- No training/fine-tuning required to add products

## Client Changes

- Results screen shows SKU name on each bounding box (when available)
- Unknown products show detection box without label (same as today)
