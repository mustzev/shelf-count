# Proto C Design: Rust Axum Server + Flutter Client

## Summary

Two-part system: a Rust Axum server that runs ML inference via ONNX Runtime, and a Flutter client that captures shelf photos and sends them to the server. Supports multiple models (YOLOv8m, DETR) via a trait-based abstraction, one model active at a time configured at startup.

## Architecture: Trait-Based Model Abstraction

```rust
pub trait DetectionModel: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, image: image::DynamicImage) -> Result<DetectionResult, AppError>;
}
```

Each model implementation owns its `ort::Session` and handles preprocessing + postprocessing internally. The server stores the active model as `Arc<dyn DetectionModel>` in Axum state.

Model selection via CLI: `--model yolo` or `--model detr` (default: `yolo`).

## Server Structure

```
proto-c-rust-flutter/server/
├── src/
│   ├── main.rs                 # CLI arg parsing, model loading, Axum server
│   ├── error.rs                # AppError type (into Axum response)
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── analyze.rs          # POST /analyze — accepts image, returns detections
│   │   └── health.rs           # GET /health — status + which model is loaded
│   └── ml/
│       ├── mod.rs              # DetectionModel trait + factory fn
│       ├── types.rs            # Detection, BoundingBox, DetectionResult (serde)
│       ├── yolo.rs             # YoloModel impl
│       └── detr.rs             # DetrModel impl
├── models/                     # .onnx files (git-ignored)
├── Cargo.toml
└── Dockerfile
```

## API Contract

### POST /analyze

Request: `multipart/form-data` with image file (JPEG).

Response:
```json
{
  "model": "yolov8m-sku110k",
  "count": 42,
  "detections": [
    {
      "label": "object",
      "confidence": 0.92,
      "bbox": { "x": 0.15, "y": 0.10, "width": 0.08, "height": 0.15 }
    }
  ],
  "inference_time_ms": 340
}
```

### GET /health

Response:
```json
{
  "status": "ok",
  "model": "yolov8m-sku110k"
}
```

Bounding box coordinates normalized 0–1.

## Model Preprocessing & Postprocessing

### YOLOv8m

- **Preprocess:** Resize to 640×640, normalize 0–1 (`pixel / 255.0`), NCHW `[1, 3, 640, 640]`
- **Postprocess:** Output `[1, 5, 8400]` row-major → confidence filter (0.25) → sort → NMS (IoU 0.5) → clamp 0–1

### DETR (SKU-110K fine-tuned)

- **Preprocess:** Resize shortest side to 800 (max 1333), ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`), NCHW
- **Postprocess:** 100 queries → softmax on class logits → threshold filter → no NMS (transformer handles suppression)
- **Note:** Exact output shape depends on the specific fine-tuned model. Inspect ONNX tensors at load time.

## Client Structure

```
proto-c-rust-flutter/client/
├── lib/
│   ├── main.dart
│   ├── config.dart               # Server URL config
│   ├── models/
│   │   └── detection.dart        # Detection types (from JSON)
│   ├── screens/
│   │   ├── camera_screen.dart
│   │   └── results_screen.dart
│   ├── services/
│   │   └── api_service.dart      # HTTP: POST /analyze, GET /health
│   └── widgets/
│       └── bounding_box_overlay.dart
├── pubspec.yaml
└── analysis_options.yaml
```

### Client Flow

1. Camera captures photo
2. `ApiService.analyze(imageBytes)` sends multipart POST
3. Parse JSON → `DetectionResult` (includes `model` field)
4. Results screen: photo + bounding box overlay + count + model name
5. Loading spinner during server request
6. Error state for network failures

### Client Dependencies

`camera`, `http`, standard Flutter.

## Error Handling

### Server

- Bad/missing image → `400` with error JSON
- Inference failure → `500` with error JSON
- Model file not found at startup → exit with clear message

### Client

- Network timeout (10s) → "Server unreachable" with retry
- Non-200 → parse and display error
- Health check on app start for connection status

## Server Dependencies

`axum`, `ort` (ONNX Runtime), `image`, `ndarray`, `serde`, `serde_json`, `clap`, `tokio`, `tower-http` (CORS/logging).

## Environment

- **Dev:** Local Mac, CPU inference, phone on same network
- **Later:** Containerize for Linux GPU server with CUDA execution provider
