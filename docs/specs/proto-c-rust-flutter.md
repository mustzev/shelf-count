# Prototype C: Rust Axum Server + Flutter Client

## Overview

Two-part system: a Rust backend that runs ML inference on shelf images via an HTTP API, and a Flutter mobile app that captures and sends images to the server.

## Tech Stack

### Server
- **Framework:** Axum (Rust)
- **Language:** Rust
- **ML Runtime:** ONNX Runtime (`ort` crate) or `tch-rs` (PyTorch bindings)
- **Model:** YOLO or EfficientDet in ONNX format
- **Image processing:** `image` crate for preprocessing

### Client
- **Framework:** Flutter
- **Language:** Dart
- **Camera:** `camera` package
- **HTTP:** `dio` or `http` package

## Requirements

### Functional — Server
- HTTP endpoint `POST /analyze` accepts an image (multipart/form-data)
- Runs object detection model on the image
- Returns JSON with bounding boxes, labels, confidence scores, and total count
- Health check endpoint `GET /health`

### Functional — Client
- Camera viewfinder fills the screen
- Tap to capture a shelf image
- Send image to Rust server for analysis
- Display bounding boxes over detected products
- Show total count of detected items
- Loading state while waiting for server response
- Error handling for network failures

### Non-functional
- Server inference time < 2 seconds per image
- End-to-end latency (capture → display results) < 5 seconds
- Server handles concurrent requests

## API Contract

### POST /analyze

**Request:**
```
Content-Type: multipart/form-data
Body: image file (JPEG/PNG)
```

**Response:**
```json
{
  "count": 42,
  "detections": [
    {
      "label": "product",
      "confidence": 0.92,
      "bbox": {
        "x": 120,
        "y": 45,
        "width": 80,
        "height": 150
      }
    }
  ],
  "inference_time_ms": 340,
  "image_dimensions": {
    "width": 1920,
    "height": 1080
  }
}
```

### GET /health

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Data Flow

```
Flutter App                          Rust Server
──────────                          ───────────
Camera capture
    → JPEG encode
    → HTTP POST /analyze  ────────→  Receive image
                                     → Decode + preprocess
                                     → ONNX Runtime inference
                                     → Post-processing (NMS)
    ← JSON response       ←────────  → Serialize results
Display bounding boxes
    + count overlay
```

## Project Structure

```
proto-c-rust-flutter/
├── server/
│   ├── src/
│   │   ├── main.rs               # Axum server setup
│   │   ├── routes/
│   │   │   ├── mod.rs
│   │   │   ├── analyze.rs        # POST /analyze handler
│   │   │   └── health.rs         # GET /health handler
│   │   ├── ml/
│   │   │   ├── mod.rs
│   │   │   ├── model.rs          # ONNX model loading + inference
│   │   │   └── postprocess.rs    # NMS, confidence filtering
│   │   └── error.rs              # Error types
│   ├── models/                   # ONNX model files
│   ├── Cargo.toml
│   └── Dockerfile
├── client/
│   ├── lib/
│   │   ├── main.dart
│   │   ├── screens/
│   │   │   ├── camera_screen.dart
│   │   │   └── results_screen.dart
│   │   ├── services/
│   │   │   ├── api_service.dart  # HTTP client for Rust server
│   │   │   └── camera_service.dart
│   │   └── widgets/
│   │       └── bounding_box_overlay.dart
│   └── pubspec.yaml
```

## Acceptance Criteria

### Server
- [ ] Axum server starts and responds to `GET /health`
- [ ] `POST /analyze` accepts an image and returns detection JSON
- [ ] ONNX model loads at startup
- [ ] Inference runs and produces bounding boxes on a test image
- [ ] Server handles errors gracefully (bad image, model failure)

### Client
- [ ] Flutter app builds and runs on iOS simulator / Android emulator
- [ ] Camera viewfinder renders with live preview
- [ ] Image capture and upload to server works
- [ ] Results screen shows bounding boxes and count from server response
- [ ] Loading and error states display correctly
