# Prototype B: Flutter + On-device ML

## Overview

Flutter app using `tflite_flutter` for on-device YOLOv8 object detection on shelf images.

## Tech Stack

- **Framework:** Flutter
- **Language:** Dart
- **Camera:** `camera` package
- **ML Runtime:** `tflite_flutter` with custom TFLite model
- **Image Processing:** `image` package (JPEG decode + resize)
- **Model:** YOLOv8m fine-tuned on SKU-110K (~99MB float32 TFLite)

## Architecture

Photo-based inference pipeline:
1. Camera captures a JPEG photo via `takePicture()`
2. `image` package decodes JPEG and resizes to 640×640
3. Pixels normalized to float32 0–1, fed to TFLite interpreter
4. YOLO postprocessing (confidence filter + NMS) in Dart
5. Results displayed with custom-painted bounding box overlay

## Data Flow

```
Capture photo (camera package)
    → Decode JPEG + resize to 640×640 (image package)
    → Normalize to float32 RGB 0–1
    → TFLite inference (tflite_flutter)
    → YOLO postprocessing (confidence filter + NMS)
    → Bounding boxes + count → Results screen
```

## Project Structure

```
proto-b-flutter/
├── lib/
│   ├── main.dart                      # App entry + tab layout
│   ├── models/
│   │   └── detection.dart             # Detection, BoundingBox types
│   ├── screens/
│   │   ├── home_screen.dart           # Tab controller (Camera + Results)
│   │   ├── camera_screen.dart         # Camera viewfinder + capture
│   │   └── results_screen.dart        # Photo + boxes + count display
│   ├── services/
│   │   ├── ml_service.dart            # TFLite model loading + inference
│   │   ├── postprocess.dart           # YOLO NMS, IoU, confidence filtering
│   │   ├── labels.dart                # Label definitions
│   │   └── storage_service.dart       # Local result storage
│   └── widgets/
│       └── bounding_box_overlay.dart   # CustomPainter for green boxes
├── test/
│   └── services/
│       └── postprocess_test.dart       # 20 unit tests
├── assets/
│   └── models/
│       └── yolov8m-sku110k.tflite     # ~99MB (git-ignored)
├── pubspec.yaml
└── analysis_options.yaml
```

## Key Implementation Details

- **TFLite output buffer:** Must match exact tensor shape as nested lists (`List<List<List<double>>>` for `[1, 5, 8400]`), not flat arrays. Inner list references may be replaced by `copyTo`, so always read from the outputs map after inference.
- **Coordinate space:** Model output coordinates are already normalized 0–1 (Ultralytics TFLite export). No division by input size needed.
- **EXIF rotation:** Flutter's `image` package and camera plugin handle rotation natively — photos arrive in the correct portrait orientation.

## Performance (Android, mid-range phone)

- Preprocess (decode + resize): ~500ms
- Inference: ~3s
- Total pipeline: ~3.5s

## Acceptance Criteria

- [x] Flutter app builds and runs on Android device
- [x] Camera viewfinder renders with live preview
- [x] Photo capture works
- [x] TFLite YOLOv8m model loads and runs inference
- [x] Bounding boxes render correctly over detected products
- [x] Item count and timing display on results screen
- [x] Postprocessing unit tests pass (20 tests)
