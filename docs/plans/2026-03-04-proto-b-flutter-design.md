# Proto B: Flutter + tflite_flutter — Design

## Decision Summary

| Decision | Choice |
|----------|--------|
| ML runtime | `tflite_flutter` ^0.11.0 via FFI |
| Placeholder model | EfficientDet-Lite0 (same as Proto A) |
| Image preprocessing | `image` package (decode + resize to 320×320 RGB) |
| State management | None — StatefulWidget + navigator arguments |
| Bounding box rendering | `CustomPainter` (single paint pass) |
| Platforms | iOS + Android from the start |
| Scope | Full parity with Proto A acceptance criteria |

## Project Structure

```
proto-b-flutter/
├── lib/
│   ├── main.dart
│   ├── screens/
│   │   ├── camera_screen.dart      # Camera viewfinder + capture button
│   │   └── results_screen.dart     # Photo + bounding boxes + count
│   ├── services/
│   │   ├── ml_service.dart         # TFLite model load + inference
│   │   └── storage_service.dart    # In-memory audit history
│   └── widgets/
│       └── bounding_box_overlay.dart
├── assets/
│   └── models/
│       └── efficientdet-lite0.tflite
├── pubspec.yaml
└── analysis_options.yaml
```

## Dependencies

```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.11.0+2
  tflite_flutter: ^0.11.0
  image: ^4.3.0
  path_provider: ^2.1.0
```

No state management library. Two screens, plain widgets, navigator args.

## Data Flow

```
Camera (camera package)
    → User taps capture
    → Save photo to temp file
    → Load image as bytes (image package)
    → Resize to 320×320, convert to RGB Uint8List
    → Feed into tflite_flutter interpreter
    → Parse EfficientDet-Lite0 outputs (4 tensors: boxes, classes, scores, count)
    → Filter by confidence threshold (0.4)
    → Navigate to results screen with detections + photo path
```

No frame processor / worklet bridge needed. Flutter captures a photo, processes it synchronously via FFI. Simpler than Proto A's worklet→JS thread architecture.

## ML Service API

```dart
class MlService {
  Interpreter? _interpreter;

  Future<void> loadModel();
  DetectionResult runInference(Uint8List imageBytes, int imageWidth, int imageHeight);
  void dispose();
}

class DetectionResult {
  final List<Detection> detections;
  final int count;
  final double inferenceTimeMs;
}

class Detection {
  final String label;
  final double confidence;
  final BoundingBox bbox; // x, y, width, height normalized 0–1
}
```

Preprocessing (resize, RGB extraction) is encapsulated inside `runInference()`. Swapping models later (EfficientDet → YOLO) only changes this file.

## Screens

**CameraScreen:** Full-screen camera preview, circular capture button, loading indicator while model loads, button disabled until ready.

**ResultsScreen:** Captured photo with green bounding boxes via `CustomPainter`, item count + inference time, "Scan Another Shelf" button.

## Key Differences from Proto A

- **No worklet bridge complexity.** `tflite_flutter` uses FFI — tensor data stays in native memory.
- **`CustomPainter` for bounding boxes** instead of absolutely positioned View elements. More performant for dense detections (100+ boxes in SKU-110K).
- **`image` package for preprocessing** instead of manual nearest-neighbor resize loop.
- **Navigator args for data passing** instead of global singleton store.
