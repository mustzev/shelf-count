# Prototype B: Flutter + On-device ML

## Overview

Flutter app using Google ML Kit for on-device object detection on shelf images.

## Tech Stack

- **Framework:** Flutter
- **Language:** Dart
- **Camera:** `camera` package
- **ML Runtime:** Google ML Kit (`google_mlkit_object_detection`) or `tflite_flutter` for custom models
- **Model:** ML Kit's built-in object detection, with option to swap in a custom TFLite model

## Requirements

### Functional
- Camera viewfinder fills the screen
- Tap to capture a shelf image
- Run ML Kit object detection on the captured image
- Display bounding boxes over detected products
- Show total count of detected items
- Save results locally (image + count + bounding box data)

### Non-functional
- Inference time < 3 seconds on a mid-range phone
- Works fully offline
- App size < 100MB

## Data Flow

```
Camera Frame
    → Image preprocessing (ML Kit handles this)
    → ML Kit object detection inference
    → DetectedObject list (bounding boxes, labels, confidence)
    → UI overlay + count display
```

## Project Structure

```
proto-b-flutter/
├── lib/
│   ├── main.dart
│   ├── screens/
│   │   ├── camera_screen.dart    # Camera viewfinder + capture
│   │   └── results_screen.dart   # Results display
│   ├── services/
│   │   ├── ml_service.dart       # ML Kit inference wrapper
│   │   └── storage_service.dart  # Local result storage
│   └── widgets/
│       └── bounding_box_overlay.dart
├── assets/
│   └── models/                   # Custom TFLite models (if needed)
├── pubspec.yaml
└── analysis_options.yaml
```

## Acceptance Criteria

- [ ] Flutter project builds and runs on iOS simulator / Android emulator
- [ ] Camera viewfinder renders with live preview
- [ ] Image capture saves a photo locally
- [ ] ML Kit model runs inference on a test image
- [ ] Bounding boxes render over detected objects
- [ ] Item count displays on results screen
