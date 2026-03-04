# Prototype A: React Native + On-device ML

## Overview

React Native app (Expo with dev client) that captures shelf photos and runs YOLOv8 object detection on-device using TensorFlow Lite.

## Tech Stack

- **Framework:** React Native with Expo (SDK 55, dev client — not Expo Go)
- **Language:** TypeScript
- **Camera:** `react-native-vision-camera` v4.x with frame processors
- **Frame Processors:** `react-native-worklets-core` (required for VisionCamera)
- **ML Runtime:** `react-native-fast-tflite` v2.x (C++/JSI)
- **Image Processing:** `expo-image-manipulator` (native resize), `jpeg-js` (JS JPEG decode)
- **File System:** `expo-file-system` (new File API)
- **Model:** YOLOv8m fine-tuned on SKU-110K (~99MB float32 TFLite)

## Architecture

Photo-based inference pipeline:
1. VisionCamera captures a JPEG photo via `takePhoto()`
2. `expo-image-manipulator` natively resizes to 640×640 (handles EXIF rotation)
3. `jpeg-js` decodes the small JPEG to RGBA pixels in JS
4. Pixels normalized to float32 0–1, fed to TFLite model
5. YOLO postprocessing (confidence filter + NMS) in JS
6. Results displayed with bounding box overlay

The frame processor is configured but idle — VisionCamera requires it for camera session setup on Android. Actual inference runs on the captured photo to ensure bounding boxes match the displayed image exactly.

## Data Flow

```
Capture photo (VisionCamera)
    → Native resize to 640×640 (expo-image-manipulator)
    → Decode small JPEG to RGBA (jpeg-js)
    → Normalize to float32 RGB 0–1
    → TFLite inference (react-native-fast-tflite)
    → YOLO postprocessing (confidence filter + NMS)
    → Bounding boxes + count → Results screen
```

## Project Structure

```
proto-a-rn/
├── app/                          # Expo Router screens
│   ├── _layout.tsx               # Root layout
│   └── (tabs)/
│       ├── _layout.tsx           # Tab bar (Camera + Results)
│       ├── index.tsx             # Camera screen (capture)
│       └── results.tsx           # Results display (photo + boxes + count)
├── components/
│   └── BoundingBoxOverlay.tsx    # Green box overlay on detected objects
├── lib/
│   ├── frameStore.ts             # In-memory result passing between screens
│   ├── hooks/
│   │   └── useCamera.ts          # Camera setup, frame processor, takePhoto
│   └── ml/
│       ├── photoInference.ts     # Photo-based inference pipeline
│       ├── postprocess.ts        # YOLO NMS, IoU, confidence filtering
│       ├── preprocess.ts         # Resize + normalize (with rotation variant)
│       ├── types.ts              # Detection, BoundingBox, DetectionResult
│       └── __tests__/
│           └── postprocess.test.ts  # 27 unit tests
├── assets/
│   └── models/
│       └── yolov8m-sku110k.tflite   # ~99MB (git-ignored)
├── app.json
├── package.json
└── tsconfig.json
```

## Key Implementation Details

- **EXIF rotation:** Android stores portrait photos as landscape JPEG with EXIF tag. `expo-image-manipulator` handles this natively during resize, so the model sees the correct portrait orientation.
- **VisionCamera config:** Requires `pixelFormat="rgb"` + `frameProcessor` props on the Camera component. Removing these causes `session/invalid-output-configuration` on Android.
- **Model output:** `[1, 5, 8400]` row-major — coordinates are already normalized 0–1 (Ultralytics TFLite export bakes normalization into the graph).

## Performance (Android, mid-range phone)

- Preprocess (native resize + JS decode): ~200ms
- Inference: ~1.3s
- Total pipeline: ~2s

## Acceptance Criteria

- [x] Expo dev client builds and runs on Android device
- [x] Camera viewfinder renders with live preview
- [x] Photo capture works
- [x] TFLite YOLOv8m model loads and runs inference
- [x] Bounding boxes render correctly over detected products
- [x] Item count and timing display on results screen
- [x] Postprocessing unit tests pass (27 tests)
