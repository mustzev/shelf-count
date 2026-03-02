# Prototype A: React Native + On-device ML

## Overview

React Native app (Expo managed workflow) that captures shelf images and runs object detection on-device using TensorFlow Lite.

## Tech Stack

- **Framework:** React Native with Expo
- **Language:** TypeScript
- **Camera:** `react-native-vision-camera` v4.x with frame processors
- **Frame Processors:** `react-native-worklets-core` (required for VisionCamera frame processors)
- **ML Runtime:** `react-native-fast-tflite` v2.x (C++/JSI, GPU-accelerated)
- **Model:** Pre-trained EfficientDet-Lite0 (~4.4MB, COCO 80 classes) in TFLite format
- **Note:** Requires Expo development build (not Expo Go) due to native dependencies

## Requirements

### Functional
- Camera viewfinder fills the screen
- Tap to capture a shelf image
- Run TFLite object detection on the captured image
- Display bounding boxes over detected products
- Show total count of detected items
- Save results locally (image + count + bounding box data)

### Non-functional
- Inference time < 3 seconds on a mid-range phone
- Works fully offline (no network required for inference)
- App size < 100MB (including bundled ML model)

## Data Flow

```
Camera Frame
    → Image preprocessing (resize, normalize)
    → TFLite model inference
    → Post-processing (NMS, confidence threshold)
    → Bounding boxes + class labels + confidence scores
    → UI overlay + count display
```

## Project Structure

```
proto-a-rn/
├── app/                  # Expo Router screens
│   ├── index.tsx         # Camera/capture screen
│   └── results.tsx       # Results display screen
├── components/           # Reusable UI components
├── lib/
│   ├── ml/               # ML inference wrapper
│   └── storage/          # Local result storage
├── assets/
│   └── models/           # TFLite model files
├── app.json
├── package.json
└── tsconfig.json
```

## Acceptance Criteria

- [ ] Expo project builds and runs on iOS simulator / Android emulator
- [ ] Camera viewfinder renders with live preview
- [ ] Image capture saves a photo locally
- [ ] TFLite model loads and runs inference on a test image
- [ ] Bounding boxes render over detected objects
- [ ] Item count displays on results screen
