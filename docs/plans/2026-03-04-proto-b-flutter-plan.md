# Proto B: Flutter + tflite_flutter — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Flutter app that captures shelf photos and runs on-device TFLite object detection, matching Proto A's functionality.

**Architecture:** Two-screen Flutter app (Camera → Results). `tflite_flutter` runs EfficientDet-Lite0 via FFI. `image` package handles preprocessing. No state management library — plain StatefulWidgets with navigator arguments.

**Tech Stack:** Flutter, Dart, `tflite_flutter` ^0.11.0, `camera` ^0.11.0, `image` ^4.3.0, `path_provider` ^2.1.0

---

## Task 0: Install Flutter SDK

**Prerequisite — do this before anything else.**

**Step 1: Install Flutter via Homebrew**

Run:
```bash
brew install --cask flutter
```

**Step 2: Verify installation**

Run:
```bash
flutter doctor
```

Expected: Flutter SDK found. There will likely be warnings about Xcode/Android Studio — that's fine for now. The key line is `Flutter • Channel stable`.

**Step 3: Accept Android licenses (if Android Studio installed)**

Run:
```bash
flutter doctor --android-licenses
```

**Step 4: Verify Xcode setup (for iOS)**

Run:
```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
```

**Step 5: Run flutter doctor again and confirm no blocking issues**

Run:
```bash
flutter doctor -v
```

Expected: At minimum, Flutter SDK and one platform (iOS or Android) should show green checkmarks.

---

## Task 1: Scaffold Flutter Project

**Files:**
- Create: `proto-b-flutter/` (entire project scaffold)

**Step 1: Create Flutter project**

Run from repo root:
```bash
cd /Users/erchis/Projects/shelf-count
flutter create --org com.shelfcount proto-b-flutter
```

**Step 2: Verify it builds**

Run:
```bash
cd proto-b-flutter
flutter run --no-hot-reload
```

Expected: Default Flutter counter app launches in simulator/emulator. Kill it with `q`.

**Step 3: Clean up default code**

Replace `lib/main.dart` with minimal shell:

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(const ShelfCountApp());
}

class ShelfCountApp extends StatelessWidget {
  const ShelfCountApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shelf Count',
      theme: ThemeData.dark(),
      home: const Scaffold(
        body: Center(
          child: Text('Shelf Count — Proto B'),
        ),
      ),
    );
  }
}
```

**Step 4: Add dependencies to pubspec.yaml**

In `pubspec.yaml`, replace the `dependencies` section:

```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.11.0+2
  tflite_flutter: ^0.11.0
  image: ^4.3.0
  path_provider: ^2.1.0
```

And add the assets section under `flutter:`:

```yaml
flutter:
  uses-material-design: true
  assets:
    - assets/models/
```

**Step 5: Create assets directory and copy model**

Run:
```bash
mkdir -p assets/models
cp ../proto-a-rn/assets/models/efficientdet-lite0-v2.tflite assets/models/efficientdet-lite0.tflite
```

If Proto A's model file doesn't exist at that path, download EfficientDet-Lite0 from TFLite model zoo:
```bash
curl -L -o assets/models/efficientdet-lite0.tflite \
  "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1.tflite"
```

**Step 6: Configure platform permissions**

**iOS** — add to `ios/Runner/Info.plist` inside the top `<dict>`:

```xml
<key>NSCameraUsageDescription</key>
<string>Camera access is needed to photograph store shelves for product counting.</string>
```

**Android** — add to `android/app/src/main/AndroidManifest.xml` inside `<manifest>` (before `<application>`):

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

**Step 7: Install dependencies**

Run:
```bash
flutter pub get
```

**Step 8: Verify it still builds**

Run:
```bash
flutter run --no-hot-reload
```

Expected: App launches showing "Shelf Count — Proto B" text.

**Step 9: Commit**

```
feat(proto-b): scaffold Flutter project with dependencies
```

---

## Task 2: Data Models

**Files:**
- Create: `lib/models/detection.dart`
- Test: `test/models/detection_test.dart`

**Step 1: Write the test**

Create `test/models/detection_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/models/detection.dart';

void main() {
  group('BoundingBox', () {
    test('stores normalized coordinates', () {
      const box = BoundingBox(x: 0.1, y: 0.2, width: 0.3, height: 0.4);
      expect(box.x, 0.1);
      expect(box.y, 0.2);
      expect(box.width, 0.3);
      expect(box.height, 0.4);
    });
  });

  group('Detection', () {
    test('stores label, confidence, and bbox', () {
      const det = Detection(
        label: 'bottle',
        confidence: 0.95,
        bbox: BoundingBox(x: 0.1, y: 0.2, width: 0.3, height: 0.4),
      );
      expect(det.label, 'bottle');
      expect(det.confidence, 0.95);
      expect(det.bbox.x, 0.1);
    });
  });

  group('DetectionResult', () {
    test('count matches detections length', () {
      const result = DetectionResult(
        detections: [
          Detection(
            label: 'bottle',
            confidence: 0.9,
            bbox: BoundingBox(x: 0, y: 0, width: 0.1, height: 0.1),
          ),
        ],
        count: 1,
        inferenceTimeMs: 42.5,
      );
      expect(result.count, 1);
      expect(result.detections.length, 1);
      expect(result.inferenceTimeMs, 42.5);
    });
  });
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
flutter test test/models/detection_test.dart
```

Expected: FAIL — cannot find `package:proto_b_flutter/models/detection.dart`.

**Step 3: Write the implementation**

Create `lib/models/detection.dart`:

```dart
class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;

  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });
}

class Detection {
  final String label;
  final double confidence;
  final BoundingBox bbox;

  const Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
  });
}

class DetectionResult {
  final List<Detection> detections;
  final int count;
  final double inferenceTimeMs;

  const DetectionResult({
    required this.detections,
    required this.count,
    required this.inferenceTimeMs,
  });
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
flutter test test/models/detection_test.dart
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```
feat(proto-b): add detection data models
```

---

## Task 3: COCO Labels

**Files:**
- Create: `lib/services/labels.dart`
- Test: `test/services/labels_test.dart`

**Step 1: Write the test**

Create `test/services/labels_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/services/labels.dart';

void main() {
  test('has 90 entries', () {
    expect(coco90Labels.length, 90);
  });

  test('first label is person', () {
    expect(coco90Labels[0], 'person');
  });

  test('unused slots are null', () {
    expect(coco90Labels[11], isNull); // COCO ID 12
    expect(coco90Labels[25], isNull); // COCO ID 26
  });

  test('bottle is at index 43', () {
    expect(coco90Labels[43], 'bottle');
  });
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
flutter test test/services/labels_test.dart
```

Expected: FAIL.

**Step 3: Write the implementation**

Create `lib/services/labels.dart`:

```dart
/// EfficientDet-Lite0 outputs 90 class slots mapping to COCO category IDs 1–90.
/// Index = model output class index (0–89). Value = label or null for unused IDs.
const List<String?> coco90Labels = [
  'person',        // 0
  'bicycle',       // 1
  'car',           // 2
  'motorcycle',    // 3
  'airplane',      // 4
  'bus',           // 5
  'train',         // 6
  'truck',         // 7
  'boat',          // 8
  'traffic light',  // 9
  'fire hydrant',  // 10
  null,            // 11 (COCO 12)
  'stop sign',     // 12
  'parking meter', // 13
  'bench',         // 14
  'bird',          // 15
  'cat',           // 16
  'dog',           // 17
  'horse',         // 18
  'sheep',         // 19
  'cow',           // 20
  'elephant',      // 21
  'bear',          // 22
  'zebra',         // 23
  'giraffe',       // 24
  null,            // 25 (COCO 26)
  'backpack',      // 26
  'umbrella',      // 27
  null,            // 28 (COCO 29)
  null,            // 29 (COCO 30)
  'handbag',       // 30
  'tie',           // 31
  'suitcase',      // 32
  'frisbee',       // 33
  'skis',          // 34
  'snowboard',     // 35
  'sports ball',   // 36
  'kite',          // 37
  'baseball bat',  // 38
  'baseball glove', // 39
  'skateboard',    // 40
  'surfboard',     // 41
  'tennis racket', // 42
  'bottle',        // 43
  null,            // 44 (COCO 45)
  'wine glass',    // 45
  'cup',           // 46
  'fork',          // 47
  'knife',         // 48
  'spoon',         // 49
  'bowl',          // 50
  'banana',        // 51
  'apple',         // 52
  'sandwich',      // 53
  'orange',        // 54
  'broccoli',      // 55
  'carrot',        // 56
  'hot dog',       // 57
  'pizza',         // 58
  'donut',         // 59
  'cake',          // 60
  'chair',         // 61
  'couch',         // 62
  'potted plant',  // 63
  'bed',           // 64
  null,            // 65 (COCO 66)
  'dining table',  // 66
  null,            // 67 (COCO 68)
  null,            // 68 (COCO 69)
  'toilet',        // 69
  null,            // 70 (COCO 71)
  'tv',            // 71
  'laptop',        // 72
  'mouse',         // 73
  'remote',        // 74
  'keyboard',      // 75
  'cell phone',    // 76
  'microwave',     // 77
  'oven',          // 78
  'toaster',       // 79
  'sink',          // 80
  'refrigerator',  // 81
  null,            // 82 (COCO 83)
  'book',          // 83
  'clock',         // 84
  'vase',          // 85
  'scissors',      // 86
  'teddy bear',    // 87
  'hair drier',    // 88
  'toothbrush',    // 89
];
```

**Step 4: Run test to verify it passes**

Run:
```bash
flutter test test/services/labels_test.dart
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```
feat(proto-b): add COCO 90-class labels
```

---

## Task 4: Postprocessing Logic

**Files:**
- Create: `lib/services/postprocess.dart`
- Test: `test/services/postprocess_test.dart`

This is the pure logic that parses EfficientDet-Lite0's 4-tensor output into `Detection` objects. Fully testable without a device.

**Step 1: Write the test**

Create `test/services/postprocess_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/models/detection.dart';
import 'package:proto_b_flutter/services/postprocess.dart';

void main() {
  // Simulated EfficientDet-Lite0 output for 2 detections out of 25 slots.
  // boxes: [y1, x1, y2, x2] normalized
  final boxes = List<double>.filled(25 * 4, 0.0);
  final classes = List<double>.filled(25, 0.0);
  final scores = List<double>.filled(25, 0.0);
  const count = 2;

  final labels = <String?>[
    'person', // 0
    'bicycle', // 1
    null, null, null, null, null, null, null, null,
    null, null, null, null, null, null, null, null,
    null, null, null, null, null, null, null, null,
    null, null, null, null, null, null, null, null,
    null, null, null, null, null, null, null, null,
    null, null,
    'bottle', // 43
  ];

  setUp(() {
    // Detection 0: person at [0.1, 0.2, 0.5, 0.6] conf 0.9
    boxes[0] = 0.1; boxes[1] = 0.2; boxes[2] = 0.5; boxes[3] = 0.6;
    classes[0] = 0.0; // person
    scores[0] = 0.9;

    // Detection 1: bottle at [0.3, 0.4, 0.7, 0.8] conf 0.7
    boxes[4] = 0.3; boxes[5] = 0.4; boxes[6] = 0.7; boxes[7] = 0.8;
    classes[1] = 43.0; // bottle
    scores[1] = 0.7;
  });

  test('parses two detections correctly', () {
    final detections = processPostNmsOutputs(
      boxes: boxes,
      classes: classes,
      scores: scores,
      count: count,
      labels: labels,
      scoreThreshold: 0.4,
    );

    expect(detections.length, 2);

    expect(detections[0].label, 'person');
    expect(detections[0].confidence, 0.9);
    // bbox: x=x1, y=y1, width=x2-x1, height=y2-y1
    expect(detections[0].bbox.x, closeTo(0.2, 0.001));
    expect(detections[0].bbox.y, closeTo(0.1, 0.001));
    expect(detections[0].bbox.width, closeTo(0.4, 0.001));
    expect(detections[0].bbox.height, closeTo(0.4, 0.001));

    expect(detections[1].label, 'bottle');
    expect(detections[1].confidence, 0.7);
  });

  test('filters by confidence threshold', () {
    final detections = processPostNmsOutputs(
      boxes: boxes,
      classes: classes,
      scores: scores,
      count: count,
      labels: labels,
      scoreThreshold: 0.8,
    );

    expect(detections.length, 1);
    expect(detections[0].label, 'person');
  });

  test('skips null labels', () {
    classes[0] = 2.0; // index 2 is beyond our short labels list → null
    scores[0] = 0.95;

    final detections = processPostNmsOutputs(
      boxes: boxes,
      classes: classes,
      scores: scores,
      count: count,
      labels: labels,
      scoreThreshold: 0.4,
    );

    // Only the bottle detection survives (person slot now has null label)
    expect(detections.length, 1);
    expect(detections[0].label, 'bottle');
  });

  test('returns empty list when count is 0', () {
    final detections = processPostNmsOutputs(
      boxes: boxes,
      classes: classes,
      scores: scores,
      count: 0,
      labels: labels,
      scoreThreshold: 0.4,
    );

    expect(detections, isEmpty);
  });
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
flutter test test/services/postprocess_test.dart
```

Expected: FAIL — cannot find `package:proto_b_flutter/services/postprocess.dart`.

**Step 3: Write the implementation**

Create `lib/services/postprocess.dart`:

```dart
import '../models/detection.dart';

/// Parse post-NMS EfficientDet-Lite0 outputs into Detection objects.
///
/// EfficientDet-Lite0 output tensors:
///   [0] boxes   — [1, maxDet, 4] as [y1, x1, y2, x2] normalized 0–1
///   [1] classes  — [1, maxDet] class index as float
///   [2] scores   — [1, maxDet] confidence 0–1
///   [3] count    — [1] number of valid detections
List<Detection> processPostNmsOutputs({
  required List<double> boxes,
  required List<double> classes,
  required List<double> scores,
  required int count,
  required List<String?> labels,
  required double scoreThreshold,
}) {
  final detections = <Detection>[];

  for (var i = 0; i < count; i++) {
    if (scores[i] < scoreThreshold) continue;

    final classIdx = classes[i].round();
    if (classIdx < 0 || classIdx >= labels.length) continue;
    final label = labels[classIdx];
    if (label == null) continue;

    final y1 = boxes[i * 4];
    final x1 = boxes[i * 4 + 1];
    final y2 = boxes[i * 4 + 2];
    final x2 = boxes[i * 4 + 3];

    detections.add(Detection(
      label: label,
      confidence: scores[i],
      bbox: BoundingBox(x: x1, y: y1, width: x2 - x1, height: y2 - y1),
    ));
  }

  return detections;
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
flutter test test/services/postprocess_test.dart
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```
feat(proto-b): add EfficientDet postprocessing logic
```

---

## Task 5: ML Service

**Files:**
- Create: `lib/services/ml_service.dart`

This wraps `tflite_flutter` interpreter loading, image preprocessing, and inference. Cannot be unit-tested without a real device (FFI requires native libraries), so we test it manually in Task 8.

**Step 1: Write the implementation**

Create `lib/services/ml_service.dart`:

```dart
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection.dart';
import 'labels.dart';
import 'postprocess.dart';

const _modelAsset = 'assets/models/efficientdet-lite0.tflite';
const _inputSize = 320;
const _confidenceThreshold = 0.4;
const _maxDetections = 25;

class MlService {
  Interpreter? _interpreter;

  bool get isLoaded => _interpreter != null;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(_modelAsset);
  }

  /// Run inference on a JPEG/PNG image.
  ///
  /// [imageBytes] — raw file bytes (not decoded pixels).
  /// Returns null if the model is not loaded or image decode fails.
  DetectionResult? runInference(Uint8List imageBytes) {
    if (_interpreter == null) return null;

    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return null;

    // Resize to 320x320
    final resized = img.copyResize(decoded, width: _inputSize, height: _inputSize);

    // Convert to [1, 320, 320, 3] uint8 input tensor
    final input = _imageToInputTensor(resized);

    // Prepare output buffers
    final boxes = List<List<double>>.generate(
      _maxDetections, (_) => List<double>.filled(4, 0),
    );
    final classes = List<double>.filled(_maxDetections, 0);
    final scores = List<double>.filled(_maxDetections, 0);
    final countArr = List<double>.filled(1, 0);

    final outputs = <int, Object>{
      0: [boxes],
      1: [classes],
      2: [scores],
      3: countArr,
    };

    final stopwatch = Stopwatch()..start();
    _interpreter!.runForMultipleInputs([input], outputs);
    stopwatch.stop();

    // Flatten boxes from List<List<double>> to List<double>
    final flatBoxes = <double>[];
    for (final box in boxes) {
      flatBoxes.addAll(box);
    }

    final detections = processPostNmsOutputs(
      boxes: flatBoxes,
      classes: classes,
      scores: scores,
      count: countArr[0].round(),
      labels: coco90Labels,
      scoreThreshold: _confidenceThreshold,
    );

    return DetectionResult(
      detections: detections,
      count: detections.length,
      inferenceTimeMs: stopwatch.elapsedMicroseconds / 1000.0,
    );
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  /// Convert an image to [1, 320, 320, 3] uint8 tensor.
  List<List<List<List<int>>>> _imageToInputTensor(img.Image image) {
    return [
      List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) {
            final pixel = image.getPixel(x, y);
            return [pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt()];
          },
        ),
      ),
    ];
  }
}
```

**Step 2: Verify it compiles**

Run:
```bash
flutter analyze lib/services/ml_service.dart
```

Expected: No errors.

**Step 3: Commit**

```
feat(proto-b): add ML service with TFLite inference
```

---

## Task 6: Camera Screen

**Files:**
- Create: `lib/screens/camera_screen.dart`
- Modify: `lib/main.dart`

**Step 1: Write CameraScreen**

Create `lib/screens/camera_screen.dart`:

```dart
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/ml_service.dart';
import 'results_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  final _mlService = MlService();
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _mlService.loadModel();

    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final backCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _controller = CameraController(backCamera, ResolutionPreset.high);
    await _controller!.initialize();
    if (mounted) setState(() {});
  }

  Future<void> _capture() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isProcessing) return;

    setState(() => _isProcessing = true);

    try {
      final photo = await _controller!.takePicture();
      final imageBytes = await photo.readAsBytes();
      final result = _mlService.runInference(imageBytes);

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ResultsScreen(
              photoPath: photo.path,
              result: result ?? const DetectionResult(
                detections: [],
                count: 0,
                inferenceTimeMs: 0,
              ),
            ),
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _mlService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller != null &&
        _controller!.value.isInitialized &&
        _mlService.isLoaded;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (_controller != null && _controller!.value.isInitialized)
            CameraPreview(_controller!)
          else
            const Center(
              child: CircularProgressIndicator(color: Colors.white),
            ),
          if (!_mlService.isLoaded)
            Positioned(
              top: 60,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      SizedBox(
                        width: 14,
                        height: 14,
                        child:
                            CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                      ),
                      SizedBox(width: 8),
                      Text('Loading model…',
                          style: TextStyle(color: Colors.white, fontSize: 13)),
                    ],
                  ),
                ),
              ),
            ),
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: ready && !_isProcessing ? _capture : null,
                child: Container(
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white.withValues(alpha: ready ? 0.3 : 0.1),
                  ),
                  child: Center(
                    child: _isProcessing
                        ? const CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 3)
                        : Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: ready
                                  ? Colors.white
                                  : Colors.white.withValues(alpha: 0.4),
                            ),
                          ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
```

**Step 2: Update main.dart**

Replace `lib/main.dart`:

```dart
import 'package:flutter/material.dart';

import 'screens/camera_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ShelfCountApp());
}

class ShelfCountApp extends StatelessWidget {
  const ShelfCountApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shelf Count',
      theme: ThemeData.dark(),
      home: const CameraScreen(),
    );
  }
}
```

**Step 3: Verify it compiles**

Run:
```bash
flutter analyze
```

Expected: No errors.

**Step 4: Commit**

```
feat(proto-b): add camera screen with capture + inference
```

---

## Task 7: Results Screen + Bounding Box Overlay

**Files:**
- Create: `lib/widgets/bounding_box_overlay.dart`
- Create: `lib/screens/results_screen.dart`

**Step 1: Write BoundingBoxOverlay widget**

Create `lib/widgets/bounding_box_overlay.dart`:

```dart
import 'package:flutter/material.dart';

import '../models/detection.dart';

class BoundingBoxOverlay extends StatelessWidget {
  final List<Detection> detections;
  final double imageWidth;
  final double imageHeight;
  final double viewWidth;
  final double viewHeight;

  const BoundingBoxOverlay({
    super.key,
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.viewWidth,
    required this.viewHeight,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size(viewWidth, viewHeight),
      painter: _BoxPainter(
        detections: detections,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  final List<Detection> detections;
  final double imageWidth;
  final double imageHeight;

  _BoxPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final textStyle = TextStyle(
      color: Colors.black,
      fontSize: 10,
      fontWeight: FontWeight.bold,
      background: Paint()..color = Colors.green,
    );

    for (final det in detections) {
      final rect = Rect.fromLTWH(
        det.bbox.x * size.width,
        det.bbox.y * size.height,
        det.bbox.width * size.width,
        det.bbox.height * size.height,
      );

      canvas.drawRect(rect, paint);

      final textSpan = TextSpan(
        text: ' ${det.label} ${(det.confidence * 100).round()}% ',
        style: textStyle,
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      textPainter.paint(canvas, Offset(rect.left, rect.top - textPainter.height));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) =>
      detections != old.detections;
}
```

**Step 2: Write ResultsScreen**

Create `lib/screens/results_screen.dart`:

```dart
import 'dart:io';

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../widgets/bounding_box_overlay.dart';

class ResultsScreen extends StatelessWidget {
  final String photoPath;
  final DetectionResult result;

  const ResultsScreen({
    super.key,
    required this.photoPath,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  return Stack(
                    children: [
                      Image.file(
                        File(photoPath),
                        width: constraints.maxWidth,
                        fit: BoxFit.contain,
                      ),
                      BoundingBoxOverlay(
                        detections: result.detections,
                        imageWidth: 1,
                        imageHeight: 1,
                        viewWidth: constraints.maxWidth,
                        viewHeight: constraints.maxHeight,
                      ),
                    ],
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Text(
                    '${result.count} items detected',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Inference: ${result.inferenceTimeMs.round()}ms',
                    style: const TextStyle(color: Colors.grey, fontSize: 14),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF007AFF),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8)),
                ),
                onPressed: () => Navigator.pop(context),
                child: const Text(
                  'Scan Another Shelf',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

**Step 3: Verify it compiles**

Run:
```bash
flutter analyze
```

Expected: No errors.

**Step 4: Commit**

```
feat(proto-b): add results screen with bounding box overlay
```

---

## Task 8: End-to-End Device Test

**No files to create — manual verification on device/simulator.**

**Step 1: Run on iOS Simulator**

Run:
```bash
flutter run -d iphone
```

Verify:
- [ ] App launches to camera preview
- [ ] "Loading model…" indicator appears briefly, then disappears
- [ ] Capture button becomes solid white when model is ready
- [ ] Tapping capture takes a photo and navigates to results screen
- [ ] Results screen shows the captured photo
- [ ] Bounding boxes appear over detected objects (if any in view)
- [ ] Item count and inference time are displayed
- [ ] "Scan Another Shelf" returns to camera

**Step 2: Run on Android Emulator**

Run:
```bash
flutter run -d emulator
```

Verify the same checklist.

**Step 3: Note inference time for comparison with Proto A**

Record the inference time displayed on the results screen. This will be compared with Proto A's numbers later.

**Step 4: Commit any fixes needed**

```
fix(proto-b): [describe what was fixed]
```

---

## Task 9: Storage Service (Optional — Matches Spec)

**Files:**
- Create: `lib/services/storage_service.dart`
- Test: `test/services/storage_service_test.dart`

**Step 1: Write the test**

Create `test/services/storage_service_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:proto_b_flutter/models/detection.dart';
import 'package:proto_b_flutter/services/storage_service.dart';

void main() {
  late StorageService storage;

  setUp(() {
    storage = StorageService();
  });

  test('starts empty', () {
    expect(storage.audits, isEmpty);
  });

  test('saves and retrieves an audit', () {
    final record = AuditRecord(
      id: '1',
      timestamp: DateTime.now().millisecondsSinceEpoch,
      photoPath: '/tmp/photo.jpg',
      result: const DetectionResult(
        detections: [],
        count: 0,
        inferenceTimeMs: 50,
      ),
    );

    storage.save(record);
    expect(storage.audits.length, 1);
    expect(storage.audits.first.id, '1');
  });

  test('most recent audit is first', () {
    storage.save(AuditRecord(
      id: '1',
      timestamp: 100,
      photoPath: '/a.jpg',
      result: const DetectionResult(detections: [], count: 0, inferenceTimeMs: 0),
    ));
    storage.save(AuditRecord(
      id: '2',
      timestamp: 200,
      photoPath: '/b.jpg',
      result: const DetectionResult(detections: [], count: 0, inferenceTimeMs: 0),
    ));

    expect(storage.audits.first.id, '2');
  });

  test('clear removes all audits', () {
    storage.save(AuditRecord(
      id: '1',
      timestamp: 100,
      photoPath: '/a.jpg',
      result: const DetectionResult(detections: [], count: 0, inferenceTimeMs: 0),
    ));
    storage.clear();
    expect(storage.audits, isEmpty);
  });
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
flutter test test/services/storage_service_test.dart
```

Expected: FAIL.

**Step 3: Write the implementation**

Create `lib/services/storage_service.dart`:

```dart
import '../models/detection.dart';

class AuditRecord {
  final String id;
  final int timestamp;
  final String photoPath;
  final DetectionResult result;

  const AuditRecord({
    required this.id,
    required this.timestamp,
    required this.photoPath,
    required this.result,
  });
}

class StorageService {
  final List<AuditRecord> _audits = [];

  List<AuditRecord> get audits => List.unmodifiable(_audits);

  void save(AuditRecord record) {
    _audits.insert(0, record);
  }

  void clear() {
    _audits.clear();
  }
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
flutter test test/services/storage_service_test.dart
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```
feat(proto-b): add in-memory audit storage service
```

---

## Task Summary

| Task | What | Test? |
|------|------|-------|
| 0 | Install Flutter SDK | `flutter doctor` |
| 1 | Scaffold project + dependencies | Manual build |
| 2 | Data models (Detection, BoundingBox, DetectionResult) | Unit tests |
| 3 | COCO labels | Unit tests |
| 4 | Postprocessing logic | Unit tests |
| 5 | ML service (TFLite load + inference) | Compiles; tested on device in Task 8 |
| 6 | Camera screen | Compiles; tested on device in Task 8 |
| 7 | Results screen + bounding box overlay | Compiles; tested on device in Task 8 |
| 8 | End-to-end device test | Manual on iOS + Android |
| 9 | Storage service | Unit tests |
