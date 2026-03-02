# Prototype A: React Native + On-device ML — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal React Native app that captures a shelf photo and runs on-device object detection to count products.

**Architecture:** Expo managed workflow with CNG (Continuous Native Generation). Camera via `react-native-vision-camera`, ML inference via `react-native-fast-tflite` using a bundled EfficientDet-Lite0 model. Two screens: camera capture and results display.

**Tech Stack:** TypeScript, Expo SDK 52+, react-native-vision-camera v4.x, react-native-fast-tflite v2.x, react-native-worklets-core, Expo Router

---

## Task 1: Scaffold Expo Project

**Files:**
- Create: `proto-a-rn/` (entire project scaffold)

**Step 1: Create Expo project**

Run from `shelf-count/`:
```bash
npx create-expo-app@latest proto-a-rn --template tabs
```

This gives us Expo Router with tab navigation already configured.

Expected: New `proto-a-rn/` directory with a working Expo project.

**Step 2: Verify it runs**

```bash
cd proto-a-rn && npx expo start
```

Expected: Metro bundler starts, QR code shown. Press `i` for iOS simulator (if available) or confirm it bundles without errors.

**Step 3: Clean up template**

Remove the default tab screens we don't need. We want two screens:
- `app/(tabs)/index.tsx` → Camera screen
- `app/(tabs)/results.tsx` → Results screen

Strip out the "explore" tab and example content. Keep the tab layout structure.

**Step 4: Commit**

```
chore(proto-a): scaffold Expo project from tabs template
```

---

## Task 2: Install Native Dependencies

**Files:**
- Modify: `proto-a-rn/package.json`
- Modify: `proto-a-rn/app.json`
- Modify: `proto-a-rn/babel.config.js`

**Step 1: Install camera + ML packages**

Run from `proto-a-rn/`:
```bash
npx expo install react-native-vision-camera react-native-worklets-core react-native-fast-tflite
```

Using `npx expo install` ensures version compatibility with the current Expo SDK.

**Step 2: Configure babel for worklets**

Modify `babel.config.js` to add the worklets plugin:

```js
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: ['react-native-worklets-core/plugin'],
  };
};
```

**Step 3: Configure Expo config plugins**

Add to `app.json` (inside `"expo"` → `"plugins"`):

```json
{
  "expo": {
    "plugins": [
      [
        "react-native-vision-camera",
        {
          "cameraPermissionText": "Shelf Count needs camera access to photograph shelves for product counting.",
          "enableMicrophonePermission": false
        }
      ],
      [
        "react-native-fast-tflite",
        {
          "enableCoreMLDelegate": true
        }
      ]
    ]
  }
}
```

**Step 4: Prebuild to generate native projects**

```bash
npx expo prebuild --clean
```

Expected: `ios/` and `android/` directories are generated with native config applied.

**Step 5: Build and run dev client (iOS)**

```bash
npx expo run:ios
```

Expected: App builds, installs on iOS simulator, and launches. This is the development client — not Expo Go.

If no iOS simulator available, try Android:
```bash
npx expo run:android
```

**Step 6: Commit**

```
feat(proto-a): install vision-camera, fast-tflite, worklets-core
```

---

## Task 3: Download and Bundle ML Model

**Files:**
- Create: `proto-a-rn/assets/models/efficientdet-lite0.tflite`
- Create: `proto-a-rn/assets/models/labels.txt`

**Step 1: Download EfficientDet-Lite0 TFLite model**

```bash
cd proto-a-rn/assets && mkdir -p models && cd models
curl -L -o efficientdet-lite0.tflite \
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
```

If the above URL doesn't work, try TensorFlow Hub:
```bash
curl -L -o efficientdet-lite0.tflite \
  "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite"
```

Expected: A `.tflite` file around 4-5 MB in the models directory.

**Step 2: Create COCO labels file**

Create `proto-a-rn/assets/models/labels.txt` with the 80 COCO class labels (one per line):

```
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```

**Step 3: Verify model file is valid**

```bash
ls -lh proto-a-rn/assets/models/efficientdet-lite0.tflite
```

Expected: File exists, size ~4-5 MB.

**Step 4: Commit**

```
feat(proto-a): bundle EfficientDet-Lite0 model and COCO labels
```

---

## Task 4: ML Inference Wrapper

**Files:**
- Create: `proto-a-rn/lib/ml/types.ts`
- Create: `proto-a-rn/lib/ml/model.ts`
- Create: `proto-a-rn/lib/ml/postprocess.ts`
- Test: `proto-a-rn/lib/ml/__tests__/postprocess.test.ts`

**Step 1: Write the detection types**

Create `proto-a-rn/lib/ml/types.ts`:

```typescript
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  label: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface DetectionResult {
  detections: Detection[];
  count: number;
  inferenceTimeMs: number;
}
```

**Step 2: Write failing test for postprocessing**

Create `proto-a-rn/lib/ml/__tests__/postprocess.test.ts`:

```typescript
import { filterByConfidence, mapOutputToDetections } from '../postprocess';

describe('filterByConfidence', () => {
  it('filters detections below threshold', () => {
    const detections = [
      { label: 'bottle', confidence: 0.9, bbox: { x: 0, y: 0, width: 10, height: 10 } },
      { label: 'cup', confidence: 0.3, bbox: { x: 20, y: 20, width: 10, height: 10 } },
      { label: 'book', confidence: 0.6, bbox: { x: 40, y: 40, width: 10, height: 10 } },
    ];
    const result = filterByConfidence(detections, 0.5);
    expect(result).toHaveLength(2);
    expect(result[0].label).toBe('bottle');
    expect(result[1].label).toBe('book');
  });
});

describe('mapOutputToDetections', () => {
  it('maps raw model output tensors to Detection objects', () => {
    const boxes = [0.1, 0.2, 0.5, 0.6]; // [y1, x1, y2, x2] normalized
    const classes = [39]; // COCO class index for "bottle"
    const scores = [0.85];
    const count = 1;
    const labels = ['bottle'];

    const result = mapOutputToDetections(boxes, classes, scores, count, labels);

    expect(result).toHaveLength(1);
    expect(result[0].label).toBe('bottle');
    expect(result[0].confidence).toBe(0.85);
    expect(result[0].bbox.x).toBeCloseTo(0.2);
    expect(result[0].bbox.y).toBeCloseTo(0.1);
    expect(result[0].bbox.width).toBeCloseTo(0.4); // x2 - x1
    expect(result[0].bbox.height).toBeCloseTo(0.4); // y2 - y1
  });
});
```

**Step 3: Run test to verify it fails**

```bash
npx jest lib/ml/__tests__/postprocess.test.ts
```

Expected: FAIL — functions not defined.

**Step 4: Implement postprocessing**

Create `proto-a-rn/lib/ml/postprocess.ts`:

```typescript
import { Detection } from './types';

export function filterByConfidence(
  detections: Detection[],
  threshold: number
): Detection[] {
  return detections.filter((d) => d.confidence >= threshold);
}

export function mapOutputToDetections(
  boxes: number[],
  classes: number[],
  scores: number[],
  count: number,
  labels: string[]
): Detection[] {
  const detections: Detection[] = [];

  for (let i = 0; i < count; i++) {
    const y1 = boxes[i * 4];
    const x1 = boxes[i * 4 + 1];
    const y2 = boxes[i * 4 + 2];
    const x2 = boxes[i * 4 + 3];

    detections.push({
      label: labels[classes[i]] ?? `class_${classes[i]}`,
      confidence: scores[i],
      bbox: {
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
      },
    });
  }

  return detections;
}
```

**Step 5: Run test to verify it passes**

```bash
npx jest lib/ml/__tests__/postprocess.test.ts
```

Expected: PASS — 2 tests passing.

**Step 6: Implement model loader**

Create `proto-a-rn/lib/ml/model.ts`:

```typescript
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { Detection, DetectionResult } from './types';
import { filterByConfidence, mapOutputToDetections } from './postprocess';

const CONFIDENCE_THRESHOLD = 0.5;

let model: TensorflowModel | null = null;
let labels: string[] = [];

export async function loadModel(): Promise<void> {
  if (model) return;

  model = await loadTensorflowModel(
    require('../../assets/models/efficientdet-lite0.tflite')
  );

  // Labels loaded separately — in production, parse from model metadata
  const labelModule = require('../../assets/models/labels.txt');
  // For now, labels are hardcoded or loaded at build time
}

export function setLabels(labelList: string[]): void {
  labels = labelList;
}

export function runInference(inputData: Float32Array): DetectionResult {
  if (!model) {
    throw new Error('Model not loaded. Call loadModel() first.');
  }

  const start = performance.now();
  const output = model.runSync([inputData]);
  const inferenceTimeMs = performance.now() - start;

  // EfficientDet-Lite0 output format:
  // output[0]: bounding boxes [N, 4] — [y1, x1, y2, x2] normalized
  // output[1]: class indices [N]
  // output[2]: confidence scores [N]
  // output[3]: number of detections (scalar)
  const boxes = Array.from(output[0] as Float32Array);
  const classIndices = Array.from(output[1] as Float32Array).map(Math.round);
  const scores = Array.from(output[2] as Float32Array);
  const numDetections = Math.round((output[3] as Float32Array)[0]);

  const allDetections = mapOutputToDetections(
    boxes,
    classIndices,
    scores,
    numDetections,
    labels
  );

  const detections = filterByConfidence(allDetections, CONFIDENCE_THRESHOLD);

  return {
    detections,
    count: detections.length,
    inferenceTimeMs,
  };
}
```

**Step 7: Commit**

```
feat(proto-a): add ML inference wrapper with postprocessing and tests
```

---

## Task 5: Camera Screen

**Files:**
- Modify: `proto-a-rn/app/(tabs)/index.tsx`
- Create: `proto-a-rn/lib/hooks/useCamera.ts`

**Step 1: Create camera hook**

Create `proto-a-rn/lib/hooks/useCamera.ts`:

```typescript
import { useRef, useState, useCallback } from 'react';
import { Camera, useCameraDevice, useCameraPermission, PhotoFile } from 'react-native-vision-camera';

export function useCamera() {
  const cameraRef = useRef<Camera>(null);
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const [photo, setPhoto] = useState<PhotoFile | null>(null);

  const takePhoto = useCallback(async () => {
    if (!cameraRef.current) return null;

    const result = await cameraRef.current.takePhoto({
      qualityPrioritization: 'speed',
    });
    setPhoto(result);
    return result;
  }, []);

  return {
    cameraRef,
    device,
    hasPermission,
    requestPermission,
    photo,
    takePhoto,
  };
}
```

**Step 2: Build the camera screen**

Replace `proto-a-rn/app/(tabs)/index.tsx`:

```typescript
import { StyleSheet, View, TouchableOpacity, Text } from 'react-native';
import { Camera } from 'react-native-vision-camera';
import { useRouter } from 'expo-router';
import { useCamera } from '../../lib/hooks/useCamera';
import { useEffect } from 'react';

export default function CameraScreen() {
  const router = useRouter();
  const { cameraRef, device, hasPermission, requestPermission, takePhoto } = useCamera();

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const handleCapture = async () => {
    const photo = await takePhoto();
    if (photo) {
      router.push({
        pathname: '/results',
        params: { photoPath: photo.path },
      });
    }
  };

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Camera permission required</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>No camera device found</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
      />
      <View style={styles.captureContainer}>
        <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
          <View style={styles.captureInner} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  message: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  captureContainer: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
  },
  captureButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
});
```

**Step 3: Build and verify camera renders on device**

```bash
npx expo run:ios
```

Expected: Camera viewfinder fills the screen, capture button visible at bottom.

**Step 4: Commit**

```
feat(proto-a): add camera screen with capture button
```

---

## Task 6: Results Screen with Bounding Boxes

**Files:**
- Modify: `proto-a-rn/app/(tabs)/results.tsx`
- Create: `proto-a-rn/components/BoundingBoxOverlay.tsx`

**Step 1: Create bounding box overlay component**

Create `proto-a-rn/components/BoundingBoxOverlay.tsx`:

```typescript
import { View, Text, StyleSheet } from 'react-native';
import { Detection } from '../lib/ml/types';

interface Props {
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
  viewWidth: number;
  viewHeight: number;
}

export function BoundingBoxOverlay({
  detections,
  imageWidth,
  imageHeight,
  viewWidth,
  viewHeight,
}: Props) {
  const scaleX = viewWidth / imageWidth;
  const scaleY = viewHeight / imageHeight;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {detections.map((det, i) => (
        <View
          key={i}
          style={[
            styles.box,
            {
              left: det.bbox.x * imageWidth * scaleX,
              top: det.bbox.y * imageHeight * scaleY,
              width: det.bbox.width * imageWidth * scaleX,
              height: det.bbox.height * imageHeight * scaleY,
            },
          ]}
        >
          <Text style={styles.label}>
            {det.label} {Math.round(det.confidence * 100)}%
          </Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00FF00',
    backgroundColor: 'transparent',
  },
  label: {
    position: 'absolute',
    top: -18,
    left: 0,
    backgroundColor: '#00FF00',
    color: '#000',
    fontSize: 10,
    fontWeight: '700',
    paddingHorizontal: 4,
    paddingVertical: 1,
  },
});
```

**Step 2: Build the results screen**

Replace `proto-a-rn/app/(tabs)/results.tsx`:

```typescript
import { useEffect, useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Image,
  ActivityIndicator,
  TouchableOpacity,
  useWindowDimensions,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { BoundingBoxOverlay } from '../../components/BoundingBoxOverlay';
import { loadModel, runInference, setLabels } from '../../lib/ml/model';
import { DetectionResult } from '../../lib/ml/types';

export default function ResultsScreen() {
  const { photoPath } = useLocalSearchParams<{ photoPath: string }>();
  const router = useRouter();
  const { width: viewWidth } = useWindowDimensions();
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState({ width: 1, height: 1 });

  useEffect(() => {
    if (!photoPath) return;

    const analyze = async () => {
      try {
        setLoading(true);
        await loadModel();
        // TODO: preprocess image to model input tensor
        // For now, this is a placeholder — real preprocessing requires
        // reading the image pixels into a Float32Array at model's expected dimensions
        // This will be connected in the integration task
        setError('Image preprocessing not yet implemented');
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    analyze();
  }, [photoPath]);

  const viewHeight = viewWidth * (imageDims.height / imageDims.width);

  return (
    <View style={styles.container}>
      {photoPath && (
        <View style={{ width: viewWidth, height: viewHeight }}>
          <Image
            source={{ uri: `file://${photoPath}` }}
            style={{ width: viewWidth, height: viewHeight }}
            onLoad={(e) => {
              const { width, height } = e.nativeEvent.source;
              setImageDims({ width, height });
            }}
          />
          {result && (
            <BoundingBoxOverlay
              detections={result.detections}
              imageWidth={imageDims.width}
              imageHeight={imageDims.height}
              viewWidth={viewWidth}
              viewHeight={viewHeight}
            />
          )}
        </View>
      )}

      <View style={styles.infoPanel}>
        {loading && <ActivityIndicator size="large" color="#007AFF" />}
        {error && <Text style={styles.error}>{error}</Text>}
        {result && (
          <>
            <Text style={styles.count}>{result.count} items detected</Text>
            <Text style={styles.timing}>
              Inference: {Math.round(result.inferenceTimeMs)}ms
            </Text>
          </>
        )}
      </View>

      <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
        <Text style={styles.backText}>Scan Another Shelf</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  infoPanel: {
    padding: 20,
    alignItems: 'center',
  },
  count: {
    color: '#fff',
    fontSize: 32,
    fontWeight: '700',
    marginBottom: 8,
  },
  timing: {
    color: '#888',
    fontSize: 14,
  },
  error: {
    color: '#FF4444',
    fontSize: 14,
    textAlign: 'center',
  },
  backButton: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  backText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
```

**Step 3: Build and verify**

```bash
npx expo run:ios
```

Expected: After capturing a photo, navigates to results screen showing the photo with loading/placeholder state.

**Step 4: Commit**

```
feat(proto-a): add results screen with bounding box overlay component
```

---

## Task 7: Image Preprocessing + End-to-End Integration

**Files:**
- Create: `proto-a-rn/lib/ml/preprocess.ts`
- Modify: `proto-a-rn/app/(tabs)/results.tsx` (connect real inference)
- Test: `proto-a-rn/lib/ml/__tests__/preprocess.test.ts`

This is the glue task — connecting the camera output to the model input. EfficientDet-Lite0 expects a 320x320x3 uint8 input tensor.

**Step 1: Write failing test for preprocessing**

Create `proto-a-rn/lib/ml/__tests__/preprocess.test.ts`:

```typescript
import { resizeNormalize } from '../preprocess';

describe('resizeNormalize', () => {
  it('returns a Uint8Array of correct size for 320x320x3', () => {
    // Simulate a small 2x2 RGBA image (8 bytes)
    const rgba = new Uint8Array([
      255, 0, 0, 255,   // red pixel
      0, 255, 0, 255,   // green pixel
      0, 0, 255, 255,   // blue pixel
      255, 255, 0, 255,  // yellow pixel
    ]);
    const result = resizeNormalize(rgba, 2, 2, 2, 2);
    // Output should be 2*2*3 = 12 bytes (RGB, no alpha)
    expect(result.length).toBe(12);
    // First pixel: R=255, G=0, B=0
    expect(result[0]).toBe(255);
    expect(result[1]).toBe(0);
    expect(result[2]).toBe(0);
  });
});
```

**Step 2: Run test to verify it fails**

```bash
npx jest lib/ml/__tests__/preprocess.test.ts
```

Expected: FAIL — function not defined.

**Step 3: Implement preprocessing**

Create `proto-a-rn/lib/ml/preprocess.ts`:

```typescript
/**
 * Strips alpha channel from RGBA pixel data to produce RGB.
 * In a full implementation, this would also resize to the model's expected dimensions.
 * For the prototype, we rely on the camera outputting near the right resolution
 * and the model accepting slight variations.
 */
export function resizeNormalize(
  rgbaPixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number
): Uint8Array {
  const rgb = new Uint8Array(targetWidth * targetHeight * 3);
  const xRatio = srcWidth / targetWidth;
  const yRatio = srcHeight / targetHeight;

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.floor(x * xRatio);
      const srcY = Math.floor(y * yRatio);
      const srcIdx = (srcY * srcWidth + srcX) * 4; // RGBA
      const dstIdx = (y * targetWidth + x) * 3;    // RGB

      rgb[dstIdx] = rgbaPixels[srcIdx];         // R
      rgb[dstIdx + 1] = rgbaPixels[srcIdx + 1]; // G
      rgb[dstIdx + 2] = rgbaPixels[srcIdx + 2]; // B
    }
  }

  return rgb;
}
```

**Step 4: Run test to verify it passes**

```bash
npx jest lib/ml/__tests__/preprocess.test.ts
```

Expected: PASS.

**Step 5: Connect inference in results screen**

Update the `analyze` function in `results.tsx` to actually read the photo pixels, preprocess, and run the model. This step may need a native image-to-pixels bridge — if `react-native-fast-tflite` doesn't provide one, we may need `expo-image-manipulator` or a custom approach.

This is the most uncertain step and may require iteration.

**Step 6: Build and verify end-to-end**

```bash
npx expo run:ios
```

Expected: Capture photo → see bounding boxes drawn on detected objects → count displayed.

**Step 7: Commit**

```
feat(proto-a): connect image preprocessing to model inference end-to-end
```

---

## Task 8: Local Storage for Audit Results

**Files:**
- Create: `proto-a-rn/lib/storage/auditStore.ts`
- Create: `proto-a-rn/lib/storage/types.ts`
- Test: `proto-a-rn/lib/storage/__tests__/auditStore.test.ts`

**Step 1: Define storage types**

Create `proto-a-rn/lib/storage/types.ts`:

```typescript
import { DetectionResult } from '../ml/types';

export interface AuditRecord {
  id: string;
  timestamp: number;
  photoPath: string;
  result: DetectionResult;
}
```

**Step 2: Write failing test**

Create `proto-a-rn/lib/storage/__tests__/auditStore.test.ts`:

```typescript
import { saveAudit, getAudits, clearAudits } from '../auditStore';

describe('auditStore', () => {
  beforeEach(() => clearAudits());

  it('saves and retrieves an audit', () => {
    const record = {
      id: '1',
      timestamp: Date.now(),
      photoPath: '/path/to/photo.jpg',
      result: { detections: [], count: 0, inferenceTimeMs: 100 },
    };
    saveAudit(record);
    const audits = getAudits();
    expect(audits).toHaveLength(1);
    expect(audits[0].id).toBe('1');
  });
});
```

**Step 3: Run test to verify it fails**

```bash
npx jest lib/storage/__tests__/auditStore.test.ts
```

Expected: FAIL.

**Step 4: Implement storage (in-memory for prototype)**

Create `proto-a-rn/lib/storage/auditStore.ts`:

```typescript
import { AuditRecord } from './types';

let audits: AuditRecord[] = [];

export function saveAudit(record: AuditRecord): void {
  audits.unshift(record);
}

export function getAudits(): AuditRecord[] {
  return [...audits];
}

export function clearAudits(): void {
  audits = [];
}
```

**Step 5: Run test to verify it passes**

```bash
npx jest lib/storage/__tests__/auditStore.test.ts
```

Expected: PASS.

**Step 6: Wire into results screen**

After inference completes, call `saveAudit()` with the result.

**Step 7: Commit**

```
feat(proto-a): add local audit storage with tests
```

---

## Summary of Tasks

| Task | What | Key Risk |
|------|------|----------|
| 1 | Scaffold Expo project | None (standard) |
| 2 | Install native deps + dev client build | Build failures on native side |
| 3 | Download + bundle ML model | Model URL availability |
| 4 | ML inference wrapper + postprocessing | Model output format assumptions |
| 5 | Camera screen | Permission handling on simulator |
| 6 | Results screen + bounding box overlay | Image scaling math |
| 7 | Image preprocessing + E2E integration | **Highest risk** — pixel extraction from photo |
| 8 | Local audit storage | None (simple) |

**Highest risk:** Task 7 (image preprocessing). Getting raw pixels from a camera photo into a format the TFLite model accepts is the trickiest part. We may need to iterate on the approach — possibly using `expo-image-manipulator` or a Frame Processor for real-time inference instead of post-capture.
