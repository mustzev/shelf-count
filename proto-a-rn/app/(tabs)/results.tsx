import { useLocalSearchParams, useRouter } from 'expo-router'
import { useEffect, useState } from 'react'
import {
  ActivityIndicator,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from 'react-native'
import { BoundingBoxOverlay } from '../../components/BoundingBoxOverlay'
import { clearFrame, getFrame } from '../../lib/frameStore'
import { loadModel, runInference, setLabels } from '../../lib/ml/model'
import { resizeFromRgb } from '../../lib/ml/preprocess'
import type { DetectionResult } from '../../lib/ml/types'
import { saveAudit } from '../../lib/storage/auditStore'

// EfficientDet-Lite0 input dimensions
const MODEL_INPUT_WIDTH = 320
const MODEL_INPUT_HEIGHT = 320

// COCO labels — order matches assets/models/labels.txt exactly.
// Hardcoded here to avoid an async file-read before we can set up labels;
// a future improvement would read from the bundled labels.txt via FileSystem.
const COCO_LABELS = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
]

export default function ResultsScreen() {
  const { photoPath } = useLocalSearchParams<{ photoPath: string }>()
  const router = useRouter()
  const { width: viewWidth } = useWindowDimensions()
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [imageDims, setImageDims] = useState<{
    width: number
    height: number
  } | null>(null)

  const photoUri = photoPath ? `file://${photoPath}` : null
  const viewHeight = imageDims
    ? viewWidth * (imageDims.height / imageDims.width)
    : viewWidth // default to square before image loads

  useEffect(() => {
    if (!photoPath) return

    const analyze = async () => {
      try {
        setLoading(true)

        // Step 1: Load TFLite model (cached — no-op on subsequent calls).
        await loadModel()

        // Step 2: Register COCO labels so runInference can map class IDs.
        setLabels(COCO_LABELS)

        // Step 3: Get raw RGB pixels from the frame store.
        const frameData = getFrame()

        if (!frameData) {
          setError('No frame data captured. Please go back and try again.')
          return
        }

        // Step 4: Resize RGB frame to model input dimensions (320×320×3).
        const inputTensor = resizeFromRgb(
          frameData.pixels,
          frameData.width,
          frameData.height,
          frameData.bytesPerRow,
          MODEL_INPUT_WIDTH,
          MODEL_INPUT_HEIGHT,
        )

        // Step 5: Run inference. Accepts Uint8Array of 320*320*3 bytes.
        const detectionResult = runInference(inputTensor)

        setResult(detectionResult)

        // Step 6: Persist the audit record.
        saveAudit({
          id: Date.now().toString(),
          timestamp: Date.now(),
          photoPath: photoPath ?? '',
          result: detectionResult,
        })
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
      }
    }

    analyze()

    return () => {
      clearFrame()
    }
  }, [photoPath])

  return (
    <View style={styles.container}>
      {photoUri && (
        <View style={{ width: viewWidth, height: viewHeight }}>
          <Image
            source={{ uri: photoUri }}
            style={{ width: viewWidth, height: viewHeight }}
            resizeMode="contain"
            onLoad={(e) => {
              const { width, height } = e.nativeEvent.source
              setImageDims({ width, height })
            }}
          />
          {result && imageDims && (
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
  )
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
})
