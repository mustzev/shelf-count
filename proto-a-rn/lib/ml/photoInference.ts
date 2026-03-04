import { manipulateAsync, SaveFormat } from 'expo-image-manipulator'
import { File } from 'expo-file-system/next'
import jpeg from 'jpeg-js'
import type { TensorflowModel } from 'react-native-fast-tflite'
import { processYoloOutputs } from './postprocess'
import type { DetectionResult } from './types'

const MODEL_INPUT_SIZE = 640
const CONFIDENCE_THRESHOLD = 0.25
const IOU_THRESHOLD = 0.5

/**
 * Run YOLO inference directly on a captured photo file.
 *
 * Uses expo-image-manipulator (native C/C++) to resize the photo to 640x640,
 * then decodes the small JPEG in JS. This is ~100x faster than decoding
 * the full-resolution photo in pure JS with jpeg-js.
 */
export async function runPhotoInference(
  photoPath: string,
  model: TensorflowModel,
): Promise<DetectionResult> {
  const start = performance.now()

  const uri = photoPath.startsWith('file://') ? photoPath : `file://${photoPath}`

  // Native resize to 640x640 — handles EXIF rotation automatically
  const resized = await manipulateAsync(
    uri,
    [{ resize: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE } }],
    { format: SaveFormat.JPEG, compress: 1 },
  )

  // Decode the small 640x640 JPEG (fast — only 410K pixels vs 12.5M)
  const resizedFile = new File(resized.uri)
  const bytes = await resizedFile.bytes()
  const decoded = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true })

  // Normalize RGBA to float32 RGB 0–1 (no resize needed, already 640x640)
  const pixelCount = decoded.width * decoded.height
  const float32Input = new Float32Array(pixelCount * 3)
  const rgba = decoded.data as unknown as Uint8Array
  for (let i = 0; i < pixelCount; i++) {
    float32Input[i * 3] = rgba[i * 4] / 255
    float32Input[i * 3 + 1] = rgba[i * 4 + 1] / 255
    float32Input[i * 3 + 2] = rgba[i * 4 + 2] / 255
  }

  // Run inference
  const inferenceStart = performance.now()
  const outputs = model.runSync([float32Input])
  const inferenceTimeMs = performance.now() - inferenceStart

  // Process YOLO output
  const raw = outputs[0] as Float32Array
  const detections = processYoloOutputs(raw, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

  const totalTimeMs = performance.now() - start
  console.log(
    `[photoInference] ${detections.length} detections (${decoded.width}x${decoded.height}), ` +
    `preprocess: ${Math.round(inferenceStart - start)}ms, ` +
    `inference: ${Math.round(inferenceTimeMs)}ms, ` +
    `total: ${Math.round(totalTimeMs)}ms`,
  )

  return {
    detections,
    count: detections.length,
    inferenceTimeMs,
    totalTimeMs,
  }
}
