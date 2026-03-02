import {
  loadTensorflowModel,
  type TensorflowModel,
} from 'react-native-fast-tflite'
import { filterByConfidence, mapOutputToDetections } from './postprocess'
import type { DetectionResult } from './types'

const CONFIDENCE_THRESHOLD = 0.5

let model: TensorflowModel | null = null
let labels: string[] = []

export async function loadModel(): Promise<void> {
  if (model) return

  model = await loadTensorflowModel(
    require('../../assets/models/efficientdet-lite0.tflite'),
  )
}

export function setLabels(labelList: string[]): void {
  labels = labelList
}

/**
 * Run inference on a preprocessed input tensor.
 *
 * EfficientDet-Lite0 input spec: [1, 320, 320, 3] uint8.
 * Pass the output of resizeNormalize() (a Uint8Array of 320*320*3 bytes)
 * directly here — no float normalisation is required for this model.
 *
 * API change from initial scaffold: parameter changed from Float32Array to
 * Uint8Array to match the model's actual input tensor type (uint8).
 */
export function runInference(inputData: Uint8Array): DetectionResult {
  if (!model) {
    throw new Error('Model not loaded. Call loadModel() first.')
  }

  const start = performance.now()
  const output = model.runSync([inputData])
  const inferenceTimeMs = performance.now() - start

  // EfficientDet-Lite0 output format:
  // output[0]: bounding boxes [N, 4] — [y1, x1, y2, x2] normalized
  // output[1]: class indices [N]
  // output[2]: confidence scores [N]
  // output[3]: number of detections (scalar)
  const boxes = Array.from(output[0] as Float32Array)
  const classIndices = Array.from(output[1] as Float32Array).map(Math.round)
  const scores = Array.from(output[2] as Float32Array)
  const numDetections = Math.round((output[3] as Float32Array)[0])

  const allDetections = mapOutputToDetections(
    boxes,
    classIndices,
    scores,
    numDetections,
    labels,
  )

  const detections = filterByConfidence(allDetections, CONFIDENCE_THRESHOLD)

  return {
    detections,
    count: detections.length,
    inferenceTimeMs,
  }
}
