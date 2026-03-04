import {
  loadTensorflowModel,
  type TensorflowModel,
} from 'react-native-fast-tflite'
import { processYoloOutputs } from './postprocess'
import type { DetectionResult } from './types'

const CONFIDENCE_THRESHOLD = 0.25
const IOU_THRESHOLD = 0.5

let model: TensorflowModel | null = null

export async function loadModel(): Promise<void> {
  if (model) return

  model = await loadTensorflowModel(
    require('../../assets/models/yolov8m-sku110k.tflite'),
  )
}

/**
 * Run inference on a preprocessed input tensor.
 *
 * YOLOv8m-SKU110K input spec: [1, 640, 640, 3] float32 NHWC, values in 0–1.
 * Pass the output of resizeNormalizeFloat() or resizeFromRgbFloat() here.
 *
 * Output tensor: [1, 5, 8400] float32 — 8400 candidate boxes with
 * (cx, cy, w, h, confidence) each in pixel space (0–640). processYoloOutputs
 * handles transposition, NMS, and coordinate normalization.
 */
export function runInference(inputData: Float32Array): DetectionResult {
  if (!model) {
    throw new Error('Model not loaded. Call loadModel() first.')
  }

  const start = performance.now()
  const output = model.runSync([inputData])
  const inferenceTimeMs = performance.now() - start

  // YOLOv8 output: single tensor [1, 5, 8400] as a flat Float32Array.
  const raw = output[0] as Float32Array

  const detections = processYoloOutputs(raw, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

  return {
    detections,
    count: detections.length,
    inferenceTimeMs,
  }
}
