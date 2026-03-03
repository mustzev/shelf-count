import type { Detection } from './types'

/**
 * Process post-NMS EfficientDet-Lite0 outputs (4-tensor format with built-in NMS).
 *
 * outputs[0]: boxes   — Float32Array [1, maxDet, 4] (y1, x1, y2, x2 normalized 0–1)
 * outputs[1]: classes — Float32Array [1, maxDet] (class index as float)
 * outputs[2]: scores  — Float32Array [1, maxDet] (confidence 0–1)
 * outputs[3]: count   — Float32Array [1] (number of valid detections)
 */
export function processPostNMSOutputs(
  boxes: Float32Array,
  classes: Float32Array,
  scores: Float32Array,
  countArr: Float32Array,
  labels: (string | null)[],
  scoreThreshold: number,
): Detection[] {
  const count = Math.round(countArr[0])
  const detections: Detection[] = []

  for (let i = 0; i < count; i++) {
    if (scores[i] < scoreThreshold) continue

    const classIdx = Math.round(classes[i])
    const label = labels[classIdx]
    if (label == null) continue

    const y1 = boxes[i * 4]
    const x1 = boxes[i * 4 + 1]
    const y2 = boxes[i * 4 + 2]
    const x2 = boxes[i * 4 + 3]

    detections.push({
      label,
      confidence: scores[i],
      bbox: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 },
    })
  }

  return detections
}
