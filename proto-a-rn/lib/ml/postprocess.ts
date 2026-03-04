import type { Detection } from './types'

// ---------------------------------------------------------------------------
// Legacy EfficientDet helper functions (kept for reference and existing tests)
// ---------------------------------------------------------------------------

/**
 * Filter a list of detections to those meeting the minimum confidence.
 */
export function filterByConfidence(
  detections: Detection[],
  threshold: number,
): Detection[] {
  return detections.filter((d) => d.confidence >= threshold)
}

/**
 * Map raw EfficientDet output arrays to Detection objects.
 *
 * @param boxes       - Flat array of [y1, x1, y2, x2] values (normalized 0–1), length count*4
 * @param classIndices - Integer class indices, length count
 * @param scores       - Confidence scores, length count
 * @param count        - Number of valid detections
 * @param labels       - Label strings indexed by class index
 */
export function mapOutputToDetections(
  boxes: number[],
  classIndices: number[],
  scores: number[],
  count: number,
  labels: string[],
): Detection[] {
  const detections: Detection[] = []
  for (let i = 0; i < count; i++) {
    const label = labels[classIndices[i]]
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

/**
 * Compute Intersection over Union (IoU) between two bounding boxes.
 * All coordinates are in the same pixel space.
 */
export function computeIoU(
  ax: number,
  ay: number,
  aw: number,
  ah: number,
  bx: number,
  by: number,
  bw: number,
  bh: number,
): number {
  const ax2 = ax + aw
  const ay2 = ay + ah
  const bx2 = bx + bw
  const by2 = by + bh

  const interX1 = Math.max(ax, bx)
  const interY1 = Math.max(ay, by)
  const interX2 = Math.min(ax2, bx2)
  const interY2 = Math.min(ay2, by2)

  const interW = Math.max(0, interX2 - interX1)
  const interH = Math.max(0, interY2 - interY1)
  const interArea = interW * interH

  if (interArea === 0) return 0

  const aArea = aw * ah
  const bArea = bw * bh
  return interArea / (aArea + bArea - interArea)
}

/**
 * Apply Non-Maximum Suppression to a list of candidate detections.
 * Detections must be sorted by confidence (highest first) before calling.
 * Returns the indices of kept detections.
 */
export function applyNMS(
  candidates: Array<{ x: number; y: number; w: number; h: number; confidence: number }>,
  iouThreshold: number,
): number[] {
  const kept: number[] = []
  const suppressed = new Uint8Array(candidates.length)

  for (let i = 0; i < candidates.length; i++) {
    if (suppressed[i]) continue
    kept.push(i)

    for (let j = i + 1; j < candidates.length; j++) {
      if (suppressed[j]) continue
      const iou = computeIoU(
        candidates[i].x,
        candidates[i].y,
        candidates[i].w,
        candidates[i].h,
        candidates[j].x,
        candidates[j].y,
        candidates[j].w,
        candidates[j].h,
      )
      if (iou > iouThreshold) {
        suppressed[j] = 1
      }
    }
  }

  return kept
}

/**
 * Process raw YOLOv8 single-tensor output into Detection[].
 *
 * YOLOv8 output tensor shape: [1, 5, 8400] — row-major Float32Array of 5 * 8400 values.
 * The 5 rows are: cx, cy, w, h, confidence. Coordinates are already normalized 0–1.
 *
 * Steps:
 *   1. Filter candidates below the confidence threshold.
 *   2. Sort survivors by confidence (descending).
 *   3. Run NMS with the given IoU threshold.
 *   4. Clamp coordinates to 0–1.
 *
 * @param raw              - Float32Array containing 5 * 8400 values (row-major [5, 8400])
 * @param confidenceThresh - Minimum confidence to keep a candidate (e.g. 0.25)
 * @param iouThreshold     - IoU threshold for NMS suppression (e.g. 0.5)
 * @returns Detection[] with label "object" and normalized bbox coordinates
 */
export function processYoloOutputs(
  raw: Float32Array,
  confidenceThresh: number,
  iouThreshold: number,
): Detection[] {
  const NUM_BOXES = 8400

  // Collect all candidates above the confidence threshold.
  // Layout is row-major [5, 8400]: value at row r, box b = raw[r * NUM_BOXES + b]
  type Candidate = { x: number; y: number; w: number; h: number; confidence: number }
  const candidates: Candidate[] = []

  for (let b = 0; b < NUM_BOXES; b++) {
    const confidence = raw[4 * NUM_BOXES + b]
    if (confidence < confidenceThresh) continue

    const cx = raw[0 * NUM_BOXES + b]
    const cy = raw[1 * NUM_BOXES + b]
    const w = raw[2 * NUM_BOXES + b]
    const h = raw[3 * NUM_BOXES + b]

    // Convert center-format to top-left corner format.
    candidates.push({ x: cx - w / 2, y: cy - h / 2, w, h, confidence })
  }

  if (candidates.length === 0) return []

  // Sort by confidence descending so NMS keeps the best box first.
  candidates.sort((a, b) => b.confidence - a.confidence)

  const keptIndices = applyNMS(candidates, iouThreshold)

  return keptIndices.map((i) => {
    const c = candidates[i]
    // Clamp coordinates to 0–1.
    const x = Math.max(0, Math.min(1, c.x))
    const y = Math.max(0, Math.min(1, c.y))
    const width = Math.max(0, Math.min(1, c.w))
    const height = Math.max(0, Math.min(1, c.h))
    return {
      label: 'object',
      confidence: c.confidence,
      bbox: { x, y, width, height },
    }
  })
}
