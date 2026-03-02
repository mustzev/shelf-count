import type { Detection } from './types'

export function filterByConfidence(
  detections: Detection[],
  threshold: number,
): Detection[] {
  return detections.filter((d) => d.confidence >= threshold)
}

export function mapOutputToDetections(
  boxes: number[],
  classes: number[],
  scores: number[],
  count: number,
  labels: string[],
): Detection[] {
  const detections: Detection[] = []

  for (let i = 0; i < count; i++) {
    const y1 = boxes[i * 4]
    const x1 = boxes[i * 4 + 1]
    const y2 = boxes[i * 4 + 2]
    const x2 = boxes[i * 4 + 3]

    detections.push({
      label: labels[classes[i]] ?? `class_${classes[i]}`,
      confidence: scores[i],
      bbox: {
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
      },
    })
  }

  return detections
}
