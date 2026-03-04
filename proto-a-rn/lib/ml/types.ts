export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Detection {
  label: string
  confidence: number
  bbox: BoundingBox
}

export interface DetectionResult {
  detections: Detection[]
  count: number
  inferenceTimeMs: number
  totalTimeMs?: number
}
