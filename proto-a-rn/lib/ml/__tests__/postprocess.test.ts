import {
  filterByConfidence,
  mapOutputToDetections,
  processYoloOutputs,
  computeIoU,
  applyNMS,
} from '../postprocess'

describe('filterByConfidence', () => {
  it('filters detections below threshold', () => {
    const detections = [
      {
        label: 'bottle',
        confidence: 0.9,
        bbox: { x: 0, y: 0, width: 10, height: 10 },
      },
      {
        label: 'cup',
        confidence: 0.3,
        bbox: { x: 20, y: 20, width: 10, height: 10 },
      },
      {
        label: 'book',
        confidence: 0.6,
        bbox: { x: 40, y: 40, width: 10, height: 10 },
      },
    ]
    const result = filterByConfidence(detections, 0.5)
    expect(result).toHaveLength(2)
    expect(result[0].label).toBe('bottle')
    expect(result[1].label).toBe('book')
  })
})

describe('mapOutputToDetections', () => {
  it('maps raw model output tensors to Detection objects', () => {
    const boxes = [0.1, 0.2, 0.5, 0.6] // [y1, x1, y2, x2] normalized
    const classes = [0] // index 0 maps to 'bottle' in the labels array
    const scores = [0.85]
    const count = 1
    const labels = ['bottle']

    const result = mapOutputToDetections(boxes, classes, scores, count, labels)

    expect(result).toHaveLength(1)
    expect(result[0].label).toBe('bottle')
    expect(result[0].confidence).toBe(0.85)
    expect(result[0].bbox.x).toBeCloseTo(0.2)
    expect(result[0].bbox.y).toBeCloseTo(0.1)
    expect(result[0].bbox.width).toBeCloseTo(0.4) // x2 - x1
    expect(result[0].bbox.height).toBeCloseTo(0.4) // y2 - y1
  })
})

// ---------------------------------------------------------------------------
// computeIoU
// ---------------------------------------------------------------------------

describe('computeIoU', () => {
  it('returns 1 for identical boxes', () => {
    expect(computeIoU(0, 0, 10, 10, 0, 0, 10, 10)).toBeCloseTo(1)
  })

  it('returns 0 for non-overlapping boxes', () => {
    // Box A: x=0, y=0, w=10, h=10  →  right edge at 10
    // Box B: x=20, y=20, w=10, h=10  →  no overlap
    expect(computeIoU(0, 0, 10, 10, 20, 20, 10, 10)).toBe(0)
  })

  it('returns 0 for adjacent but non-overlapping boxes', () => {
    // Box A ends at x=10; Box B starts at x=10 — zero-width intersection.
    expect(computeIoU(0, 0, 10, 10, 10, 0, 10, 10)).toBe(0)
  })

  it('computes IoU for partially overlapping boxes', () => {
    // Box A: (0,0) 10x10 → area 100
    // Box B: (5,5) 10x10 → area 100
    // Intersection: (5,5) 5x5 = 25
    // Union: 100 + 100 - 25 = 175
    expect(computeIoU(0, 0, 10, 10, 5, 5, 10, 10)).toBeCloseTo(25 / 175)
  })

  it('computes IoU when one box is fully inside the other', () => {
    // Outer: (0,0) 10x10 → area 100
    // Inner: (2,2)  6x6  → area 36
    // Intersection: 36  Union: 64 + 36 = 100
    expect(computeIoU(0, 0, 10, 10, 2, 2, 6, 6)).toBeCloseTo(36 / 100)
  })
})

// ---------------------------------------------------------------------------
// applyNMS
// ---------------------------------------------------------------------------

describe('applyNMS', () => {
  it('keeps the single candidate when there is only one', () => {
    const candidates = [{ x: 0, y: 0, w: 10, h: 10, confidence: 0.9 }]
    expect(applyNMS(candidates, 0.5)).toEqual([0])
  })

  it('keeps both boxes when they do not overlap', () => {
    const candidates = [
      { x: 0, y: 0, w: 10, h: 10, confidence: 0.9 },
      { x: 50, y: 50, w: 10, h: 10, confidence: 0.8 },
    ]
    const kept = applyNMS(candidates, 0.5)
    expect(kept).toEqual([0, 1])
  })

  it('suppresses the lower-confidence duplicate when IoU exceeds threshold', () => {
    // Two nearly identical boxes — IoU ~ 1.
    const candidates = [
      { x: 0, y: 0, w: 10, h: 10, confidence: 0.9 },
      { x: 0, y: 0, w: 10, h: 10, confidence: 0.7 },
    ]
    const kept = applyNMS(candidates, 0.5)
    expect(kept).toEqual([0])
  })

  it('keeps both boxes when IoU is below threshold', () => {
    // Slight overlap but IoU < 0.5.
    // Box A: (0,0) 10x10, Box B: (8,8) 10x10
    // Intersection: (8,8) 2x2 = 4, Union = 196, IoU ≈ 0.02
    const candidates = [
      { x: 0, y: 0, w: 10, h: 10, confidence: 0.9 },
      { x: 8, y: 8, w: 10, h: 10, confidence: 0.8 },
    ]
    const kept = applyNMS(candidates, 0.5)
    expect(kept).toEqual([0, 1])
  })
})

// ---------------------------------------------------------------------------
// processYoloOutputs
// ---------------------------------------------------------------------------

/**
 * Build a minimal [1, 5, 8400] raw Float32Array with a small number of boxes
 * set to known values and the rest zeroed (confidence=0, filtered out).
 *
 * Layout: row-major [5, 8400].
 *   row 0 → cx values:   raw[0 * 8400 + b]
 *   row 1 → cy values:   raw[1 * 8400 + b]
 *   row 2 → w  values:   raw[2 * 8400 + b]
 *   row 3 → h  values:   raw[3 * 8400 + b]
 *   row 4 → conf values: raw[4 * 8400 + b]
 */
function buildRaw(
  boxes: Array<{ cx: number; cy: number; w: number; h: number; conf: number }>,
): Float32Array {
  const NUM_BOXES = 8400
  const raw = new Float32Array(5 * NUM_BOXES) // all zeros
  for (let i = 0; i < boxes.length; i++) {
    const { cx, cy, w, h, conf } = boxes[i]
    raw[0 * NUM_BOXES + i] = cx
    raw[1 * NUM_BOXES + i] = cy
    raw[2 * NUM_BOXES + i] = w
    raw[3 * NUM_BOXES + i] = h
    raw[4 * NUM_BOXES + i] = conf
  }
  return raw
}

describe('processYoloOutputs', () => {
  it('returns an empty array when no candidates exceed the threshold', () => {
    const raw = new Float32Array(5 * 8400) // all zeros → all confidence = 0
    const result = processYoloOutputs(raw, 0.25, 0.5)
    expect(result).toHaveLength(0)
  })

  it('returns a single detection for one above-threshold box', () => {
    // cx=0.5, cy=0.5, w=0.2, h=0.2, conf=0.8 (already normalized)
    // x = 0.5 - 0.1 = 0.4,  y = 0.5 - 0.1 = 0.4
    const raw = buildRaw([{ cx: 0.5, cy: 0.5, w: 0.2, h: 0.2, conf: 0.8 }])
    const result = processYoloOutputs(raw, 0.25, 0.5)

    expect(result).toHaveLength(1)
    expect(result[0].label).toBe('object')
    expect(result[0].confidence).toBeCloseTo(0.8)
    expect(result[0].bbox.x).toBeCloseTo(0.4)
    expect(result[0].bbox.y).toBeCloseTo(0.4)
    expect(result[0].bbox.width).toBeCloseTo(0.2)
    expect(result[0].bbox.height).toBeCloseTo(0.2)
  })

  it('filters out boxes below the confidence threshold', () => {
    const raw = buildRaw([
      { cx: 0.2, cy: 0.2, w: 0.08, h: 0.08, conf: 0.1 }, // below 0.25
      { cx: 0.5, cy: 0.5, w: 0.08, h: 0.08, conf: 0.9 }, // kept
    ])
    const result = processYoloOutputs(raw, 0.25, 0.5)
    expect(result).toHaveLength(1)
    expect(result[0].confidence).toBeCloseTo(0.9)
  })

  it('suppresses duplicate boxes via NMS', () => {
    const raw = buildRaw([
      { cx: 0.5, cy: 0.5, w: 0.3, h: 0.3, conf: 0.9 },
      { cx: 0.51, cy: 0.51, w: 0.3, h: 0.3, conf: 0.7 },
    ])
    const result = processYoloOutputs(raw, 0.25, 0.5)
    expect(result).toHaveLength(1)
    expect(result[0].confidence).toBeCloseTo(0.9)
  })

  it('keeps two well-separated boxes after NMS', () => {
    const raw = buildRaw([
      { cx: 0.1, cy: 0.1, w: 0.06, h: 0.06, conf: 0.85 },
      { cx: 0.9, cy: 0.9, w: 0.06, h: 0.06, conf: 0.75 },
    ])
    const result = processYoloOutputs(raw, 0.25, 0.5)
    expect(result).toHaveLength(2)
  })

  it('handles a full-image box', () => {
    // cx=0.5, cy=0.5, w=1.0, h=1.0 → x=0, y=0, width=1, height=1
    const raw = buildRaw([{ cx: 0.5, cy: 0.5, w: 1.0, h: 1.0, conf: 0.9 }])
    const result = processYoloOutputs(raw, 0.25, 0.5)

    expect(result).toHaveLength(1)
    expect(result[0].bbox.x).toBeCloseTo(0)
    expect(result[0].bbox.y).toBeCloseTo(0)
    expect(result[0].bbox.width).toBeCloseTo(1)
    expect(result[0].bbox.height).toBeCloseTo(1)
  })

  it('clamps out-of-bounds coordinates to [0, 1]', () => {
    // cx=0.02, cy=0.02, w=1.0, h=1.0 → x = 0.02 - 0.5 = -0.48 → clamped to 0
    const raw = buildRaw([{ cx: 0.02, cy: 0.02, w: 1.0, h: 1.0, conf: 0.9 }])
    const result = processYoloOutputs(raw, 0.25, 0.5)

    expect(result).toHaveLength(1)
    expect(result[0].bbox.x).toBeGreaterThanOrEqual(0)
    expect(result[0].bbox.y).toBeGreaterThanOrEqual(0)
    expect(result[0].bbox.width).toBeLessThanOrEqual(1)
    expect(result[0].bbox.height).toBeLessThanOrEqual(1)
  })

  it('returns detections sorted by confidence (highest first) after NMS', () => {
    const raw = buildRaw([
      { cx: 0.1, cy: 0.1, w: 0.05, h: 0.05, conf: 0.6 },
      { cx: 0.5, cy: 0.5, w: 0.05, h: 0.05, conf: 0.9 },
      { cx: 0.9, cy: 0.9, w: 0.05, h: 0.05, conf: 0.75 },
    ])
    const result = processYoloOutputs(raw, 0.25, 0.5)

    expect(result).toHaveLength(3)
    expect(result[0].confidence).toBeGreaterThanOrEqual(result[1].confidence)
    expect(result[1].confidence).toBeGreaterThanOrEqual(result[2].confidence)
  })
})
