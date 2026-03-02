import { filterByConfidence, mapOutputToDetections } from '../postprocess'

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
