import { resizeFromRgb, resizeNormalize } from '../preprocess'

describe('resizeNormalize', () => {
  it('returns a Uint8Array of correct size for target dimensions', () => {
    // Simulate a small 2x2 RGBA image (8 bytes per row, 4 bytes per pixel)
    const rgba = new Uint8Array([
      255,
      0,
      0,
      255, // red pixel
      0,
      255,
      0,
      255, // green pixel
      0,
      0,
      255,
      255, // blue pixel
      255,
      255,
      0,
      255, // yellow pixel
    ])
    const result = resizeNormalize(rgba, 2, 2, 2, 2)
    // Output should be 2*2*3 = 12 bytes (RGB, no alpha)
    expect(result.length).toBe(12)
    // First pixel: R=255, G=0, B=0
    expect(result[0]).toBe(255)
    expect(result[1]).toBe(0)
    expect(result[2]).toBe(0)
  })

  it('resizes from larger to smaller dimensions', () => {
    // 4x4 RGBA → 2x2 RGB (nearest neighbor sampling)
    const rgba = new Uint8Array(4 * 4 * 4) // 64 bytes
    // Fill with a known pattern: top-left quadrant red, rest green
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const i = (y * 4 + x) * 4
        if (x < 2 && y < 2) {
          rgba[i] = 255
          rgba[i + 1] = 0
          rgba[i + 2] = 0
          rgba[i + 3] = 255 // red
        } else {
          rgba[i] = 0
          rgba[i + 1] = 255
          rgba[i + 2] = 0
          rgba[i + 3] = 255 // green
        }
      }
    }
    const result = resizeNormalize(rgba, 4, 4, 2, 2)
    expect(result.length).toBe(12) // 2*2*3
    // Top-left pixel should be red
    expect(result[0]).toBe(255)
    expect(result[1]).toBe(0)
    expect(result[2]).toBe(0)
  })
})

describe('resizeFromRgb', () => {
  it('handles 2x2 identity (no padding)', () => {
    const input = new Uint8Array([
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
    ])
    const result = resizeFromRgb(input, 2, 2, 6, 2, 2)
    expect(Array.from(result)).toEqual([
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
    ])
  })

  it('handles row padding (bytesPerRow > width * 3)', () => {
    const input = new Uint8Array([
      10, 20, 30, 40, 50, 60, 0, 0, 70, 80, 90, 100, 110, 120, 0, 0,
    ])
    const result = resizeFromRgb(input, 2, 2, 8, 2, 2)
    expect(Array.from(result)).toEqual([
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
    ])
  })

  it('downscales 4x4 to 2x2', () => {
    const input = new Uint8Array(4 * 12) // all zeros
    input[0] = 255 // pixel (0,0) R
    const result = resizeFromRgb(input, 4, 4, 12, 2, 2)
    expect(result[0]).toBe(255)
    expect(result.length).toBe(2 * 2 * 3)
  })
})
