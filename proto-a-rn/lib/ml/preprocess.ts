/**
 * Strips alpha channel from RGBA pixel data and resizes to target dimensions.
 * Uses nearest-neighbor sampling for simplicity and speed.
 *
 * EfficientDet-Lite0 expects a 320x320x3 uint8 input tensor, so the typical
 * call is:
 *   resizeNormalize(rgbaPixels, srcWidth, srcHeight, 320, 320)
 *
 * @param rgbaPixels - Raw RGBA byte array from the camera (4 bytes per pixel)
 * @param srcWidth   - Source image width in pixels
 * @param srcHeight  - Source image height in pixels
 * @param targetWidth  - Desired output width (e.g. 320)
 * @param targetHeight - Desired output height (e.g. 320)
 * @returns Uint8Array of size targetWidth * targetHeight * 3 (RGB, no alpha)
 */
export function resizeNormalize(
  rgbaPixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number,
): Uint8Array {
  const rgb = new Uint8Array(targetWidth * targetHeight * 3)
  const xRatio = srcWidth / targetWidth
  const yRatio = srcHeight / targetHeight

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.floor(x * xRatio)
      const srcY = Math.floor(y * yRatio)
      const srcIdx = (srcY * srcWidth + srcX) * 4 // RGBA stride
      const dstIdx = (y * targetWidth + x) * 3 // RGB stride

      rgb[dstIdx] = rgbaPixels[srcIdx] // R
      rgb[dstIdx + 1] = rgbaPixels[srcIdx + 1] // G
      rgb[dstIdx + 2] = rgbaPixels[srcIdx + 2] // B
      // alpha channel (srcIdx + 3) intentionally discarded
    }
  }

  return rgb
}

/**
 * Resizes RGB pixel data (3 bytes/pixel) with optional row padding to target dimensions.
 * Uses nearest-neighbor sampling for simplicity and speed.
 *
 * The frame processor provides RGB data where bytesPerRow may be larger than
 * width * 3 due to memory alignment padding. This function handles that stride
 * correctly, unlike resizeNormalize which assumes tightly-packed RGBA input.
 *
 * @param rgbPixels   - Raw RGB byte array from the frame processor (3 bytes per pixel, padded rows)
 * @param srcWidth    - Source image width in pixels
 * @param srcHeight   - Source image height in pixels
 * @param bytesPerRow - Actual row stride in bytes (>= srcWidth * 3)
 * @param targetWidth  - Desired output width (e.g. 320)
 * @param targetHeight - Desired output height (e.g. 320)
 * @returns Uint8Array of size targetWidth * targetHeight * 3 (RGB, no padding)
 */
export function resizeFromRgb(
  rgbPixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  bytesPerRow: number,
  targetWidth: number,
  targetHeight: number,
): Uint8Array {
  const rgb = new Uint8Array(targetWidth * targetHeight * 3)
  const xRatio = srcWidth / targetWidth
  const yRatio = srcHeight / targetHeight

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.floor(x * xRatio)
      const srcY = Math.floor(y * yRatio)
      const srcIdx = srcY * bytesPerRow + srcX * 3 // RGB stride with row padding
      const dstIdx = (y * targetWidth + x) * 3

      rgb[dstIdx] = rgbPixels[srcIdx]
      rgb[dstIdx + 1] = rgbPixels[srcIdx + 1]
      rgb[dstIdx + 2] = rgbPixels[srcIdx + 2]
    }
  }

  return rgb
}
