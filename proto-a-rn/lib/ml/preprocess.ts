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

// ---------------------------------------------------------------------------
// Float32 variants for YOLOv8 (input normalized to 0–1)
// ---------------------------------------------------------------------------

/**
 * Strips alpha channel from RGBA pixel data, resizes to target dimensions, and
 * normalizes pixel values to the range 0–1.
 *
 * YOLOv8 expects a [1, 640, 640, 3] float32 NHWC input tensor with values in
 * 0–1, so the typical call is:
 *   resizeNormalizeFloat(rgbaPixels, srcWidth, srcHeight, 640, 640)
 *
 * @param rgbaPixels   - Raw RGBA byte array from the camera (4 bytes per pixel)
 * @param srcWidth     - Source image width in pixels
 * @param srcHeight    - Source image height in pixels
 * @param targetWidth  - Desired output width (e.g. 640)
 * @param targetHeight - Desired output height (e.g. 640)
 * @returns Float32Array of size targetWidth * targetHeight * 3, values in [0, 1]
 */
export function resizeNormalizeFloat(
  rgbaPixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  const out = new Float32Array(targetWidth * targetHeight * 3)
  const xRatio = srcWidth / targetWidth
  const yRatio = srcHeight / targetHeight

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.floor(x * xRatio)
      const srcY = Math.floor(y * yRatio)
      const srcIdx = (srcY * srcWidth + srcX) * 4 // RGBA stride
      const dstIdx = (y * targetWidth + x) * 3

      out[dstIdx] = rgbaPixels[srcIdx] / 255 // R
      out[dstIdx + 1] = rgbaPixels[srcIdx + 1] / 255 // G
      out[dstIdx + 2] = rgbaPixels[srcIdx + 2] / 255 // B
      // alpha channel (srcIdx + 3) intentionally discarded
    }
  }

  return out
}

/**
 * Resize RGBA pixels to a square target while simultaneously applying a 90° CW
 * rotation. Used when jpeg-js decodes an Android portrait photo whose raw pixel
 * data is stored in landscape orientation (EXIF orientation 6).
 *
 * By rotating during resize we avoid a separate rotation pass and — more
 * importantly — the model sees the same portrait-oriented image that Flutter /
 * Proto B would see, producing identical detection quality.
 *
 * @param rgbaPixels   - Raw RGBA byte array (landscape orientation)
 * @param srcWidth     - Stored landscape width  (e.g. 4080)
 * @param srcHeight    - Stored landscape height (e.g. 3072)
 * @param targetSize   - Square target dimension  (e.g. 640)
 * @returns Float32Array of size targetSize * targetSize * 3, values in [0, 1]
 */
export function resizeNormalizeFloatRotated(
  rgbaPixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  targetSize: number,
): Float32Array {
  const out = new Float32Array(targetSize * targetSize * 3)
  // After 90° CW rotation the portrait dimensions are srcHeight × srcWidth.
  const xRatio = srcHeight / targetSize // portrait width  / target
  const yRatio = srcWidth / targetSize  // portrait height / target

  for (let ty = 0; ty < targetSize; ty++) {
    for (let tx = 0; tx < targetSize; tx++) {
      // Position in the portrait view
      const px = Math.floor(tx * xRatio)
      const py = Math.floor(ty * yRatio)
      // Map portrait → stored landscape (inverse of 90° CW)
      const sx = py
      const sy = srcHeight - 1 - px

      const srcIdx = (sy * srcWidth + sx) * 4 // RGBA stride
      const dstIdx = (ty * targetSize + tx) * 3

      out[dstIdx] = rgbaPixels[srcIdx] / 255
      out[dstIdx + 1] = rgbaPixels[srcIdx + 1] / 255
      out[dstIdx + 2] = rgbaPixels[srcIdx + 2] / 255
    }
  }

  return out
}

/**
 * Resizes padded-row RGB pixel data to target dimensions and normalizes values
 * to the range 0–1.
 *
 * Mirrors resizeFromRgb but returns Float32Array instead of Uint8Array. Use
 * this when the frame processor supplies uint8 RGB data and the model expects
 * float32 input normalized to 0–1 (e.g. YOLOv8).
 *
 * @param rgbPixels    - Raw RGB byte array (3 bytes per pixel, padded rows)
 * @param srcWidth     - Source image width in pixels
 * @param srcHeight    - Source image height in pixels
 * @param bytesPerRow  - Actual row stride in bytes (>= srcWidth * 3)
 * @param targetWidth  - Desired output width (e.g. 640)
 * @param targetHeight - Desired output height (e.g. 640)
 * @returns Float32Array of size targetWidth * targetHeight * 3, values in [0, 1]
 */
export function resizeFromRgbFloat(
  rgbPixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  bytesPerRow: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  const out = new Float32Array(targetWidth * targetHeight * 3)
  const xRatio = srcWidth / targetWidth
  const yRatio = srcHeight / targetHeight

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.floor(x * xRatio)
      const srcY = Math.floor(y * yRatio)
      const srcIdx = srcY * bytesPerRow + srcX * 3 // RGB stride with row padding
      const dstIdx = (y * targetWidth + x) * 3

      out[dstIdx] = rgbPixels[srcIdx] / 255
      out[dstIdx + 1] = rgbPixels[srcIdx + 1] / 255
      out[dstIdx + 2] = rgbPixels[srcIdx + 2] / 255
    }
  }

  return out
}
