export interface FrameData {
  pixels: Uint8Array
  width: number
  height: number
  bytesPerRow: number
}

let stored: FrameData | null = null

export function setFrame(data: FrameData): void {
  stored = data
}

export function getFrame(): FrameData | null {
  return stored
}

export function clearFrame(): void {
  stored = null
}
