import type { DetectionResult } from './ml/types'

let stored: DetectionResult | null = null

export function setResult(data: DetectionResult): void {
  stored = data
}

export function getResult(): DetectionResult | null {
  return stored
}

export function clearResult(): void {
  stored = null
}
