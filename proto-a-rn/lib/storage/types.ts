import type { DetectionResult } from '../ml/types'

export interface AuditRecord {
  id: string
  timestamp: number
  photoPath: string
  result: DetectionResult
}
