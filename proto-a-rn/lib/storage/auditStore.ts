import type { AuditRecord } from './types'

let audits: AuditRecord[] = []

export function saveAudit(record: AuditRecord): void {
  audits.unshift(record)
}

export function getAudits(): AuditRecord[] {
  return [...audits]
}

export function clearAudits(): void {
  audits = []
}
