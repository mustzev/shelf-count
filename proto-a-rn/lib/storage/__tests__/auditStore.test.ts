import { clearAudits, getAudits, saveAudit } from '../auditStore'

describe('auditStore', () => {
  beforeEach(() => clearAudits())

  it('saves and retrieves an audit', () => {
    const record = {
      id: '1',
      timestamp: Date.now(),
      photoPath: '/path/to/photo.jpg',
      result: { detections: [], count: 0, inferenceTimeMs: 100 },
    }
    saveAudit(record)
    const audits = getAudits()
    expect(audits).toHaveLength(1)
    expect(audits[0].id).toBe('1')
  })

  it('stores audits in reverse chronological order', () => {
    saveAudit({
      id: '1',
      timestamp: 1000,
      photoPath: '/a.jpg',
      result: { detections: [], count: 0, inferenceTimeMs: 50 },
    })
    saveAudit({
      id: '2',
      timestamp: 2000,
      photoPath: '/b.jpg',
      result: { detections: [], count: 3, inferenceTimeMs: 75 },
    })
    const audits = getAudits()
    expect(audits[0].id).toBe('2')
    expect(audits[1].id).toBe('1')
  })

  it('returns a copy, not a reference', () => {
    saveAudit({
      id: '1',
      timestamp: Date.now(),
      photoPath: '/x.jpg',
      result: { detections: [], count: 0, inferenceTimeMs: 100 },
    })
    const a = getAudits()
    const b = getAudits()
    expect(a).not.toBe(b)
  })
})
