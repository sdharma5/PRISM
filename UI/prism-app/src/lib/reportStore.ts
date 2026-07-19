'use client'

// The last completed assessment, kept in the browser so it survives navigation
// and refreshes. The API is stateless and keeps nothing between requests, so the
// client holds the result.
//
// localStorage rather than server-side: this is identifiable clinical data and a
// prototype shouldn't outlive the user's own browser with a copy of it.

import type { WebsitePCOSProfileResponse } from '@/types/api'

const KEY = 'prism.lastAssessment.v1'

export interface StoredAssessment {
  report: WebsitePCOSProfileResponse
  /** The answers that produced it, so the form can be re-opened pre-filled. */
  answers: Record<string, string | boolean | undefined>
  savedAt: string
}

export function saveAssessment(
  report: WebsitePCOSProfileResponse,
  answers: Record<string, string | boolean | undefined> = {},
): void {
  if (typeof window === 'undefined') return
  try {
    const payload: StoredAssessment = {
      report,
      answers,
      savedAt: new Date().toISOString(),
    }
    window.localStorage.setItem(KEY, JSON.stringify(payload))
    // `storage` only fires in other tabs, so notify this one too.
    window.dispatchEvent(new CustomEvent('prism:assessment-changed'))
  } catch {
    // Full or disabled storage shouldn't lose the result that's already on screen.
  }
}

export function loadAssessment(): StoredAssessment | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as StoredAssessment
    // Older contract, missing fields the UI now reads — discard rather than
    // half-render it.
    if (!parsed?.report?.pcos_assessment || !parsed.report.report_id) return null
    return parsed
  } catch {
    return null
  }
}

export function clearStoredAssessment(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(KEY)
    window.dispatchEvent(new CustomEvent('prism:assessment-changed'))
  } catch {
    /* nothing to clear */
  }
}

/** Subscribe to changes, including from other tabs. */
export function onAssessmentChange(handler: () => void): () => void {
  if (typeof window === 'undefined') return () => {}
  const storage = (e: StorageEvent) => e.key === KEY && handler()
  window.addEventListener('prism:assessment-changed', handler)
  window.addEventListener('storage', storage)
  return () => {
    window.removeEventListener('prism:assessment-changed', handler)
    window.removeEventListener('storage', storage)
  }
}
