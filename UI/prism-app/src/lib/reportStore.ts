'use client'

// The last completed assessment, kept in the browser so it survives navigation
// and refreshes. The API is stateless and keeps nothing between requests, so the
// client holds the result.
//
// localStorage rather than server-side: this is identifiable clinical data and a
// prototype shouldn't outlive the user's own browser with a copy of it.

import type { WebsitePMOSProfileResponse } from '@/types/api'

const KEY = 'prism.lastAssessment.v1'

export interface StoredAssessment {
  report: WebsitePMOSProfileResponse
  /** The answers that produced it, so the form can be re-opened pre-filled. */
  answers: Record<string, string | boolean | undefined>
  savedAt: string
}

export function saveAssessment(
  report: WebsitePMOSProfileResponse,
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
    if (!parsed?.report?.pmos_assessment || !parsed.report.report_id) return null
    return parsed
  } catch {
    return null
  }
}

// -- injected temporal series ----------------------------------------------
//
// The demo-temporal inject and the main analysis both write the same assessment
// key, so whichever runs last wins. Persisting the injected days here lets the
// main run re-send them, so running an analysis after injecting keeps the
// longitudinal branch instead of silently dropping it.

const TEMPORAL_KEY = 'prism.injectedTemporal.v1'

export function saveInjectedTemporal(days: unknown[]): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(TEMPORAL_KEY, JSON.stringify(days))
  } catch {
    /* full or disabled storage — the inject's own report is still on screen */
  }
}

export function loadInjectedTemporal(): unknown[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(TEMPORAL_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function clearInjectedTemporal(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(TEMPORAL_KEY)
  } catch {
    /* nothing to clear */
  }
}

// -- handled events --------------------------------------------------------
//
// The event ledger persists across sessions, so re-opening the form re-fetches
// the same proposed events and would ask the patient to review them again even
// when they added nothing new. Recording which event ids the patient has already
// dealt with lets the review step surface only genuinely new proposals.

const REVIEWED_KEY = 'prism.reviewedEvents.v1'

export function loadReviewedEventIds(): Set<string> {
  if (typeof window === 'undefined') return new Set()
  try {
    const raw = window.localStorage.getItem(REVIEWED_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return new Set(Array.isArray(parsed) ? (parsed as string[]) : [])
  } catch {
    return new Set()
  }
}

export function markEventsReviewed(ids: string[]): void {
  if (typeof window === 'undefined' || ids.length === 0) return
  try {
    const current = loadReviewedEventIds()
    for (const id of ids) current.add(id)
    window.localStorage.setItem(REVIEWED_KEY, JSON.stringify(Array.from(current)))
  } catch {
    /* full or disabled storage — worst case the prompt reappears once */
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
