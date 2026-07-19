'use client'

// One report per page, shared by every section. Per-panel fetching would run
// inference several times and could render panels from different runs side by
// side.

import { useCallback, useEffect, useState } from 'react'

import { ApiError, apiMode } from './apiClient'
import { getModelStatus, getPatientReport } from './api'
import { loadAssessment, onAssessmentChange } from './reportStore'
import type { ModelStatusResponse, WebsitePMOSProfileResponse } from '@/types/api'

export interface ReportState {
  report: WebsitePMOSProfileResponse | null
  status: ModelStatusResponse | null
  loading: boolean
  /** Set when the report itself could not be produced. */
  error: ApiError | null
  /** Status call failed but the report didn't — don't blank the page for it. */
  statusError: ApiError | null
  /** Nothing run yet and nothing stored. Not an error — prompt for data instead. */
  noAssessment: boolean
  refresh: () => void
}

function asApiError(cause: unknown): ApiError {
  return cause instanceof ApiError
    ? cause
    : new ApiError(cause instanceof Error ? cause.message : 'Unexpected error', 0, cause)
}

export function usePatientReport(patientId?: string): ReportState {
  const [report, setReport] = useState<WebsitePMOSProfileResponse | null>(null)
  const [status, setStatus] = useState<ModelStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ApiError | null>(null)
  const [statusError, setStatusError] = useState<ApiError | null>(null)
  const [noAssessment, setNoAssessment] = useState(false)
  const [nonce, setNonce] = useState(0)

  const refresh = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    setStatusError(null)
    setNoAssessment(false)

    // Stored assessment wins -- the API keeps nothing between requests, so a
    // re-request without the answers would score an empty bundle.
    const stored = patientId ? null : loadAssessment()

    const reportPromise = stored
      ? Promise.resolve(stored.report)
      : apiMode() === 'mock' || patientId
        ? getPatientReport(patientId)
        : Promise.resolve(null)

    Promise.allSettled([reportPromise, getModelStatus()])
      .then(([reportResult, statusResult]) => {
        if (cancelled) return

        if (reportResult.status === 'fulfilled') {
          if (reportResult.value) {
            setReport(reportResult.value)
          } else {
            setNoAssessment(true)
          }
        } else {
          setError(asApiError(reportResult.reason))
        }

        if (statusResult.status === 'fulfilled') {
          setStatus(statusResult.value)
        } else {
          setStatusError(asApiError(statusResult.reason))
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [patientId, nonce])

  // Re-read when an assessment is saved or cleared, including from another tab.
  useEffect(() => onAssessmentChange(() => setNonce((n) => n + 1)), [])

  return { report, status, loading, error, statusError, noAssessment, refresh }
}

/** Whether a named branch may be presented as working. */
export function branchAvailable(
  status: ModelStatusResponse | null,
  branch: 'static_clinical' | 'temporal_state' | 'ovarian_ultrasound',
): boolean {
  // Unknown is not "available" — otherwise a failed status call advertises a
  // gated branch.
  return Boolean(status?.[branch]?.available)
}

export function branchReason(
  status: ModelStatusResponse | null,
  branch: 'static_clinical' | 'temporal_state' | 'ovarian_ultrasound',
): string | null {
  return status?.[branch]?.reason ?? null
}

export { apiMode }
