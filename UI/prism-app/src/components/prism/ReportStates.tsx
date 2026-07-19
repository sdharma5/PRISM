'use client'

// Loading, failure and partial-result states (§17).
//
// Two failure surfaces: ReportError replaces the page when there's nothing to
// show, PartialNotice sits above a complete report and names what was missing.
//
// Wording separates "couldn't reach the service" from "the service rejected
// this input" — otherwise the user can't tell whether retrying would help.

import { AlertTriangle, Loader2, RefreshCw, WifiOff } from 'lucide-react'

import { Card } from './Primitives'
import type { ApiError } from '@/lib/apiClient'

export function ReportLoading({ message = 'Running PRISM analysis…' }: { message?: string }) {
  return (
    <Card className="flex items-center gap-3">
      <Loader2 className="h-4 w-4 animate-spin text-neutral-400" />
      <div>
        <p className="text-sm font-medium text-neutral-800">{message}</p>
        <p className="mt-0.5 text-xs text-neutral-500">
          Loading models and scoring the available evidence.
        </p>
      </div>
    </Card>
  )
}

function describe(error: ApiError): { title: string; body: string; retryable: boolean } {
  if (error.status === 0) {
    return {
      title: 'The analysis service is unreachable',
      body:
        'PRISM could not connect to the inference API. If you are running it locally, ' +
        'check that the service is started.',
      retryable: true,
    }
  }
  if (error.status === 422) {
    return {
      title: 'The submitted information could not be processed',
      body:
        'The analysis service rejected this input as invalid. Retrying will not help ' +
        'until the input is corrected.',
      retryable: false,
    }
  }
  if (error.status === 503) {
    return {
      title: 'This analysis is not currently available',
      body:
        typeof error.detail === 'object' && error.detail !== null && 'reason' in error.detail
          ? String((error.detail as { reason?: unknown }).reason)
          : 'A required model branch is not available for inference.',
      retryable: false,
    }
  }
  return {
    title: 'The analysis could not be completed',
    body: error.message,
    retryable: true,
  }
}

export function ReportError({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const { title, body, retryable } = describe(error)
  const Icon = error.status === 0 ? WifiOff : AlertTriangle

  return (
    <Card className="border-rose-200 bg-rose-50/50">
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 h-5 w-5 shrink-0 text-rose-500" />
        <div className="flex-1">
          <p className="text-sm font-semibold text-rose-900">{title}</p>
          <p className="mt-1 text-sm text-rose-800/80">{body}</p>
          {retryable && onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-700"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Try again
            </button>
          )}
        </div>
      </div>
    </Card>
  )
}

/** Leads with what was produced, then what wasn't. */
export function PartialNotice({
  missing,
  warnings,
}: {
  missing: string[]
  warnings: string[]
}) {
  if (missing.length === 0 && warnings.length === 0) return null

  const readable = missing.map((m) =>
    m === 'ovarian_ultrasound'
      ? 'ultrasound'
      : m === 'longitudinal_hormonal_state'
        ? 'longitudinal'
        : m.replace(/_/g, ' '),
  )

  return (
    <div className="rounded-2xl border border-sky-200 bg-sky-50/60 p-4">
      {readable.length > 0 && (
        <p className="text-sm text-sky-900">
          Your profile was generated from the available evidence.{' '}
          {readable.join(' and ')} analysis was unavailable.
        </p>
      )}
      {warnings.length > 0 && (
        <ul className="mt-2 space-y-1">
          {warnings.map((warning) => (
            <li key={warning} className="text-xs text-sky-800/80">
              {warning}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
