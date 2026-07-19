'use client'

// One proposed event, decided in place during intake review.
//
// The actions sit in the collapsed row rather than behind the expander: this is
// a step the patient must act on before running, and a control they have to go
// looking for reads as optional. The detail behind the chevron is the evidence
// for the decision, not the decision itself.

import { useState } from 'react'
import { Check, ChevronDown, ChevronRight, X } from 'lucide-react'

import ConfidenceBar from '@/components/ConfidenceBar'
import type { HormonalHealthEvent } from '@/types'

export default function ReviewEventRow({
  event,
  onConfirm,
  onReject,
}: {
  event: HormonalHealthEvent
  onConfirm: (id: string) => void
  onReject: (id: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const awaiting =
    event.confirmationStatus === 'awaiting_patient_confirmation' ||
    event.confirmationStatus === 'awaiting_clinician_confirmation'
  const rejected = event.confirmationStatus === 'rejected'

  return (
    <div
      className={`rounded-lg border transition-opacity ${
        rejected ? 'border-neutral-200 opacity-50' : 'border-neutral-200'
      }`}
    >
      <div className="flex items-start justify-between gap-3 px-3 py-2.5">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex min-w-0 flex-1 items-start gap-2 text-left"
        >
          {expanded ? (
            <ChevronDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-neutral-400" />
          ) : (
            <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-neutral-400" />
          )}
          <span className="min-w-0">
            <span className="block truncate text-sm font-medium text-neutral-900">
              {event.variableName}
            </span>
            {event.value != null && event.missingnessStatus !== 'not_collected' && (
              <span className="mt-0.5 block text-xs text-neutral-600">
                {String(event.value)}
                {event.unit ? ` ${event.unit}` : ''}
              </span>
            )}
          </span>
        </button>

        <div className="flex shrink-0 items-center gap-1.5">
          {awaiting ? (
            <>
              <button
                type="button"
                onClick={() => onConfirm(event.eventId)}
                className="inline-flex items-center gap-1 rounded-lg border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700 hover:bg-emerald-100"
              >
                <Check className="h-3 w-3" /> Accept
              </button>
              <button
                type="button"
                onClick={() => onReject(event.eventId)}
                className="inline-flex items-center gap-1 rounded-lg border border-red-200 bg-red-50 px-2.5 py-1 text-xs font-medium text-red-600 hover:bg-red-100"
              >
                <X className="h-3 w-3" /> Reject
              </button>
            </>
          ) : (
            <span
              className={`text-xs font-medium ${
                rejected ? 'text-neutral-400' : 'text-emerald-700'
              }`}
            >
              {rejected ? 'Not sent' : 'Included'}
            </span>
          )}
        </div>
      </div>

      {expanded && (
        <div className="space-y-2 border-t border-neutral-100 px-3 pb-3 pt-2">
          {event.evidenceText && (
            <div>
              <p className="mb-1 text-xs text-neutral-400">Evidence text</p>
              <p className="rounded-lg border border-neutral-100 bg-neutral-50 p-2 text-xs text-neutral-600">
                &ldquo;{event.evidenceText}&rdquo;
              </p>
            </div>
          )}
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
            <div>
              <p className="text-neutral-400">Canonical code</p>
              <p className="mt-0.5 font-mono text-neutral-700">{event.canonicalVariableCode}</p>
            </div>
            <div>
              <p className="text-neutral-400">Source</p>
              <p className="mt-0.5 capitalize text-neutral-700">{event.modality}</p>
            </div>
          </div>
          <div>
            <p className="mb-1 text-xs text-neutral-400">Extraction confidence</p>
            <ConfidenceBar value={event.extractionConfidence} size="sm" />
          </div>
        </div>
      )}
    </div>
  )
}
