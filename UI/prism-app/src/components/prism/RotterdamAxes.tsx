'use client'

// Rotterdam axes (§8).
//
// The clinical/biochemical hyperandrogenism split is preserved, not collapsed —
// otherwise "hyperandrogenism: met" reads as a lab result when the only evidence
// is self-reported acne.
//
// `not_assessable` is styled neutrally: never obtained ≠ looked for and absent.

import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

import { Card, SectionHeading, StatusPill } from './Primitives'
import {
  androgenicEvidenceSentence,
  axisLabel,
  axisStatusLabel,
  axisTone,
  humanizeCode,
} from '@/lib/present'
import type { AxisView, WebsitePCOSProfileResponse } from '@/types/api'

/** Clinical reading order; anything unlisted follows, alphabetically. */
const AXIS_ORDER = [
  'ovulatory_dysfunction',
  'hyperandrogenism_clinical',
  'hyperandrogenism_biochemical',
  'polycystic_ovarian_morphology',
]

function axisRank(key: string): number {
  const index = AXIS_ORDER.indexOf(key)
  return index === -1 ? AXIS_ORDER.length : index
}

function AxisRow({ axisKey, axis }: { axisKey: string; axis: AxisView }) {
  const [open, setOpen] = useState(false)
  const hasDetail =
    Boolean(axis.supporting_evidence?.length) ||
    Boolean(axis.missing_evidence?.length) ||
    Boolean(axis.caveats?.length)

  return (
    <div className="border-t border-neutral-100 py-4 first:border-t-0 first:pt-0">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-neutral-900">{axisLabel(axisKey)}</p>
          {axis.reason && <p className="mt-1 text-sm text-neutral-500">{axis.reason}</p>}
        </div>
        <StatusPill tone={axisTone(axis.status)}>{axisStatusLabel(axis.status)}</StatusPill>
      </div>

      {axis.biochemical_evidence_available === false && (
        <p className="mt-2 text-xs font-medium text-neutral-500">
          Biochemical androgenic evidence: not available
        </p>
      )}

      {hasDetail && (
        <>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-neutral-500 hover:text-neutral-800"
            aria-expanded={open}
          >
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            {open ? 'Hide evidence' : 'Show evidence'}
          </button>

          {open && (
            <div className="mt-3 space-y-3 rounded-xl bg-neutral-50 p-4 text-sm">
              {Boolean(axis.supporting_evidence?.length) && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    Available evidence supports
                  </p>
                  <ul className="mt-1 space-y-1 text-neutral-700">
                    {axis.supporting_evidence!.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {Boolean(axis.missing_evidence?.length) && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    Not available
                  </p>
                  <p className="mt-1 text-neutral-600">
                    {axis.missing_evidence!.map(humanizeCode).join(', ')}
                  </p>
                </div>
              )}

              {Boolean(axis.caveats?.length) && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    Caveats
                  </p>
                  <ul className="mt-1 space-y-1 text-neutral-600">
                    {axis.caveats!.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function RotterdamAxes({ report }: { report: WebsitePCOSProfileResponse }) {
  const axes = Object.entries(report.rotterdam_axes ?? {}).sort(
    ([a], [b]) => axisRank(a) - axisRank(b),
  )
  const androgenic = androgenicEvidenceSentence(report.androgenic_evidence_source)

  return (
    <Card>
      <SectionHeading
        title="Rotterdam-axis evidence"
        subtitle="Assessed against the 2023 International Evidence-based Guideline."
      />

      <div className="rounded-xl bg-neutral-50 p-4">
        <p className="text-sm font-medium text-neutral-800">{androgenic.headline}</p>
        {androgenic.caveat && (
          <p className="mt-1 text-sm text-neutral-500">{androgenic.caveat}</p>
        )}
      </div>

      <div className="mt-4">
        {axes.length === 0 ? (
          <p className="text-sm text-neutral-500">No axes were assessed for this patient.</p>
        ) : (
          axes.map(([key, axis]) => <AxisRow key={key} axisKey={key} axis={axis} />)
        )}
      </div>
    </Card>
  )
}
