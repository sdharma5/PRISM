'use client'

// Source data and model provenance (§6.8, §13, §18).
//
// Branch availability comes from /api/v1/models/status, never hardcoded (§14),
// so a branch switched off server-side goes dark here without a redeploy.
//
// Ultrasound is a deliberate gate, not an absence — its reason is shown verbatim
// rather than paraphrased into something reassuring.

import { useState } from 'react'
import { Database, ChevronDown, ChevronRight, FlaskConical, Watch, Image as ImageIcon } from 'lucide-react'

import { Card, SectionHeading, StatusPill } from './Primitives'
import { formatCoverage, humanizeCode } from '@/lib/present'
import type { ModelStatusResponse, WebsitePCOSProfileResponse } from '@/types/api'

const BRANCH_META = {
  static_clinical: { label: 'Clinical & laboratory', Icon: FlaskConical },
  temporal_state: { label: 'Longitudinal measurements', Icon: Watch },
  ovarian_ultrasound: { label: 'Ovarian ultrasound', Icon: ImageIcon },
} as const

type BranchKey = keyof typeof BRANCH_META

export default function SourceData({
  report,
  status,
}: {
  report: WebsitePCOSProfileResponse
  status: ModelStatusResponse | null
}) {
  const [open, setOpen] = useState(false)

  return (
    <Card>
      <SectionHeading
        title="Source data and models"
        subtitle="Every displayed claim traces to one of these."
      />

      <ul className="space-y-3">
        {(Object.keys(BRANCH_META) as BranchKey[]).map((key) => {
          const meta = BRANCH_META[key]
          const branch = status?.[key]
          const contributed = report.available_modalities?.some((m) =>
            key === 'static_clinical'
              ? m === 'static_clinical'
              : key === 'temporal_state'
                ? m === 'longitudinal_hormonal_state'
                : m === 'ovarian_ultrasound',
          )

          const experimental = branch && branch.trained && !branch.validated_for_inference

          return (
            <li key={key} className="rounded-xl border border-neutral-200 p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3">
                  <meta.Icon className="mt-0.5 h-4 w-4 shrink-0 text-neutral-400" />
                  <div>
                    <p className="text-sm font-semibold text-neutral-900">{meta.label}</p>
                    <p className="mt-0.5 text-xs text-neutral-500">
                      {branch?.version ?? 'Version unknown'}
                    </p>
                  </div>
                </div>
                {contributed ? (
                  <StatusPill tone="ok">Contributed</StatusPill>
                ) : experimental ? (
                  <StatusPill tone="warn">Experimental — not available</StatusPill>
                ) : (
                  <StatusPill tone="neutral">Not provided</StatusPill>
                )}
              </div>

              {/* Verbatim. */}
              {branch?.reason && (
                <p className="mt-3 rounded-lg bg-neutral-50 p-3 text-xs text-neutral-600">
                  {branch.reason}
                </p>
              )}
            </li>
          )
        })}
      </ul>

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-neutral-500 hover:text-neutral-800"
        aria-expanded={open}
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        Provenance detail
      </button>

      {open && (
        <div className="mt-3 space-y-4 rounded-xl bg-neutral-50 p-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Report
            </p>
            <dl className="mt-1 space-y-1 text-xs text-neutral-600">
              <div className="flex gap-2">
                <dt className="text-neutral-400">Report ID</dt>
                <dd className="font-tabular">{report.report_id}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-neutral-400">Generated</dt>
                <dd>{new Date(report.generated_at).toLocaleString()}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-neutral-400">Schema</dt>
                <dd className="font-tabular">{report.schema_version}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-neutral-400">Combination</dt>
                <dd>{report.provenance?.combination_mode ?? 'unknown'}</dd>
              </div>
            </dl>
          </div>

          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Contributing components
            </p>
            <ul className="mt-1 space-y-1.5">
              {(report.provenance?.records ?? []).map((record, index) => (
                <li key={`${record.label}-${index}`} className="text-xs text-neutral-600">
                  <span className="font-medium text-neutral-800">
                    {humanizeCode(record.label)}
                  </span>
                  {' · '}
                  {record.origin === 'model_estimate' ? 'Model estimate' : 'Rule-based'}
                  {record.model_version ? ` · ${record.model_version}` : ''}
                  {record.confidence != null
                    ? ` · confidence ${formatCoverage(record.confidence)}`
                    : ''}
                </li>
              ))}
            </ul>
          </div>

          {/* §6.10 — the guideline axes are thresholds, not learned. */}
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
                Learned components
              </p>
              <p className="mt-1 text-xs text-neutral-600">
                {report.learned_components_used?.join(', ') || 'None'}
              </p>
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
                Rule-based components
              </p>
              <p className="mt-1 text-xs text-neutral-600">
                {report.rule_based_components_used?.join(', ') || 'None'}
              </p>
            </div>
          </div>

          <p className="flex items-start gap-2 text-xs text-neutral-400">
            <Database className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            Cross-modal evidence is combined by transparent rules, not a jointly
            trained fusion model.
          </p>
        </div>
      )}
    </Card>
  )
}
