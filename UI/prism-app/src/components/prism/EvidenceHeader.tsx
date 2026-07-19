'use client'

// Profile header (§7). The score is the thing most likely to be read as a
// diagnosis, so the framing renders with it at the same prominence.
//
// No static branch means no score — say so rather than showing a zero or a dash.

import { AlertTriangle, FlaskConical, Info } from 'lucide-react'

import { Card, Meter, Stat, StatusPill } from './Primitives'
import {
  evidenceLabel,
  evidenceTone,
  formatCoverage,
  formatScore,
} from '@/lib/present'
import type { WebsitePCOSProfileResponse } from '@/types/api'

export default function EvidenceHeader({
  report,
}: {
  report: WebsitePCOSProfileResponse
}) {
  const assessment = report.pcos_assessment
  const available = assessment.available
  const score = assessment.calibrated_model_score ?? assessment.raw_model_score

  return (
    <Card className="relative overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
            PCOS-related evidence
          </p>
          <div className="mt-2 flex items-baseline gap-3">
            <span className="text-3xl font-semibold tracking-tight text-neutral-900">
              {evidenceLabel(assessment.evidence_level)}
            </span>
            {available && (
              <span className="font-tabular text-lg text-neutral-500">
                Model score {formatScore(score)}
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <StatusPill tone="neutral">Research prototype</StatusPill>
          <StatusPill tone="neutral">Not a diagnosis</StatusPill>
        </div>
      </div>

      {available ? (
        <>
          <dl className="mt-6 grid grid-cols-2 gap-6 sm:grid-cols-4">
            <Stat
              label="Data coverage"
              value={formatCoverage(report.modality_coverage)}
              hint="Sources contributing"
            />
            <Stat
              label="Feature coverage"
              value={formatCoverage(assessment.feature_coverage)}
              hint="Observed, not imputed"
            />
            <Stat
              label="Calibrated"
              value={assessment.calibrated ? 'Yes' : 'No'}
              hint={assessment.calibrated ? 'Platt scaling' : 'Raw score'}
            />
            <Stat
              label="Last updated"
              value={new Date(report.generated_at).toLocaleDateString()}
              hint={new Date(report.generated_at).toLocaleTimeString()}
            />
          </dl>

          <div className="mt-4">
            <Meter
              value={assessment.feature_coverage}
              tone={evidenceTone(assessment.evidence_level)}
            />
            {/* Stated in words too — a low bar next to a confident band is easy
                to miss. */}
            <p className="mt-2 text-xs text-neutral-500">
              This score used {formatCoverage(assessment.feature_coverage)} observed
              values; the remainder were filled with training-set medians.
            </p>
          </div>

          {assessment.source && (
            <p className="mt-4 flex items-start gap-2 text-sm text-neutral-600">
              <FlaskConical className="mt-0.5 h-4 w-4 shrink-0 text-neutral-400" />
              <span>
                The learned score comes from the{' '}
                <strong className="font-semibold text-neutral-800">
                  static-clinical branch
                </strong>
                . {assessment.qualifier}
              </span>
            </p>
          )}
        </>
      ) : (
        <div className="mt-6 rounded-xl border border-amber-200 bg-amber-50/70 p-4">
          <p className="flex items-start gap-2 text-sm text-amber-900">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              A whole-profile model score could not be calculated because sufficient
              clinical evidence was not available.
            </span>
          </p>
          {assessment.unavailable_reason && (
            <p className="mt-2 pl-6 text-sm text-amber-800/80">
              {assessment.unavailable_reason}
            </p>
          )}
        </div>
      )}

      <p className="mt-5 flex items-start gap-2 border-t border-neutral-100 pt-4 text-xs text-neutral-500">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>{report.disclaimer}</span>
      </p>
    </Card>
  )
}
