'use client'

// Profile similarities and the stability verdict (§10, §11).
//
// Two outcomes, both normal: a named pattern when the assignment is determinate
// AND survives the stability checks, or no name plus the reason it was withheld.
// Indeterminate is not an error state and isn't styled like one — similarities
// are still shown either way.
//
// Never say "subtype": these are similarities to literature patterns.

import { ShieldCheck, ShieldAlert, ShieldQuestion } from 'lucide-react'

import { Card, Meter, SectionHeading, Stat, StatusPill } from './Primitives'
import { formatCoverage, phenotypeVerdict, profileLabel, stabilityLabel } from '@/lib/present'
import type { WebsitePMOSProfileResponse } from '@/types/api'

const STABILITY_ICON = {
  stable: ShieldCheck,
  moderately_stable: ShieldQuestion,
  unstable: ShieldAlert,
  not_assessed: ShieldQuestion,
} as const

const STABILITY_TONE = {
  stable: 'ok',
  moderately_stable: 'info',
  unstable: 'warn',
  not_assessed: 'neutral',
} as const

export default function PhenotypeProfile({
  report,
}: {
  report: WebsitePMOSProfileResponse
}) {
  const phenotype = report.phenotype
  const verdict = phenotypeVerdict(report)
  const stability = phenotype?.stability
  const label = (stability?.label ?? 'not_assessed') as keyof typeof STABILITY_ICON
  const Icon = STABILITY_ICON[label]

  const similarities = Object.entries(phenotype?.profile_similarities ?? {}).sort(
    ([, a], [, b]) => b - a,
  )

  return (
    <Card>
      <SectionHeading
        title="Phenotype-profile similarity"
        subtitle={phenotype?.interpretation_note}
      />

      {/* Same visual weight either way. */}
      <div
        className={
          verdict.resolved
            ? 'rounded-xl border border-sky-200 bg-sky-50/70 p-4'
            : 'rounded-xl border border-neutral-200 bg-neutral-50 p-4'
        }
      >
        <p className="text-sm font-semibold text-neutral-900">{verdict.headline}</p>
        {verdict.body && <p className="mt-1 text-sm text-neutral-600">{verdict.body}</p>}

        {!verdict.resolved && Boolean(phenotype?.indeterminate_reasons?.length) && (
          <ul className="mt-3 space-y-1 text-sm text-neutral-600">
            {phenotype!.indeterminate_reasons!.map((reason) => (
              <li key={reason} className="flex gap-2">
                <span aria-hidden className="text-neutral-400">
                  •
                </span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Shown in both states — withholding the label isn't a reason to hide
          the evidence behind it. */}
      <div className="mt-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
          Similarity to catalogued patterns
        </p>
        {similarities.length === 0 ? (
          <p className="mt-2 text-sm text-neutral-500">
            Similarity was not computed for this patient.
          </p>
        ) : (
          <ul className="mt-3 space-y-3">
            {similarities.map(([key, value]) => (
              <li key={key}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm text-neutral-700">{profileLabel(key)}</span>
                  <span className="font-tabular text-sm text-neutral-500">
                    {value.toFixed(2)}
                  </span>
                </div>
                <Meter
                  value={value}
                  tone={verdict.resolved && key === phenotype?.dominant_profile ? 'info' : 'neutral'}
                  className="mt-1"
                />
              </li>
            ))}
          </ul>
        )}
        <p className="mt-3 text-xs text-neutral-400">
          Similarity scores are not calibrated probabilities.
        </p>
      </div>

      {/* Beside the result, not in a tooltip (§11). Plain language first. */}
      <div className="mt-6 border-t border-neutral-100 pt-5">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-2">
            <Icon className="mt-0.5 h-4 w-4 shrink-0 text-neutral-400" />
            <div>
              <p className="text-sm font-semibold text-neutral-900">
                Stability: {stabilityLabel(stability?.label)}
              </p>
              {stability?.plain_language && (
                <p className="mt-1 text-sm text-neutral-600">{stability.plain_language}</p>
              )}
            </div>
          </div>
          <StatusPill tone={STABILITY_TONE[label]}>{stabilityLabel(stability?.label)}</StatusPill>
        </div>

        <dl className="mt-4 grid grid-cols-2 gap-6 sm:grid-cols-3">
          <Stat
            label="Stability score"
            value={stability?.stability_score?.toFixed(2) ?? '—'}
          />
          <Stat
            label="Bootstrap agreement"
            value={formatCoverage(stability?.bootstrap_agreement)}
          />
          <Stat
            label="Profile flip rate"
            value={
              stability?.profile_flip_rate == null
                ? '—'
                : formatCoverage(stability.profile_flip_rate)
            }
            hint="Feature-removal sensitivity"
          />
        </dl>

        {Boolean(stability?.unstable_domains?.length) && (
          <p className="mt-3 text-sm text-neutral-600">
            <span className="font-medium text-neutral-800">Sensitive to:</span>{' '}
            {stability!.unstable_domains!.join(', ')} — removing these changes the
            leading pattern.
          </p>
        )}

        {stability?.withheld_reason && (
          <p className="mt-3 rounded-lg bg-neutral-50 p-3 text-sm text-neutral-600">
            <span className="font-medium text-neutral-800">Why no profile was named:</span>{' '}
            {stability.withheld_reason}
          </p>
        )}
      </div>
    </Card>
  )
}
