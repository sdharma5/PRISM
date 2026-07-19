'use client'

// Longitudinal current state (§12). Its own section, and deliberately not the
// dashboard headline — it's the prettiest output and the least central claim.
//
// Every estimate shows its method in words. LOCF repeats the last observation
// back, so it must never read as a prediction.

import { Card, Meter, NotAssessed, SectionHeading, StatusPill } from './Primitives'
import { formatCoverage, humanizeCode } from '@/lib/present'
import type { WebsitePCOSProfileResponse } from '@/types/api'

const PHASE_LABEL: Record<string, string> = {
  menstrual: 'Menstrual',
  follicular: 'Follicular',
  peri_ovulatory: 'Peri-ovulatory',
  luteal: 'Luteal',
  unknown: 'Unknown',
}

export default function CurrentState({ report }: { report: WebsitePCOSProfileResponse }) {
  const state = report.current_state

  if (!state?.available) {
    return (
      <Card>
        <SectionHeading title="Current longitudinal hormonal-state estimate" />
        <NotAssessed reason={state?.unavailable_reason} />
      </Card>
    )
  }

  const phases = Object.entries(state.cycle_phase_probabilities ?? {}).sort(
    ([, a], [, b]) => b - a,
  )
  const hormones = Object.entries(state.hormone_estimates ?? {})

  return (
    <Card>
      <SectionHeading
        title="Current longitudinal hormonal-state estimate"
        subtitle="A current-state estimate from recent measurements. Not a PCOS diagnosis."
        action={
          <StatusPill tone="neutral">
            {formatCoverage(state.input_coverage)} input coverage
          </StatusPill>
        }
      />

      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
          Predicted cycle phase
        </p>
        <p className="mt-1 text-lg font-semibold text-neutral-900">
          {PHASE_LABEL[state.predicted_cycle_phase ?? 'unknown'] ??
            humanizeCode(state.predicted_cycle_phase ?? 'unknown')}
        </p>

        <ul className="mt-3 space-y-2">
          {phases.map(([phase, probability]) => (
            <li key={phase}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="text-neutral-700">{PHASE_LABEL[phase] ?? humanizeCode(phase)}</span>
                <span className="font-tabular text-neutral-500">
                  {formatCoverage(probability)}
                </span>
              </div>
              <Meter
                value={probability}
                tone={phase === state.predicted_cycle_phase ? 'info' : 'neutral'}
                className="mt-1"
              />
            </li>
          ))}
        </ul>
      </div>

      {hormones.length > 0 && (
        <div className="mt-6 border-t border-neutral-100 pt-5">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Hormone estimates
          </p>
          <div className="mt-3 space-y-4">
            {hormones.map(([code, estimate]) => (
              <div key={code}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm font-medium text-neutral-800">
                    {estimate.display_name}
                  </span>
                  <span className="font-tabular text-sm text-neutral-900">
                    {estimate.value?.toFixed(2) ?? '—'}
                    {estimate.unit ? ` ${estimate.unit}` : ''}
                  </span>
                </div>
                {/* Method at the same size as the value. */}
                <p className="mt-0.5 text-xs text-neutral-500">{estimate.method}</p>
                {estimate.interval_low != null && estimate.interval_high != null && (
                  <p className="mt-0.5 font-tabular text-xs text-neutral-400">
                    Uncertainty interval {estimate.interval_low.toFixed(2)} –{' '}
                    {estimate.interval_high.toFixed(2)}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {Boolean(state.methods_used?.length) && (
        <p className="mt-5 text-xs text-neutral-400">
          Methods used: {state.methods_used!.join(' · ')}
        </p>
      )}
    </Card>
  )
}
