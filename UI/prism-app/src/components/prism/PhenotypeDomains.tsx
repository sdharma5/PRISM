'use client'

// Continuous domain scores (§9) — the primary phenotype output, so they come
// first and get the most space. Profile similarity is secondary and lives in a
// quieter component.
//
// Unassessable domains show a hatched track, not a zero-length bar: zero is the
// cohort average, which is a real finding.

import { useState } from 'react'

import { Card, SectionHeading, StatusPill, ZScoreBar } from './Primitives'
import { formatCoverage, formatZScore, humanizeCode, orderedDomains, Z_SCORE_NOTE } from '@/lib/present'
import type { DomainScoreView, WebsitePMOSProfileResponse } from '@/types/api'

const EVIDENCE_TONE = {
  symptoms: 'info',
  biochemical: 'ok',
  imaging: 'neutral',
  mixed: 'neutral',
} as const

function DomainRow({ domainKey, domain }: { domainKey: string; domain: DomainScoreView }) {
  const [open, setOpen] = useState(false)
  const tone = EVIDENCE_TONE[(domain.evidence_source ?? 'mixed') as keyof typeof EVIDENCE_TONE]

  return (
    <div className="border-t border-neutral-100 py-4 first:border-t-0 first:pt-0">
      <div className="flex items-baseline justify-between gap-4">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-neutral-900">
            {domain.label ?? humanizeCode(domainKey)}
          </p>
          <p className="mt-0.5 text-xs text-neutral-500">
            {formatCoverage(domain.coverage)} of variables observed
            {domain.evidence_source ? ` · ${domain.evidence_source}` : ''}
          </p>
        </div>
        <span
          className="font-tabular text-sm font-semibold text-neutral-900"
          title={domain.available ? Z_SCORE_NOTE : undefined}
        >
          {formatZScore(domain.available ? domain.score : null)}
        </span>
      </div>

      <div className="mt-2">
        <ZScoreBar score={domain.available ? domain.score : null} />
      </div>

      {domain.qualifier && (
        <p className="mt-2 text-xs font-medium text-sky-700">{domain.qualifier}</p>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mt-2 text-xs font-medium text-neutral-500 hover:text-neutral-800"
        aria-expanded={open}
      >
        {open ? 'Hide variables' : 'Show variables'}
      </button>

      {open && (
        <div className="mt-2 grid gap-3 rounded-xl bg-neutral-50 p-4 text-sm sm:grid-cols-2">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Supporting observations
            </p>
            <p className="mt-1 text-neutral-700">
              {domain.observed_variables?.length
                ? domain.observed_variables.map(humanizeCode).join(', ')
                : 'None'}
            </p>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
              Missing variables
            </p>
            <p className="mt-1 text-neutral-600">
              {domain.missing_variables?.length
                ? domain.missing_variables.map(humanizeCode).join(', ')
                : 'None'}
            </p>
          </div>
          <div className="sm:col-span-2">
            <StatusPill tone={tone}>
              {domain.available ? 'Assessed' : 'Not assessable'}
            </StatusPill>
          </div>
        </div>
      )}
    </div>
  )
}

export default function PhenotypeDomains({
  report,
}: {
  report: WebsitePMOSProfileResponse
}) {
  const domains = orderedDomains(report.phenotype?.domain_scores)
  const assessed = domains.filter(([, d]) => d.available).length

  return (
    <Card>
      <SectionHeading
        title="Phenotype domain scores"
        subtitle={Z_SCORE_NOTE}
        action={
          <StatusPill tone="neutral">
            {assessed} of {domains.length} assessed
          </StatusPill>
        }
      />
      {domains.length === 0 ? (
        <p className="text-sm text-neutral-500">No domain scores were produced.</p>
      ) : (
        domains.map(([key, domain]) => (
          <DomainRow key={key} domainKey={key} domain={domain} />
        ))
      )}
    </Card>
  )
}
