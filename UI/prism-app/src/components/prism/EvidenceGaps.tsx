'use client'

// Missing and conflicting evidence (§6.6).
//
// Strictly what the assessment couldn't determine, never advice. "No androgen
// assay was available" is a fact about the report; "you should get an androgen
// panel" is a clinical recommendation this system can't make.

import { AlertCircle, HelpCircle } from 'lucide-react'

import { Card, SectionHeading } from './Primitives'
import { humanizeCode } from '@/lib/present'
import type { WebsitePCOSProfileResponse } from '@/types/api'

export default function EvidenceGaps({ report }: { report: WebsitePCOSProfileResponse }) {
  const missing = report.missing_evidence ?? []
  const conflicts = report.conflicting_evidence ?? []

  if (missing.length === 0 && conflicts.length === 0) {
    return null
  }

  return (
    <Card>
      <SectionHeading
        title="Missing and conflicting evidence"
        subtitle="What this assessment could not determine, and why."
      />

      {conflicts.length > 0 && (
        <div className="mb-5">
          <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-amber-700">
            <AlertCircle className="h-3.5 w-3.5" />
            Sources disagree
          </p>
          <ul className="mt-2 space-y-2">
            {conflicts.map((conflict, index) => (
              <li
                key={`${conflict.variable_code ?? 'conflict'}-${index}`}
                className="rounded-xl border border-amber-200 bg-amber-50/60 p-3 text-sm text-amber-900"
              >
                {conflict.detail}
                {Boolean(conflict.modalities?.length) && (
                  <span className="mt-1 block text-xs text-amber-800/70">
                    Between: {conflict.modalities!.join(', ')}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {missing.length > 0 && (
        <div>
          <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-neutral-500">
            <HelpCircle className="h-3.5 w-3.5" />
            Not available for this assessment
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {missing.map((code) => (
              <span
                key={code}
                className="rounded-lg bg-neutral-100 px-2.5 py-1 text-xs font-medium text-neutral-600"
              >
                {humanizeCode(code)}
              </span>
            ))}
          </div>
          <p className="mt-3 text-xs text-neutral-500">
            These were not measured or not provided. Their absence limits which
            conclusions could be reached; it is not itself a finding.
          </p>
        </div>
      )}
    </Card>
  )
}
