// Wording rules for anything clinical, kept in one place so components can't
// drift. A model score isn't a probability of having PMOS, a z-score isn't a
// percentage, and an unassessable axis isn't a negative result.

import type {
  AxisView,
  DomainScoreView,
  PmosAssessmentView,
  WebsitePMOSProfileResponse,
} from '@/types/api'

// Derived from the generated interfaces: OpenAPI inlines these unions into the
// properties that use them, so there's no named schema to generate.
type AxisStatus = NonNullable<AxisView['status']>
type EvidenceLevel = NonNullable<PmosAssessmentView['evidence_level']>

/** Human label for an evidence band. */
export function evidenceLabel(level: EvidenceLevel | string | undefined): string {
  switch (level) {
    case 'low':
      return 'Low'
    case 'moderate':
      return 'Moderate'
    case 'elevated':
      return 'Elevated'
    case 'high':
      return 'High'
    default:
      // Not "Low" — absence of a finding isn't a finding.
      return 'Not available'
  }
}

export function evidenceTone(level: EvidenceLevel | string | undefined):
  | 'neutral'
  | 'info'
  | 'warn'
  | 'alert' {
  switch (level) {
    case 'high':
      return 'alert'
    case 'elevated':
      return 'warn'
    case 'moderate':
      return 'info'
    default:
      return 'neutral'
  }
}

/** Two decimals, no percent sign. Never "N% chance of PMOS". */
export function formatScore(score: number | null | undefined): string {
  return score == null ? '—' : score.toFixed(2)
}

/** Format a 0-1 coverage fraction as a percentage. Coverage genuinely is one. */
export function formatCoverage(value: number | null | undefined): string {
  return value == null ? '—' : `${Math.round(value * 100)}%`
}

/**
 * Format a domain composite.
 *
 * Signed, because direction carries the meaning: +1.9 and −1.9 are opposite
 * findings and an unsigned "1.9" is ambiguous.
 */
export function formatZScore(score: number | null | undefined): string {
  if (score == null) return 'Not assessed'
  const sign = score > 0 ? '+' : ''
  return `${sign}${score.toFixed(2)}`
}

export const Z_SCORE_NOTE =
  'Standard deviations from the training cohort average. 0 is average; this is not a percentage.'

// -- axis vocabulary -------------------------------------------------------

export const AXIS_LABELS: Record<string, string> = {
  ovulatory_dysfunction: 'Ovulatory dysfunction',
  hyperandrogenism_clinical: 'Hyperandrogenism — clinical',
  hyperandrogenism_biochemical: 'Hyperandrogenism — biochemical',
  polycystic_ovarian_morphology: 'Polycystic ovarian morphology',
}

export function axisLabel(key: string): string {
  return AXIS_LABELS[key] ?? key.replace(/_/g, ' ')
}

export function axisStatusLabel(status: AxisStatus | string): string {
  switch (status) {
    case 'met':
      return 'Met'
    case 'not_met':
      return 'Not met'
    case 'uncertain':
      return 'Uncertain'
    case 'not_assessable':
      return 'Not assessable'
    default:
      return String(status).replace(/_/g, ' ')
  }
}

/** `not_assessable` is neutral, not negative — the evidence was never obtained. */
export function axisTone(status: AxisStatus | string): 'neutral' | 'info' | 'warn' | 'ok' {
  switch (status) {
    case 'met':
      return 'warn'
    case 'not_met':
      return 'ok'
    case 'uncertain':
      return 'info'
    default:
      return 'neutral'
  }
}

// -- androgenic evidence ---------------------------------------------------

export function androgenicEvidenceSentence(
  source: string | null | undefined,
): { headline: string; caveat: string | null } {
  switch (source) {
    case 'symptoms_only':
      return {
        headline: 'Clinical androgenic evidence: supported by reported symptoms',
        caveat: 'Biochemical androgen measurements were unavailable.',
      }
    case 'biochemical_only':
      return {
        headline: 'Biochemical androgenic evidence: supported by measured androgens',
        caveat: 'No cutaneous androgenic signs were recorded.',
      }
    case 'both':
      return {
        headline: 'Androgenic evidence: supported by symptoms and measured androgens',
        caveat: null,
      }
    default:
      return {
        headline: 'Androgenic evidence: not available',
        caveat: 'Neither cutaneous signs nor an androgen assay were scored.',
      }
  }
}

// -- phenotype -------------------------------------------------------------

export interface PhenotypeVerdict {
  /** True when a named, stable profile may be shown. */
  resolved: boolean
  headline: string
  body: string | null
}

/**
 * What the phenotype section may claim. A named profile needs the assignment to
 * be both determinate and stable; either alone isn't enough. Indeterminate is a
 * normal outcome, not an error.
 */
export function phenotypeVerdict(report: WebsitePMOSProfileResponse): PhenotypeVerdict {
  const phenotype = report.phenotype
  const dominant = phenotype?.dominant_profile
  const resolved = Boolean(dominant) && Boolean(phenotype?.stable_dominant_profile)

  if (resolved && dominant) {
    return {
      resolved: true,
      headline: `Most similar stable pattern: ${profileLabel(dominant)}`,
      body:
        report.androgenic_evidence_source === 'symptoms_only'
          ? 'Biochemical androgen measurements were unavailable.'
          : null,
    }
  }

  return {
    resolved: false,
    headline: 'No stable dominant profile',
    body:
      'PRISM found similarities to several phenotype patterns, but the leading ' +
      'pattern changed across resampling or evidence-removal checks.',
  }
}

/** "Androgenic-leaning pattern", never "androgenic PMOS subtype". */
export function profileLabel(key: string): string {
  const base = key.replace(/_leaning$/, '').replace(/_/g, '–')
  const pretty = base.charAt(0).toUpperCase() + base.slice(1)
  return key.endsWith('_leaning') ? `${pretty}-leaning pattern` : `${pretty} pattern`
}

export function stabilityLabel(label: string | undefined): string {
  switch (label) {
    case 'stable':
      return 'Stable'
    case 'moderately_stable':
      return 'Moderately stable'
    case 'unstable':
      return 'Unstable'
    default:
      return 'Not assessed'
  }
}

// -- domains ---------------------------------------------------------------

/** Registry display order — dict order isn't a contract. */
export function orderedDomains(
  domains: Record<string, DomainScoreView> | undefined,
): Array<[string, DomainScoreView]> {
  return Object.entries(domains ?? {}).sort(
    ([, a], [, b]) => (a.display_order ?? 0) - (b.display_order ?? 0),
  )
}

/** Turn a canonical variable code into something readable. */
export function humanizeCode(code: string): string {
  return code.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
