'use client'

import { humanizeCode, axisLabel } from './present'
import type { WebsitePCOSProfileResponse } from '@/types/api'

export interface AskDoctorItem {
  question: string
  why: string
  urgency: 'high' | 'medium' | 'low'
}

const MISSING_TO_QUESTION: Record<string, { question: string; why: string; urgency: AskDoctorItem['urgency'] }> = {
  total_testosterone:       { question: 'Order a total testosterone blood test', why: 'Needed to assess androgen excess, a core Rotterdam criterion', urgency: 'high' },
  free_testosterone:        { question: 'Order a free testosterone or FAI calculation', why: 'Free testosterone is a more sensitive androgen marker than total', urgency: 'high' },
  dheas:                    { question: 'Order a DHEA-sulfate (DHEAS) blood test', why: 'Rules out adrenal androgen excess as an alternative diagnosis', urgency: 'high' },
  shbg:                     { question: 'Order a SHBG (sex hormone binding globulin) test', why: 'Low SHBG amplifies androgen effect and indicates insulin resistance', urgency: 'high' },
  anti_mullerian_hormone:   { question: 'Order an AMH (Anti-Müllerian Hormone) test', why: 'Elevated AMH is a strong marker of polycystic ovarian reserve', urgency: 'high' },
  luteinizing_hormone:      { question: 'Order LH and FSH hormone levels', why: 'LH:FSH ratio > 2 supports PCOS diagnosis', urgency: 'medium' },
  follicle_stimulating_hormone: { question: 'Order FSH to pair with LH', why: 'Used to rule out premature ovarian insufficiency', urgency: 'medium' },
  fasting_glucose:          { question: 'Order a fasting plasma glucose test', why: 'Screens for insulin resistance and pre-diabetes, common in PCOS', urgency: 'high' },
  fasting_insulin:          { question: 'Order a fasting insulin level', why: 'Quantifies insulin resistance when combined with glucose (HOMA-IR)', urgency: 'high' },
  hdl_cholesterol:          { question: 'Request a full lipid panel including HDL', why: 'Metabolic syndrome risk is elevated in PCOS', urgency: 'medium' },
  triglycerides:            { question: 'Request a full lipid panel including triglycerides', why: 'Elevated triglycerides are a metabolic syndrome marker', urgency: 'medium' },
  cycle_length:             { question: 'Discuss your menstrual cycle history in detail', why: 'Cycle length and regularity are required to assess ovulatory dysfunction', urgency: 'high' },
  cycle_irregularity:       { question: 'Discuss cycle irregularity patterns', why: 'Irregular cycles are one of the three Rotterdam diagnostic criteria', urgency: 'high' },
  menstrual_frequency_per_year: { question: 'Report how many periods you had in the last 12 months', why: 'Fewer than 8–9 per year indicates oligo-anovulation', urgency: 'high' },
  waist_circumference:      { question: 'Ask for a waist and hip measurement at your visit', why: 'Waist-to-hip ratio indicates central adiposity and metabolic risk', urgency: 'low' },
  systolic_blood_pressure:  { question: 'Have your blood pressure measured at the visit', why: 'Hypertension is part of metabolic syndrome screening', urgency: 'low' },
}

const MODALITY_TO_QUESTION: Record<string, { question: string; why: string; urgency: AskDoctorItem['urgency'] }> = {
  ovarian_ultrasound: {
    question: 'Request a transvaginal or pelvic ultrasound',
    why: 'Polycystic ovarian morphology (≥20 follicles per ovary or volume ≥10 mL) is the third Rotterdam criterion and cannot be assessed without imaging',
    urgency: 'high',
  },
}

export function deriveAskDoctorItems(report: WebsitePCOSProfileResponse | null): AskDoctorItem[] {
  if (!report) return []

  const items: AskDoctorItem[] = []
  const seen = new Set<string>()

  function add(q: { question: string; why: string; urgency: AskDoctorItem['urgency'] }) {
    if (!seen.has(q.question)) {
      seen.add(q.question)
      items.push(q)
    }
  }

  // From specific missing variables
  for (const code of report.missing_evidence ?? []) {
    const mapped = MISSING_TO_QUESTION[code]
    if (mapped) add(mapped)
  }

  // From axes that couldn't be assessed
  for (const [key, axis] of Object.entries(report.rotterdam_axes ?? {})) {
    if (axis.status === 'not_assessable') {
      for (const missing of axis.missing_evidence ?? []) {
        const mapped = MISSING_TO_QUESTION[missing]
        if (mapped) add(mapped)
      }
      // Ultrasound axis specifically
      if (key === 'ovarian_morphology') {
        add(MODALITY_TO_QUESTION['ovarian_ultrasound'])
      }
    }
  }

  // From missing modalities
  for (const mod of report.missing_modalities ?? []) {
    const mapped = MODALITY_TO_QUESTION[mod]
    if (mapped) add(mapped)
  }

  // Sort by urgency
  const order = { high: 0, medium: 1, low: 2 }
  return items.sort((a, b) => order[a.urgency] - order[b.urgency])
}

export function formatAnswersForSummary(
  answers: Record<string, string | boolean | undefined>
): { label: string; value: string }[] {
  return Object.entries(answers)
    .filter(([, v]) => v !== undefined && v !== '' && v !== null)
    .map(([code, val]) => ({
      label: humanizeCode(code),
      value: typeof val === 'boolean' ? (val ? 'Yes' : 'No') : String(val),
    }))
}
