/**
 * The app's data layer.
 *
 * The PRISM section routes every call through `./apiClient`, which owns the
 * mock/HTTP mode switch and the generated response types. The legacy section
 * below serves the pre-PRISM demo pages (`/dashboard`, `/cycle`, `/chat`) whose
 * endpoints have no counterpart on the FastAPI service; it keeps its own local
 * fetch so it is obvious at a glance which calls reach a real backend and which
 * do not.
 */

import type {
  User,
  CycleDay,
  ChatMessage,
  PRISMInsight,
  IntakeAnswer,
  PatientProfile,
  HormonalHealthEvent,
  PhenotypeDomain,
  ProfileSimilarity,
  StabilityReport,
  TemporalState,
  RecommendationReport,
  SpeechPipelineResult,
} from '@/types'

import { ApiError, apiBaseUrl, apiFetch, apiMode, apiPost, DEFAULT_DEMO_PATIENT } from './apiClient'
import type {
  ModelStatusResponse,
  WebsitePCOSProfileResponse,
} from '@/types/api'

export type { WebsitePCOSProfileResponse } from '@/types/api'


const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? ''

/**
 * Fetch for the legacy demo endpoints only.
 *
 * Named distinctly from the PRISM client's `apiFetch` because the two are not
 * interchangeable: this one falls back to in-file stubs, which is exactly the
 * behaviour the PRISM path must never have.
 */
async function legacyFetch<T>(path: string, init?: RequestInit): Promise<T> {
  if (!BASE_URL) return stub<T>(path, init)
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`)
  return res.json()
}

// ── Stubs (removed when BASE_URL is set) ────────────────────────────────────

function stub<T>(path: string, init?: RequestInit): T {
  if (path === '/api/user') return MOCK_USER as T
  if (path === '/api/cycle') return MOCK_CYCLE as T
  if (path === '/api/insights') return MOCK_INSIGHTS as T
  if (path === '/api/chat' && init?.method === 'POST') return MOCK_CHAT_REPLY as T
  return {} as T
}

// ── Legacy API (existing pages) ──────────────────────────────────────────────

export const getUser = (): Promise<User> => legacyFetch('/api/user')

export const getCycleDays = (): Promise<CycleDay[]> => legacyFetch('/api/cycle')

export const getInsights = (): Promise<PRISMInsight[]> => legacyFetch('/api/insights')

export const sendChat = (messages: ChatMessage[]): Promise<ChatMessage> =>
  legacyFetch('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ messages }),
  })

export const submitIntake = (answers: IntakeAnswer[]): Promise<{ ok: boolean }> =>
  legacyFetch('/api/intake', {
    method: 'POST',
    body: JSON.stringify({ answers }),
  })

// ── PRISM API ────────────────────────────────────────────────────────────────
//
// Everything below goes through `apiClient.apiFetch`, so mock and HTTP modes
// share one code path and one set of generated types.
//
// `getPatientReport` is the single source of truth. The narrower getters are
// derived from it rather than fetched separately, so they cannot drift out of
// agreement with the report they summarise.

/** Which model branches are usable. Never hardcode availability in a component. */
export const getModelStatus = (): Promise<ModelStatusResponse> =>
  apiFetch('/api/v1/models/status')

function eventToSnake(e: HormonalHealthEvent): Record<string, unknown> {
  return {
    event_id: e.eventId,
    patient_id: e.patientId,
    variable_name: e.variableName,
    canonical_variable_code: e.canonicalVariableCode,
    value: e.value,
    unit: e.unit ?? null,
    observed_at: e.observedAt ?? null,
    start_at: e.startAt ?? null,
    end_at: e.endAt ?? null,
    modality: e.modality,
    provenance: e.provenance,
    extraction_confidence: e.extractionConfidence,
    confirmation_status: e.confirmationStatus,
    missingness_status: e.missingnessStatus,
    negated: e.negated,
    historical: e.historical,
    uncertain: e.uncertain,
    source_file_id: e.sourceFileId ?? null,
    source_page: e.sourcePage ?? null,
    evidence_text: e.evidenceText ?? null,
    parser_version: e.parserVersion ?? null,
    model_version: e.modelVersion ?? null,
    schema_version: e.schemaVersion,
  }
}

/** Run inference for one patient. The primary call. */
export const getPatientReport = (
  patientId: string = DEFAULT_DEMO_PATIENT,
  body: Record<string, unknown> = {},
): Promise<WebsitePCOSProfileResponse> => {
  const payload = { patient_id: patientId, ...body }
  if (Array.isArray(payload.confirmed_events)) {
    payload.confirmed_events = (payload.confirmed_events as HormonalHealthEvent[]).map(eventToSnake)
  }
  return apiPost('/api/v1/patients/infer', payload)
}

export const getEvents = async (
  patientId: string = DEFAULT_DEMO_PATIENT,
): Promise<HormonalHealthEvent[]> => {
  // Backend returns snake_case; frontend type is camelCase.
  const raw = await apiFetch<Record<string, unknown>[]>(`/api/v1/events/${patientId}`)
  return raw.map(e => ({
    eventId: (e.event_id ?? e.eventId) as string,
    patientId: (e.patient_id ?? e.patientId) as string,
    variableName: (e.variable_name ?? e.variableName) as string,
    canonicalVariableCode: (e.canonical_variable_code ?? e.canonicalVariableCode) as string,
    value: e.value,
    unit: e.unit as string | undefined,
    observedAt: (e.observed_at ?? e.observedAt) as string | undefined,
    startAt: (e.start_at ?? e.startAt) as string | undefined,
    endAt: (e.end_at ?? e.endAt) as string | undefined,
    modality: (e.modality) as HormonalHealthEvent['modality'],
    provenance: (e.provenance) as HormonalHealthEvent['provenance'],
    extractionConfidence: (e.extraction_confidence ?? e.extractionConfidence ?? 0) as number,
    confirmationStatus: ((e.confirmation_status ?? e.confirmationStatus) ?? 'not_required') as HormonalHealthEvent['confirmationStatus'],
    missingnessStatus: ((e.missingness_status ?? e.missingnessStatus) ?? 'observed') as HormonalHealthEvent['missingnessStatus'],
    negated: (e.negated ?? false) as boolean,
    historical: (e.historical ?? false) as boolean,
    uncertain: (e.uncertain ?? false) as boolean,
    sourceFileId: (e.source_file_id ?? e.sourceFileId) as string | undefined,
    sourcePage: (e.source_page ?? e.sourcePage) as number | undefined,
    evidenceText: (e.evidence_text ?? e.evidenceText) as string | undefined,
    parserVersion: (e.parser_version ?? e.parserVersion) as string | undefined,
    modelVersion: (e.model_version ?? e.modelVersion) as string | undefined,
    schemaVersion: (e.schema_version ?? e.schemaVersion ?? '1.0.0') as string,
  }))
}

// -- derived views ---------------------------------------------------------

const evidenceQuality = (coverage: number | null | undefined): 'low' | 'moderate' | 'high' => {
  if (coverage == null) return 'low'
  if (coverage >= 0.66) return 'high'
  if (coverage >= 0.33) return 'moderate'
  return 'low'
}

export const getPatientProfile = async (
  patientId: string = DEFAULT_DEMO_PATIENT,
): Promise<PatientProfile> => {
  const report = await getPatientReport(patientId)
  return {
    patientId: report.patient_id,
    displayName: report.patient_id,
    lastUpdated: report.generated_at,
    profileCompleteness: report.modality_coverage ?? 0,
    evidenceQuality: evidenceQuality(report.modality_coverage),
    confirmedCount: report.supporting_evidence?.length ?? 0,
    awaitingCount: 0,
    conflictCount: report.conflicting_evidence?.length ?? 0,
    missingHighValueCount: report.missing_evidence?.length ?? 0,
  }
}

export const getPhenotypeDomains = async (
  patientId: string = DEFAULT_DEMO_PATIENT,
): Promise<PhenotypeDomain[]> => {
  const report = await getPatientReport(patientId)
  const domains = report.phenotype?.domain_scores ?? {}
  return Object.entries(domains)
    .sort(([, a], [, b]) => (a.display_order ?? 0) - (b.display_order ?? 0))
    .map(([key, domain]) => ({
      key,
      name: domain.label ?? key,
      // Absent stays absent. The legacy type demands a number, so an
      // unassessable domain reports 0 coverage and is flagged unavailable via
      // an empty score rather than being given a fabricated composite.
      score: domain.score ?? 0,
      confidence: domain.coverage ?? 0,
      coverage: domain.coverage ?? 0,
      supportingEvidence: domain.observed_variables ?? [],
      missingVariables: domain.missing_variables ?? [],
    }))
}

export const getProfileSimilarities = async (
  patientId: string = DEFAULT_DEMO_PATIENT,
): Promise<ProfileSimilarity[]> => {
  const report = await getPatientReport(patientId)
  const stability = report.phenotype?.stability?.stability_score ?? 0
  return Object.entries(report.phenotype?.profile_similarities ?? {})
    .sort(([, a], [, b]) => b - a)
    .map(([label, probability]) => ({ label, probability, stability }))
}

export const getStabilityReport = async (
  patientId: string = DEFAULT_DEMO_PATIENT,
): Promise<StabilityReport> => {
  const report = await getPatientReport(patientId)
  const stability = report.phenotype?.stability
  const label = stability?.label ?? 'not_assessed'
  return {
    overallStability:
      label === 'stable' ? 'high' : label === 'moderately_stable' ? 'moderate'
        : label === 'unstable' ? 'low' : 'indeterminate',
    bootstrapAgreement: stability?.bootstrap_agreement ?? 0,
    subtypeFlipRate: stability?.profile_flip_rate ?? 0,
    sensitiveVariables: (stability?.unstable_domains ?? []).map((variable) => ({
      variable,
      flipRateWithout: stability?.profile_flip_rate ?? 0,
    })),
    abstained: report.phenotype?.indeterminate ?? true,
    abstentionReason:
      stability?.withheld_reason ?? report.phenotype?.indeterminate_reasons?.[0],
  }
}

export const getTemporalState = async (
  patientId: string = DEFAULT_DEMO_PATIENT,
): Promise<TemporalState> => {
  const report = await getPatientReport(patientId)
  const state = report.current_state
  const p = state?.cycle_phase_probabilities ?? {}
  const coverage = state?.input_coverage ?? 0
  return {
    date: report.generated_at,
    phaseDistribution: {
      menstrual: p.menstrual ?? 0,
      follicular: p.follicular ?? 0,
      periOvulatory: p.peri_ovulatory ?? 0,
      luteal: p.luteal ?? 0,
    },
    coverage,
    uncertainty: coverage >= 0.66 ? 'low' : coverage >= 0.33 ? 'moderate' : 'high',
    missingStreams: report.missing_modalities ?? [],
  }
}

// -- personalised recommendations (Tavily + LLM) ---------------------------
//
// POST /api/v1/patients/{id}/recommendations
// Body: { patient_context?: string }
// The backend calls PersonalisedRecommender.recommend() and returns a
// RecommendationReport.  Set TAVILY_API_KEY + LLM_API_KEY in the server env
// (free Groq key from console.groq.com; Tavily promo code: HackNationJuly).

export const MOCK_RECOMMENDATIONS: RecommendationReport = {
  patient_id: 'demo-maya-chen-001',
  summary:
    'Based on your current findings — irregular cycles and elevated androgenic markers — the most evidence-supported steps involve a combination of structured aerobic exercise, a low-glycaemic diet, and regular monitoring with your clinician.',
  recommendations: [
    {
      category: 'lifestyle',
      title: 'Aim for 150 min aerobic exercise per week',
      body:
        'Moderate aerobic exercise (brisk walking, cycling, swimming) for at least 150 minutes per week is recommended by the 2023 International PCOS Guideline to improve insulin sensitivity and menstrual regularity. Discuss a safe starting point with your clinician.',
      evidence_level: 'guideline-backed',
      sources: ['https://www.monash.edu/medicine/sphpm/mchri/pcos/guideline'],
      caveats: [],
    },
    {
      category: 'nutrition',
      title: 'Try a low-glycaemic index eating pattern',
      body:
        'A low-GI diet can reduce insulin resistance and help regulate cycles. Swap refined carbohydrates for wholegrains, legumes, and non-starchy vegetables. Ask your doctor or a registered dietitian to help you plan this.',
      evidence_level: 'guideline-backed',
      sources: ['https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6734597/'],
      caveats: ['Individual responses vary; a dietitian can tailor the plan.'],
    },
    {
      category: 'monitoring',
      title: 'Track your cycle length every month',
      body:
        'Logging the first day of each period lets your clinician see whether lifestyle changes are improving cycle regularity. Even a simple calendar app counts. Discuss a threshold (e.g. cycles consistently > 35 days) that should prompt a check-in.',
      evidence_level: 'expert-opinion',
      sources: [],
      caveats: [],
    },
    {
      category: 'clinical',
      title: 'Ask about an androgen panel at your next visit',
      body:
        "Your assessment flagged elevated androgenic markers. A lab panel (total testosterone, DHEAS, SHBG) with your clinician's reference ranges can clarify whether levels have changed. Do not self-interpret results — lab ranges differ between assays.",
      evidence_level: 'guideline-backed',
      sources: ['https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6734597/'],
      caveats: ["Biochemical thresholds are assay-specific; only your lab's range applies."],
    },
  ],
  search_queries_used: [
    'PCOS evidence-based self-management guidelines 2024',
    'PCOS ovulatory dysfunction lifestyle intervention evidence 2024',
    'PCOS elevated androgens testosterone management diet exercise',
  ],
  warnings: [],
}

export const getRecommendations = async (
  patientId: string = DEFAULT_DEMO_PATIENT,
  patientContext = '',
): Promise<RecommendationReport> => {
  // Try the Next.js server-side route first (calls Tavily + Groq directly;
  // works even when the Python backend is not running).
  try {
    const res = await fetch('/api/recommendations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId, patient_context: patientContext }),
    })
    if (res.ok) return res.json()
  } catch { /* fall through */ }

  // If Next.js route unavailable, try the FastAPI backend.
  try {
    return await apiPost<RecommendationReport>(`/api/v1/patients/${patientId}/recommendations`, {
      patient_context: patientContext,
    })
  } catch { /* fall through */ }

  // Final fallback: static mock so the page always renders something.
  return MOCK_RECOMMENDATIONS
}

// -- event review ----------------------------------------------------------
//
// The ledger is append-only: a review decision is stored as a new revision
// rather than an edit, so the history of a value stays reconstructable. There
// is no confirmation route on the API yet, so these currently only resolve --
// but they go through the client, so wiring the route later is a one-line
// change here rather than a hunt through components.

export const appendEvents = (events: HormonalHealthEvent[]): Promise<unknown> =>
  apiPost('/api/v1/events', { events })

export const confirmEvent = async (eventId: string): Promise<void> => {
  await apiPost(`/api/v1/events/${eventId}/review`, {
    confirmation_status: 'confirmed',
    reviewed_by: 'patient',
  })
}

export const rejectEvent = async (eventId: string): Promise<void> => {
  await apiPost(`/api/v1/events/${eventId}/review`, {
    confirmation_status: 'rejected',
    reviewed_by: 'patient',
  })
}

// -- ingestion jobs --------------------------------------------------------

export const createSpeechJob = (patientId: string = DEFAULT_DEMO_PATIENT) =>
  apiPost('/api/v1/jobs/speech', { patient_id: patientId })

export const createDocumentJob = (patientId: string = DEFAULT_DEMO_PATIENT) =>
  apiPost('/api/v1/jobs/documents', { patient_id: patientId })

export const uploadDocumentFile = async (
  file: File,
  patientId: string = DEFAULT_DEMO_PATIENT,
): Promise<unknown> => {
  const form = new FormData()
  form.append('patient_id', patientId)
  form.append('file', file)
  const res = await fetch(`${apiBaseUrl()}/api/v1/jobs/documents/upload`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new ApiError(`Document upload failed: ${res.statusText}`, res.status)
  return res.json()
}

export const createUltrasoundJob = (patientId: string = DEFAULT_DEMO_PATIENT) =>
  apiPost('/api/v1/jobs/ultrasound', { patient_id: patientId })

export const getDocumentJob = (jobId: string) =>
  apiFetch(`/api/v1/jobs/documents/${jobId}`)

export const getSpeechJob = (jobId: string) => apiFetch(`/api/v1/jobs/speech/${jobId}`)

export const getUltrasoundJob = (jobId: string) =>
  apiFetch(`/api/v1/jobs/ultrasound/${jobId}`)

// ── Mock data ────────────────────────────────────────────────────────────────

const MOCK_USER: User = {
  id: 'u1',
  name: 'Alex',
  email: 'alex@example.com',
  age: 29,
  cycleLength: 28,
  periodLength: 5,
  lastPeriodStart: '2026-07-01',
}

const today = new Date('2026-07-18')

const MOCK_CYCLE: CycleDay[] = Array.from({ length: 28 }, (_, i) => {
  const d = new Date(today)
  d.setDate(d.getDate() - 17 + i)
  const day = i + 1
  const phase =
    day <= 5  ? 'menstrual'  :
    day <= 13 ? 'follicular' :
    day <= 16 ? 'ovulatory'  : 'luteal'
  return {
    date: d.toISOString().slice(0, 10),
    phase,
    dayOfCycle: day,
    symptoms: day <= 3 ? ['cramps', 'fatigue'] : day === 14 ? ['bloating'] : [],
    mood: day <= 5 ? 2 : day <= 14 ? 4 : day <= 16 ? 5 : 3,
    energy: day <= 5 ? 2 : day <= 14 ? 4 : day <= 16 ? 5 : 3,
    flow: day <= 5 ? (day <= 2 ? 'heavy' : 'medium') : undefined,
  }
})

const MOCK_INSIGHTS: PRISMInsight[] = [
  {
    id: 'i1',
    type: 'cycle',
    title: "You're in your luteal phase",
    body: 'Progesterone is rising. You may notice lower energy and mood shifts toward the end of this week — this is normal.',
    confidence: 0.91,
    date: '2026-07-18',
  },
  {
    id: 'i2',
    type: 'lifestyle',
    title: 'Magnesium may help with late-luteal symptoms',
    body: 'Based on your logged symptoms, supplementing 300 mg magnesium glycinate in the evenings has clinical support for PMS relief.',
    confidence: 0.78,
    date: '2026-07-17',
  },
  {
    id: 'i3',
    type: 'hormone',
    title: 'Estrogen window opens in ~10 days',
    body: 'Your follicular phase starts around July 28. This is typically your highest-energy, best-mood window — a good time to schedule demanding work.',
    confidence: 0.85,
    date: '2026-07-18',
  },
]

const MOCK_CHAT_REPLY: ChatMessage = {
  id: 'r1',
  role: 'assistant',
  content:
    "Based on your cycle data, you're currently in day 17 of your luteal phase. The fatigue you're describing is common as progesterone peaks this week. Gentle movement, consistent sleep, and reducing caffeine can help. Would you like more detail on any of these?",
  timestamp: new Date().toISOString(),
  citations: ['Endocrine Society 2023', 'Prior & Vigna 1987'],
}

// -- speech ----------------------------------------------------------------

/**
 * Send a recording for transcription and clinical-event extraction.
 *
 * Uses the PRISM client's base URL and mode rather than its own fetch, so the
 * recorder points at the same backend as everything else. It does NOT go
 * through `apiFetch`: that helper sets a JSON content-type, and a multipart
 * upload must let the browser set its own boundary header.
 *
 * Every returned event is `proposed`. Nothing enters the record until the
 * patient confirms it.
 */
export async function submitSpeechRecording(
  audio: File | Blob,
  opts?: { patientId?: string; language?: string; signal?: AbortSignal },
): Promise<SpeechPipelineResult> {
  if (apiMode() === 'mock') {
    throw new ApiError(
      'Voice transcription needs the live API. Set NEXT_PUBLIC_PRISM_API_MODE=http.',
      503,
      { reason: 'Speech transcription is unavailable in mock mode.' },
    )
  }

  const form = new FormData()
  const file =
    audio instanceof File
      ? audio
      : new File([audio], `prism-voice-${Date.now()}.webm`, {
          type: audio.type || 'audio/webm',
        })
  form.append('audio', file)
  if (opts?.patientId) form.append('patient_id', opts.patientId)
  if (opts?.language) form.append('language', opts.language)

  let res: Response
  try {
    res = await fetch(`${apiBaseUrl()}/api/v1/speech/transcribe`, {
      method: 'POST',
      body: form,
      signal: opts?.signal,
    })
  } catch (cause) {
    throw new ApiError(`Could not reach the PRISM API at ${apiBaseUrl()}.`, 0, cause)
  }

  if (!res.ok) {
    let detail: unknown
    let message = `Speech transcription failed (${res.status}).`
    try {
      detail = await res.json()
      const inner = (detail as { detail?: { error?: { message?: string } } })?.detail?.error
      if (inner?.message) message = inner.message
    } catch {
      /* keep the default message */
    }
    throw new ApiError(message, res.status, detail)
  }

  const result = (await res.json()) as SpeechPipelineResult

  // Store proposed events in the ledger so they appear in the timeline.
  // They arrive as awaiting_patient_confirmation; the patient reviews and
  // confirms or rejects them there.
  if (result.events && result.events.length > 0) {
    appendEvents(result.events).catch(() => {
      // Non-fatal: the transcript is still returned even if storage fails.
    })
  }

  return result
}

// -- intake ----------------------------------------------------------------

/** One question on the intake form, defined by the backend's variable registry. */
export interface IntakeField {
  code: string
  label: string
  canonical_name: string
  type: 'number' | 'boolean' | 'text'
  unit: string | null
  min: number | null
  max: number | null
  help_text: string | null
  description: string | null
}

export interface IntakeGroup {
  key: string
  title: string
  description: string
  fields: IntakeField[]
}

export interface IntakeSchema {
  groups: IntakeGroup[]
  dropped_unknown_codes: string[]
  guidance: string
}

/**
 * Field definitions for the intake form.
 *
 * Fetched rather than hardcoded so labels, units and valid ranges come from
 * `registry/variables.yaml`. A form that restates them locally drifts from the
 * registry, and a wrong unit is an undetectable hundredfold error downstream.
 */
export const getIntakeSchema = (): Promise<IntakeSchema> =>
  apiFetch('/api/v1/intake/schema')
