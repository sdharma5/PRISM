export type ConfirmationStatus =
  | 'confirmed'
  | 'awaiting_patient_confirmation'
  | 'awaiting_clinician_confirmation'
  | 'rejected'
  | 'not_required'

export type ProvenanceType =
  | 'patient_confirmed'
  | 'clinician_confirmed'
  | 'document_extracted'
  | 'device_measured'
  | 'dataset_provided'
  | 'model_measured'
  | 'model_inferred'

export type MissingnessStatus =
  | 'observed'
  | 'not_collected'
  | 'not_available'
  | 'not_applicable'
  | 'extraction_failed'
  | 'intentionally_masked'

export type ProcessingStage =
  | 'queued'
  | 'validating'
  | 'preprocessing'
  | 'running_model'
  | 'postprocessing'
  | 'awaiting_review'
  | 'completed'
  | 'failed'

export interface HormonalHealthEvent {
  eventId: string
  patientId: string
  variableName: string
  canonicalVariableCode: string
  value: unknown
  unit?: string
  observedAt?: string
  startAt?: string
  endAt?: string
  modality: 'symptom' | 'cycle' | 'lab' | 'medication' | 'wearable' | 'cgm' | 'ultrasound' | 'document' | 'model'
  provenance: ProvenanceType
  extractionConfidence: number
  confirmationStatus: ConfirmationStatus
  /** Who reviewed it. Machine-extracted evidence cannot be `confirmed` without
   *  this -- the backend rejects the whole request (schemas/event.py). */
  reviewedBy?: string | null
  reviewedAt?: string | null
  missingnessStatus: MissingnessStatus
  negated: boolean
  historical: boolean
  uncertain: boolean
  sourceFileId?: string
  sourcePage?: number
  evidenceText?: string
  parserVersion?: string
  modelVersion?: string
  schemaVersion: string
}

export interface ProcessingJob<T = unknown> {
  id: string
  type: 'speech' | 'document' | 'ultrasound' | 'temporal'
  stage: ProcessingStage
  progress: number
  message: string
  createdAt: string
  updatedAt: string
  result?: T
  error?: { code: string; message: string }
}

export interface PhenotypeDomain {
  name: string
  key: string
  score: number
  confidence: number
  coverage: number
  supportingEvidence: string[]
  missingVariables: string[]
}

export interface ProfileSimilarity {
  label: string
  probability: number
  stability: number
}

export interface StabilityReport {
  overallStability: 'low' | 'moderate' | 'high' | 'indeterminate'
  bootstrapAgreement: number
  subtypeFlipRate: number
  sensitiveVariables: Array<{ variable: string; flipRateWithout: number }>
  abstained: boolean
  abstentionReason?: string
}

export interface TemporalState {
  date: string
  cycleDay?: number
  phaseDistribution: { menstrual: number; follicular: number; periOvulatory: number; luteal: number }
  coverage: number
  uncertainty: 'low' | 'moderate' | 'high'
  missingStreams: string[]
}

export interface PatientProfile {
  patientId: string
  displayName: string
  lastUpdated: string
  profileCompleteness: number
  evidenceQuality: 'low' | 'moderate' | 'high'
  confirmedCount: number
  awaitingCount: number
  conflictCount: number
  missingHighValueCount: number
}

// Backward-compatible types for existing pages
export interface User {
  id: string
  name: string
  email: string
  age: number
  cycleLength: number
  periodLength: number
  lastPeriodStart: string
}

export interface CycleDay {
  date: string
  phase: 'menstrual' | 'follicular' | 'ovulatory' | 'luteal'
  dayOfCycle: number
  symptoms: string[]
  mood: number
  energy: number
  flow?: 'light' | 'medium' | 'heavy'
  notes?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  citations?: string[]
}

export interface PRISMInsight {
  id: string
  type: 'cycle' | 'hormone' | 'lifestyle' | 'alert'
  title: string
  body: string
  confidence: number
  date: string
}

export interface IntakeAnswer {
  questionId: string
  value: string | string[] | number
}

// ── Personalised recommendations (Tavily + LLM) ─────────────────────────────
// Field names match the Python API's snake_case serialization.

export type RecommendationCategory = 'lifestyle' | 'clinical' | 'monitoring' | 'nutrition'
export type EvidenceLevel = 'guideline-backed' | 'observational' | 'expert-opinion'

export interface Recommendation {
  category: RecommendationCategory
  title: string
  body: string
  evidence_level: EvidenceLevel
  /** Tavily source URLs that grounded this recommendation. */
  sources: string[]
  caveats: string[]
}

export interface RecommendationReport {
  patient_id: string
  summary: string
  recommendations: Recommendation[]
  search_queries_used: string[]
  warnings: string[]
}

// ── Speech pipeline response types ──────────────────────────────────────────
// Mirror the backend ingestion/speech/ contracts (Transcript + ExtractionResult)
// closely enough that the voice page can render real backend output without
// reshaping. Fields are optional where the backend may omit them.

export interface TranscriptWord {
  text: string
  start_seconds: number
  end_seconds: number
  confidence?: number
}

export interface TranscriptSegment {
  segment_id: string
  speaker_role: string
  text: string
  start_seconds: number
  end_seconds: number
  words?: TranscriptWord[]
}

export interface SpeechTranscript {
  recording_id: string
  language?: string
  text: string
  segments?: TranscriptSegment[]
  engine?: string
  engine_version?: string
}

export interface SpeechEvidenceSpan {
  text: string
  char_start: number
  char_end: number
  start_seconds?: number | null
  end_seconds?: number | null
  segment_id?: string | null
}

export interface ExtractedSpeechEvent {
  extraction_id: string
  recording_id: string
  patient_id: string
  canonical_code: string
  variable_name: string
  value: unknown
  unit?: string | null
  surface_form: string
  category?: string
  negated: boolean
  historical: boolean
  uncertain: boolean
  temporality?: 'current' | 'historical' | 'unknown'
  attribution?: 'patient' | 'family_member' | 'other' | 'unknown'
  relation?: string | null
  speaker_role?: string
  severity?: 'mild' | 'moderate' | 'severe' | null
  evidence: SpeechEvidenceSpan
  extraction_confidence: number
  warnings?: string[]
}

export interface SpeechPipelineResult {
  recording_id: string
  patient_id: string
  transcript: SpeechTranscript
  events: ExtractedSpeechEvent[]
  /** Backend may report unsupported extractions (e.g. ungrounded spans). */
  unsupported_count?: number
  /** Audio quality score from ingestion/speech/audio.py, if computed. */
  audio_quality_score?: number
  warnings?: string[]
}

export interface SpeechSubmissionError {
  code: string
  message: string
}
