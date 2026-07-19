// GENERATED FILE -- DO NOT EDIT BY HAND.
//
// Source: docs/openapi.yaml (from apps/api/schemas/).
// Regenerate: python scripts/export_openapi.py
//             python scripts/generate_frontend_types.py
//
// These mirror the server's response contract exactly. Editing them here does
// not change the server; it only hides a mismatch until runtime.

/** One Rotterdam axis. */
export interface AxisView {
  status: "met" | "not_met" | "uncertain" | "not_assessable"
  level?: string | null
  supporting_evidence?: string[]
  missing_evidence?: string[]
  evidence_source?: string | null
  biochemical_evidence_available?: boolean | null
  reason?: string | null
  caveats?: string[]
  threshold_sources?: Record<string, string>
}

export interface Body_transcribe_api_v1_speech_transcribe_post {
  audio: string
  patient_id?: string
  language?: string
}

/** What the service can honestly say about one model branch. */
export interface BranchStatusView {
  available: boolean
  trained: boolean
  persisted: boolean
  validated_for_inference: boolean
  version?: string | null
  implementation?: string | null
  reason?: string | null
}

export interface CalibrationStatusView {
  available: boolean
  method?: string | null
  note?: string | null
}

/** Two sources disagreeing about the same variable. */
export interface ConflictView {
  detail: string
  variable_code?: string | null
  modalities?: string[]
  severity?: string | null
}

/** The longitudinal branch's read on where the patient is right now. */
export interface CurrentStateView {
  available?: boolean
  predicted_cycle_phase?: string | null
  cycle_phase_probabilities?: Record<string, number>
  hormone_estimates?: Record<string, HormoneEstimateView>
  input_coverage?: number | null
  methods_used?: string[]
  confidence?: number | null
  observed_days?: number | null
  unavailable_reason?: string | null
}

/** One continuous phenotype domain. */
export interface DomainScoreView {
  label?: string | null
  score?: number | null
  scale?: "cohort_z_score"
  available?: boolean
  coverage?: number | null
  evidence_source?: string | null
  qualifier?: string | null
  observed_variables?: string[]
  missing_variables?: string[]
  display_order?: number
}

/** One or more events to append. */
export interface EventBatch {
  events: HormonalHealthEvent[]
}

export interface EventBatchResult {
  stored: number
  patient_ids: string[]
}

/** One piece of supporting evidence, structured rather than a raw string. */
export interface EvidenceStatementView {
  statement: string
  variable_code?: string | null
  axis?: string | null
  guideline_source?: string | null
}

export interface HTTPValidationError {
  detail?: ValidationError[]
}

/** One observation about one patient, with full provenance. */
export interface HormonalHealthEvent {
  event_id?: string
  patient_id: string
  variable_name: string
  canonical_variable_code: string
  value?: unknown
  unit?: string | null
  raw_value?: unknown
  raw_unit?: string | null
  observed_at?: string | null
  start_at?: string | null
  end_at?: string | null
  modality: "questionnaire" | "patient_voice" | "clinician_voice" | "laboratory" | "clinical_document" | "ultrasound_report" | "ultrasound_image" | "wearable" | "cgm" | "menstrual_history" | "medication" | "diagnosis_history"
  provenance: "patient_confirmed" | "clinician_confirmed" | "document_extracted" | "device_measured" | "dataset_provided" | "model_measured" | "model_inferred"
  extraction_confidence: number
  confirmation_status: "confirmed" | "awaiting_patient_confirmation" | "awaiting_clinician_confirmation" | "rejected" | "not_required"
  reviewed_by?: string | null
  reviewed_at?: string | null
  missingness_status?: "observed" | "not_collected" | "not_available" | "not_applicable" | "extraction_failed" | "intentionally_masked"
  negated?: boolean
  historical?: boolean
  uncertain?: boolean
  source_dataset?: string | null
  source_file_id?: string | null
  source_file_hash?: string | null
  source_page?: number | null
  source_time_start_seconds?: number | null
  source_time_end_seconds?: number | null
  evidence_text?: string | null
  parser_version?: string | null
  model_version?: string | null
  schema_version?: string
}

/** One temporal hormone estimate, with how it was produced. */
export interface HormoneEstimateView {
  code: string
  display_name: string
  value?: number | null
  method?: string | null
  method_code?: string | null
  interval_low?: number | null
  interval_high?: number | null
  unit?: string | null
}

/** A job's observable state. */
export interface JobRecord {
  job_id: string
  kind: "documents" | "speech" | "ultrasound"
  patient_id: string
  status: "queued" | "processing" | "completed" | "failed" | "unavailable"
  created_at: string
  updated_at: string
  reason?: string | null
  result?: Record<string, unknown> | null
}

/** What a client sends to open a job. */
export interface JobSubmission {
  patient_id: string
  source_ids?: string[]
  note?: string | null
}

/** Body of ``GET /api/v1/models/status``. */
export interface ModelStatusResponse {
  schema_version?: string
  static_clinical: BranchStatusView
  temporal_state: BranchStatusView
  ovarian_ultrasound: BranchStatusView
  calibration: CalibrationStatusView
  speech?: SpeechStatusView | null
  warnings?: string[]
}

/** One row per participant per day. */
export interface ParticipantDay {
  participant_id: string
  study_day: number
  calendar_date?: string | null
  cycle_day?: number | null
  cycle_phase?: "menstrual" | "follicular" | "peri_ovulatory" | "luteal" | "unknown"
  values?: Record<string, number | null>
  is_observed?: Record<string, boolean>
  time_since_last_observed?: Record<string, number>
  daily_symptoms?: Record<string, boolean>
  source_dataset?: string | null
}

/** The main inference request. */
export interface PatientInferenceRequest {
  patient_id: string
  /** Canonical variable code -> value, for values entered directly rather than arriving as confirmed events. Codes must exist in registry/variables.yaml; unknown codes are dropped at the encoder boundary rather than treated as evidence. */
  clinical_features?: Record<string, number | string | boolean | null> | null
  /** Events the patient or a clinician has confirmed. */
  confirmed_events?: HormonalHealthEvent[]
  /** Daily longitudinal observations. Named `temporal_observations` for the API surface; the internal type is ParticipantDay. */
  temporal_observations?: ParticipantDay[]
  /** Completed ultrasound job identifiers. Accepted so the contract is stable, but the ultrasound branch is gated off for inference; see GET /api/v1/models/status. */
  ultrasound_job_ids?: string[]
  requested_adapter?: "pmos"
}

/** The learned whole-patient score, with the conditions for reading it. */
export interface PmosAssessmentView {
  available: boolean
  raw_model_score?: number | null
  calibrated_model_score?: number | null
  evidence_level?: "low" | "moderate" | "elevated" | "high" | "not_available"
  source?: string | null
  qualifier?: string | null
  unavailable_reason?: string | null
  feature_coverage?: number | null
  calibrated?: boolean
}

/** Continuous domains first; soft similarities second. */
export interface PhenotypeView {
  domain_scores?: Record<string, DomainScoreView>
  profile_similarities?: Record<string, number>
  dominant_profile?: string | null
  stable_dominant_profile?: boolean
  indeterminate?: boolean
  indeterminate_reasons?: string[]
  status?: "stable_dominant_profile" | "no_stable_dominant_profile"
  stability?: StabilityView
  interpretation_note?: string
}

/** Where one displayed claim came from, for the detail drawer. */
export interface ProvenanceRecordView {
  label: string
  origin: "patient_reported" | "clinician_confirmed" | "document_extracted" | "device_measured" | "model_estimate" | "rule_based_interpretation"
  source_id?: string | null
  observed_at?: string | null
  confirmation_status?: string | null
  model_version?: string | null
  method?: string | null
  confidence?: number | null
}

/** Traceability for the report as a whole. */
export interface ProvenanceView {
  records?: ProvenanceRecordView[]
  provenance_ids?: string[]
  model_versions?: Record<string, string>
  combination_mode?: string | null
  clinician_review_status?: string | null
}

/** Whether voice transcription is usable. */
export interface SpeechStatusView {
  available: boolean
  model?: string | null
  reason?: string | null
}

/** How fragile the profile assignment is, in plain terms first. */
export interface StabilityView {
  label?: "stable" | "moderately_stable" | "unstable" | "not_assessed"
  plain_language?: string | null
  stability_score?: number | null
  bootstrap_agreement?: number | null
  profile_flip_rate?: number | null
  unstable_domains?: string[]
  withheld_reason?: string | null
}

/** Static-clinical branch only. */
export interface StaticInferenceRequest {
  patient_id: string
  clinical_features?: Record<string, number | string | boolean | null> | null
  confirmed_events?: HormonalHealthEvent[]
}

/** Longitudinal branch only. */
export interface TemporalInferenceRequest {
  patient_id: string
  temporal_observations?: ParticipantDay[]
}

/** Ultrasound branch only. Gated off; see the router for the 503 it returns. */
export interface UltrasoundInferenceRequest {
  patient_id: string
  job_ids?: string[]
}

export interface ValidationError {
  loc: string | number[]
  msg: string
  type: string
  input?: unknown
  ctx?: Record<string, unknown>
}

/** Everything one patient-facing report needs, and nothing the UI must guess. */
export interface WebsitePMOSProfileResponse {
  schema_version?: string
  report_id: string
  patient_id: string
  generated_at: string
  modality_coverage?: number | null
  pmos_assessment: PmosAssessmentView
  rotterdam_axes?: Record<string, AxisView>
  phenotype?: PhenotypeView
  current_state?: CurrentStateView
  androgenic_evidence_source?: string | null
  supporting_evidence?: EvidenceStatementView[]
  conflicting_evidence?: ConflictView[]
  missing_evidence?: string[]
  available_modalities?: string[]
  missing_modalities?: string[]
  learned_components_used?: string[]
  rule_based_components_used?: string[]
  provenance?: ProvenanceView
  warnings?: string[]
  is_diagnosis?: false
  disclaimer?: string
}

/** Every path this API serves. */
export type ApiPath =
  | '/api/v1/events'
  | '/api/v1/events/{patient_id}'
  | '/api/v1/health'
  | '/api/v1/intake/schema'
  | '/api/v1/jobs/documents'
  | '/api/v1/jobs/documents/{job_id}'
  | '/api/v1/jobs/speech'
  | '/api/v1/jobs/speech/{job_id}'
  | '/api/v1/jobs/ultrasound'
  | '/api/v1/jobs/ultrasound/{job_id}'
  | '/api/v1/models/status'
  | '/api/v1/patients/infer'
  | '/api/v1/patients/infer/static'
  | '/api/v1/patients/infer/temporal'
  | '/api/v1/patients/infer/ultrasound'
  | '/api/v1/speech/transcribe'
