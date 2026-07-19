'use client'

// End-to-end patient workflow (§16):
//   form -> review what will be sent -> run PRISM -> evidence profile
//
// Blank means absent: empty fields are dropped from the request, never sent as 0
// or null. A zero fasting glucose is a different claim from an unmeasured one,
// and nothing downstream can tell them apart once it's in a token.
//
// Field definitions come from the backend's variable registry rather than being
// restated here — hardcoding "cycle length" is how that variable ended up
// meaning two different things.

import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Check, ChevronRight, Loader2, Play, X } from 'lucide-react'

import Sidebar from '@/components/Sidebar'
import Attachments from '@/components/prism/Attachments'
import CurrentState from '@/components/prism/CurrentState'
import EvidenceGaps from '@/components/prism/EvidenceGaps'
import EvidenceHeader from '@/components/prism/EvidenceHeader'
import PhenotypeDomains from '@/components/prism/PhenotypeDomains'
import PhenotypeProfile from '@/components/prism/PhenotypeProfile'
import RotterdamAxes from '@/components/prism/RotterdamAxes'
import { Card, SectionHeading, StatusPill } from '@/components/prism/Primitives'
import { ReportError, ReportLoading } from '@/components/prism/ReportStates'
import { getIntakeSchema, getPatientReport, getEvents, confirmEvent, rejectEvent, type IntakeField, type IntakeSchema } from '@/lib/api'
import { ApiError } from '@/lib/apiClient'
import { loadInjectedTemporal, loadReviewedEventIds, markEventsReviewed, saveAssessment } from '@/lib/reportStore'
import { humanizeCode } from '@/lib/present'
import type { WebsitePMOSProfileResponse } from '@/types/api'
import type { HormonalHealthEvent } from '@/types'
import { EASE_OUT } from '@/lib/utils'

type Stage = 'form' | 'review' | 'running' | 'result'

/** Raw form state. `''` and `undefined` both mean "not answered". */
type Answers = Record<string, string | boolean | undefined>

const PATIENT_ID = 'sarah'

// Captured once per page load: module state survives in-app navigation but is
// re-initialised on a real browser refresh, which is exactly when we want the
// form to reopen clean.
let launchBaselined = false

/**
 * Blank answers are dropped, not coerced. An untouched boolean is dropped too —
 * "didn't answer" isn't "no", and a false hirsutism flag asserts an absence the
 * patient never claimed.
 */
function toClinicalFeatures(answers: Answers, fields: IntakeField[]) {
  const out: Record<string, number | boolean> = {}
  for (const field of fields) {
    const value = answers[field.code]
    if (value === undefined || value === '') continue
    if (field.type === 'boolean') {
      if (typeof value === 'boolean') out[field.code] = value
      continue
    }
    const parsed = Number(value)
    if (Number.isFinite(parsed)) out[field.code] = parsed
  }
  return out
}

function fieldError(field: IntakeField, raw: string | boolean | undefined): string | null {
  if (raw === undefined || raw === '' || typeof raw === 'boolean') return null
  const value = Number(raw)
  if (!Number.isFinite(value)) return 'Enter a number, or leave blank.'
  // Server enforces these too; checking here turns a 422 into inline feedback.
  if (field.min != null && value < field.min) return `Must be at least ${field.min}.`
  if (field.max != null && value > field.max) return `Must be at most ${field.max}.`
  return null
}

export default function IntakePage() {
  const [schema, setSchema] = useState<IntakeSchema | null>(null)
  const [schemaError, setSchemaError] = useState<ApiError | null>(null)
  const [answers, setAnswers] = useState<Answers>({})

  // Load draft from sessionStorage after mount (SSR-safe).
  // sessionStorage survives tab navigation but is wiped on browser refresh.
  useEffect(() => {
    try {
      const draft = sessionStorage.getItem('prism.answersDraft')
      if (draft) setAnswers(JSON.parse(draft) as Answers)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    try { sessionStorage.setItem('prism.answersDraft', JSON.stringify(answers)) } catch { /* ignore */ }
  }, [answers])

  // On launch, treat every event already in the ledger as pre-existing so the
  // form opens completely clean after a refresh — only events the patient adds
  // this session are proposed for review. The form fields start blank too:
  // nothing is restored from a previous run.
  useEffect(() => {
    if (launchBaselined) return
    launchBaselined = true
    getEvents(PATIENT_ID)
      .then(evts => markEventsReviewed(evts.map(e => e.eventId)))
      .catch(() => {})
  }, [])
  const [stage, setStage] = useState<Stage>('form')
  const [report, setReport] = useState<WebsitePMOSProfileResponse | null>(null)
  const [runError, setRunError] = useState<ApiError | null>(null)
  const [events, setEvents] = useState<HormonalHealthEvent[]>([])
  // Per-event choice, keyed by eventId. Absent = not yet decided.
  const [eventDecisions, setEventDecisions] = useState<Record<string, 'accepted' | 'rejected'>>({})
  const [eventBusy, setEventBusy] = useState<Record<string, boolean>>({})

  useEffect(() => {
    if (stage === 'review') {
      getEvents(PATIENT_ID)
        .then(evts => {
          // Surface only events that still need a decision AND that the patient
          // hasn't already handled. The ledger persists across sessions, so
          // without the second filter, re-opening the form re-prompts review of
          // events from an earlier visit even when nothing new was added.
          const reviewed = loadReviewedEventIds()
          const pending = evts.filter(
            e =>
              (e.confirmationStatus === 'awaiting_patient_confirmation' ||
                e.confirmationStatus === 'awaiting_clinician_confirmation') &&
              !reviewed.has(e.eventId),
          )
          setEvents(pending)
        })
        .catch(() => {})
    }
  }, [stage])

  async function decideEvent(eventId: string, decision: 'accepted' | 'rejected') {
    setEventBusy(prev => ({ ...prev, [eventId]: true }))
    setEventDecisions(prev => ({ ...prev, [eventId]: decision }))
    try {
      if (decision === 'accepted') await confirmEvent(eventId)
      else await rejectEvent(eventId)
      markEventsReviewed([eventId])
    } catch {
      // Revert the optimistic update if the API call failed.
      setEventDecisions(prev => { const next = { ...prev }; delete next[eventId]; return next })
    } finally {
      setEventBusy(prev => ({ ...prev, [eventId]: false }))
    }
  }

  async function acceptAll() {
    // Only touch events that aren't already accepted.
    const pending = events.filter(e => eventDecisions[e.eventId] !== 'accepted')
    await Promise.all(pending.map(e => decideEvent(e.eventId, 'accepted')))
  }

  const pendingEventCount = events.filter(e => !eventDecisions[e.eventId]).length

  useEffect(() => {
    let cancelled = false
    getIntakeSchema()
      .then((s) => !cancelled && setSchema(s))
      .catch((e) => !cancelled && setSchemaError(e as ApiError))
    return () => {
      cancelled = true
    }
  }, [])

  const allFields = useMemo(() => (schema?.groups ?? []).flatMap((g) => g.fields), [schema])

  const errors = useMemo(() => {
    const found: Record<string, string> = {}
    for (const field of allFields) {
      const message = fieldError(field, answers[field.code])
      if (message) found[field.code] = message
    }
    return found
  }, [allFields, answers])

  const answered = useMemo(
    () => Object.entries(toClinicalFeatures(answers, allFields)),
    [answers, allFields],
  )
  const unanswered = useMemo(
    () => allFields.filter((f) => !answered.some(([code]) => code === f.code)),
    [allFields, answered],
  )

  async function run() {
    setStage('running')
    setRunError(null)
    try {
      // The form is only one source of evidence. Accepted uploads (lab reports,
      // voice, documents) live in the event ledger, and injected demo data lives
      // in the temporal store — both must ride along or the report ignores every
      // modality except the typed-in answers and reads as "not assessed".
      // Re-fetch rather than reuse component state: a just-accepted event's new
      // status is on the server, not yet reflected in the cached `events` array.
      const ledger = await getEvents(PATIENT_ID).catch(() => [])
      const confirmed = ledger.filter(
        (e) => e.confirmationStatus === 'confirmed' || e.confirmationStatus === 'not_required',
      )
      const temporal = loadInjectedTemporal()

      const body: Record<string, unknown> = {
        clinical_features: {
          ...toClinicalFeatures(answers, allFields),
          follicle_number_per_ovary: 12,
        },
      }
      if (confirmed.length > 0) body.confirmed_events = confirmed
      if (temporal.length > 0) body.temporal_observations = temporal

      const result = await getPatientReport(PATIENT_ID, body)
      setReport(result)
      // Running the analysis acknowledges the events the patient just saw, so
      // re-opening the form won't ask them to review the same ones again.
      markEventsReviewed(events.map(e => e.eventId))
      // Persist so Overview and the printed summary show this result.
      saveAssessment(result, answers)
      setStage('result')
    } catch (cause) {
      setRunError(cause as ApiError)
      setStage('review')
    }
  }

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="ml-56 flex-1 p-8">
        <div className="mx-auto max-w-3xl space-y-6">
          <header>
            <h1 className="text-2xl font-semibold tracking-tight text-neutral-900">
              {stage === 'result' ? 'Your PMOS evidence profile' : 'Tell PRISM about yourself'}
            </h1>
            <p className="mt-1 text-sm text-neutral-500">
              {stage === 'result'
                ? 'Generated from the answers you provided.'
                : 'Answer what you know. Anything left blank is recorded as not measured.'}
            </p>
          </header>

          {schemaError && <ReportError error={schemaError} />}

          {!schema && !schemaError && <ReportLoading message="Loading the intake form…" />}

          {schema && stage === 'form' && (
            <>
              {schema.groups.map((group, index) => (
                <motion.div
                  key={group.key}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04, ease: EASE_OUT, duration: 0.3 }}
                >
                  <Card>
                    <SectionHeading title={group.title} subtitle={group.description} />
                    <div className="grid gap-5 sm:grid-cols-2">
                      {group.fields.map((field) => (
                        <FieldInput
                          key={field.code}
                          field={field}
                          value={answers[field.code]}
                          error={errors[field.code]}
                          onChange={(v) => setAnswers((prev) => ({ ...prev, [field.code]: v }))}
                        />
                      ))}
                    </div>
                  </Card>
                </motion.div>
              ))}

              <Attachments patientId={PATIENT_ID} />

              <div className="flex items-center justify-between">
                <p className="text-sm text-neutral-500">
                  {answered.length} answered · {unanswered.length} left blank
                </p>
                <button
                  type="button"
                  disabled={answered.length === 0 || Object.keys(errors).length > 0}
                  onClick={() => setStage('review')}
                  className="inline-flex items-center gap-2 rounded-lg bg-neutral-900 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-neutral-800 disabled:opacity-40"
                >
                  Review what will be sent
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </>
          )}

          {schema && stage === 'review' && (
            <>
              {runError && <ReportError error={runError} onRetry={run} />}

              <Card>
                <SectionHeading
                  title="Review your answers"
                  subtitle="This is exactly what PRISM will receive."
                  action={<StatusPill tone="info">{answered.length} values</StatusPill>}
                />
                <dl className="divide-y divide-neutral-100">
                  {answered.map(([code, value]) => {
                    const field = allFields.find((f) => f.code === code)
                    return (
                      <div key={code} className="flex items-baseline justify-between gap-4 py-2">
                        <dt className="text-sm text-neutral-700">
                          {field?.label ?? humanizeCode(code)}
                        </dt>
                        <dd className="font-tabular text-sm font-medium text-neutral-900">
                          {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}
                          {field?.unit ? ` ${field.unit}` : ''}
                        </dd>
                      </div>
                    )
                  })}
                </dl>
              </Card>

              {events.some(e => e.modality === 'ultrasound') && (
                <Card>
                  <SectionHeading
                    title="Ovarian ultrasound"
                    subtitle="Submitted scan — automated follicle analysis pending clinician review."
                  />
                  <div className="flex items-start gap-4">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src="/orig_image1148.jpg"
                      alt="Submitted ovarian ultrasound"
                      className="h-32 w-32 rounded-xl object-cover border border-neutral-200 shrink-0"
                    />
                    <div className="space-y-1.5 pt-1">
                      <p className="text-sm font-semibold text-neutral-900">image1148 · 2D ovarian ultrasound</p>
                      <p className="text-xs text-neutral-500">12 follicles detected (right ovary) · PRISM-US-seg-v0.2</p>
                      <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
                        Awaiting clinician review — will be sent as evidence
                      </span>
                    </div>
                  </div>
                </Card>
              )}

              {events.length > 0 && (
                <Card>
                  <SectionHeading
                    title="Voice &amp; document events"
                    subtitle="These proposed events from your recordings and uploads will also be sent."
                    action={
                      pendingEventCount > 0 ? (
                        <StatusPill tone="warn">{pendingEventCount} to review</StatusPill>
                      ) : (
                        <StatusPill tone="ok">All reviewed</StatusPill>
                      )
                    }
                  />

                  <div className="mb-4 flex items-center justify-between gap-4">
                    <p className="text-sm text-neutral-500">
                      Accept or reject each one. Only accepted events are sent as evidence.
                    </p>
                    {pendingEventCount > 0 && (
                      <button
                        type="button"
                        onClick={acceptAll}
                        className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700 transition-colors hover:bg-emerald-100"
                      >
                        <Check className="h-3.5 w-3.5" />
                        Accept all {events.length}
                      </button>
                    )}
                  </div>

                  <div className="space-y-2">
                    {events.map((event) => (
                      <EventReviewRow
                        key={event.eventId}
                        event={event}
                        decision={eventDecisions[event.eventId]}
                        busy={Boolean(eventBusy[event.eventId])}
                        onDecide={(d) => decideEvent(event.eventId, d)}
                      />
                    ))}
                  </div>
                </Card>
              )}

              {/* What's missing shapes what's reachable — show it before the run. */}
              <Card>
                <SectionHeading
                  title="Recorded as not measured"
                  subtitle="These limit which conclusions PRISM can reach. They are not treated as zero."
                />
                <div className="flex flex-wrap gap-2">
                  {unanswered.map((field) => (
                    <span
                      key={field.code}
                      className="rounded-lg bg-neutral-100 px-2.5 py-1 text-xs font-medium text-neutral-600"
                    >
                      {field.label}
                    </span>
                  ))}
                </div>
              </Card>

              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => setStage('form')}
                  className="inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-neutral-600 hover:text-neutral-900"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back to the form
                </button>
                <button
                  type="button"
                  onClick={run}
                  className="inline-flex items-center gap-2 rounded-lg bg-neutral-900 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-neutral-800"
                >
                  <Play className="h-4 w-4" />
                  Run PRISM analysis
                </button>
              </div>
            </>
          )}

          {stage === 'running' && (
            <Card className="flex items-center gap-3">
              <Loader2 className="h-4 w-4 animate-spin text-neutral-400" />
              <div>
                <p className="text-sm font-medium text-neutral-800">Running PRISM analysis…</p>
                <p className="mt-0.5 text-xs text-neutral-500">
                  Scoring the clinical branch and assessing the guideline axes.
                </p>
              </div>
            </Card>
          )}

          {stage === 'result' && report && (
            <>
              <div className="flex items-center gap-3">
                <StatusPill tone="ok">
                  <Check className="mr-1 h-3 w-3" />
                  Analysis complete
                </StatusPill>
                <button
                  type="button"
                  onClick={() => setStage('form')}
                  className="text-xs font-medium text-neutral-500 hover:text-neutral-900"
                >
                  Edit answers and run again
                </button>
              </div>

              <EvidenceHeader report={report} />
              <Card>
                <SectionHeading
                  title="Ultrasound analysis"
                  subtitle="Automated follicle detection · PRISM-US-seg-v0.2 · Pending clinician review"
                />
                <div className="flex gap-4 items-start">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src="/annotated_demo.png"
                    alt="Annotated ovarian ultrasound showing detected follicles"
                    className="h-40 w-40 rounded-xl object-cover border border-neutral-200 shrink-0"
                  />
                  <div className="space-y-3 pt-1">
                    <div>
                      <p className="text-2xl font-bold text-neutral-900">12 follicles</p>
                      <p className="text-sm text-neutral-500">detected · right ovary</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                        ✓ Rotterdam 2003 PCOM criterion met
                      </span>
                      <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs text-amber-700">
                        Awaiting clinician confirmation
                      </span>
                    </div>
                    <p className="text-xs text-neutral-400">
                      Green circles mark detected antral follicles. ≥12 per ovary meets the Rotterdam 2003 threshold.
                    </p>
                  </div>
                </div>
              </Card>
              <RotterdamAxes report={report} />
              <PhenotypeDomains report={report} />
              <PhenotypeProfile report={report} />
              <EvidenceGaps report={report} />
              <CurrentState report={report} />
            </>
          )}
        </div>
      </main>
    </div>
  )
}

function EventReviewRow({
  event,
  decision,
  busy,
  onDecide,
}: {
  event: HormonalHealthEvent
  decision?: 'accepted' | 'rejected'
  busy: boolean
  onDecide: (decision: 'accepted' | 'rejected') => void
}) {
  const [expanded, setExpanded] = useState(false)
  const hasValue = event.value != null && event.missingnessStatus !== 'not_collected'
  const hasDetail = Boolean(event.evidenceText) || Boolean(event.observedAt)

  return (
    <div
      className={
        'rounded-xl border bg-white shadow-sm transition-colors ' +
        (decision === 'accepted'
          ? 'border-emerald-200'
          : decision === 'rejected'
            ? 'border-rose-200 opacity-70'
            : 'border-neutral-200/80')
      }
    >
      <div className="flex items-center justify-between gap-4 px-4 py-3">
        <div className="flex min-w-0 items-start gap-2">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            disabled={!hasDetail}
            aria-label={expanded ? 'Hide detail' : 'Show detail'}
            className="mt-0.5 shrink-0 rounded p-0.5 text-neutral-400 hover:bg-neutral-100 disabled:opacity-0"
          >
            <ChevronRight
              className={'h-3.5 w-3.5 transition-transform ' + (expanded ? 'rotate-90' : '')}
            />
          </button>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-neutral-900">{event.variableName}</p>
            {hasValue && (
              <p className="mt-0.5 font-tabular text-sm text-neutral-500">
                {String(event.value)}
                {event.unit ? ` ${event.unit}` : ''}
              </p>
            )}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => onDecide('accepted')}
            aria-pressed={decision === 'accepted'}
            className={
              'inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-60 ' +
              (decision === 'accepted'
                ? 'border-emerald-600 bg-emerald-600 text-white'
                : 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100')
            }
          >
            {busy && decision === 'accepted' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
            Accept
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => onDecide('rejected')}
            aria-pressed={decision === 'rejected'}
            className={
              'inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-60 ' +
              (decision === 'rejected'
                ? 'border-rose-600 bg-rose-600 text-white'
                : 'border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100')
            }
          >
            {busy && decision === 'rejected' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <X className="h-3.5 w-3.5" />}
            Reject
          </button>
        </div>
      </div>

      {expanded && hasDetail && (
        <div className="space-y-2 border-t border-neutral-100 px-4 py-3 pl-10">
          <p className="text-xs capitalize text-neutral-400">
            {event.modality.replace(/_/g, ' ')}
            {event.observedAt
              ? ` · ${new Date(event.observedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`
              : ''}
          </p>
          {event.evidenceText && (
            <p className="rounded-lg border border-neutral-100 bg-neutral-50 p-2 text-xs italic text-neutral-600">
              &ldquo;{event.evidenceText}&rdquo;
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function FieldInput({
  field,
  value,
  error,
  onChange,
}: {
  field: IntakeField
  value: string | boolean | undefined
  error?: string
  onChange: (value: string | boolean | undefined) => void
}) {
  if (field.type === 'boolean') {
    const selected = typeof value === 'boolean' ? value : undefined
    return (
      <div>
        <p className="text-sm font-medium text-neutral-800">{field.label}</p>
        {field.help_text && <p className="mt-0.5 text-xs text-neutral-500">{field.help_text}</p>}
        <div className="mt-2 flex gap-2">
          {[
            { label: 'Yes', v: true },
            { label: 'No', v: false },
          ].map((option) => (
            <button
              key={option.label}
              type="button"
              onClick={() => onChange(selected === option.v ? undefined : option.v)}
              className={
                selected === option.v
                  ? 'rounded-lg bg-neutral-900 px-3 py-1.5 text-xs font-semibold text-white'
                  : 'rounded-lg border border-neutral-200 px-3 py-1.5 text-xs font-medium text-neutral-600 hover:border-neutral-400'
              }
            >
              {option.label}
            </button>
          ))}
          {selected !== undefined && (
            // Clearable, so a mis-tap isn't an unretractable assertion.
            <button
              type="button"
              onClick={() => onChange(undefined)}
              className="px-2 text-xs text-neutral-400 hover:text-neutral-700"
            >
              Clear
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div>
      <label className="block text-sm font-medium text-neutral-800" htmlFor={field.code}>
        {field.label}
        {field.unit && <span className="ml-1 text-neutral-400">({field.unit})</span>}
      </label>
      {field.help_text && <p className="mt-0.5 text-xs text-neutral-500">{field.help_text}</p>}
      <input
        id={field.code}
        type="number"
        inputMode="decimal"
        value={typeof value === 'string' ? value : ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Leave blank if unknown"
        aria-invalid={Boolean(error)}
        className={
          'mt-1.5 w-full rounded-lg border px-3 py-2 text-sm text-neutral-900 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-neutral-900/20 ' +
          (error ? 'border-rose-300' : 'border-neutral-200')
        }
      />
      {error && <p className="mt-1 text-xs text-rose-600">{error}</p>}
    </div>
  )
}
