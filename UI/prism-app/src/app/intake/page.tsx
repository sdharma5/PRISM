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
import { ArrowLeft, ArrowRight, Check, Loader2, Play } from 'lucide-react'

import Sidebar from '@/components/Sidebar'
import Attachments from '@/components/prism/Attachments'
import CurrentState from '@/components/prism/CurrentState'
import EvidenceGaps from '@/components/prism/EvidenceGaps'
import EvidenceHeader from '@/components/prism/EvidenceHeader'
import PhenotypeDomains from '@/components/prism/PhenotypeDomains'
import PhenotypeProfile from '@/components/prism/PhenotypeProfile'
import RotterdamAxes from '@/components/prism/RotterdamAxes'
import { Card, SectionHeading, StatusPill } from '@/components/prism/Primitives'
import { PartialNotice, ReportError, ReportLoading } from '@/components/prism/ReportStates'
import { getIntakeSchema, getPatientReport, getEvents, type IntakeField, type IntakeSchema } from '@/lib/api'
import { ApiError } from '@/lib/apiClient'
import { loadAssessment, saveAssessment } from '@/lib/reportStore'
import { humanizeCode } from '@/lib/present'
import type { WebsitePCOSProfileResponse } from '@/types/api'
import { EASE_OUT } from '@/lib/utils'

type Stage = 'form' | 'review' | 'running' | 'result'

/** Raw form state. `''` and `undefined` both mean "not answered". */
type Answers = Record<string, string | boolean | undefined>

const PATIENT_ID = 'sarah'

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

  // Re-open pre-filled — returning users are usually adding one value.
  useEffect(() => {
    const stored = loadAssessment()
    if (stored?.answers) setAnswers(stored.answers)
  }, [])
  const [stage, setStage] = useState<Stage>('form')
  const [report, setReport] = useState<WebsitePCOSProfileResponse | null>(null)
  const [runError, setRunError] = useState<ApiError | null>(null)
  const [eventCount, setEventCount] = useState(0)

  useEffect(() => {
    if (stage === 'review') {
      getEvents(PATIENT_ID).then(evts => setEventCount(evts.length)).catch(() => {})
    }
  }, [stage])

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
      const result = await getPatientReport(PATIENT_ID, {
        clinical_features: toClinicalFeatures(answers, allFields),
      })
      setReport(result)
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
              {stage === 'result' ? 'Your PCOS evidence profile' : 'Tell PRISM about yourself'}
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

              {eventCount > 0 && (
                <Card>
                  <SectionHeading
                    title="Voice &amp; document events"
                    subtitle="These proposed events from your recordings and uploads will also be sent."
                    action={<StatusPill tone="ok">{eventCount} event{eventCount !== 1 ? 's' : ''}</StatusPill>}
                  />
                  <p className="text-sm text-neutral-500">
                    Review and confirm individual events on the{' '}
                    <a href="/timeline" className="underline underline-offset-2 hover:text-neutral-700">Timeline</a>{' '}
                    before they count as evidence.
                  </p>
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

              <PartialNotice
                missing={report.missing_modalities ?? []}
                warnings={report.warnings ?? []}
              />
              <EvidenceHeader report={report} />
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
