'use client'

// Optional attachments, part of the single data-entry flow (§16).
//
// Everything here reports what actually happened to the file: voice transcribes
// on the server, and the rest submit a real job and render whatever the service
// returns rather than a local progress animation.

import { useRef, useState } from 'react'
import { AlertCircle, Check, FileText, FlaskConical, Image as ImageIcon, Loader2, Mic, Square, Upload, Watch } from 'lucide-react'

import { Card, SectionHeading, StatusPill } from './Primitives'
import { createUltrasoundJob, getEvents, getPatientReport, submitSpeechRecording, uploadDocumentFile } from '@/lib/api'
import { ApiError, DEFAULT_DEMO_PATIENT, apiMode } from '@/lib/apiClient'
import { saveAssessment } from '@/lib/reportStore'
import type { SpeechPipelineResult } from '@/types'

type JobKind = 'documents' | 'ultrasound'

interface JobOutcome {
  status: string
  reason?: string | null
  fileName: string
}

export default function Attachments({
  patientId,
  onSpeechEvents,
}: {
  patientId: string
  onSpeechEvents?: (result: SpeechPipelineResult) => void
}) {
  return (
    <Card>
      <SectionHeading
        title="Add files or a recording"
        subtitle="All optional. Anything you add here is reviewed before it counts as evidence."
      />
      <div className="space-y-5">
        <VoiceBlock patientId={patientId} onResult={onSpeechEvents} />
        <FileBlock
          kind="documents"
          label="Lab report or clinic letter"
          hint="PDF or image of an actual report."
          Icon={FileText}
          patientId={patientId}
          demoPdfPath="/demo-lab-report.pdf"
        />
        <FileBlock
          kind="ultrasound"
          label="Ovarian ultrasound"
          hint="Imaging is experimental and is not used to score your result."
          Icon={ImageIcon}
          patientId={patientId}
          experimental
        />
        {apiMode() !== 'mock' && (
          <>
            <div className="flex items-center gap-3 pt-1">
              <div className="h-px flex-1 bg-neutral-200" />
              <span className="text-xs font-medium uppercase tracking-wide text-neutral-400">
                Add wearable or Fitbit data
              </span>
              <div className="h-px flex-1 bg-neutral-200" />
            </div>
            <TemporalDemoBlock patientId={patientId} />
          </>
        )}
      </div>
    </Card>
  )
}

function VoiceBlock({
  patientId,
  onResult,
}: {
  patientId: string
  onResult?: (result: SpeechPipelineResult) => void
}) {
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<SpeechPipelineResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  async function start() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data)
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setBusy(true)
        try {
          const transcript = await submitSpeechRecording(blob, { patientId })
          setResult(transcript)
          onResult?.(transcript)
        } catch (cause) {
          setError(cause instanceof ApiError ? cause.message : 'Transcription failed.')
        } finally {
          setBusy(false)
        }
      }
      recorder.start()
      recorderRef.current = recorder
      setRecording(true)
    } catch {
      // Usually a denied permission prompt — the user's choice, not a fault.
      setError('Microphone access was not granted, so nothing was recorded.')
    }
  }

  function stop() {
    recorderRef.current?.stop()
    setRecording(false)
  }

  return (
    <div className="rounded-xl border border-neutral-200 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <Mic className="mt-0.5 h-4 w-4 shrink-0 text-neutral-400" />
          <div>
            <p className="text-sm font-semibold text-neutral-900">Describe your symptoms aloud</p>
            <p className="mt-0.5 text-xs text-neutral-500">
              Transcribed on the server. Anything it picks up is proposed, never recorded
              automatically.
            </p>
          </div>
        </div>
        {busy ? (
          <StatusPill tone="info">
            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
            Transcribing
          </StatusPill>
        ) : recording ? (
          <button
            type="button"
            onClick={stop}
            className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white"
          >
            <Square className="h-3 w-3" />
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={start}
            className="inline-flex items-center gap-1.5 rounded-lg border border-neutral-200 px-3 py-1.5 text-xs font-medium text-neutral-700 hover:border-neutral-400"
          >
            <Mic className="h-3 w-3" />
            Record
          </button>
        )}
      </div>

      {error && <p className="mt-3 text-xs text-amber-700">{error}</p>}

      {result && (
        <div className="mt-3 rounded-lg bg-neutral-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Transcript
          </p>
          <p className="mt-1 text-sm text-neutral-700">
            {result.transcript?.text || '(no speech detected)'}
          </p>
          <p className="mt-2 text-xs text-neutral-500">
            {result.events?.length ?? 0} proposed observation
            {(result.events?.length ?? 0) === 1 ? '' : 's'} — review them in Timeline before they
            count as evidence.
          </p>
        </div>
      )}
    </div>
  )
}

function FileBlock({
  kind,
  label,
  hint,
  Icon,
  patientId,
  experimental = false,
  demoPdfPath,
}: {
  kind: JobKind
  label: string
  hint: string
  Icon: React.ElementType
  patientId: string
  experimental?: boolean
  demoPdfPath?: string
}) {
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState<JobOutcome | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  async function loadAndSubmitDemo() {
    if (!demoPdfPath) return
    setBusy(true)
    try {
      const res = await fetch(demoPdfPath)
      const blob = await res.blob()
      const file = new File([blob], 'demo-lab-report.pdf', { type: 'application/pdf' })
      await submit(file)
    } catch (cause) {
      setOutcome({ status: 'failed', reason: 'Could not load demo file.', fileName: 'demo-lab-report.pdf' })
      setBusy(false)
    }
  }

  async function submit(file: File) {
    setBusy(true)
    try {
      const job =
        kind === 'documents'
          ? await uploadDocumentFile(file, patientId)
          : await createUltrasoundJob(patientId)
      const body = job as { status?: string; reason?: string | null; result?: { extracted?: number; events_stored?: number } }
      setOutcome({
        status: body.status ?? 'unknown',
        reason: body.result
          ? `Extracted ${body.result.extracted ?? 0} lab value(s), stored ${body.result.events_stored ?? 0} event(s).`
          : body.reason ?? null,
        fileName: file.name,
      })
    } catch (cause) {
      setOutcome({
        status: 'failed',
        reason: cause instanceof ApiError ? cause.message : 'The upload could not be submitted.',
        fileName: file.name,
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-xl border border-neutral-200 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <Icon className="mt-0.5 h-4 w-4 shrink-0 text-neutral-400" />
          <div>
            <p className="text-sm font-semibold text-neutral-900">{label}</p>
            <p className="mt-0.5 text-xs text-neutral-500">{hint}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {experimental && <StatusPill tone="warn">Experimental</StatusPill>}
          {demoPdfPath && (
            <button
              type="button"
              disabled={busy}
              onClick={loadAndSubmitDemo}
              className="inline-flex items-center gap-1.5 rounded-lg border border-violet-200 bg-violet-50 px-3 py-1.5 text-xs font-semibold text-violet-700 hover:bg-violet-100 disabled:opacity-50"
            >
              {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <FlaskConical className="h-3 w-3" />}
              Try demo report
            </button>
          )}
          <button
            type="button"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-neutral-200 px-3 py-1.5 text-xs font-medium text-neutral-700 hover:border-neutral-400 disabled:opacity-50"
          >
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Upload className="h-3 w-3" />}
            Choose file
          </button>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) void submit(file)
        }}
      />

      {outcome && (
        <div className="mt-3 rounded-lg bg-neutral-50 p-3">
          <p className="flex items-center gap-1.5 text-sm font-medium text-neutral-800">
            {outcome.status === 'completed' ? (
              <Check className="h-3.5 w-3.5 text-emerald-600" />
            ) : (
              <AlertCircle className="h-3.5 w-3.5 text-amber-600" />
            )}
            {outcome.fileName}
          </p>
          {/* The service's own words. */}
          {outcome.reason && <p className="mt-1 text-xs text-neutral-600">{outcome.reason}</p>}
          {outcome.status !== 'completed' && (
            <p className="mt-1 text-xs text-neutral-400">
              Status: {outcome.status}. This file did not contribute to your result.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function buildDemoTemporalDays(patientId: string) {
  const days = []
  for (let i = 0; i < 30; i++) {
    const cycleDay = (i % 35) + 1
    const follicular = cycleDay <= 28
    const phase = cycleDay <= 5 ? 'menstrual' : follicular ? 'follicular' : 'luteal'
    const lh = follicular ? 4 + Math.sin(i * 0.3) * 2 : 3.5
    const e3g = follicular ? 80 + cycleDay * 3 : 90
    const pdg = follicular ? 1.5 : 2.0
    const hr = 68 + Math.sin(i * 0.2) * 3
    const temp = phase === 'luteal' ? 34.1 : 33.8
    const hrv = 52 + Math.sin(i * 0.4) * 8
    const glucose = 108 + Math.sin(i * 0.15) * 10
    days.push({
      participant_id: patientId,
      study_day: i + 1,
      calendar_date: new Date(Date.now() - (29 - i) * 86400000).toISOString().slice(0, 10),
      cycle_day: cycleDay,
      cycle_phase: phase,
      values: { lh, e3g, pdg, resting_heart_rate: hr, wrist_temperature: temp, hrv_rmssd: hrv, mean_glucose: glucose },
      is_observed: { lh: true, e3g: true, pdg: i % 3 === 0, resting_heart_rate: true, wrist_temperature: true, hrv_rmssd: true, mean_glucose: i % 2 === 0 },
      time_since_last_observed: { lh: 1, e3g: 1, pdg: i % 3 === 0 ? 1 : (i % 3) + 1, resting_heart_rate: 1, wrist_temperature: 1, hrv_rmssd: 1, mean_glucose: i % 2 === 0 ? 1 : 2 },
      daily_symptoms: { bloating: cycleDay <= 5, fatigue: cycleDay > 20 },
    })
  }
  return days
}

function TemporalDemoBlock({ patientId }: { patientId: string }) {
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run() {
    setBusy(true)
    setError(null)
    try {
      const temporalObs = buildDemoTemporalDays(patientId)
      const events = await getEvents(patientId)
      const confirmed = events.filter(e => e.confirmationStatus === 'confirmed' || e.confirmationStatus === 'not_required')
      const result = await getPatientReport(patientId, { confirmed_events: confirmed, temporal_observations: temporalObs })
      saveAssessment(result, {})
      setDone(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-xl border border-neutral-200 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <Watch className="mt-0.5 h-4 w-4 shrink-0 text-neutral-400" />
          <div>
            <p className="text-sm font-semibold text-neutral-900">Add wearable / Fitbit / temporal data</p>
            <p className="mt-0.5 text-xs text-neutral-500">
              Injects 30 days of synthetic hormone + wearable data to demo the longitudinal state model.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={run}
          disabled={busy || done}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-violet-200 bg-violet-50 px-3 py-1.5 text-xs font-semibold text-violet-700 hover:bg-violet-100 disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : done ? <Check className="h-3 w-3" /> : <FlaskConical className="h-3 w-3" />}
          {busy ? 'Running…' : done ? 'Injected' : 'Demo inject'}
        </button>
      </div>
      {done && <p className="mt-2 text-xs text-emerald-700">Temporal data injected — view results on the <a href="/overview" className="underline">Overview</a> page.</p>}
      {error && <p className="mt-2 text-xs text-rose-600">{error}</p>}
    </div>
  )
}
