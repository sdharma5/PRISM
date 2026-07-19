'use client'

import { useEffect, useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity, FlaskConical, Pill, Watch, FileText, Cpu, Mic, Image,
  AlertTriangle, Check, X, ChevronDown, ChevronUp,
} from 'lucide-react'
import Sidebar from '@/components/Sidebar'
import ConfidenceBar from '@/components/ConfidenceBar'
import type { HormonalHealthEvent, ConfirmationStatus } from '@/types'
import { getEvents, confirmEvent, rejectEvent } from '@/lib/api'

const ICON: Record<string, React.ElementType> = {
  laboratory: FlaskConical, lab: FlaskConical,
  symptom: Activity, cycle: Activity, menstrual_history: Activity, cgm: Activity,
  medication: Pill,
  wearable: Watch,
  ultrasound: Image, ultrasound_image: Image, ultrasound_report: FileText,
  document: FileText, clinical_document: FileText, questionnaire: FileText,
  patient_voice: Mic, clinician_voice: Mic,
  model: Cpu, diagnosis_history: Cpu,
}

const DOT_COLOR: Record<string, string> = {
  laboratory: 'bg-blue-500', lab: 'bg-blue-500',
  symptom: 'bg-rose-400', cycle: 'bg-sky-400', menstrual_history: 'bg-sky-400',
  medication: 'bg-teal-400',
  wearable: 'bg-emerald-400',
  cgm: 'bg-orange-400',
  ultrasound: 'bg-cyan-400', ultrasound_image: 'bg-cyan-400', ultrasound_report: 'bg-cyan-400',
  document: 'bg-amber-400', clinical_document: 'bg-amber-400', questionnaire: 'bg-violet-400',
  patient_voice: 'bg-rose-400', clinician_voice: 'bg-rose-400',
  model: 'bg-neutral-400', diagnosis_history: 'bg-neutral-400',
}

const STATUS_STYLE: Record<ConfirmationStatus, string> = {
  confirmed: 'bg-emerald-50 border-emerald-200 text-emerald-700',
  awaiting_patient_confirmation: 'bg-amber-50 border-amber-200 text-amber-700',
  awaiting_clinician_confirmation: 'bg-amber-50 border-amber-200 text-amber-700',
  rejected: 'bg-red-50 border-red-200 text-red-500',
  not_required: 'bg-neutral-100 border-neutral-200 text-neutral-500',
}

const STATUS_LABEL: Record<ConfirmationStatus, string> = {
  confirmed: 'Confirmed',
  awaiting_patient_confirmation: 'Awaiting review',
  awaiting_clinician_confirmation: 'Awaiting review',
  rejected: 'Rejected',
  not_required: 'Auto',
}

const CACHE_KEY = 'prism:timeline:events'

function sortEvents(evts: HormonalHealthEvent[]) {
  return [...evts].sort((a, b) => (b.observedAt ?? '').localeCompare(a.observedAt ?? ''))
}

function dateLabel(iso: string | undefined) {
  if (!iso) return 'Date not recorded'
  return new Date(iso).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
}

function groupByDate(events: HormonalHealthEvent[]): [string, HormonalHealthEvent[]][] {
  const map = new Map<string, HormonalHealthEvent[]>()
  for (const e of events) {
    const key = e.observedAt ? e.observedAt.slice(0, 10) : '__undated__'
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(e)
  }
  return Array.from(map.entries())
}

function EventCard({
  event,
  onConfirm,
  onReject,
  showConflict,
}: {
  event: HormonalHealthEvent
  onConfirm: (id: string) => void
  onReject: (id: string) => void
  showConflict: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  const Icon = ICON[event.modality] ?? Activity
  const dotColor = DOT_COLOR[event.modality] ?? 'bg-neutral-400'
  const isAwaiting =
    event.confirmationStatus === 'awaiting_patient_confirmation' ||
    event.confirmationStatus === 'awaiting_clinician_confirmation'
  const isRejected = event.confirmationStatus === 'rejected'

  return (
    <div className="flex gap-4 items-start">
      {/* Dot */}
      <div className="relative flex flex-col items-center mt-3.5 shrink-0">
        <div className={`w-3 h-3 rounded-full ring-2 ring-white ${dotColor} ${isRejected ? 'opacity-40' : ''}`} />
      </div>

      {/* Card */}
      <div className={`flex-1 mb-4 rounded-xl border bg-white shadow-sm transition-opacity ${isRejected ? 'opacity-50' : ''}`}>
        {/* Header row */}
        <div className="flex items-start justify-between gap-3 px-4 pt-3 pb-3">
          <div className="flex items-start gap-3 min-w-0">
            <Icon size={15} className="mt-0.5 shrink-0 text-neutral-400" />
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-semibold text-neutral-900">{event.variableName}</span>
                {showConflict && <AlertTriangle size={12} className="text-amber-400 shrink-0" />}
              </div>
              {event.value != null && event.missingnessStatus !== 'not_collected' && (
                <p className="text-sm text-neutral-600 mt-0.5">
                  {String(event.value)}{event.unit ? ` ${event.unit}` : ''}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <span className={`inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-medium ${STATUS_STYLE[event.confirmationStatus]}`}>
              {STATUS_LABEL[event.confirmationStatus]}
            </span>
            <button
              onClick={() => setExpanded(v => !v)}
              className="p-1 rounded hover:bg-neutral-100 text-neutral-400"
            >
              {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>
          </div>
        </div>

        {/* Accept / Reject — always visible when awaiting */}
        {isAwaiting && (
          <div className="flex gap-2 px-4 pb-3">
            <button
              onClick={() => onConfirm(event.eventId)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium hover:bg-emerald-100 transition-colors"
            >
              <Check size={12} /> Accept
            </button>
            <button
              onClick={() => onReject(event.eventId)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-50 border border-red-200 text-red-600 text-xs font-medium hover:bg-red-100 transition-colors"
            >
              <X size={12} /> Reject
            </button>
          </div>
        )}

        {/* Expandable detail */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="overflow-hidden"
            >
              <div className="border-t border-neutral-100 px-4 py-3 space-y-3">
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                  <div>
                    <p className="text-neutral-400">Canonical code</p>
                    <p className="text-neutral-700 font-mono mt-0.5">{event.canonicalVariableCode}</p>
                  </div>
                  <div>
                    <p className="text-neutral-400">Modality</p>
                    <p className="text-neutral-700 mt-0.5 capitalize">{event.modality}</p>
                  </div>
                  <div>
                    <p className="text-neutral-400">Provenance</p>
                    <p className="text-neutral-700 mt-0.5">{event.provenance}</p>
                  </div>
                  {event.sourcePage != null && (
                    <div>
                      <p className="text-neutral-400">Source page</p>
                      <p className="text-neutral-700 mt-0.5">{event.sourcePage}</p>
                    </div>
                  )}
                </div>
                {event.evidenceText && (
                  <div>
                    <p className="text-xs text-neutral-400 mb-1">Evidence text</p>
                    <p className="text-xs text-neutral-600 bg-neutral-50 rounded-lg p-2 border border-neutral-100">
                      &ldquo;{event.evidenceText}&rdquo;
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-xs text-neutral-400 mb-1">Extraction confidence</p>
                  <ConfidenceBar value={event.extractionConfidence} size="sm" />
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {event.uncertain && <span className="px-2 py-0.5 rounded-full bg-amber-50 border border-amber-200 text-amber-700 text-xs">Uncertain</span>}
                  {event.negated && <span className="px-2 py-0.5 rounded-full bg-neutral-100 border border-neutral-200 text-neutral-500 text-xs">Negated</span>}
                  {event.historical && <span className="px-2 py-0.5 rounded-full bg-neutral-100 border border-neutral-200 text-neutral-500 text-xs">Historical</span>}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default function TimelinePage() {
  const [events, setEvents] = useState<HormonalHealthEvent[]>(() => {
    try {
      const cached = sessionStorage.getItem(CACHE_KEY)
      return cached ? sortEvents(JSON.parse(cached)) : []
    } catch { return [] }
  })

  useEffect(() => {
    getEvents().then(evts => {
      const sorted = sortEvents(evts)
      setEvents(sorted)
      try { sessionStorage.setItem(CACHE_KEY, JSON.stringify(sorted)) } catch {}
    })
  }, [])

  const conflictCodes = useMemo(() => {
    const codes = events.map(e => e.canonicalVariableCode)
    return new Set(codes.filter((c, i) => codes.indexOf(c) !== i))
  }, [events])

  function handleConfirm(id: string) {
    confirmEvent(id)
    setEvents(prev => prev.map(e => e.eventId === id ? { ...e, confirmationStatus: 'confirmed' as const } : e))
  }

  function handleReject(id: string) {
    rejectEvent(id)
    setEvents(prev => prev.map(e => e.eventId === id ? { ...e, confirmationStatus: 'rejected' as const } : e))
  }

  const groups = useMemo(() => groupByDate(events), [events])

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="flex-1 p-8 ml-56">
        <div className="max-w-2xl mx-auto">
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-neutral-900">Evidence timeline</h1>
            <p className="text-sm text-neutral-500 mt-1">
              {events.length} event{events.length !== 1 ? 's' : ''} · accept or reject proposed observations before they count as evidence
            </p>
          </div>

          {events.length === 0 && (
            <div className="text-center py-16 text-neutral-400 text-sm">
              No events yet — upload a lab report or record a voice note to get started.
            </div>
          )}

          {/* Timeline */}
          <div className="relative">
            {/* Vertical line */}
            {events.length > 0 && (
              <div className="absolute left-[5px] top-4 bottom-0 w-px bg-neutral-200" />
            )}

            {groups.map(([dateKey, group]) => (
              <div key={dateKey} className="mb-2">
                {/* Date label */}
                <div className="flex items-center gap-3 mb-3 pl-7">
                  <span className="text-xs font-semibold uppercase tracking-wide text-neutral-400">
                    {dateKey === '__undated__' ? 'Date not recorded' : dateLabel(dateKey)}
                  </span>
                </div>

                {/* Events for this date */}
                {group.map((evt, i) => (
                  <motion.div
                    key={evt.eventId}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.03, duration: 0.2 }}
                  >
                    <EventCard
                      event={evt}
                      onConfirm={handleConfirm}
                      onReject={handleReject}
                      showConflict={conflictCodes.has(evt.canonicalVariableCode) && evt.missingnessStatus === 'observed'}
                    />
                  </motion.div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}
