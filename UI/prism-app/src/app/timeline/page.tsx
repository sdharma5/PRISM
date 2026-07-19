'use client'

// Horizontal evidence timeline: time runs left to right, oldest first, and the
// events observed at each point stack beneath it.
//
// Confirmation state is deliberately absent. Accepting and rejecting happens
// once, during intake review, where the decision immediately feeds the run.
// Showing status here invited a second, disconnected place to reason about it.

import { useEffect, useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity, FlaskConical, Pill, Watch, FileText, Cpu, Mic, Image,
  ChevronDown, ChevronUp,
} from 'lucide-react'
import Sidebar from '@/components/Sidebar'
import ConfidenceBar from '@/components/ConfidenceBar'
import type { HormonalHealthEvent } from '@/types'
import { getEvents } from '@/lib/api'

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

const UNDATED = '__undated__'

/** Oldest first: a timeline that reads left to right should start at the left. */
function sortEvents(evts: HormonalHealthEvent[]) {
  return [...evts].sort((a, b) => (a.observedAt ?? '').localeCompare(b.observedAt ?? ''))
}

function dateLabel(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
  })
}

/** One column per observation date, undated events collected at the far end. */
function groupByDate(events: HormonalHealthEvent[]): [string, HormonalHealthEvent[]][] {
  const map = new Map<string, HormonalHealthEvent[]>()
  for (const e of events) {
    const key = e.observedAt ? e.observedAt.slice(0, 10) : UNDATED
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(e)
  }
  return Array.from(map.entries()).sort(([a], [b]) => {
    if (a === UNDATED) return 1
    if (b === UNDATED) return -1
    return a.localeCompare(b)
  })
}

function EventCard({ event }: { event: HormonalHealthEvent }) {
  const [expanded, setExpanded] = useState(false)
  const Icon = ICON[event.modality] ?? Activity

  return (
    <div className="rounded-xl border border-neutral-200 bg-white shadow-sm">
      <div className="flex items-start justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-start gap-2">
          <Icon size={14} className="mt-0.5 shrink-0 text-neutral-400" />
          <div className="min-w-0">
            <span className="text-sm font-semibold text-neutral-900">{event.variableName}</span>
            {event.value != null && event.missingnessStatus !== 'not_collected' && (
              <p className="mt-0.5 font-tabular text-sm text-neutral-600">
                {String(event.value)}{event.unit ? ` ${event.unit}` : ''}
              </p>
            )}
          </div>
        </div>
        <button
          onClick={() => setExpanded(v => !v)}
          className="shrink-0 rounded p-1 text-neutral-400 hover:bg-neutral-100"
          aria-label={expanded ? 'Hide detail' : 'Show detail'}
        >
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            <div className="space-y-3 border-t border-neutral-100 px-3 py-3">
              <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
                <div>
                  <p className="text-neutral-400">Canonical code</p>
                  <p className="mt-0.5 break-all font-mono text-neutral-700">
                    {event.canonicalVariableCode}
                  </p>
                </div>
                <div>
                  <p className="text-neutral-400">Modality</p>
                  <p className="mt-0.5 capitalize text-neutral-700">{event.modality}</p>
                </div>
                <div>
                  <p className="text-neutral-400">Provenance</p>
                  <p className="mt-0.5 text-neutral-700">{event.provenance}</p>
                </div>
                {event.sourcePage != null && (
                  <div>
                    <p className="text-neutral-400">Source page</p>
                    <p className="mt-0.5 text-neutral-700">{event.sourcePage}</p>
                  </div>
                )}
              </div>
              {event.evidenceText && (
                <div>
                  <p className="mb-1 text-xs text-neutral-400">Evidence text</p>
                  <p className="rounded-lg border border-neutral-100 bg-neutral-50 p-2 text-xs text-neutral-600">
                    &ldquo;{event.evidenceText}&rdquo;
                  </p>
                </div>
              )}
              <div>
                <p className="mb-1 text-xs text-neutral-400">Extraction confidence</p>
                <ConfidenceBar value={event.extractionConfidence} size="sm" />
              </div>
              <div className="flex flex-wrap gap-1.5">
                {event.uncertain && <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-xs text-amber-700">Uncertain</span>}
                {event.negated && <span className="rounded-full border border-neutral-200 bg-neutral-100 px-2 py-0.5 text-xs text-neutral-500">Negated</span>}
                {event.historical && <span className="rounded-full border border-neutral-200 bg-neutral-100 px-2 py-0.5 text-xs text-neutral-500">Historical</span>}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function TimelinePage() {
  const [events, setEvents] = useState<HormonalHealthEvent[]>([])

  useEffect(() => {
    getEvents().then(evts => setEvents(sortEvents(evts)))
  }, [])

  const groups = useMemo(() => groupByDate(events), [events])

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="ml-56 flex-1 p-8">
        <div className="mx-auto max-w-6xl">
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-neutral-900">Evidence timeline</h1>
            <p className="mt-1 text-sm text-neutral-500">
              {events.length} observation{events.length !== 1 ? 's' : ''} across{' '}
              {groups.length} point{groups.length !== 1 ? 's' : ''} in time
            </p>
          </div>

          {/* Submitted ultrasound scan — shown whenever ultrasound events are present */}
          {events.some(e => e.modality === 'ultrasound') && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25 }}
              className="mb-6 rounded-xl border border-neutral-200 bg-white p-4"
            >
              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-neutral-500">Submitted ultrasound scan</p>
              <div className="flex items-start gap-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/orig_image1148.jpg"
                  alt="Submitted ovarian ultrasound"
                  className="h-32 w-32 rounded-lg object-cover border border-neutral-200 shrink-0"
                />
                <div className="space-y-1.5 pt-1">
                  <p className="text-sm font-semibold text-neutral-900">Ovarian ultrasound · image1148</p>
                  <p className="text-xs text-neutral-500">2D transvaginal scan · May 22 2026</p>
                  <p className="text-xs text-neutral-500">Submitted for automated follicle analysis — results below</p>
                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
                    Awaiting clinician review
                  </span>
                </div>
              </div>
            </motion.div>
          )}

          {events.length === 0 ? (
            <div className="py-16 text-center text-sm text-neutral-400">
              No events yet — upload a lab report or record a voice note to get started.
            </div>
          ) : (
            // Horizontal scroll lives here so the page body never scrolls
            // sideways, however many dates there are.
            <div className="-mx-2 overflow-x-auto px-2 pb-4">
              <div className="flex min-w-min">
                {groups.map(([dateKey, group], col) => (
                  <motion.div
                    key={dateKey}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: col * 0.05, duration: 0.25 }}
                    // No flex gap: the rail below must run unbroken between
                    // columns, so the spacing is padding inside each one.
                    className="w-72 shrink-0 pr-5"
                  >
                    <p className="mb-2 text-sm font-bold tracking-tight text-neutral-900">
                      {dateKey === UNDATED ? 'Date not recorded' : dateLabel(dateKey)}
                    </p>

                    {/* The axis: a continuous rail with one marker per point. */}
                    <div className="relative mb-4 h-3">
                      <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-neutral-200" />
                      <div
                        className={`absolute left-0 top-1/2 h-3 w-3 -translate-y-1/2 rounded-full ring-2 ring-neutral-50 ${
                          DOT_COLOR[group[0]?.modality] ?? 'bg-neutral-400'
                        }`}
                      />
                    </div>

                    <div className="space-y-2">
                      {group.map(evt => (
                        <EventCard key={evt.eventId} event={evt} />
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
