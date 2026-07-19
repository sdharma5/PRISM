'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity,
  FlaskConical,
  Pill,
  Watch,
  FileText,
  Cpu,
  Mic,
  Image,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
} from 'lucide-react'
import type { HormonalHealthEvent } from '@/types'
import ConfirmationBadge from './ConfirmationBadge'
import ProvenanceBadge from './ProvenanceBadge'
import ConfidenceBar from './ConfidenceBar'

const MODALITY_ICON: Record<string, React.ElementType> = {
  // frontend aliases
  symptom: Activity,
  cycle: Activity,
  lab: FlaskConical,
  medication: Pill,
  wearable: Watch,
  cgm: Activity,
  ultrasound: Image,
  document: FileText,
  model: Cpu,
  // backend canonical values
  laboratory: FlaskConical,
  questionnaire: FileText,
  patient_voice: Mic,
  clinician_voice: Mic,
  clinical_document: FileText,
  ultrasound_report: FileText,
  ultrasound_image: Image,
  menstrual_history: Activity,
  medication: Pill,
  diagnosis_history: Cpu,
}

const MODALITY_COLORS: Record<string, string> = {
  symptom: 'text-rose-400',
  cycle: 'text-sky-400',
  lab: 'text-blue-400',
  laboratory: 'text-blue-400',
  medication: 'text-teal-400',
  wearable: 'text-emerald-400',
  cgm: 'text-orange-400',
  ultrasound: 'text-cyan-400',
  ultrasound_image: 'text-cyan-400',
  ultrasound_report: 'text-cyan-400',
  document: 'text-amber-400',
  clinical_document: 'text-amber-400',
  model: 'text-neutral-400',
  questionnaire: 'text-violet-400',
  patient_voice: 'text-rose-400',
  clinician_voice: 'text-rose-400',
  menstrual_history: 'text-sky-400',
  diagnosis_history: 'text-neutral-400',
}

interface Props {
  event: HormonalHealthEvent
  onConfirm?: (id: string) => void
  onReject?: (id: string) => void
  showConflict?: boolean
}

export default function EventRow({ event, onConfirm, onReject, showConflict }: Props) {
  const [expanded, setExpanded] = useState(false)
  const Icon = MODALITY_ICON[event.modality] ?? Activity
  const iconColor = MODALITY_COLORS[event.modality] ?? 'text-neutral-400'
  const isAwaiting =
    event.confirmationStatus === 'awaiting_patient_confirmation' ||
    event.confirmationStatus === 'awaiting_clinician_confirmation'

  const dateStr = event.observedAt
    ? new Date(event.observedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    : 'Date not recorded'

  return (
    <div className="border border-neutral-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-neutral-50 transition-colors text-left"
      >
        <Icon size={16} className={iconColor} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-neutral-900 truncate">{event.variableName}</span>
            {showConflict && (
              <AlertTriangle size={12} className="text-amber-400 shrink-0" />
            )}
          </div>
          <p className="text-xs text-neutral-400 mt-0.5">{dateStr}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {event.missingnessStatus !== 'not_collected' && event.value !== null && (
            <span className="text-sm text-neutral-700">
              {String(event.value)}{event.unit ? ` ${event.unit}` : ''}
            </span>
          )}
          {event.missingnessStatus === 'not_collected' && (
            <span className="text-xs text-neutral-400 italic">not collected</span>
          )}
          <ProvenanceBadge provenance={event.provenance} />
          <ConfirmationBadge status={event.confirmationStatus} />
          {expanded ? <ChevronDown size={14} className="text-neutral-400" /> : <ChevronRight size={14} className="text-neutral-400" />}
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-2 border-t border-neutral-100 space-y-3">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <p className="text-neutral-400 mb-0.5">Canonical code</p>
                  <p className="text-neutral-700 font-mono">{event.canonicalVariableCode}</p>
                </div>
                <div>
                  <p className="text-neutral-400 mb-0.5">Modality</p>
                  <p className="text-neutral-700 capitalize">{event.modality}</p>
                </div>
                {event.sourceFileId && (
                  <div>
                    <p className="text-neutral-400 mb-0.5">Source file</p>
                    <p className="text-neutral-700 font-mono truncate">{event.sourceFileId}</p>
                  </div>
                )}
                {event.sourcePage != null && (
                  <div>
                    <p className="text-neutral-400 mb-0.5">Source page</p>
                    <p className="text-neutral-700">{event.sourcePage}</p>
                  </div>
                )}
              </div>
              {event.evidenceText && (
                <div>
                  <p className="text-xs text-neutral-400 mb-1">Evidence text</p>
                  <p className="text-xs text-neutral-700 bg-neutral-50 rounded-lg p-2 border border-neutral-200">
                    &ldquo;{event.evidenceText}&rdquo;
                  </p>
                </div>
              )}
              <div>
                <p className="text-xs text-neutral-400 mb-1">Extraction confidence</p>
                <ConfidenceBar value={event.extractionConfidence} size="sm" />
              </div>
              <div className="flex flex-wrap gap-2 text-xs text-neutral-400">
                {event.uncertain && <span className="px-2 py-0.5 rounded-full bg-amber-50 border border-amber-200 text-amber-700">Uncertain</span>}
                {event.negated && <span className="px-2 py-0.5 rounded-full bg-neutral-100 border border-neutral-200 text-neutral-500">Negated</span>}
                {event.historical && <span className="px-2 py-0.5 rounded-full bg-neutral-100 border border-neutral-200 text-neutral-500">Historical</span>}
              </div>
              {isAwaiting && (onConfirm || onReject) && (
                <div className="flex gap-2 pt-1">
                  {onConfirm && (
                    <button
                      onClick={() => onConfirm(event.eventId)}
                      className="px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium hover:bg-emerald-100 transition-colors"
                    >
                      Confirm
                    </button>
                  )}
                  {onReject && (
                    <button
                      onClick={() => onReject(event.eventId)}
                      className="px-3 py-1.5 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs font-medium hover:bg-red-100 transition-colors"
                    >
                      Reject
                    </button>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
