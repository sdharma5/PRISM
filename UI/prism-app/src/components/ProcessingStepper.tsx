'use client'

import { motion } from 'framer-motion'
import { Check, Loader2, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ProcessingStage } from '@/types'

interface Step {
  stage: ProcessingStage
  label: string
}

interface Props {
  steps: Step[]
  currentStage: ProcessingStage
}

const ORDER: ProcessingStage[] = [
  'queued',
  'validating',
  'preprocessing',
  'running_model',
  'postprocessing',
  'awaiting_review',
  'completed',
]

function stageIndex(stage: ProcessingStage): number {
  if (stage === 'failed') return -1
  return ORDER.indexOf(stage)
}

export default function ProcessingStepper({ steps, currentStage }: Props) {
  const currentIdx = stageIndex(currentStage)
  const failed = currentStage === 'failed'

  return (
    <div className="space-y-2">
      {steps.map((step, i) => {
        const stepIdx = stageIndex(step.stage)
        const done = !failed && currentIdx > stepIdx
        const active = !failed && currentIdx === stepIdx
        const pending = failed ? false : currentIdx < stepIdx

        return (
          <motion.div
            key={step.stage}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04 }}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg text-sm',
              done && 'text-emerald-600',
              active && 'text-neutral-900 bg-neutral-100',
              pending && 'text-neutral-400',
              failed && step.stage === currentStage && 'text-red-600'
            )}
          >
            <div className="w-5 h-5 shrink-0 flex items-center justify-center">
              {done && <Check size={14} className="text-emerald-600" />}
              {active && (
                <Loader2 size={14} className="text-neutral-700 animate-spin" />
              )}
              {pending && (
                <div className="w-2 h-2 rounded-full bg-neutral-300" />
              )}
              {failed && step.stage === currentStage && (
                <AlertCircle size={14} className="text-red-600" />
              )}
            </div>
            <span>{step.label}</span>
          </motion.div>
        )
      })}
    </div>
  )
}
