'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronRight, ChevronLeft, Check, Sparkles } from 'lucide-react'
import { useRouter } from 'next/navigation'
import PrototypeBadge from '@/components/PrototypeBadge'

const GOALS = [
  'Understand irregular cycles',
  'Organize PMOS-related evidence',
  'Track symptoms across modalities',
  'Share organized data with a clinician',
  'Participate in hormonal health research',
  'Understand how lab values relate to each other',
]

const AVAILABLE_DATA = [
  'Symptom history',
  'Lab results (PDFs or typed values)',
  'Pelvic ultrasound images',
  'Cycle dates and flow logs',
  'Wearable data (Fitbit, Apple Watch, etc.)',
  'CGM / glucose data',
  'Medication history',
  'Clinical notes',
]

const TOTAL_STEPS = 5

export default function OnboardingPage() {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [selectedGoals, setSelectedGoals] = useState<Set<string>>(new Set())
  const [selectedData, setSelectedData] = useState<Set<string>>(new Set())
  const [consented, setConsented] = useState(false)

  function toggleGoal(g: string) {
    setSelectedGoals(prev => {
      const next = new Set(prev)
      next.has(g) ? next.delete(g) : next.add(g)
      return next
    })
  }

  function toggleData(d: string) {
    setSelectedData(prev => {
      const next = new Set(prev)
      next.has(d) ? next.delete(d) : next.add(d)
      return next
    })
  }

  function handleNext() {
    if (step < TOTAL_STEPS) setStep(s => s + 1)
    else router.push('/intake')
  }

  function handleBack() {
    if (step > 1) setStep(s => s - 1)
  }

  const canProceed =
    (step === 1) ||
    (step === 2 && selectedGoals.size > 0) ||
    (step === 3 && selectedData.size > 0) ||
    (step === 4 && consented) ||
    step === 5

  return (
    <div className="min-h-screen bg-neutral-50 flex flex-col items-center justify-center p-6">
      {/* Progress */}
      <div className="w-full max-w-lg mb-8">
        <div className="flex items-center justify-between text-xs text-neutral-500 mb-2">
          <span>Step {step} of {TOTAL_STEPS}</span>
          <PrototypeBadge />
        </div>
        <div className="h-1 bg-neutral-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-neutral-700 rounded-full"
            animate={{ width: `${(step / TOTAL_STEPS) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      <div className="w-full max-w-lg">
        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.18 }}
              className="bg-white rounded-lg border border-neutral-200 p-8"
            >
              <Sparkles size={24} className="text-neutral-500 mb-4" />
              <h1 className="text-2xl font-semibold text-neutral-900 mb-3">Welcome to PRISM</h1>
              <div className="space-y-3 text-sm text-neutral-500">
                <p>
                  PRISM is a research prototype that organizes your hormonal health evidence into a
                  structured, explainable profile — not a medical device or diagnostic tool.
                </p>
                <p>
                  It collects data from multiple sources (labs, symptoms, wearables, ultrasound) and
                  presents what is known, what is missing, and how confident the model is — without
                  making clinical claims.
                </p>
                <p className="text-neutral-400">
                  If you have a health concern, please work with a qualified clinician. PRISM can help
                  you organize evidence to bring to those conversations.
                </p>
              </div>
            </motion.div>
          )}

          {step === 2 && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.18 }}
              className="bg-white rounded-lg border border-neutral-200 p-8"
            >
              <h1 className="text-2xl font-semibold text-neutral-900 mb-2">What are you hoping to do?</h1>
              <p className="text-sm text-neutral-500 mb-5">Select all that apply</p>
              <div className="space-y-2">
                {GOALS.map(goal => (
                  <button
                    key={goal}
                    onClick={() => toggleGoal(goal)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-md border text-sm text-left transition-colors ${
                      selectedGoals.has(goal)
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-neutral-50 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100'
                    }`}
                  >
                    <div className={`w-4 h-4 rounded border shrink-0 flex items-center justify-center transition-colors ${
                      selectedGoals.has(goal) ? 'bg-neutral-900 border-neutral-900' : 'border-neutral-300'
                    }`}>
                      {selectedGoals.has(goal) && <Check size={10} className="text-white" />}
                    </div>
                    {goal}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {step === 3 && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.18 }}
              className="bg-white rounded-lg border border-neutral-200 p-8"
            >
              <h1 className="text-2xl font-semibold text-neutral-900 mb-2">What data do you have?</h1>
              <p className="text-sm text-neutral-500 mb-5">This helps PRISM show the most relevant input options first</p>
              <div className="space-y-2">
                {AVAILABLE_DATA.map(item => (
                  <button
                    key={item}
                    onClick={() => toggleData(item)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-md border text-sm text-left transition-colors ${
                      selectedData.has(item)
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-neutral-50 text-neutral-500 hover:text-neutral-700 hover:bg-neutral-100'
                    }`}
                  >
                    <div className={`w-4 h-4 rounded border shrink-0 flex items-center justify-center transition-colors ${
                      selectedData.has(item) ? 'bg-neutral-900 border-neutral-900' : 'border-neutral-300'
                    }`}>
                      {selectedData.has(item) && <Check size={10} className="text-white" />}
                    </div>
                    {item}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {step === 4 && (
            <motion.div
              key="step4"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.18 }}
              className="bg-white rounded-lg border border-neutral-200 p-8"
            >
              <h1 className="text-2xl font-semibold text-neutral-900 mb-2">Prototype acknowledgment</h1>
              <div className="bg-amber-50 border border-amber-200 rounded-md p-3 mb-5 text-xs text-amber-700">
                Prototype — backend enforcement pending. These consent placeholders are for UX design purposes only.
                No real data consent or storage is implemented in this prototype.
              </div>
              <div className="space-y-3 text-sm text-neutral-500 mb-6">
                <p>By continuing, you acknowledge:</p>
                <ul className="space-y-2">
                  {[
                    'PRISM is a research prototype, not a medical device',
                    'Evidence patterns shown are not clinical diagnoses',
                    'Data in this demo is synthetic and not stored',
                    'You should consult a qualified clinician for health decisions',
                  ].map(item => (
                    <li key={item} className="flex items-start gap-2">
                      <Check size={14} className="text-neutral-500 mt-0.5 shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
              <button
                onClick={() => setConsented(!consented)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-md border text-sm text-left transition-colors ${
                  consented
                    ? 'border-neutral-900 bg-neutral-900 text-white'
                    : 'border-neutral-200 bg-neutral-50 text-neutral-500 hover:text-neutral-700'
                }`}
              >
                <div className={`w-4 h-4 rounded border shrink-0 flex items-center justify-center transition-colors ${
                  consented ? 'bg-neutral-900 border-neutral-900' : 'border-neutral-300'
                }`}>
                  {consented && <Check size={10} className="text-white" />}
                </div>
                I understand and want to continue
              </button>
            </motion.div>
          )}

          {step === 5 && (
            <motion.div
              key="step5"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.18 }}
              className="bg-white rounded-lg border border-neutral-200 p-8 text-center"
            >
              <div className="w-16 h-16 rounded-full bg-neutral-100 border border-neutral-200 flex items-center justify-center mx-auto mb-5">
                <Sparkles size={28} className="text-neutral-700" />
              </div>
              <h1 className="text-2xl font-semibold text-neutral-900 mb-3">Ready to build your profile</h1>
              <p className="text-sm text-neutral-500">
                Add your data using the intake form to generate your PMOS evidence profile.
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex items-center gap-3 mt-6">
          <button
            onClick={handleBack}
            disabled={step === 1}
            className="flex items-center gap-2 px-4 py-2.5 rounded-md bg-neutral-100 border border-neutral-200 text-neutral-500 text-sm font-medium disabled:opacity-40 hover:bg-neutral-200 active:scale-[0.97] transition-all duration-150"
          >
            <ChevronLeft size={16} /> Back
          </button>
          <button
            onClick={handleNext}
            disabled={!canProceed}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-md bg-neutral-900 hover:bg-neutral-800 text-white text-sm font-medium disabled:opacity-40 active:scale-[0.97] transition-colors duration-100"
          >
            {step === TOTAL_STEPS ? (
              <><Check size={16} /> Start my profile</>
            ) : (
              <>Continue <ChevronRight size={16} /></>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
