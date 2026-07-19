'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowRight, Layers, FlaskConical, Watch, Image, MessageSquare, AlertTriangle } from 'lucide-react'

const INPUTS = [
  { icon: MessageSquare, label: 'Symptom questionnaires' },
  { icon: FlaskConical, label: 'Lab reports (PDF/image)' },
  { icon: Image, label: 'Ultrasound images' },
  { icon: Watch, label: 'Wearable streams' },
  { icon: Layers, label: 'Clinical documents' },
]

export default function Home() {
  return (
    <main className="flex flex-col min-h-screen bg-white">
      {/* Nav */}
      <nav className="fixed top-0 w-full z-50 bg-white border-b border-neutral-200 px-6 py-4 flex items-center justify-between">
        <span className="font-bold text-lg text-neutral-900 tracking-tight">PRISM</span>
        <div className="flex gap-4 items-center">
          <Link href="/research" className="text-sm text-neutral-500 hover:text-neutral-700 transition-colors">
            How it works
          </Link>
          <Link
            href="/onboarding"
            className="text-sm px-4 py-1.5 rounded-md bg-neutral-900 text-white font-medium hover:bg-neutral-800 active:scale-[0.97] transition-colors duration-100"
          >
            Build my profile
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="flex-1 flex flex-col items-center justify-center text-center px-6 pt-32 pb-20 max-w-3xl mx-auto w-full">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700 text-xs font-medium mb-8">
            Research prototype · Not a medical device
          </div>
          <h1 className="text-4xl md:text-6xl font-semibold text-neutral-900 tracking-tight mb-6 leading-tight">
            Your hormonal health is a timeline,<br />
            <span className="text-neutral-500">not a snapshot.</span>
          </h1>
          <p className="text-neutral-500 text-lg max-w-xl mx-auto mb-10 leading-relaxed">
            PRISM organizes your hormonal health evidence — labs, symptoms, cycles, wearables, ultrasound —
            into a structured, transparent profile. It shows what the data supports, what is missing,
            and how confident the model is.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link
              href="/onboarding"
              className="flex items-center gap-2 px-6 py-3 rounded-md bg-neutral-900 hover:bg-neutral-800 text-white font-semibold active:scale-[0.97] transition-colors duration-100"
            >
              Build my profile <ArrowRight size={16} />
            </Link>
            <Link
              href="/research"
              className="flex items-center gap-2 px-6 py-3 rounded-md bg-neutral-50 border border-neutral-200 text-neutral-800 font-semibold hover:bg-neutral-100 active:scale-[0.97] transition-all duration-150"
            >
              See how PRISM works
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Problem statement */}
      <section className="py-16 px-6 border-t border-neutral-200">
        <div className="max-w-3xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.2 }}
          >
            <h2 className="text-xl font-semibold text-neutral-900 mb-4">
              The problem with fragmented evidence
            </h2>
            <p className="text-neutral-500 leading-relaxed mb-4">
              A typical patient with irregular cycles might have: a lab report from one clinic,
              cycle logs in a phone app, an ultrasound report from two years ago, wearable data
              in a fitness tracker, and symptoms described across three different appointments.
            </p>
            <p className="text-neutral-500 leading-relaxed">
              None of these systems talk to each other. No single view shows what is known, what
              is contradictory, and what is missing. PRISM addresses that — not by replacing the
              clinician, but by organizing what is already there.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Supported inputs */}
      <section className="py-16 px-6 bg-neutral-50 border-t border-neutral-200">
        <div className="max-w-3xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.2 }}
          >
            <h2 className="text-xl font-semibold text-neutral-900 mb-2">What PRISM can organize</h2>
            <p className="text-sm text-neutral-400 mb-6">
              Each source is processed independently. Missing data is marked as missing — not imputed.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {INPUTS.map((item, i) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, y: 4 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.04 }}
                  className="flex items-center gap-3 bg-white border border-neutral-200 rounded-lg px-4 py-3"
                >
                  <item.icon size={16} className="text-neutral-500 shrink-0" />
                  <span className="text-sm text-neutral-700">{item.label}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Transparency */}
      <section className="py-16 px-6 border-t border-neutral-200">
        <div className="max-w-3xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.2 }}
          >
            <h2 className="text-xl font-semibold text-neutral-900 mb-4">Uncertainty is explicit</h2>
            <div className="grid grid-cols-3 gap-4">
              {[
                { label: 'Provenance', desc: 'Every data point shows where it came from and how it was extracted.' },
                { label: 'Confidence', desc: 'Extraction confidence scores are shown, not hidden behind polished outputs.' },
                { label: 'Missingness', desc: 'Missing data is labeled with why it is missing — not silently excluded.' },
              ].map(item => (
                <div key={item.label} className="bg-neutral-50 rounded-lg border border-neutral-200 p-4">
                  <p className="text-sm font-medium text-neutral-800 mb-1">{item.label}</p>
                  <p className="text-xs text-neutral-500">{item.desc}</p>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Non-diagnostic disclaimer */}
      <section className="py-10 px-6 border-t border-neutral-200">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-start gap-3 text-xs text-neutral-400">
            <AlertTriangle size={14} className="text-neutral-400 mt-0.5 shrink-0" />
            <p>
              PRISM is a research prototype and is not a medical device, clinical decision support system,
              or diagnostic tool. It does not produce diagnoses, make treatment recommendations, or replace
              clinical judgment. All pattern descriptions represent statistical organization of available evidence.
              Consult a qualified clinician for any health decisions.
            </p>
          </div>
        </div>
      </section>
    </main>
  )
}
