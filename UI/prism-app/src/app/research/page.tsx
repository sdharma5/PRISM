'use client'

import { motion } from 'framer-motion'
import { Layers, GitBranch, Database, FlaskConical, Shield, ChevronRight } from 'lucide-react'
import Sidebar from '@/components/Sidebar'
import PrototypeBadge from '@/components/PrototypeBadge'

const SECTIONS = [
  {
    icon: Layers,
    title: 'Modular pipeline architecture',
    body: `Each data modality — symptom questionnaires, lab documents, 2D ultrasound images, wearable streams — runs through an independent extraction pipeline. These pipelines share a universal output schema but have separate parser versions, model versions, and confidence calibration. This means a v0.2 ultrasound segmentation model can be swapped independently of the NLP lab parser without breaking downstream aggregation.`,
  },
  {
    icon: Database,
    title: 'Universal event schema',
    body: `Every data point, regardless of source, is normalized into a HormonalHealthEvent with: canonical variable code, provenance type, extraction confidence, confirmation status, and missingness status. This schema makes the data collection process observable and auditable. It also avoids the temptation to silently impute or discard uncertain values — every decision about missingness is explicit.`,
  },
  {
    icon: GitBranch,
    title: 'Trait vs. state architecture',
    body: `PRISM distinguishes between stable trait-level patterns (phenotype domains, which are relatively stable across cycles) and dynamic state estimates (current hormonal phase, which changes day to day). Many existing tools conflate these. Treating them separately allows appropriate uncertainty representation — high-confidence stable traits alongside acknowledged uncertainty in real-time estimates when wearable data coverage is incomplete.`,
  },
  {
    icon: FlaskConical,
    title: 'Missingness-aware design',
    body: `The model does not treat missing data as zero. Missing AMH is not the same as low AMH. Missing fasting insulin does not default to "normal insulin." Every variable has a MissingnessStatus that distinguishes: not collected (not ordered), not available (ordered but not returned), not applicable (irrelevant for this patient's context), and extraction failed (the document existed but could not be parsed). This prevents the system from silently producing biased estimates.`,
  },
  {
    icon: Shield,
    title: 'Non-diagnostic scope',
    body: `PRISM does not produce diagnoses. The system organizes evidence into pattern domains and profile similarities — statistical descriptions of how a patient's data clusters relative to training distributions. Whether that constitutes a clinical diagnosis is a clinician's determination. PRISM is explicitly designed as research infrastructure, not a clinical decision support tool in the regulated sense.`,
  },
  {
    icon: ChevronRight,
    title: 'Open benchmark plan',
    body: `We intend to release benchmark datasets for each pipeline: (1) NLP extraction F1 on de-identified lab reports, (2) follicle count MAE on paired clinician-verified ultrasound images, (3) phase prediction accuracy vs. serum hormone ground truth. Model cards will accompany each released component, including training data sources, known failure modes, and demographic coverage gaps.`,
  },
]

export default function ResearchPage() {
  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="flex-1 p-8 ml-56">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <PrototypeBadge />
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-neutral-900">Research infrastructure</h1>
          <p className="text-sm text-neutral-500 mt-1 max-w-xl">
            PRISM is built as reusable infrastructure for hormonal health research. This page explains the
            architectural decisions that distinguish it from a consumer app wrapped around a classifier.
          </p>
        </motion.div>

        {/* Key properties */}
        <div className="grid grid-cols-4 gap-px bg-neutral-200 rounded-lg overflow-hidden mb-8 border border-neutral-200">
          {[
            { label: 'Modalities', value: '5+', note: 'Symptom, lab, ultrasound, wearable, document' },
            { label: 'Schema version', value: '0.1.0', note: 'Semantic versioned' },
            { label: 'Pipeline components', value: '5', note: 'NLP, ultrasound, temporal, wearable, CGM' },
            { label: 'Missingness statuses', value: '6', note: 'Granular, not binary' },
          ].map(item => (
            <div key={item.label} className="bg-white px-5 py-4">
              <div className="text-2xl font-semibold text-neutral-900 font-tabular mb-0.5">{item.value}</div>
              <div className="text-xs font-medium text-neutral-700">{item.label}</div>
              <div className="text-xs text-neutral-400 mt-0.5">{item.note}</div>
            </div>
          ))}
        </div>

        {/* Sections */}
        <div className="max-w-2xl divide-y divide-neutral-200">
          {SECTIONS.map((section, i) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.18, delay: i * 0.04 }}
              className="py-6 first:pt-0"
            >
              <div className="flex items-center gap-2.5 mb-2">
                <section.icon size={14} className="text-neutral-400 shrink-0" />
                <h2 className="text-sm font-semibold text-neutral-800">{section.title}</h2>
              </div>
              <p className="text-sm text-neutral-500 leading-relaxed pl-6">{section.body}</p>
            </motion.div>
          ))}
        </div>

        {/* Schema spec note */}
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18, delay: 0.28 }}
          className="mt-8 border-l-2 border-neutral-300 pl-5 max-w-2xl"
        >
          <h3 className="text-sm font-medium text-neutral-800 mb-1.5">On multimodal data pairing</h3>
          <p className="text-sm text-neutral-500 leading-relaxed">
            Current multimodal health AI benchmarks often require simultaneous data across all modalities —
            a constraint that makes datasets unrepresentative of real patient records, where data is
            fragmentary and asynchronous. PRISM explicitly avoids requiring multimodal pairing. Each modality
            contributes independently to the evidence schema; the aggregation layer handles sparsity
            through missingness-aware inference rather than imputation or exclusion.
          </p>
        </motion.div>
      </main>
    </div>
  )
}
