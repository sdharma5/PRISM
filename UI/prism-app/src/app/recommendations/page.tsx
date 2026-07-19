'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Lightbulb, ExternalLink, RefreshCw, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react'
import Sidebar from '@/components/Sidebar'
import PrototypeBadge from '@/components/PrototypeBadge'
import { getRecommendations, MOCK_RECOMMENDATIONS } from '@/lib/api'
import type { RecommendationReport, Recommendation, RecommendationCategory } from '@/types'

const CATEGORY_COLOURS: Record<RecommendationCategory, string> = {
  lifestyle:  'bg-emerald-50 text-emerald-700 border-emerald-200',
  nutrition:  'bg-amber-50 text-amber-700 border-amber-200',
  monitoring: 'bg-sky-50 text-sky-700 border-sky-200',
  clinical:   'bg-violet-50 text-violet-700 border-violet-200',
}

const EVIDENCE_LABEL: Record<string, string> = {
  'guideline-backed': 'Guideline-backed',
  observational:      'Observational study',
  'expert-opinion':   'Expert opinion',
}

function EvidencePip({ level }: { level: string }) {
  const colours =
    level === 'guideline-backed' ? 'bg-emerald-500'
    : level === 'observational'  ? 'bg-amber-400'
    : 'bg-neutral-400'
  return (
    <span className="flex items-center gap-1.5 text-xs text-neutral-500">
      <span className={`w-2 h-2 rounded-full ${colours}`} />
      {EVIDENCE_LABEL[level] ?? level}
    </span>
  )
}

function RecommendationCard({ rec, index }: { rec: Recommendation; index: number }) {
  const [open, setOpen] = useState(false)
  const colourClass = CATEGORY_COLOURS[rec.category] ?? 'bg-neutral-50 text-neutral-600 border-neutral-200'

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="bg-white border border-neutral-200 rounded-md overflow-hidden"
    >
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full text-left p-4 flex items-start gap-3 hover:bg-neutral-50 transition-colors"
      >
        <span className={`shrink-0 mt-0.5 px-2 py-0.5 rounded text-[11px] font-semibold border capitalize ${colourClass}`}>
          {rec.category}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-neutral-900 leading-snug">{rec.title}</p>
          <EvidencePip level={rec.evidence_level} />
        </div>
        <span className="shrink-0 text-neutral-400 mt-0.5">
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="border-t border-neutral-100 px-4 py-3 space-y-3">
              <p className="text-sm text-neutral-700 leading-relaxed">{rec.body}</p>

              {rec.caveats.length > 0 && (
                <div className="bg-amber-50 border border-amber-200 rounded p-2.5">
                  {rec.caveats.map((c, i) => (
                    <p key={i} className="text-xs text-amber-800">{c}</p>
                  ))}
                </div>
              )}

              {rec.sources.length > 0 && (
                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-neutral-400 uppercase tracking-wide">Sources</p>
                  {rec.sources.map((url, i) => (
                    <a
                      key={i}
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-xs text-sky-600 hover:text-sky-800 transition-colors truncate"
                    >
                      <ExternalLink size={10} className="shrink-0" />
                      <span className="truncate">{url}</span>
                    </a>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function RecommendationsPage() {
  const [report, setReport] = useState<RecommendationReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [context, setContext] = useState('')
  const [isMock, setIsMock] = useState(false)

  async function load(ctx = context) {
    setLoading(true)
    setError('')
    setIsMock(false)
    try {
      const result = await getRecommendations(undefined, ctx)
      setReport(result)
      setIsMock(result === MOCK_RECOMMENDATIONS)
    } catch {
      setError('Could not load recommendations. Try again in a moment.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load('') }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="ml-56 flex-1 p-8">
        <div className="mx-auto max-w-2xl space-y-6">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Lightbulb size={18} className="text-amber-500" />
                <h1 className="text-2xl font-semibold tracking-tight text-neutral-900">
                  Personalised Recommendations
                </h1>
              </div>
              <p className="mt-1 text-sm text-neutral-500">
                Evidence-grounded suggestions based on your assessment findings. Discuss each with your clinician before acting.
              </p>
            </div>
            <PrototypeBadge />
          </div>

          {/* Context input */}
          <div className="bg-white border border-neutral-200 rounded-md p-4 space-y-3">
            <label className="block text-xs font-semibold text-neutral-500 uppercase tracking-wide">
              Optional: add context for more tailored suggestions
            </label>
            <textarea
              value={context}
              onChange={e => setContext(e.target.value)}
              placeholder={`e.g. "I want to conceive in the next year" or "I'm vegetarian and can't tolerate metformin"`}
              rows={2}
              className="w-full text-sm text-neutral-800 placeholder-neutral-400 border border-neutral-200 rounded-lg px-3 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
            />
            <button
              onClick={() => load(context)}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-neutral-900 hover:bg-neutral-800 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
            >
              <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
              {loading ? 'Generating…' : 'Get recommendations'}
            </button>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-sm text-rose-600 bg-rose-50 border border-rose-200 rounded-md p-3">
              <AlertCircle size={14} className="shrink-0" />
              {error}
            </div>
          )}

          {isMock && !loading && (
            <div className="flex items-center gap-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
              <AlertCircle size={12} className="shrink-0" />
              Showing example recommendations. Set <code className="font-mono bg-amber-100 px-1 rounded">TAVILY_API_KEY</code> and <code className="font-mono bg-amber-100 px-1 rounded">LLM_API_KEY</code> in <code className="font-mono bg-amber-100 px-1 rounded">.env.local</code> to enable live search-grounded suggestions.
            </div>
          )}

          {report && !loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              {/* Summary */}
              {report.summary && (
                <div className="bg-white border border-neutral-200 rounded-md p-4">
                  <p className="text-[10px] font-semibold text-neutral-400 uppercase tracking-wide mb-2">Summary</p>
                  <p className="text-sm text-neutral-700 leading-relaxed">{report.summary}</p>
                </div>
              )}

              {/* Recommendations */}
              <div className="space-y-2">
                {report.recommendations.map((rec, i) => (
                  <RecommendationCard key={i} rec={rec} index={i} />
                ))}
              </div>

              {/* Footer */}
              <div className="flex items-start gap-2 text-xs text-neutral-400 bg-white border border-neutral-200 rounded-md p-3">
                <AlertCircle size={12} className="shrink-0 mt-0.5" />
                <span>
                  These suggestions are generated by an AI research prototype using web-retrieved evidence. They do not constitute medical advice and must be reviewed with a qualified clinician before any action is taken.
                </span>
              </div>

              {report.warnings.length > 0 && (
                <details className="text-xs text-neutral-400">
                  <summary className="cursor-pointer hover:text-neutral-600">
                    {report.warnings.length} generation {report.warnings.length === 1 ? 'warning' : 'warnings'}
                  </summary>
                  <ul className="mt-1 space-y-0.5 pl-3">
                    {report.warnings.map((w, i) => <li key={i}>{w}</li>)}
                  </ul>
                </details>
              )}
            </motion.div>
          )}
        </div>
      </main>
    </div>
  )
}
