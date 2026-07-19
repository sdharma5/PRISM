'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  MapPin, Phone, ExternalLink, Search, Printer,
  FileText, Stethoscope, ChevronDown, ChevronUp, AlertCircle
} from 'lucide-react'
import Sidebar from '@/components/Sidebar'
import PrototypeBadge from '@/components/PrototypeBadge'
import { searchProviders, specialtyLabel, type NPIProvider, type Specialty } from '@/lib/npi'
import VisitSummary from '@/components/prism/VisitSummary'
import { usePatientReport } from '@/lib/usePatientReport'
import { axisLabel, evidenceLabel } from '@/lib/present'
import { deriveAskDoctorItems, type AskDoctorItem } from '@/lib/askDoctor'
import { loadAssessment } from '@/lib/reportStore'
import type { WebsitePMOSProfileResponse } from '@/types/api'
import { INSURERS, loadInsurancePlan, saveInsurancePlan, getInsurer, type Insurer } from '@/lib/insurers'

const US_STATES = [
  'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
  'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
  'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
  'VA','WA','WV','WI','WY','DC',
]

const SPECIALTIES: { key: Specialty; label: string; why: string }[] = [
  {
    key: 'reproductive_endo',
    label: 'Reproductive Endocrinologist',
    why: 'Best for cycle irregularity, AMH, and ovarian morphology concerns',
  },
  {
    key: 'obgyn',
    label: 'OB-GYN',
    why: 'First point of contact for menstrual and hormonal concerns',
  },
  {
    key: 'endocrinology',
    label: 'Endocrinologist',
    why: 'Best for metabolic and androgen concerns (insulin, testosterone)',
  },
  {
    key: 'dermatology',
    label: 'Dermatologist',
    why: 'For acne and hair-related androgenic symptoms',
  },
]

function generateScript(
  specialty: Specialty,
  provider: NPIProvider,
  insurer?: Insurer,
  report?: WebsitePMOSProfileResponse | null,
): string {
  // Derived from the actual assessment. This script is read aloud to a clinic,
  // so inventing lab values here would have the patient assert measurements
  // they never had -- the same failure as the printed summary, spoken instead.
  const met = Object.entries(report?.rotterdam_axes ?? {})
    .filter(([, axis]) => axis.status === 'met')
    .map(([key]) => axisLabel(key).toLowerCase())
    .join(' and ')

  const symptoms = met || 'symptoms I have been tracking'

  const specialtyIntro: Record<Specialty, string> = {
    reproductive_endo: "I'm looking to see a reproductive endocrinologist regarding irregular menstrual cycles and some hormone lab results.",
    obgyn: "I'm looking to schedule an appointment regarding irregular periods and some hormonal symptoms I've been tracking.",
    endocrinology: "I'm looking to see an endocrinologist regarding elevated androgen levels and some metabolic concerns.",
    dermatology: "I'm looking to see a dermatologist about hormonal acne and hair changes that I believe may be related to a hormonal imbalance.",
  }

  const insuranceLine = insurer && insurer.id !== 'other'
    ? `I have ${insurer.name} insurance. Before I book, can you confirm that ${provider.name} is in-network for my plan?`
    : `Before I book, can you tell me which insurance plans you currently accept?`

  return `Hello, my name is [Your Name] and I'm calling to schedule an appointment with ${provider.name}.

${specialtyIntro[specialty]}

I have been tracking my symptoms and have a summary I can bring with me. My main concerns include ${symptoms}.

${insuranceLine}

I'll be bringing a written summary of my symptom timeline, lab values, and cycle history to the appointment.

A few questions before I book:
1. Does ${provider.name} see patients for hormonal cycle irregularities?
2. Are new patients currently being accepted?
3. What is the typical wait time for a new patient appointment?
4. What should I bring to my first visit?

Thank you — I look forward to speaking with you.`
}

export default function Care() {
  const [city, setCity] = useState('')
  const [state, setState] = useState('')
  const [specialty, setSpecialty] = useState<Specialty>('reproductive_endo')
  const [results, setResults] = useState<NPIProvider[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [searched, setSearched] = useState(false)
  const [scriptFor, setScriptFor] = useState<NPIProvider | null>(null)
  const [insurancePlanId, setInsurancePlanId] = useState('')
  const printRef = useRef<HTMLDivElement>(null)
  const { report } = usePatientReport()
  const [answers, setAnswers] = useState<Record<string, string | boolean | undefined>>({})

  useEffect(() => {
    setInsurancePlanId(loadInsurancePlan())
    const stored = loadAssessment()
    if (stored?.answers) setAnswers(stored.answers)
  }, [])

  const askDoctorItems = deriveAskDoctorItems(report)

  const insurer = getInsurer(insurancePlanId)

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!city.trim() || !state) return
    setLoading(true)
    setError('')
    setSearched(false)
    try {
      const providers = await searchProviders(city, state, specialty)
      setResults(providers)
      setSearched(true)
    } catch {
      setError('Could not reach the NPI registry. Check your connection and try again.')
    } finally {
      setLoading(false)
    }
  }

  function printSummary() {
    window.print()
  }

  return (
    <>
      {/* Print-only summary — hidden on screen, shown when printing */}
      <div ref={printRef} className="hidden print:block print-summary">
        <VisitSummary report={report} answers={answers} askDoctorItems={askDoctorItems} />
      </div>

      <div className="flex min-h-screen bg-neutral-50 print:hidden">
        <Sidebar />
        <main className="flex-1 p-8 ml-56">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h1 className="text-2xl font-semibold text-neutral-900">Find Care</h1>
              <p className="text-sm text-neutral-500 mt-1">
                Search for specialists near you using the NPI registry, and generate a visit summary to bring to your appointment.
              </p>
            </div>
            <PrototypeBadge />
          </div>

          {/* Three-column layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left column: search + results */}
            <div className="space-y-5">
              {/* Specialty selector */}
              <div className="bg-white rounded-md p-5 border border-neutral-200">
                <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Specialty</p>
                <div className="space-y-2">
                  {SPECIALTIES.map(s => (
                    <label
                      key={s.key}
                      className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer border transition-colors ${
                        specialty === s.key
                          ? 'bg-neutral-900 border-neutral-900 text-white'
                          : 'border-neutral-200 hover:border-neutral-300 text-neutral-500'
                      }`}
                    >
                      <input
                        type="radio"
                        name="specialty"
                        value={s.key}
                        checked={specialty === s.key}
                        onChange={() => setSpecialty(s.key)}
                        className="mt-0.5 accent-neutral-700"
                      />
                      <div>
                        <p className="text-sm font-medium">{s.label}</p>
                        <p className={`text-xs mt-0.5 ${specialty === s.key ? 'text-neutral-300' : 'text-neutral-500'}`}>{s.why}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Insurance plan */}
              <div className="bg-white rounded-md p-5 border border-neutral-200">
                <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Your insurance plan</p>
                <select
                  value={insurancePlanId}
                  onChange={e => { setInsurancePlanId(e.target.value); saveInsurancePlan(e.target.value) }}
                  className="w-full bg-white border border-neutral-300 rounded-lg px-3 py-2 text-sm text-neutral-800 focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
                >
                  <option value="">Select your plan…</option>
                  {INSURERS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}
                </select>
                {insurer && (
                  <div className="mt-2 flex items-start gap-2">
                    <span className="text-xs text-neutral-800 font-medium">{insurer.name}</span>
                    {insurer.note && <span className="text-xs text-neutral-500">— {insurer.note}</span>}
                  </div>
                )}
                {!insurancePlanId && (
                  <p className="text-xs text-neutral-400 mt-2">
                    Select your plan to get coverage check links and a tailored phone script.
                  </p>
                )}
              </div>

              {/* Location search */}
              <form onSubmit={handleSearch} className="bg-white rounded-md p-5 border border-neutral-200">
                <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-3">Location</p>
                <div className="flex gap-3">
                  <input
                    value={city}
                    onChange={e => setCity(e.target.value)}
                    placeholder="City"
                    required
                    className="flex-1 bg-white border border-neutral-300 rounded-lg px-3 py-2 text-sm text-neutral-800 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
                  />
                  <select
                    value={state}
                    onChange={e => setState(e.target.value)}
                    required
                    className="w-20 bg-white border border-neutral-300 rounded-lg px-2 py-2 text-sm text-neutral-800 focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
                  >
                    <option value="">ST</option>
                    {US_STATES.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                  <button
                    type="submit"
                    disabled={loading || !city.trim() || !state}
                    className="flex items-center gap-2 px-4 py-2 bg-neutral-900 hover:bg-neutral-800 disabled:opacity-40 text-white text-sm font-medium rounded-lg transition-colors active:scale-[0.97]"
                  >
                    <Search size={14} />
                    {loading ? 'Searching…' : 'Search'}
                  </button>
                </div>
                {error && (
                  <div className="flex items-center gap-2 mt-3 text-xs text-rose-500">
                    <AlertCircle size={13} /> {error}
                  </div>
                )}
              </form>

              {/* Results */}
              <AnimatePresence>
                {searched && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    {results.length === 0 ? (
                      <div className="bg-white rounded-md p-6 border border-neutral-200 text-center">
                        <p className="text-neutral-500 text-sm">No {specialtyLabel(specialty)}s found in {city}, {state}.</p>
                        <p className="text-neutral-400 text-xs mt-1">Try a nearby larger city or a different specialty.</p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <p className="text-xs text-neutral-500">{results.length} providers found via NPI registry</p>
                        {results.map((p, i) => (
                          <ProviderCard
                            key={p.npi}
                            report={report}
                            provider={p}
                            index={i}
                            onScript={() => setScriptFor(scriptFor?.npi === p.npi ? null : p)}
                            scriptOpen={scriptFor?.npi === p.npi}
                            specialty={specialty}
                            insurer={insurer}
                          />
                        ))}
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Middle column: ask your doctor */}
            <div className="space-y-4">
              {askDoctorItems.length > 0
                ? <AskDoctorSection items={askDoctorItems} />
                : <div className="bg-white rounded-md p-5 border border-neutral-200 text-xs text-neutral-400">Run an analysis on the intake form to see what to discuss with your doctor.</div>
              }
            </div>

            {/* Right column: visit summary */}
            <div className="space-y-4">
              <div className="bg-white rounded-md p-5 border border-neutral-200 sticky top-8">
                <div className="flex items-center gap-2 mb-3">
                  <FileText size={15} className="text-neutral-500" />
                  <p className="text-sm font-semibold text-neutral-900">Visit Summary</p>
                </div>
                <p className="text-xs text-neutral-500 mb-4">
                  A one-page document to bring to your appointment — includes symptom timeline, confirmed lab values, phenotype domain scores, and a disclaimer.
                </p>

                <div className="space-y-2 mb-4">
                  {[
                    report
                      ? `PMOS-related evidence: ${evidenceLabel(report.pmos_assessment.evidence_level)}`
                      : 'No assessment run yet',
                    `${Object.values(report?.rotterdam_axes ?? {}).filter(a => a.status === 'met').length} Rotterdam axes met`,
                    `${Object.values(report?.phenotype?.domain_scores ?? {}).filter(d => d.available).length} phenotype domains assessed`,
                    `${report?.missing_evidence?.length ?? 0} variables not available`,
                    'Research prototype disclaimer',
                  ].map(item => (
                    <div key={item} className="flex items-center gap-2 text-xs text-neutral-500">
                      <span className="w-1.5 h-1.5 rounded-full bg-neutral-400 shrink-0" />
                      {item}
                    </div>
                  ))}
                </div>

                <button
                  onClick={printSummary}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-neutral-900 hover:bg-neutral-800 text-white text-sm font-medium rounded-lg border border-neutral-900 transition-colors"
                >
                  <Printer size={14} />
                  Print / Save as PDF
                </button>
                <p className="text-xs text-neutral-400 mt-2 text-center">
                  Use your browser&apos;s &ldquo;Save as PDF&rdquo; option
                </p>
              </div>

            </div>
          </div>
        </main>
      </div>
    </>
  )
}

function ProviderCard({
  report,
  provider, index, onScript, scriptOpen, specialty, insurer
}: {
  report: WebsitePMOSProfileResponse | null
  provider: NPIProvider
  index: number
  onScript: () => void
  scriptOpen: boolean
  specialty: Specialty
  insurer?: Insurer
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04 }}
      className="bg-white rounded-md border border-neutral-200 overflow-hidden"
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Stethoscope size={13} className="text-neutral-400 shrink-0" />
              <p className="text-sm font-semibold text-neutral-900">
                {provider.name}{provider.credential ? `, ${provider.credential}` : ''}
              </p>
            </div>
            <p className="text-xs text-neutral-500 mt-0.5 ml-5">{provider.specialty}</p>
          </div>
          <span className="text-xs text-neutral-400 shrink-0">NPI {provider.npi}</span>
        </div>

        <div className="mt-3 space-y-1.5 ml-5">
          {provider.address && (
            <div className="flex items-start gap-2 text-xs text-neutral-500">
              <MapPin size={11} className="mt-0.5 shrink-0 text-neutral-400" />
              <span>{provider.address}, {provider.city}, {provider.state} {provider.zip}</span>
            </div>
          )}
          {provider.phone && (
            <div className="flex items-center gap-2 text-xs text-neutral-500">
              <Phone size={11} className="shrink-0 text-neutral-400" />
              <a href={`tel:${provider.phone}`} className="hover:text-neutral-700 transition-colors">
                {provider.phone}
              </a>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-3 ml-5">
          <a
            href={provider.mapsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-neutral-400 hover:text-neutral-700 transition-colors"
          >
            <ExternalLink size={11} /> View on Maps
          </a>
          {insurer && (
            <>
              <span className="text-neutral-300">·</span>
              <a
                href={insurer.searchUrl(provider.name, provider.city, provider.state)}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-emerald-600 hover:text-emerald-700 transition-colors"
                title={insurer.prefilled ? `Opens ${insurer.name} directory pre-filled with this provider` : `Opens ${insurer.name} provider directory`}
              >
                <ExternalLink size={11} />
                Check {insurer.name} coverage
                {!insurer.prefilled && <span className="text-neutral-400 ml-0.5">(search manually)</span>}
              </a>
            </>
          )}
          <span className="text-neutral-300">·</span>
          <button
            onClick={onScript}
            className="flex items-center gap-1 text-xs text-neutral-400 hover:text-neutral-700 transition-colors"
          >
            <Phone size={11} />
            {scriptOpen ? 'Hide script' : 'Phone script'}
            {scriptOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {scriptOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="border-t border-neutral-200 p-4 bg-neutral-50">
              <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-2">
                Suggested phone script
              </p>
              <pre className="text-xs text-neutral-700 whitespace-pre-wrap leading-relaxed font-sans">
                {generateScript(specialty, provider, insurer, report)}
              </pre>
              <p className="text-xs text-neutral-400 mt-3">
                Edit this before calling — replace [Your Name] and adjust details to fit your situation.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}



const URGENCY_STYLE: Record<AskDoctorItem['urgency'], string> = {
  high:   'bg-red-50 border-red-200 text-red-700',
  medium: 'bg-amber-50 border-amber-200 text-amber-700',
  low:    'bg-neutral-100 border-neutral-200 text-neutral-500',
}

function AskDoctorSection({ items }: { items: AskDoctorItem[] }) {
  return (
    <div className="bg-white rounded-md p-5 border border-neutral-200">
      <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">
        Questions for your doctor
      </p>
      <p className="text-xs text-neutral-400 mb-4">
        These tests and conversations would allow PRISM to reach more conclusive findings. Bring this list to your appointment — it&apos;s also included in your printed visit summary.
      </p>
      <div className="space-y-3">
        {items.map((item, i) => (
          <div key={i} className="flex items-start gap-3">
            <span className={`shrink-0 mt-0.5 px-2 py-0.5 rounded-full border text-xs font-medium ${URGENCY_STYLE[item.urgency]}`}>
              {item.urgency}
            </span>
            <div>
              <p className="text-sm font-medium text-neutral-900">{item.question}</p>
              <p className="text-xs text-neutral-500 mt-0.5">{item.why}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '24px' }}>
      <h2 style={{ fontSize: '14px', fontWeight: 'bold', borderBottom: '1px solid #ddd', paddingBottom: '4px', marginBottom: '10px' }}>
        {title}
      </h2>
      {children}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', padding: '3px 0', borderBottom: '1px solid #f3f4f6' }}>
      <span style={{ color: '#555' }}>{label}</span>
      <span style={{ fontWeight: 'bold' }}>{value}</span>
    </div>
  )
}
