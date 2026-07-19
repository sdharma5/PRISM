'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'

import Sidebar from '@/components/Sidebar'
import CurrentState from '@/components/prism/CurrentState'
import EvidenceGaps from '@/components/prism/EvidenceGaps'
import EvidenceHeader from '@/components/prism/EvidenceHeader'
import PhenotypeDomains from '@/components/prism/PhenotypeDomains'
import PhenotypeProfile from '@/components/prism/PhenotypeProfile'
import { Card } from '@/components/prism/Primitives'
import { ReportError, ReportLoading } from '@/components/prism/ReportStates'
import RotterdamAxes from '@/components/prism/RotterdamAxes'
import SourceData from '@/components/prism/SourceData'
import { usePatientReport } from '@/lib/usePatientReport'
import { EASE_OUT } from '@/lib/utils'

const SECTION = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
}

export default function OverviewPage() {
  const { report: activeReport, status, loading, error, noAssessment, refresh } = usePatientReport()

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="ml-56 flex-1 p-8">
        <div className="mx-auto max-w-3xl space-y-6">
          <header>
            <h1 className="text-2xl font-semibold tracking-tight text-neutral-900">
              Your PMOS evidence profile
            </h1>
            <p className="mt-1 text-sm text-neutral-500">
              What the available evidence supports, and what it could not determine.
            </p>
          </header>

          {loading && !activeReport && <ReportLoading />}

          {!loading && error && !activeReport && <ReportError error={error} onRetry={refresh} />}

          {!loading && !error && noAssessment && !activeReport && (
            <Card>
              <p className="text-sm font-medium text-neutral-900">No assessment yet</p>
              <p className="mt-1 text-sm text-neutral-500">
                Add your data and run an analysis. Your result will appear here and stay
                until you run a new one.
              </p>
              <Link
                href="/intake"
                className="mt-4 inline-flex items-center gap-2 rounded-lg bg-neutral-900 px-4 py-2 text-sm font-semibold text-white hover:bg-neutral-800"
              >
                Add your data
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Card>
          )}

          {activeReport && (
            <>
              {[
                <EvidenceHeader key="header" report={activeReport} />,
                <RotterdamAxes key="axes" report={activeReport} />,
                <PhenotypeDomains key="domains" report={activeReport} />,
                <PhenotypeProfile key="profile" report={activeReport} />,
                <EvidenceGaps key="gaps" report={activeReport} />,
                <CurrentState key="state" report={activeReport} />,
                <SourceData key="sources" report={activeReport} status={status} />,
              ].map((section, index) => (
                <motion.div
                  key={section.key}
                  initial={SECTION.initial}
                  animate={SECTION.animate}
                  transition={{ delay: index * 0.04, ease: EASE_OUT, duration: 0.35 }}
                >
                  {section}
                </motion.div>
              ))}
            </>
          )}
        </div>
      </main>
    </div>
  )
}
