'use client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, TrendingUp, Activity, Calendar } from 'lucide-react'
import { useStore } from '@/lib/store'
import { getUser, getCycleDays, getInsights } from '@/lib/api'
import Sidebar from '@/components/Sidebar'
import PhaseChip from '@/components/PhaseChip'
import PrototypeBadge from '@/components/PrototypeBadge'
import { formatDate, PHASE_META } from '@/lib/utils'
import Link from 'next/link'

export default function Dashboard() {
  const { user, setUser, cycleDays, setCycleDays, insights, setInsights } = useStore()

  useEffect(() => {
    getUser().then(setUser)
    getCycleDays().then(setCycleDays)
    getInsights().then(setInsights)
  }, [])

  const today = cycleDays.find(d => d.date === new Date().toISOString().slice(0, 10))
    ?? cycleDays[cycleDays.length - 1]

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="flex-1 p-8 ml-56">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <PrototypeBadge />
            <Link href="/overview" className="text-xs text-neutral-500 hover:text-neutral-700 transition-colors underline underline-offset-2">
              Full evidence profile in Overview →
            </Link>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-neutral-900">
            Hello, {user?.name ?? '…'}
          </h1>
          <p className="text-sm text-neutral-500 mt-1">{formatDate(new Date().toISOString())} · Cycle day {today?.dayOfCycle ?? '—'}</p>
        </div>

        {/* Phase banner */}
        {today && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-lg p-6 mb-6 text-white"
            style={{ background: `linear-gradient(135deg, ${PHASE_META[today.phase].color} 0%, #c44cf0 100%)` }}
          >
            <div className="flex items-center gap-3 mb-2">
              <PhaseChip phase={today.phase} />
              <span className="text-white/80 text-sm">Phase</span>
            </div>
            <div className="flex gap-8 mt-4">
              <Metric label="Mood" value={today.mood} max={5} />
              <Metric label="Energy" value={today.energy} max={5} />
              {today.flow && <Metric label="Flow" value={today.flow} />}
            </div>
          </motion.div>
        )}

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <StatCard icon={Calendar} label="Cycle length" value={`${user?.cycleLength ?? 28} days`} />
          <StatCard icon={Activity} label="Period length" value={`${user?.periodLength ?? 5} days`} />
          <StatCard icon={TrendingUp} label="Next period" value={nextPeriod(user?.lastPeriodStart, user?.cycleLength)} />
        </div>

        {/* Insights */}
        <h2 className="text-[10px] font-semibold uppercase tracking-widest text-neutral-400 mb-3">PRISM Insights</h2>
        <div className="space-y-3">
          {insights.map((ins, i) => (
            <motion.div
              key={ins.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              className="bg-white rounded-lg p-5 border border-neutral-200"
            >
              <div className="flex items-start gap-3">
                <Sparkles className="w-4 h-4 text-neutral-400 mt-0.5 shrink-0" />
                <div>
                  <p className="text-sm font-medium text-neutral-900">{ins.title}</p>
                  <p className="text-sm text-neutral-500 mt-1">{ins.body}</p>
                  <p className="text-xs text-neutral-400 mt-2">{Math.round(ins.confidence * 100)}% confidence</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  )
}

function Metric({ label, value, max }: { label: string; value: string | number; max?: number }) {
  return (
    <div>
      <p className="text-white/70 text-xs mb-1">{label}</p>
      <p className="text-white font-semibold">
        {typeof value === 'number' && max ? `${value}/${max}` : value}
      </p>
    </div>
  )
}

function StatCard({ icon: Icon, label, value }: { icon: React.ElementType; label: string; value: string }) {
  return (
    <div className="bg-white rounded-lg p-4 border border-neutral-200">
      <Icon className="w-4 h-4 text-neutral-400 mb-2" />
      <p className="text-xs text-neutral-500">{label}</p>
      <p className="text-sm font-semibold text-neutral-900 mt-0.5">{value}</p>
    </div>
  )
}

function nextPeriod(lastStart?: string, cycleLen = 28): string {
  if (!lastStart) return '—'
  const d = new Date(lastStart)
  d.setDate(d.getDate() + cycleLen)
  return formatDate(d.toISOString())
}
