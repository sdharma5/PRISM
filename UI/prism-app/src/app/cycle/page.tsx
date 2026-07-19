'use client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useStore } from '@/lib/store'
import { getCycleDays } from '@/lib/api'
import Sidebar from '@/components/Sidebar'
import PhaseChip from '@/components/PhaseChip'
import { formatDate, PHASE_META, cn } from '@/lib/utils'

export default function Cycle() {
  const { cycleDays, setCycleDays } = useStore()

  useEffect(() => {
    getCycleDays().then(setCycleDays)
  }, [])

  const today = new Date().toISOString().slice(0, 10)

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="flex-1 p-8 ml-56">
        <h1 className="text-2xl font-semibold text-neutral-900 mb-6">Cycle Tracker</h1>

        {/* Phase legend */}
        <div className="flex gap-3 mb-6 flex-wrap">
          {(Object.entries(PHASE_META) as [string, typeof PHASE_META[keyof typeof PHASE_META]][]).map(([k, v]) => (
            <div key={k} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-white text-neutral-700 border border-neutral-200">
              <span className="w-2 h-2 rounded-full" style={{ background: v.color }} />
              {v.label}
            </div>
          ))}
        </div>

        {/* Calendar grid */}
        <div className="grid grid-cols-7 gap-2">
          {['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].map(d => (
            <div key={d} className="text-center text-xs text-neutral-500 font-medium pb-1">{d}</div>
          ))}
          {cycleDays.map((day, i) => {
            const meta = PHASE_META[day.phase]
            const isToday = day.date === today
            return (
              <motion.div
                key={day.date}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.015 }}
                className={cn(
                  'rounded-md p-2 text-center cursor-default border bg-white',
                  isToday ? 'ring-2 ring-neutral-900/20 border-neutral-300' : 'border-neutral-200'
                )}
              >
                <p className="text-xs font-semibold text-neutral-800">{formatDate(day.date)}</p>
                <p className="text-xs mt-0.5" style={{ color: meta.color }}>D{day.dayOfCycle}</p>
                {day.symptoms.length > 0 && (
                  <p className="text-xs text-neutral-400 truncate mt-0.5">{day.symptoms[0]}</p>
                )}
              </motion.div>
            )
          })}
        </div>

        {/* Log section */}
        <h2 className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mt-8 mb-3">Recent Days</h2>
        <div className="space-y-2">
          {[...cycleDays].reverse().slice(0, 7).map(day => (
            <div key={day.date} className="bg-white rounded-md p-4 border border-neutral-200 flex items-center gap-4">
              <div className="w-14 text-center">
                <p className="text-xs text-neutral-500">{formatDate(day.date)}</p>
                <p className="text-sm font-semibold text-neutral-800">D{day.dayOfCycle}</p>
              </div>
              <PhaseChip phase={day.phase} />
              <div className="flex gap-4 ml-auto text-xs text-neutral-500">
                <span>Mood {day.mood}/5</span>
                <span>Energy {day.energy}/5</span>
                {day.flow && <span>Flow: {day.flow}</span>}
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}
