'use client'

import { useEffect, useState } from 'react'
import { useStore } from '@/lib/store'
import { getUser } from '@/lib/api'
import Sidebar from '@/components/Sidebar'
import { cn } from '@/lib/utils'
import { INSURERS, loadInsurancePlan, saveInsurancePlan } from '@/lib/insurers'

export default function Settings() {
  const { user, setUser } = useStore()
  const [saved, setSaved] = useState(false)
  const [insurancePlan, setInsurancePlan] = useState('')

  useEffect(() => {
    getUser().then(setUser)
    setInsurancePlan(loadInsurancePlan())
  }, [])

  function handleSave(e: React.FormEvent) {
    e.preventDefault()
    saveInsurancePlan(insurancePlan)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="flex min-h-screen bg-neutral-50">
      <Sidebar />
      <main className="flex-1 p-8 ml-56 max-w-xl">
        <h1 className="text-3xl font-semibold tracking-tight text-neutral-900 mb-6">Settings</h1>

        <form onSubmit={handleSave} className="space-y-5 bg-white rounded-lg p-6 border border-neutral-200">
          <Field label="Display name" defaultValue={user?.name ?? ''} name="name" />
          <Field label="Email" defaultValue={user?.email ?? ''} name="email" type="email" />
          <Field label="Cycle length (days)" defaultValue={String(user?.cycleLength ?? 28)} name="cycleLength" type="number" />
          <Field label="Period length (days)" defaultValue={String(user?.periodLength ?? 5)} name="periodLength" type="number" />
          <Field label="Last period start" defaultValue={user?.lastPeriodStart ?? ''} name="lastPeriodStart" type="date" />

          <div>
            <label className="block text-xs font-medium text-neutral-500 mb-1">Insurance plan</label>
            <select
              value={insurancePlan}
              onChange={e => setInsurancePlan(e.target.value)}
              className="w-full bg-white border border-neutral-300 rounded-md px-3 py-2 text-sm text-neutral-800 focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
            >
              <option value="">Select your plan…</option>
              {INSURERS.map(i => (
                <option key={i.id} value={i.id}>{i.name}</option>
              ))}
            </select>
            <p className="text-xs text-neutral-400 mt-1">
              Used to generate coverage check links on the Find Care page.
            </p>
          </div>

          <button
            type="submit"
            className={cn(
              'w-full py-2.5 rounded-md text-sm font-medium transition-all active:scale-[0.97]',
              saved
                ? 'bg-emerald-600 text-white'
                : 'bg-neutral-900 hover:bg-neutral-800 text-white'
            )}
          >
            {saved ? 'Saved!' : 'Save changes'}
          </button>
        </form>

        <div className="mt-6 bg-white rounded-lg p-6 border border-neutral-200">
          <h2 className="text-sm font-semibold text-neutral-800 mb-3">Data & Privacy</h2>
          <p className="text-xs text-neutral-500 mb-4">
            This is a research prototype. Settings are stored locally in your browser only.
            No data is sent to any server.
          </p>
          <button className="text-xs text-rose-500 hover:text-rose-600 transition-colors">
            Clear all local data
          </button>
        </div>
      </main>
    </div>
  )
}

function Field({
  label, name, defaultValue, type = 'text'
}: { label: string; name: string; defaultValue: string; type?: string }) {
  return (
    <div>
      <label className="block text-xs font-medium text-neutral-500 mb-1">{label}</label>
      <input
        name={name}
        type={type}
        defaultValue={defaultValue}
        className="w-full bg-white border border-neutral-300 rounded-md px-3 py-2 text-sm text-neutral-800 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
      />
    </div>
  )
}
