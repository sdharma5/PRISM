'use client'

import { FlaskConical } from 'lucide-react'

export default function PrototypeBadge() {
  return (
    <span className="inline-flex items-center gap-1.5 bg-neutral-100 border border-neutral-300 text-neutral-500 text-xs px-2 py-0.5 rounded-full font-medium">
      <FlaskConical size={11} />
      Research prototype · Not a diagnosis
    </span>
  )
}
