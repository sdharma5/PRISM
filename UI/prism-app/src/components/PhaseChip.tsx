import { cn, PHASE_META } from '@/lib/utils'
import type { CycleDay } from '@/types'

export default function PhaseChip({ phase }: { phase: CycleDay['phase'] }) {
  const m = PHASE_META[phase]
  return (
    <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', m.bg, m.text)}>
      {m.label}
    </span>
  )
}
