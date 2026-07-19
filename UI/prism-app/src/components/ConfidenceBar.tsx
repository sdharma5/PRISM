import { cn } from '@/lib/utils'

interface Props {
  value: number // 0–1
  showLabel?: boolean
  size?: 'sm' | 'md'
}

export default function ConfidenceBar({ value, showLabel = true, size = 'sm' }: Props) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 80 ? 'bg-emerald-500' :
    pct >= 55 ? 'bg-amber-500' :
    'bg-red-500'

  return (
    <div className="flex items-center gap-2">
      <div className={cn('flex-1 rounded-full bg-neutral-200', size === 'sm' ? 'h-1.5' : 'h-2')}>
        <div
          className={cn('h-full rounded-full', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs text-neutral-500 w-8 text-right shrink-0">{pct}%</span>
      )}
    </div>
  )
}
