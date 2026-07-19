'use client'

// Shared panel building blocks, mostly so "not assessed" looks the same
// everywhere.

import { cn } from '@/lib/utils'

export type Tone = 'neutral' | 'info' | 'ok' | 'warn' | 'alert'

const TONE_PILL: Record<Tone, string> = {
  neutral: 'bg-neutral-100 text-neutral-600 ring-neutral-200',
  info: 'bg-sky-50 text-sky-700 ring-sky-200',
  ok: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  warn: 'bg-amber-50 text-amber-800 ring-amber-200',
  alert: 'bg-rose-50 text-rose-700 ring-rose-200',
}

export function StatusPill({
  children,
  tone = 'neutral',
  className,
}: {
  children: React.ReactNode
  tone?: Tone
  className?: string
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset',
        TONE_PILL[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

export function Card({
  children,
  className,
  as: Tag = 'section',
}: {
  children: React.ReactNode
  className?: string
  as?: React.ElementType
}) {
  return (
    <Tag
      className={cn(
        'rounded-2xl border border-neutral-200/80 bg-white p-6 shadow-sm',
        className,
      )}
    >
      {children}
    </Tag>
  )
}

export function SectionHeading({
  title,
  subtitle,
  action,
}: {
  title: string
  subtitle?: string
  action?: React.ReactNode
}) {
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div>
        <h2 className="text-base font-semibold text-neutral-900">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-neutral-500">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}

/** 0-1 bar. Null renders hatched, so "not measured" ≠ "measured as zero". */
export function Meter({
  value,
  tone = 'info',
  className,
}: {
  value: number | null | undefined
  tone?: Tone
  className?: string
}) {
  const TONE_FILL: Record<Tone, string> = {
    neutral: 'bg-neutral-400',
    info: 'bg-sky-500',
    ok: 'bg-emerald-500',
    warn: 'bg-amber-500',
    alert: 'bg-rose-500',
  }

  if (value == null) {
    return (
      <div
        className={cn(
          'h-2 w-full rounded-full bg-[repeating-linear-gradient(45deg,#f1f1f1,#f1f1f1_4px,#e4e4e7_4px,#e4e4e7_8px)]',
          className,
        )}
        role="img"
        aria-label="Not measured"
      />
    )
  }

  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className={cn('h-2 w-full overflow-hidden rounded-full bg-neutral-100', className)}>
      <div
        className={cn('h-full rounded-full transition-[width] duration-500', TONE_FILL[tone])}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

/** Signed z-score, centred — zero is the cohort average, not the floor. */
export function ZScoreBar({ score, max = 3 }: { score: number | null | undefined; max?: number }) {
  if (score == null) {
    return <Meter value={null} />
  }
  const clamped = Math.max(-max, Math.min(max, score))
  const half = (Math.abs(clamped) / max) * 50
  const positive = clamped >= 0

  return (
    <div className="relative h-2 w-full rounded-full bg-neutral-100">
      <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-neutral-300" />
      <div
        className={cn(
          'absolute top-0 h-full rounded-full',
          positive ? 'bg-amber-500' : 'bg-sky-500',
        )}
        style={
          positive
            ? { left: '50%', width: `${half}%` }
            : { right: '50%', width: `${half}%` }
        }
      />
    </div>
  )
}

/** Small label/value row used throughout the metric blocks. */
export function Stat({
  label,
  value,
  hint,
}: {
  label: string
  value: React.ReactNode
  hint?: string
}) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-neutral-500">{label}</dt>
      <dd className="mt-1 font-tabular text-lg font-semibold text-neutral-900">{value}</dd>
      {hint && <p className="mt-0.5 text-xs text-neutral-400">{hint}</p>}
    </div>
  )
}

/** For sections with nothing to show, where the reason matters. */
export function NotAssessed({ reason }: { reason?: string | null }) {
  return (
    <div className="rounded-xl border border-dashed border-neutral-200 bg-neutral-50/60 p-4">
      <p className="text-sm font-medium text-neutral-700">Not assessed</p>
      {reason && <p className="mt-1 text-sm text-neutral-500">{reason}</p>}
    </div>
  )
}
