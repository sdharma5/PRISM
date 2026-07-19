import type { ConfirmationStatus } from '@/types'
import { cn } from '@/lib/utils'

const META: Record<ConfirmationStatus, { label: string; className: string }> = {
  confirmed: {
    label: 'Confirmed',
    className: 'bg-emerald-50 border-emerald-200 text-emerald-700',
  },
  awaiting_patient_confirmation: {
    label: 'Awaiting you',
    className: 'bg-amber-50 border-amber-200 text-amber-700',
  },
  awaiting_clinician_confirmation: {
    label: 'Awaiting clinician',
    className: 'bg-orange-50 border-orange-200 text-orange-700',
  },
  rejected: {
    label: 'Rejected',
    className: 'bg-red-50 border-red-200 text-red-700',
  },
  not_required: {
    label: 'Auto',
    className: 'bg-neutral-100 border-neutral-200 text-neutral-500',
  },
}

export default function ConfirmationBadge({ status }: { status: ConfirmationStatus }) {
  const { label, className } = META[status]
  return (
    <span className={cn('inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-medium', className)}>
      {label}
    </span>
  )
}
