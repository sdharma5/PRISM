import type { ProvenanceType } from '@/types'
import { cn } from '@/lib/utils'

const META: Record<ProvenanceType, { label: string; className: string }> = {
  patient_confirmed: {
    label: 'Patient',
    className: 'bg-blue-50 border-blue-200 text-blue-700',
  },
  clinician_confirmed: {
    label: 'Clinician',
    className: 'bg-cyan-50 border-cyan-200 text-cyan-700',
  },
  document_extracted: {
    label: 'Document',
    className: 'bg-neutral-100 border-neutral-300 text-neutral-600',
  },
  device_measured: {
    label: 'Device',
    className: 'bg-teal-50 border-teal-200 text-teal-700',
  },
  dataset_provided: {
    label: 'Dataset',
    className: 'bg-neutral-100 border-neutral-300 text-neutral-500',
  },
  model_measured: {
    label: 'Model (measured)',
    className: 'bg-neutral-50 border-neutral-200 text-neutral-500',
  },
  model_inferred: {
    label: 'Model (inferred)',
    className: 'bg-fuchsia-50 border-fuchsia-200 text-fuchsia-700',
  },
}

export default function ProvenanceBadge({ provenance }: { provenance: ProvenanceType }) {
  const { label, className } = META[provenance]
  return (
    <span className={cn('inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-medium', className)}>
      {label}
    </span>
  )
}
