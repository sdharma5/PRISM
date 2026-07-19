import type { LucideIcon } from 'lucide-react'
import Link from 'next/link'

interface Props {
  icon: LucideIcon
  title: string
  description: string
  ctaLabel?: string
  ctaHref?: string
}

export default function EmptyState({ icon: Icon, title, description, ctaLabel, ctaHref }: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="w-12 h-12 rounded-lg bg-neutral-100 border border-neutral-200 flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-neutral-400" />
      </div>
      <h3 className="text-sm font-medium text-neutral-800 mb-1">{title}</h3>
      <p className="text-sm text-neutral-500 max-w-sm">{description}</p>
      {ctaLabel && ctaHref && (
        <Link
          href={ctaHref}
          className="mt-4 px-4 py-2 rounded-md bg-neutral-900 hover:bg-neutral-800 text-white text-sm font-medium active:scale-[0.97] transition-all duration-150"
        >
          {ctaLabel}
        </Link>
      )}
    </div>
  )
}
