'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  Settings, BarChart2, Clock, BookOpen, Stethoscope, Lightbulb, ClipboardList,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const PRIMARY_NAV = [
  // Intake first -- the entry point, and the only route where a user's own data
  // produces a result.
  { href: '/intake',             icon: ClipboardList, label: 'Add your data' },
  { href: '/overview',           icon: BarChart2,   label: 'Overview' },
  { href: '/timeline',           icon: Clock,       label: 'Timeline' },
  { href: '/care',               icon: Stethoscope, label: 'Find Care' },
  { href: '/recommendations',   icon: Lightbulb,   label: 'Recommendations' },
]

const SECONDARY_NAV = [
  { href: '/research',  icon: BookOpen,        label: 'Research' },
]

export default function Sidebar() {
  const path = usePathname()

  function isActive(href: string) {
    if (href === '/profile/phenotypes') return path.startsWith('/profile')
    return path.startsWith(href)
  }

  return (
    <aside className="fixed left-0 top-0 h-full w-56 bg-white border-r border-neutral-200 flex flex-col py-6 px-3 z-40">
      <div className="mb-8 px-3">
        <span className="font-bold text-sm tracking-widest text-neutral-900 uppercase">PRISM</span>
        <p className="text-[9px] text-neutral-400 leading-tight mt-1">
          Platform for Reusable, Interpretable,<br />Structured Multimodal Evidence
        </p>
      </div>

      <nav className="flex-1 space-y-0.5">
        <p className="text-[10px] text-neutral-400 font-semibold uppercase tracking-widest px-3 mb-2">
          Evidence
        </p>
        {PRIMARY_NAV.map(({ href, icon: Icon, label }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
              isActive(href)
                ? 'bg-neutral-100 text-neutral-900'
                : 'text-neutral-400 hover:text-neutral-700 hover:bg-neutral-50'
            )}
          >
            <Icon size={14} strokeWidth={isActive(href) ? 2 : 1.5} />
            {label}
          </Link>
        ))}

        <div className="my-3 border-t border-neutral-200" />

        <p className="text-[10px] text-neutral-400 font-semibold uppercase tracking-widest px-3 mb-2">
          Tools
        </p>
        {SECONDARY_NAV.map(({ href, icon: Icon, label }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
              isActive(href)
                ? 'bg-neutral-100 text-neutral-900'
                : 'text-neutral-400 hover:text-neutral-700 hover:bg-neutral-50'
            )}
          >
            <Icon size={14} strokeWidth={isActive(href) ? 2 : 1.5} />
            {label}
          </Link>
        ))}
      </nav>

      <Link
        href="/settings"
        className={cn(
          'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors',
          path.startsWith('/settings')
            ? 'bg-neutral-100 text-neutral-900'
            : 'text-neutral-400 hover:text-neutral-700 hover:bg-neutral-50'
        )}
      >
        <Settings size={14} strokeWidth={path.startsWith('/settings') ? 2 : 1.5} />
        Settings
      </Link>
    </aside>
  )
}
