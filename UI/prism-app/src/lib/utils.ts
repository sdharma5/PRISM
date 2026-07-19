import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

// Emil: ease-out for entering elements (fast start, feels responsive)
export const EASE_OUT = [0, 0, 0.2, 1] as const

// Standard entrance: fade + 8px rise, 200ms ease-out
export const ENTRANCE = { opacity: 0, y: 8 } as const
export const ENTRANCE_VISIBLE = { opacity: 1, y: 0 } as const
export const TRANSITION_BASE = { duration: 0.2, ease: EASE_OUT } as const

export const PHASE_META = {
  menstrual:  { label: 'Menstrual',  color: '#f43f5e', bg: 'bg-rose-100',   text: 'text-rose-700'   },
  follicular: { label: 'Follicular', color: '#38bdf8', bg: 'bg-sky-100',    text: 'text-sky-700'    },
  ovulatory:  { label: 'Ovulatory',  color: '#34d399', bg: 'bg-emerald-100',text: 'text-emerald-700'},
  luteal:     { label: 'Luteal',     color: '#f59e0b', bg: 'bg-amber-100',  text: 'text-amber-700'  },
} as const
