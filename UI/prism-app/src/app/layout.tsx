import type { Metadata } from 'next'
import { Plus_Jakarta_Sans } from 'next/font/google'
import './globals.css'

const sans = Plus_Jakarta_Sans({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'PRISM — Personalized Reproductive & Integrated Systemic Model',
  description: 'AI-powered hormonal health insights, personalized to your cycle.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`dark ${sans.variable}`}>
      <body className="min-h-screen antialiased font-sans">{children}</body>
    </html>
  )
}
