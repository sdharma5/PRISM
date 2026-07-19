'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Sparkles } from 'lucide-react'
import { useStore } from '@/lib/store'
import { sendChat } from '@/lib/api'
import Sidebar from '@/components/Sidebar'
import { cn } from '@/lib/utils'

const WELCOME: import('@/types').ChatMessage = {
  id: 'welcome',
  role: 'assistant',
  content: "Hi! I'm PRISM, your personalized hormonal health assistant. Ask me anything about your cycle, symptoms, or what to expect this week.",
  timestamp: new Date().toISOString(),
}

export default function Chat() {
  const { messages, setMessages } = useStore()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const all = messages.length ? messages : [WELCOME]

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [all])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMsg: import('@/types').ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString(),
    }
    const next = [...all, userMsg]
    setMessages(next)
    setInput('')
    setLoading(true)

    try {
      const reply = await sendChat(next)
      setMessages([...next, reply])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen bg-white">
      <Sidebar />
      <main className="flex-1 flex flex-col ml-56 h-screen">
        <header className="px-8 py-5 border-b border-neutral-200 flex items-center gap-2 bg-white">
          <Sparkles className="w-4 h-4 text-neutral-500" />
          <span className="font-semibold text-neutral-900">Ask PRISM</span>
        </header>

        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-4 bg-neutral-50">
          <AnimatePresence initial={false}>
            {all.map(msg => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}
              >
                <div
                  className={cn(
                    'max-w-[70%] rounded-lg px-4 py-3 text-sm leading-relaxed',
                    msg.role === 'user'
                      ? 'bg-neutral-900 text-white'
                      : 'bg-white border border-neutral-200 text-neutral-800'
                  )}
                >
                  {msg.content}
                  {msg.citations?.length ? (
                    <p className="text-xs mt-2 opacity-60">
                      Sources: {msg.citations.join(' · ')}
                    </p>
                  ) : null}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
              <div className="bg-white border border-neutral-200 rounded-lg px-4 py-3">
                <span className="flex gap-1">
                  {[0, 1, 2].map(i => (
                    <motion.span
                      key={i}
                      className="w-1.5 h-1.5 rounded-full bg-neutral-400"
                      animate={{ y: [0, -4, 0] }}
                      transition={{ duration: 0.6, delay: i * 0.15, repeat: Infinity }}
                    />
                  ))}
                </span>
              </div>
            </motion.div>
          )}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={submit} className="px-8 py-5 border-t border-neutral-200 flex gap-3 bg-white">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about your cycle, symptoms, or hormones…"
            className="flex-1 rounded-md border border-neutral-300 bg-white px-4 py-2.5 text-sm text-neutral-800 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="rounded-md bg-neutral-900 hover:bg-neutral-800 text-white px-4 py-2.5 disabled:opacity-40 active:scale-[0.97] transition-colors duration-100"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </main>
    </div>
  )
}
