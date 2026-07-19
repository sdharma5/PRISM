import { create } from 'zustand'
import type { ChatMessage, CycleDay, PRISMInsight, User } from '@/types'

interface PRISMState {
  user: User | null
  cycleDays: CycleDay[]
  insights: PRISMInsight[]
  messages: ChatMessage[]
  setUser: (u: User) => void
  setCycleDays: (days: CycleDay[]) => void
  setInsights: (ins: PRISMInsight[]) => void
  setMessages: (msgs: ChatMessage[]) => void
}

export const useStore = create<PRISMState>(set => ({
  user: null,
  cycleDays: [],
  insights: [],
  messages: [],
  setUser: u => set({ user: u }),
  setCycleDays: days => set({ cycleDays: days }),
  setInsights: ins => set({ insights: ins }),
  setMessages: msgs => set({ messages: msgs }),
}))

export const usePRISMStore = useStore
