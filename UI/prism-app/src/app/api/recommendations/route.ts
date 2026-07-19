/**
 * Next.js server-side recommendations route.
 *
 * Calls Tavily Search API then an OpenAI-compatible LLM (default: Groq free
 * tier) to produce patient-personalised recommendations grounded in current
 * evidence. Falls back to a structured empty response when keys are absent so
 * the client can degrade gracefully.
 *
 * Env vars (set in .env.local):
 *   TAVILY_API_KEY   — get free key at tavily.com (promo: HackNationJuly)
 *   LLM_API_KEY      — get free Groq key at console.groq.com
 *   LLM_BASE_URL     — default: https://api.groq.com/openai/v1
 *   LLM_MODEL        — default: llama-3.3-70b-versatile
 */

import { NextRequest, NextResponse } from 'next/server'
import type { RecommendationReport, Recommendation } from '@/types'

const TAVILY = 'https://api.tavily.com/search'
const LLM_BASE = process.env.LLM_BASE_URL ?? 'https://api.groq.com/openai/v1'
const LLM_MODEL = process.env.LLM_MODEL ?? 'llama-3.3-70b-versatile'

const BASE_QUERIES = [
  'PCOS evidence-based self-management guidelines 2024',
  'PCOS ovulatory dysfunction lifestyle intervention evidence 2024',
  'PCOS elevated androgens testosterone management diet exercise',
]

const SYSTEM_PROMPT = `You are a health information assistant inside a hormonal-health research tool.
You help patients understand evidence-based actions to discuss with their clinician.
You are NOT a clinician and are NOT providing a medical opinion.

Hard rules — violation means the output will be discarded:
1. Never use the words "diagnose", "diagnosis", or state that the patient has PCOS.
2. Every recommendation must include "discuss with your clinician" or "ask your doctor".
3. Never invent numerical thresholds not in the provided search excerpts.
4. If a search result does not support a recommendation, do not cite it.
5. Tone: warm, direct, plain English. Explain any medical term you use.

Return ONLY valid JSON (no markdown fences) matching this exact schema:
{
  "summary": "<2-3 sentence overview, plain language>",
  "recommendations": [
    {
      "category": "lifestyle|clinical|monitoring|nutrition",
      "title": "<short imperative title>",
      "body": "<1-3 sentence actionable recommendation including clinician qualifier>",
      "evidence_level": "guideline-backed|observational|expert-opinion",
      "source_indices": [<1-based indices into the Search Evidence block>],
      "caveats": ["<optional caveat string>"]
    }
  ]
}
Produce 3-5 recommendations. Prefer guideline-backed where evidence exists.`

type TavilyHit = { title: string; url: string; content: string }

async function searchTavily(
  queries: string[],
  apiKey: string,
): Promise<{ hits: TavilyHit[]; warnings: string[] }> {
  const hits: TavilyHit[] = []
  const warnings: string[] = []
  for (const query of queries) {
    try {
      const res = await fetch(TAVILY, {
        method: 'POST',
        headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, max_results: 3, search_depth: 'basic', include_answer: false }),
      })
      if (!res.ok) { warnings.push(`Tavily ${res.status} for "${query}"`); continue }
      const data = await res.json()
      for (const r of data.results ?? []) {
        hits.push({ title: r.title ?? '', url: r.url ?? '', content: r.content ?? '' })
      }
    } catch (e) {
      warnings.push(`Tavily error for "${query}": ${e}`)
    }
  }
  return { hits, warnings }
}

function buildSearchContext(hits: TavilyHit[]): string {
  if (!hits.length) return '(No search results — recommendations will be general.)'
  return hits
    .map((h, i) => `[${i + 1}] ${h.title}\n    URL: ${h.url}\n    ${h.content.slice(0, 320)}`)
    .join('\n\n')
}

async function synthesize(
  patientContext: string,
  searchContext: string,
  hits: TavilyHit[],
  apiKey: string,
): Promise<{ summary: string; recommendations: Recommendation[]; warnings: string[] }> {
  const warnings: string[] = []
  const userMsg =
    `## Patient context\n${patientContext || 'Patient with irregular cycles and elevated androgenic markers.'}\n\n` +
    `## Search Evidence\n${searchContext}`

  let raw: string
  try {
    const res = await fetch(`${LLM_BASE.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: LLM_MODEL,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: userMsg },
        ],
        temperature: 0.3,
        max_tokens: 1400,
      }),
    })
    if (!res.ok) {
      const txt = await res.text()
      warnings.push(`LLM ${res.status}: ${txt.slice(0, 200)}`)
      return { summary: '', recommendations: [], warnings }
    }
    const data = await res.json()
    raw = (data.choices?.[0]?.message?.content ?? '').trim()
  } catch (e) {
    warnings.push(`LLM network error: ${e}`)
    return { summary: '', recommendations: [], warnings }
  }

  // Strip accidental markdown fences
  if (raw.startsWith('```')) raw = raw.split('\n').slice(1).join('\n').split('```')[0].trim()

  let parsed: { summary: string; recommendations: Array<{
    category: string; title: string; body: string; evidence_level: string
    source_indices: number[]; caveats: string[]
  }> }
  try {
    parsed = JSON.parse(raw)
  } catch {
    warnings.push(`LLM non-JSON response: ${raw.slice(0, 300)}`)
    return { summary: '', recommendations: [], warnings }
  }

  const FORBIDDEN = ['diagnos', 'you have pcos', 'confirmed pcos']
  const recs: Recommendation[] = []
  for (const item of parsed.recommendations ?? []) {
    if (FORBIDDEN.some(p => item.body.toLowerCase().includes(p))) {
      warnings.push(`Recommendation dropped (forbidden language): ${item.body.slice(0, 80)}`)
      continue
    }
    const sources = (item.source_indices ?? [])
      .map((i: number) => hits[i - 1]?.url)
      .filter(Boolean) as string[]
    recs.push({
      category: (item.category ?? 'clinical') as Recommendation['category'],
      title: item.title ?? '',
      body: item.body ?? '',
      evidence_level: (item.evidence_level ?? 'expert-opinion') as Recommendation['evidence_level'],
      sources,
      caveats: item.caveats ?? [],
    })
  }

  return { summary: parsed.summary ?? '', recommendations: recs, warnings }
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}))
  const patientContext: string = body.patient_context ?? ''

  const tavilyKey = process.env.TAVILY_API_KEY ?? ''
  const llmKey = process.env.LLM_API_KEY ?? ''

  if (!tavilyKey || !llmKey) {
    return NextResponse.json(
      { error: 'TAVILY_API_KEY and LLM_API_KEY must be set in .env.local' },
      { status: 503 },
    )
  }

  const { hits, warnings: searchWarnings } = await searchTavily(BASE_QUERIES, tavilyKey)
  const { summary, recommendations, warnings: synthWarnings } = await synthesize(
    patientContext, buildSearchContext(hits), hits, llmKey,
  )

  const report: RecommendationReport = {
    patient_id: body.patient_id ?? 'demo-maya-chen-001',
    summary,
    recommendations,
    search_queries_used: BASE_QUERIES,
    warnings: [...searchWarnings, ...synthWarnings],
  }

  return NextResponse.json(report)
}
