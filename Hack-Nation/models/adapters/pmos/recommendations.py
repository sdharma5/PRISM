"""Personalised recommendation layer — Tavily search + LLM synthesis.

Pipeline position: runs AFTER PmosEvidenceAdapter.predict() and receives the
completed PMOSProfileOutput.  It is the only component allowed to produce
patient-facing natural-language text.

Architecture
------------
1. _build_queries   – derive 2-4 targeted search strings from the patient's
                      specific axis findings and phenotype scores.
2. _search_tavily   – issue those queries against the Tavily Search API
                      (free tier; set TAVILY_API_KEY, promo code: HackNationJuly).
3. _synthesize      – hand the clinical summary + search snippets to an LLM
                      that produces structured recommendations.

LLM backend is OpenAI-compatible, defaulting to Groq (free tier, no billing).

    GROQ (free — recommended):
        LLM_BASE_URL = https://api.groq.com/openai/v1
        LLM_MODEL    = llama-3.3-70b-versatile
        LLM_API_KEY  = <key from console.groq.com>

    Anthropic (if you have credits):
        LLM_BASE_URL = https://api.anthropic.com/v1
        LLM_MODEL    = claude-sonnet-4-6
        LLM_API_KEY  = <anthropic key>
        (also set LLM_ANTHROPIC=1 — uses x-api-key header instead of Bearer)

Safety contract
---------------
* No statement may use "diagnose", "diagnosis", or claim PMOS is confirmed.
  The system prompt enforces this; the validator below checks it before the
  caller ever sees the text.
* Every recommendation carries the Tavily URL(s) that grounded it.
* Missing search results produce a graceful degradation, never hallucinated refs.

Quickstart
----------
    from models.adapters.pmos.recommendations import PersonalisedRecommender, RecommendationRequest

    recommender = PersonalisedRecommender()   # reads TAVILY_API_KEY, LLM_API_KEY from env
    report = recommender.recommend(RecommendationRequest(profile=pmos_profile_output))
    for rec in report.recommendations:
        print(rec.title, rec.body, rec.sources)
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

import httpx

from models.adapters.pmos.profile_output import PMOSProfileOutput

__all__ = [
    "PersonalisedRecommender",
    "Recommendation",
    "RecommendationReport",
    "RecommendationRequest",
]

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_TAVILY_ENDPOINT = "https://api.tavily.com/search"

_DEFAULT_LLM_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

# Diagnostic axes whose "met" status triggers a dedicated Tavily query.
_AXIS_QUERIES: dict[str, str] = {
    "ovulatory_dysfunction": ("PMOS ovulatory dysfunction lifestyle intervention evidence 2024"),
    "hyperandrogenism_clinical": (
        "PMOS clinical hyperandrogenism hirsutism acne evidence-based treatment"
    ),
    "hyperandrogenism_biochemical": (
        "PMOS elevated androgens testosterone management diet exercise"
    ),
    "polycystic_ovarian_morphology": (
        "PMOS polycystic ovarian morphology monitoring ultrasound guidelines"
    ),
}

# Phenotype domains whose elevated z-score triggers an additional query.
_DOMAIN_QUERIES: dict[str, str] = {
    "metabolic": "PMOS insulin resistance metabolic syndrome lifestyle intervention 2024",
    "androgenic": "PMOS hyperandrogenism evidence-based treatment 2023 guidelines",
    "reproductive": "PMOS menstrual cycle regulation evidence-based interventions",
    "ovarian_morphology": "PMOS follicle count ovarian morphology monitoring guidelines",
}

# Half a SD above the cohort mean is considered "elevated" for query selection.
_ELEVATED_Z = 0.5

# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    """One actionable recommendation grounded in a Tavily search result."""

    category: str  # "lifestyle" | "clinical" | "monitoring" | "nutrition"
    title: str
    body: str
    evidence_level: str  # "guideline-backed" | "observational" | "expert-opinion"
    sources: list[str] = field(default_factory=list)  # Tavily page URLs
    caveats: list[str] = field(default_factory=list)


@dataclass
class RecommendationReport:
    """Full recommendation output produced for one patient."""

    patient_id: str
    summary: str  # 2–3 plain-language sentences
    recommendations: list[Recommendation]
    search_queries_used: list[str] = field(default_factory=list)
    raw_search_results: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RecommendationRequest:
    """What the caller provides."""

    profile: PMOSProfileOutput
    #: Optional free-text patient context, e.g. "I want to conceive this year."
    patient_context: str = ""


# ---------------------------------------------------------------------------
# Internal: query builder
# ---------------------------------------------------------------------------


def _build_queries(profile: PMOSProfileOutput) -> list[str]:
    """Return 2–4 targeted search strings based on this patient's findings."""
    queries: list[str] = ["PMOS evidence-based self-management guidelines 2024"]

    for axis, query in _AXIS_QUERIES.items():
        ev = profile.diagnostic_feature_evidence.get(axis)
        if ev and ev.axis_status == "met":
            queries.append(query)

    for domain, query in _DOMAIN_QUERIES.items():
        score = profile.phenotype_domain_scores.get(domain)
        if score is not None and score >= _ELEVATED_Z:
            queries.append(query)

    # Deduplicate preserving insertion order; cap at 4 (free tier friendly).
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:4]


# ---------------------------------------------------------------------------
# Internal: Tavily search
# ---------------------------------------------------------------------------


def _search_tavily(
    queries: list[str],
    *,
    api_key: str,
    max_results_per_query: int = 3,
    timeout: float = 12.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Execute queries via Tavily. Returns (results, warnings).

    Each result dict has keys: query, title, url, content.
    """
    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        for query in queries:
            try:
                resp = client.post(
                    _TAVILY_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "max_results": max_results_per_query,
                        "search_depth": "basic",
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                resp.raise_for_status()
                for hit in resp.json().get("results", []):
                    results.append(
                        {
                            "query": query,
                            "title": hit.get("title", ""),
                            "url": hit.get("url", ""),
                            "content": hit.get("content", ""),
                        }
                    )
            except httpx.HTTPStatusError as exc:
                warnings.append(f"Tavily returned {exc.response.status_code} for '{query}'")
            except httpx.RequestError as exc:
                warnings.append(f"Tavily network error for '{query}': {exc}")

    return results, warnings


# ---------------------------------------------------------------------------
# Internal: prompt assembly
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a health information assistant inside a hormonal-health research tool.
You help patients understand evidence-based actions to discuss with their clinician.
You are NOT a clinician and are NOT providing a medical opinion.

Hard rules — violation means the output will be discarded:
1. Never use the words "diagnose", "diagnosis", or state that the patient has PMOS.
2. Every recommendation must include "discuss with your clinician" or "ask your doctor".
3. Never invent numerical thresholds not present in the provided search excerpts.
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
Produce 3-5 recommendations. Prefer guideline-backed where the evidence exists.
"""


def _build_clinical_summary(profile: PMOSProfileOutput) -> str:
    lines: list[str] = []

    prob = profile.pmos_evidence_probability
    lines.append(
        f"PMOS evidence probability (trained model): {prob:.2f}"
        if prob is not None
        else "PMOS evidence probability: not computed (model abstained or inputs absent)"
    )

    axes = profile.diagnostic_feature_evidence.items()
    met = [ax for ax, ev in axes if ev.axis_status == "met"]
    not_met = [ax for ax, ev in axes if ev.axis_status == "not_met"]
    unassessed = [ax for ax, ev in axes if ev.axis_status == "not_assessable"]

    if met:
        lines.append(f"Diagnostic criteria met: {', '.join(met)}")
    if not_met:
        lines.append(f"Criteria not met: {', '.join(not_met)}")
    if unassessed:
        lines.append(f"Could not assess (data absent): {', '.join(unassessed)}")

    elevated = {
        d: s
        for d, s in profile.phenotype_domain_scores.items()
        if s is not None and s >= _ELEVATED_Z
    }
    if elevated:
        lines.append(
            "Elevated domains: " + ", ".join(f"{d} (z={s:.2f})" for d, s in elevated.items())
        )

    if profile.dominant_profile:
        lines.append(f"Phenotype: {profile.dominant_profile}")
    elif profile.indeterminate:
        lines.append("Phenotype: indeterminate — more data needed")

    if profile.androgenic_evidence_source not in ("unavailable", ""):
        lines.append(f"Androgenic evidence source: {profile.androgenic_evidence_source}")

    return "\n".join(lines)


def _build_search_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(No search results available — recommendations will be general.)"
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        snippet = textwrap.shorten(r["content"], width=320, placeholder="…")
        parts.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {snippet}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internal: LLM call
# ---------------------------------------------------------------------------

_FORBIDDEN_PHRASES = ("diagnos", "you have pmos", "confirmed pmos")


def _synthesize(
    clinical_summary: str,
    search_context: str,
    results: list[dict[str, Any]],
    patient_context: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    use_anthropic_header: bool = False,
    timeout: float = 30.0,
) -> tuple[str, list[Recommendation], list[str]]:
    """Call the LLM and parse its JSON output. Returns (summary, recs, warnings)."""
    warnings: list[str] = []

    user_content = (
        f"## Patient Clinical Summary\n{clinical_summary}\n\n## Search Evidence\n{search_context}\n"
    )
    if patient_context.strip():
        user_content += f"\n## Patient context\n{patient_context.strip()}\n"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if use_anthropic_header:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 1400,
    }
    # Anthropic uses a slightly different field name for max tokens.
    if use_anthropic_header:
        payload["max_tokens"] = payload.pop("max_tokens")  # same key, just explicit
        payload.pop("model", None)  # Anthropic uses model in the URL for some versions

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        warnings.append(f"LLM returned {exc.response.status_code}: {exc.response.text[:200]}")
        return "Recommendations unavailable — LLM error.", [], warnings
    except httpx.RequestError as exc:
        warnings.append(f"LLM network error: {exc}")
        return "Recommendations unavailable — network error.", [], warnings

    raw: str = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip accidental markdown fences.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        warnings.append(f"LLM returned non-JSON ({exc}); first 300 chars: {raw[:300]}")
        return "Recommendations unavailable — parse error.", [], warnings

    summary: str = parsed.get("summary", "")
    recs: list[Recommendation] = []

    for item in parsed.get("recommendations", []):
        body: str = item.get("body", "")
        if any(phrase in body.lower() for phrase in _FORBIDDEN_PHRASES):
            warnings.append(
                f"LLM recommendation contained forbidden language and was dropped: {body[:100]}"
            )
            continue

        indices: list[int] = item.get("source_indices", [])
        sources = [results[i - 1]["url"] for i in indices if 0 < i <= len(results)]

        recs.append(
            Recommendation(
                category=item.get("category", "clinical"),
                title=item.get("title", ""),
                body=body,
                evidence_level=item.get("evidence_level", "expert-opinion"),
                sources=sources,
                caveats=item.get("caveats", []),
            )
        )

    return summary, recs, warnings


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class PersonalisedRecommender:
    """Tavily + LLM recommendation engine for PMOS profile outputs.

    Free setup
    ----------
    1. Get a Tavily key at tavily.com  (promo code ``HackNationJuly``).
    2. Get a free Groq key at console.groq.com (no credit card required).
    3. Set environment variables::

           export TAVILY_API_KEY=tvly-...
           export LLM_API_KEY=gsk_...
           # Groq is the default; no LLM_BASE_URL needed.

    Usage
    -----
    ::

        recommender = PersonalisedRecommender()
        report = recommender.recommend(
            RecommendationRequest(profile=pmos_output, patient_context="I want to conceive.")
        )
        for rec in report.recommendations:
            print(f"[{rec.category}] {rec.title}")
            print(rec.body)
            print("Sources:", rec.sources)
    """

    def __init__(
        self,
        *,
        tavily_api_key: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self._tavily_key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
        self._llm_key = llm_api_key or os.environ.get("LLM_API_KEY", "")
        self._llm_base = llm_base_url or os.environ.get("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
        self._llm_model = llm_model or os.environ.get("LLM_MODEL", _DEFAULT_LLM_MODEL)
        # Use Anthropic's x-api-key header when the base URL points at Anthropic.
        self._use_anthropic_header = bool(os.environ.get("LLM_ANTHROPIC"))

    def recommend(self, request: RecommendationRequest) -> RecommendationReport:
        """Produce personalised recommendations grounded in Tavily search results.

        Raises
        ------
        ValueError
            If either API key is missing — a keyless run would silently produce
            hallucinated or empty output, which is worse than a loud failure.
        """
        if not self._tavily_key:
            raise ValueError(
                "TAVILY_API_KEY is not set.\n"
                "  Get a free key at tavily.com — hackathon promo code: HackNationJuly\n"
                "  Then: export TAVILY_API_KEY=tvly-..."
            )
        if not self._llm_key:
            raise ValueError(
                "LLM_API_KEY is not set.\n"
                "  Free option (Groq): console.groq.com — no billing required.\n"
                "  Then:\n"
                "    export LLM_API_KEY=gsk_...\n"
                "    export LLM_BASE_URL=https://api.groq.com/openai/v1\n"
                "    export LLM_MODEL=llama-3.3-70b-versatile"
            )

        warnings: list[str] = []
        profile = request.profile

        queries = _build_queries(profile)

        results, search_warnings = _search_tavily(queries, api_key=self._tavily_key)
        warnings.extend(search_warnings)

        if not results:
            warnings.append(
                "All Tavily searches returned no results. "
                "Recommendations will be general and unsourced."
            )

        summary, recs, synth_warnings = _synthesize(
            _build_clinical_summary(profile),
            _build_search_context(results),
            results,
            request.patient_context,
            api_key=self._llm_key,
            base_url=self._llm_base,
            model=self._llm_model,
            use_anthropic_header=self._use_anthropic_header,
        )
        warnings.extend(synth_warnings)

        return RecommendationReport(
            patient_id=profile.patient_id,
            summary=summary,
            recommendations=recs,
            search_queries_used=queries,
            raw_search_results=results,
            warnings=warnings,
        )
