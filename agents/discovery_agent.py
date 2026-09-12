"""Gemini Search-grounded opportunity discovery from a CandidateProfile.

Two-stage flow
--------------
Stage 1 — Grounded research
    Gemini + Google Search grounding collects current web information.
    Returns prose text with citations; NOT parsed as JSON.

Stage 2 — Structured extraction
    The prose from Stage 1 is passed to a second Gemini call WITHOUT the
    search tool, using response_mime_type="application/json" and a bounded
    output token limit to extract a clean Opportunity array.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from models import CandidateProfile, Opportunity

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

VALID_TYPES = {"job", "internship", "scholarship", "hackathon", "fellowship"}

_TYPE_ALIASES: dict[str, str] = {
    "full-time": "job",
    "full time": "job",
    "part-time": "job",
    "part time": "job",
    "contract": "job",
    "graduate": "fellowship",
    "postdoc": "fellowship",
    "research": "fellowship",
    "grant": "scholarship",
    "bursary": "scholarship",
    "award": "scholarship",
    "competition": "hackathon",
    "contest": "hackathon",
    "challenge": "hackathon",
    "co-op": "internship",
    "coop": "internship",
    "placement": "internship",
    "traineeship": "internship",
}


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class DiscoveryAgentError(RuntimeError):
    """Base exception for opportunity discovery failures."""


class DiscoveryConfigurationError(DiscoveryAgentError):
    """Raised when the Gemini client cannot be configured."""


class DiscoveryAPIError(DiscoveryAgentError):
    """Raised when a Gemini API call fails."""


class DiscoveryResponseError(DiscoveryAgentError):
    """Raised when Gemini returns an unusable response."""


class DiscoveryValidationError(DiscoveryAgentError):
    """Raised when a parsed opportunity violates the Opportunity schema."""


# ---------------------------------------------------------------------------
# Gemini client (mirrors profile_agent pattern)
# ---------------------------------------------------------------------------


def _get_gemini_client(api_key: str) -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise DiscoveryConfigurationError(
            "Opportunity discovery requires the 'google-genai' package."
        ) from exc
    return genai.Client(api_key=api_key), types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(url: str) -> str:
    """Return a 16-char deterministic ID derived from the normalised URL."""
    normalised = url.strip().lower().rstrip("/")
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def _normalize_type(raw: str) -> str:
    """Map any source wording to one of the five canonical opportunity types."""
    lowered = raw.strip().lower()
    if lowered in VALID_TYPES:
        return lowered
    for alias, canonical in _TYPE_ALIASES.items():
        if alias in lowered:
            return canonical
    return "job"  # safest fallback for unknown employment-like types


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _deduplicate(opportunities: list[Opportunity]) -> list[Opportunity]:
    """Keep the first occurrence of each normalised URL."""
    seen: set[str] = set()
    unique: list[Opportunity] = []
    for opp in opportunities:
        key = opp.url.strip().lower().rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append(opp)
    return unique


# ---------------------------------------------------------------------------
# Profile → search queries
# ---------------------------------------------------------------------------


def _build_search_queries(candidate: CandidateProfile) -> list[str]:
    """Generate focused, profile-driven search queries."""
    queries: list[str] = []
    location_hint = f" in {candidate.location}" if candidate.location else ""
    skills_hint = (
        f" {', '.join(candidate.skills[:3])}" if candidate.skills else ""
    )
    edu_hint = candidate.education[0] if candidate.education else ""

    if candidate.skills:
        queries.append(f"{candidate.skills[0]} internship 2024 2025{location_hint}")
        queries.append(f"{candidate.skills[0]} job entry level{location_hint}")

    if edu_hint:
        queries.append(
            f"scholarship for {edu_hint} students 2024 2025{location_hint}"
        )

    if candidate.skills:
        queries.append(
            f"{skills_hint.strip()} hackathon 2024 2025 open registration"
        )

    if candidate.interests:
        queries.append(
            f"{candidate.interests[0]} fellowship 2024 2025{location_hint}"
        )

    if not queries:
        queries.append(f"internship opportunities 2024 2025{location_hint}")

    return queries[:5]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _research_prompt(candidate: CandidateProfile, queries: list[str]) -> str:
    """Stage 1 prompt: instructs Gemini to search and summarise findings."""
    profile_summary = (
        f"Name: {candidate.name}\n"
        f"Location: {candidate.location or 'not specified'}\n"
        f"Education: {'; '.join(candidate.education) or 'not specified'}\n"
        f"Skills: {', '.join(candidate.skills) or 'not specified'}\n"
        f"Experience: {'; '.join(candidate.experience) or 'not specified'}\n"
        f"Interests: {', '.join(candidate.interests) or 'not specified'}\n"
        f"Years of experience: {candidate.years_of_experience or 'not specified'}"
    )
    queries_block = "\n".join(f"- {q}" for q in queries)

    return f"""
You are an opportunity research assistant. Use Google Search to find CURRENT,
real opportunities relevant to the candidate profile below.

CANDIDATE PROFILE:
{profile_summary}

SEARCH QUERIES TO EXECUTE:
{queries_block}

INSTRUCTIONS:
Search for real, currently open opportunities using the queries above.
For each opportunity found, report:
- The exact title
- The organization name
- The opportunity type (job, internship, scholarship, hackathon, or fellowship)
- A brief description (1-3 sentences from the actual listing)
- Any stated requirements
- The deadline if mentioned
- The location if mentioned
- The EXACT source URL from the search result (must be a real http/https URL)

Do NOT invent any details. Only report what the search results actually contain.
""".strip()


def _extraction_prompt(research_text: str, max_results: int) -> str:
    """Stage 2 prompt: converts prose research into a structured JSON array."""
    return f"""
You are a data extraction assistant. Convert the opportunity research below into
a JSON array of up to {max_results} opportunity objects.

RESEARCH TEXT:
---
{research_text}
---

RULES:
1. Extract ONLY opportunities that have a real, verifiable http/https URL in the research text.
2. Do NOT invent URLs, organization names, deadlines, requirements, or any other field.
3. If a field is not present in the research text, use null for optional strings and [] for lists.
4. Normalize the type field to exactly one of: job, internship, scholarship, hackathon, fellowship.
5. Set id to empty string "" (it will be replaced programmatically).

Return a JSON array where each element has exactly these fields:
{{
  "id": "",
  "title": "string",
  "organization": "string",
  "type": "job|internship|scholarship|hackathon|fellowship",
  "description": "string",
  "requirements": ["string"],
  "deadline": "string or null",
  "location": "string or null",
  "url": "real http/https URL from the research",
  "source": "domain name or null"
}}

Return ONLY the JSON array. No markdown fences, no explanation.
""".strip()


# ---------------------------------------------------------------------------
# Two-stage Gemini calls
# ---------------------------------------------------------------------------


def _run_grounded_research(
    client: Any,
    types: Any,
    model_name: str,
    prompt: str,
) -> str:
    """Stage 1: call Gemini with Google Search grounding; return prose text."""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0,
                max_output_tokens=4096,
            ),
        )
    except Exception as exc:
        raise DiscoveryAPIError(
            f"Gemini grounded research call failed: {exc}"
        ) from exc

    try:
        text = response.text
    except AttributeError as exc:
        raise DiscoveryResponseError(
            "Gemini grounded research returned no text."
        ) from exc

    if not text or not text.strip():
        raise DiscoveryResponseError(
            "Gemini grounded research returned an empty response."
        )

    return text


def _run_structured_extraction(
    client: Any,
    types: Any,
    model_name: str,
    research_text: str,
    max_results: int,
) -> list[dict]:
    """Stage 2: extract structured JSON from prose; no grounding tool used."""
    prompt = _extraction_prompt(research_text, max_results)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        raise DiscoveryAPIError(
            f"Gemini structured extraction call failed: {exc}"
        ) from exc

    try:
        raw_text = response.text
    except AttributeError as exc:
        raise DiscoveryResponseError(
            "Gemini structured extraction returned no text."
        ) from exc

    if not raw_text or not raw_text.strip():
        raise DiscoveryResponseError(
            "Gemini structured extraction returned an empty response."
        )

    try:
        raw_list = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise DiscoveryResponseError(
            f"Gemini structured extraction response is not valid JSON: {exc}"
            f"\nRaw: {raw_text[:300]}"
        ) from exc

    if not isinstance(raw_list, list):
        raise DiscoveryResponseError(
            "Gemini structured extraction response is not a JSON array of opportunities."
        )

    return raw_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_opportunities(
    candidate: CandidateProfile,
    max_results: int = 10,
) -> list[Opportunity]:
    """Discover current opportunities relevant to the candidate profile.

    Stage 1: Gemini + Google Search grounding collects raw web research.
    Stage 2: A second Gemini call (no grounding, structured JSON output)
             converts the research into validated Opportunity objects.
    """
    if not isinstance(candidate, CandidateProfile):
        raise TypeError("candidate must be a CandidateProfile instance.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise DiscoveryConfigurationError(
            "GEMINI_API_KEY is not set. Configure it before discovering opportunities."
        )

    client, types = _get_gemini_client(api_key)
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    queries = _build_search_queries(candidate)

    # Stage 1 — grounded research (prose output, may contain citations)
    research_text = _run_grounded_research(
        client, types, model_name, _research_prompt(candidate, queries)
    )

    # Stage 2 — structured extraction (JSON output, no grounding tool)
    raw_list = _run_structured_extraction(
        client, types, model_name, research_text, max_results
    )

    opportunities: list[Opportunity] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue

        url = item.get("url", "") or ""
        if not _is_valid_url(url):
            continue

        item["type"] = _normalize_type(item.get("type", ""))
        item["id"] = _stable_id(url)

        try:
            opp = Opportunity.model_validate(item)
        except ValidationError:
            continue

        opportunities.append(opp)

    opportunities = _deduplicate(opportunities)
    return opportunities[:max_results]
