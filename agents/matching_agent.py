"""Gemini-powered eligibility and matching engine.

Evaluates a CandidateProfile against a list of Opportunity objects in a single
bounded Gemini call and returns a validated list[MatchResult].
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from models import CandidateProfile, MatchResult, Opportunity

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class MatchingAgentError(RuntimeError):
    """Base exception for matching agent failures."""


class MatchingConfigurationError(MatchingAgentError):
    """Raised when the Gemini client cannot be configured."""


class MatchingAPIError(MatchingAgentError):
    """Raised when the Gemini API call fails."""


class MatchingResponseError(MatchingAgentError):
    """Raised when Gemini returns an unusable response."""


class MatchingValidationError(MatchingAgentError):
    """Raised when a parsed result violates the MatchResult schema or integrity rules."""


# ---------------------------------------------------------------------------
# Gemini client (mirrors profile_agent / discovery_agent pattern)
# ---------------------------------------------------------------------------


def _get_gemini_client(api_key: str) -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise MatchingConfigurationError(
            "Eligibility matching requires the 'google-genai' package."
        ) from exc
    return genai.Client(api_key=api_key), types


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _matching_prompt(
    candidate: CandidateProfile,
    opportunities: list[Opportunity],
) -> str:
    profile_block = (
        f"Name: {candidate.name}\n"
        f"Location: {candidate.location or 'not specified'}\n"
        f"Education: {'; '.join(candidate.education) or 'none listed'}\n"
        f"Skills: {', '.join(candidate.skills) or 'none listed'}\n"
        f"Experience: {'; '.join(candidate.experience) or 'none listed'}\n"
        f"Projects: {'; '.join(candidate.projects) or 'none listed'}\n"
        f"Certifications: {'; '.join(candidate.certifications) or 'none listed'}\n"
        f"Interests: {', '.join(candidate.interests) or 'none listed'}\n"
        f"Years of experience: {candidate.years_of_experience if candidate.years_of_experience is not None else 'not specified'}"
    )

    opp_blocks: list[str] = []
    for opp in opportunities:
        reqs = "; ".join(opp.requirements) if opp.requirements else "none stated"
        opp_blocks.append(
            f"ID: {opp.id}\n"
            f"Title: {opp.title}\n"
            f"Organization: {opp.organization}\n"
            f"Type: {opp.type}\n"
            f"Description: {opp.description}\n"
            f"Requirements: {reqs}\n"
            f"Location: {opp.location or 'not specified'}\n"
            f"Deadline: {opp.deadline or 'not specified'}"
        )
    opportunities_block = "\n\n".join(opp_blocks)

    return f"""
You are an eligibility and matching evaluator. Evaluate the candidate profile
against each opportunity listed below. Use ONLY the data provided — do not
invent skills, experience, education, requirements, or any other information.

CANDIDATE PROFILE:
{profile_block}

OPPORTUNITIES:
{opportunities_block}

SCORING RUBRIC (total 0–100):
- Skills match:        40 points
- Education match:     20 points
- Experience match:    20 points
- Requirements match:  10 points
- Location/other fit:  10 points

ELIGIBILITY RULE:
Set eligible=true ONLY when all explicitly stated hard/mandatory requirements
that can be evaluated are satisfied by the candidate profile.
A hard requirement is one that is clearly stated as mandatory (e.g. required
degree, required years of experience, mandatory skill, citizenship restriction,
explicit location restriction). If no disqualifying hard requirement is
identified, eligible may be true. Do NOT invent hard requirements.

STRENGTHS: List concrete candidate-to-opportunity matches with evidence.
GAPS: List concrete missing or weak areas with evidence.
REASONS: Explain the score, eligibility decision, major strengths, and gaps.

Return a JSON array — one object per opportunity — in this exact shape:
[
  {{
    "opportunity_id": "<exact Opportunity ID from above>",
    "match_score": <float 0.0 to 100.0>,
    "eligible": <true or false>,
    "strengths": ["<concrete strength>", ...],
    "gaps": ["<concrete gap>", ...],
    "reasons": ["<concise reason>", ...]
  }},
  ...
]

Rules:
1. opportunity_id must be copied EXACTLY from the ID field above.
2. match_score must be a float between 0.0 and 100.0 inclusive.
3. Every input opportunity must have exactly one result in the array.
4. Return ONLY the JSON array. No markdown, no explanation.
""".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_opportunities(
    candidate: CandidateProfile,
    opportunities: list[Opportunity],
) -> list[MatchResult]:
    """Evaluate candidate eligibility and match score for each opportunity.

    Uses a single bounded Gemini call to assess all opportunities at once.
    Returns a validated list[MatchResult] with opportunity_id linkage intact.
    """
    if not isinstance(candidate, CandidateProfile):
        raise TypeError("candidate must be a CandidateProfile instance.")

    if not isinstance(opportunities, list):
        raise TypeError("opportunities must be a list of Opportunity instances.")

    for i, opp in enumerate(opportunities):
        if not isinstance(opp, Opportunity):
            raise TypeError(
                f"opportunities[{i}] is not an Opportunity instance: {type(opp)!r}"
            )

    if not opportunities:
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MatchingConfigurationError(
            "GEMINI_API_KEY is not set. Configure it before evaluating opportunities."
        )

    client, types = _get_gemini_client(api_key)
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    valid_ids = {opp.id for opp in opportunities}

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=_matching_prompt(candidate, opportunities),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        raise MatchingAPIError(
            f"Gemini failed during eligibility matching: {exc}"
        ) from exc

    try:
        raw_text = response.text
    except AttributeError as exc:
        raise MatchingResponseError(
            "Gemini returned no text response for eligibility matching."
        ) from exc

    if not raw_text or not raw_text.strip():
        raise MatchingResponseError(
            "Gemini returned an empty response for eligibility matching."
        )

    try:
        raw_list = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        raise MatchingResponseError(
            f"Gemini matching response is not valid JSON: {exc}"
            f"\nRaw: {raw_text[:300]}"
        ) from exc

    if not isinstance(raw_list, list):
        raise MatchingResponseError(
            "Gemini matching response is not a JSON array."
        )

    results: list[MatchResult] = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise MatchingValidationError(
                f"Matching result item is not a dict: {item!r}"
            )

        # Verify opportunity_id linkage before model validation
        opp_id = item.get("opportunity_id", "")
        if opp_id not in valid_ids:
            raise MatchingValidationError(
                f"Gemini returned unknown opportunity_id {opp_id!r}. "
                f"Valid IDs: {sorted(valid_ids)}"
            )

        # Validate match_score range before model validation
        score = item.get("match_score")
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 100.0):
            raise MatchingValidationError(
                f"match_score {score!r} for opportunity_id {opp_id!r} is outside "
                "the required 0–100 range."
            )

        try:
            result = MatchResult.model_validate(item)
        except ValidationError as exc:
            raise MatchingValidationError(
                f"Gemini result for opportunity_id {opp_id!r} does not satisfy "
                f"MatchResult: {exc}"
            ) from exc

        results.append(result)

    return results
