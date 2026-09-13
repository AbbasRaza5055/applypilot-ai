import json
import os
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

from models.ats import ATSReport
from models.opportunity import Opportunity
from models.profile import CandidateProfile
from models.resume_optimization import ResumeOptimizationResult


class ResumeOptimizerError(Exception):
    """Base exception for resume optimizer errors."""


class ResumeOptimizerConfigurationError(ResumeOptimizerError):
    """Raised when the optimizer is not configured correctly."""


class ResumeOptimizerAPIError(ResumeOptimizerError):
    """Raised when the Gemini API call fails."""


class ResumeOptimizerResponseError(ResumeOptimizerError):
    """Raised when Gemini returns an invalid response."""


def _get_gemini_client(api_key: str) -> tuple[Any, Any]:
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key), types


def optimize_resume(
    candidate: CandidateProfile,
    opportunity: Opportunity,
    resume_text: str,
    ats_report: ATSReport,
) -> ResumeOptimizationResult:
    """
    Optimize a candidate's resume for a specific opportunity.

    The optimizer may improve wording, structure, clarity, and relevance,
    but must never fabricate candidate information.
    """

    if not isinstance(candidate, CandidateProfile):
        raise TypeError("candidate must be a CandidateProfile")

    if not isinstance(opportunity, Opportunity):
        raise TypeError("opportunity must be an Opportunity")

    if not isinstance(ats_report, ATSReport):
        raise TypeError("ats_report must be an ATSReport")

    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ResumeOptimizerResponseError("resume_text cannot be empty")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ResumeOptimizerConfigurationError(
            "GEMINI_API_KEY is not configured"
        )

    prompt = f"""
You are the Resume Optimization Agent for ApplyPilot AI.

Your task is to optimize the candidate's existing resume for ONE specific
opportunity.

STRICT TRUTHFULNESS RULE:
You may improve wording, structure, clarity, relevance, and keyword alignment,
but you MUST NOT invent or fabricate any information.

NEVER invent:
- skills
- degrees
- certifications
- jobs
- employers
- years of experience
- projects
- responsibilities
- achievements
- awards
- technologies
- measurable results

You may only use facts supported by the supplied resume and candidate profile.

CANDIDATE PROFILE:
{candidate.model_dump_json(indent=2)}

SELECTED OPPORTUNITY:
{opportunity.model_dump_json(indent=2)}

ATS REPORT:
{ats_report.model_dump_json(indent=2)}

ORIGINAL RESUME:
{resume_text}

OPTIMIZATION GOALS:
1. Improve alignment with the selected opportunity.
2. Use relevant missing keywords only when they are truthfully supported
   by the candidate's existing information.
3. Improve clarity and professional wording.
4. Strengthen weak sections without changing factual meaning.
5. Remove or reduce irrelevant or redundant content when appropriate.
6. Preserve the candidate's actual background.
7. Return the COMPLETE optimized resume.

Return ONLY valid JSON with exactly these fields:

{{
  "optimized_resume": "complete optimized resume text",
  "changed_sections": [],
  "added_keywords": [],
  "removed_or_weak_content": [],
  "recommendations": []
}}

All five fields are REQUIRED.

Each list item must be concise and grounded in the supplied information.
"""

    try:
        client, types = _get_gemini_client(api_key)

        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        raise ResumeOptimizerAPIError(
            "Gemini API request failed"
        ) from exc

    raw_text = getattr(response, "text", None)

    if not raw_text:
        raise ResumeOptimizerResponseError(
            "Gemini returned an empty response"
        )

    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ResumeOptimizerResponseError(
            "Gemini returned invalid JSON"
        ) from exc

    required_fields = {
        "optimized_resume",
        "changed_sections",
        "added_keywords",
        "removed_or_weak_content",
        "recommendations",
    }

    if not isinstance(data, dict):
        raise ResumeOptimizerResponseError(
            "Gemini response must be a JSON object"
        )

    if not required_fields.issubset(data.keys()):
        missing_fields = sorted(required_fields - data.keys())

        raise ResumeOptimizerResponseError(
            f"Gemini response is missing required fields: {missing_fields}"
        )

    optimized_resume = data.get("optimized_resume")
    if not isinstance(optimized_resume, str) or not optimized_resume.strip():
        raise ResumeOptimizerResponseError(
            "optimized_resume must be a non-empty string"
        )

    try:
        return ResumeOptimizationResult.model_validate(data)
    except Exception as exc:
        raise ResumeOptimizerResponseError(
            "Gemini response does not match ResumeOptimizationResult"
        ) from exc