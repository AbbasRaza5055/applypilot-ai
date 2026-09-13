"""Unit tests for AP-006 Application Agent."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents import application_agent
from models.application import ApplicationPackage
from models.ats import ATSReport
from models.opportunity import Opportunity
from models.profile import CandidateProfile
from models.resume_optimization import ResumeOptimizationResult


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _make_candidate(**kwargs) -> CandidateProfile:
    defaults = dict(
        name="Jane Doe",
        email="jane@example.com",
        education=["BS Computer Science, State University"],
        skills=["Python", "SQL", "Data Analysis"],
        experience=["Data Analyst Intern at Acme Corp"],
        projects=["Sales dashboard using Python and SQL"],
        certifications=["Google Data Analytics Certificate"],
        years_of_experience=1.0,
    )
    defaults.update(kwargs)
    return CandidateProfile(**defaults)


def _make_opportunity(**kwargs) -> Opportunity:
    defaults = dict(
        id="opp_001",
        title="Data Analyst Intern",
        organization="Example Corp",
        type="internship",
        description="Analyze data using Python, SQL, and Tableau.",
        requirements=["Python", "SQL", "Tableau"],
        url="https://example.com/data-analyst",
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


def _make_optimized_resume(**kwargs) -> ResumeOptimizationResult:
    defaults = dict(
        optimized_resume="Jane Doe\nBS Computer Science\nSkills: Python, SQL",
        changed_sections=["Skills"],
        added_keywords=["data analysis"],
        removed_or_weak_content=[],
        recommendations=["Highlight SQL projects"],
    )
    defaults.update(kwargs)
    return ResumeOptimizationResult(**defaults)


def _make_ats_report(**kwargs) -> ATSReport:
    defaults = dict(
        ats_score=80.0,
        strengths=["Strong Python background"],
        issues=["Tableau not mentioned"],
        matched_keywords=["Python", "SQL"],
        missing_keywords=["Tableau"],
        recommendations=["Add Tableau experience if applicable"],
    )
    defaults.update(kwargs)
    return ATSReport(**defaults)


def _make_package_payload(**kwargs) -> dict:
    defaults = dict(
        cover_letter="Dear Hiring Manager at Example Corp, I am applying for the Data Analyst Intern role.",
        sop="My background in Python and SQL drives my interest in data analysis.",
        application_answers=["I am interested in this role because of my Python experience."],
        required_documents=["Resume", "Transcript"],
        checklist=["Submit resume", "Attach transcript"],
    )
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Mock helper
# ---------------------------------------------------------------------------


def _mock_gemini(monkeypatch, response_text: str):
    captured = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(text=response_text)

    client = SimpleNamespace(models=FakeModels())
    types = SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(application_agent, "_get_gemini_client", lambda api_key: (client, types))
    return captured


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_generate_application_package_returns_validated_package(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))

    result = application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
    )

    assert isinstance(result, ApplicationPackage)
    assert "Example Corp" in result.cover_letter
    assert result.sop
    assert isinstance(result.application_answers, list)
    assert isinstance(result.required_documents, list)
    assert isinstance(result.checklist, list)


# ---------------------------------------------------------------------------
# Exactly one Gemini call
# ---------------------------------------------------------------------------


def test_exactly_one_gemini_call(monkeypatch):
    call_count = {"n": 0}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class CountingModels:
        def generate_content(self, **kwargs):
            call_count["n"] += 1
            return SimpleNamespace(text=json.dumps(_make_package_payload()))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        application_agent,
        "_get_gemini_client",
        lambda api_key: (
            SimpleNamespace(models=CountingModels()),
            SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig),
        ),
    )

    application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
    )

    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# JSON mode, temperature, bounded max_output_tokens
# ---------------------------------------------------------------------------


def test_json_mode_temperature_and_bounded_tokens(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))

    application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
    )

    cfg = captured["config"]
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["temperature"] == 0
    assert isinstance(cfg["max_output_tokens"], int)
    assert cfg["max_output_tokens"] == application_agent.MAX_OUTPUT_TOKENS
    assert cfg["max_output_tokens"] <= 8192


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(application_agent.ApplicationConfigurationError, match="GEMINI_API_KEY"):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


# ---------------------------------------------------------------------------
# API failure
# ---------------------------------------------------------------------------


def test_api_failure_raises_controlled_error(monkeypatch):
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("credential test-key failed")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        application_agent,
        "_get_gemini_client",
        lambda api_key: (
            SimpleNamespace(models=BrokenModels()),
            SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig),
        ),
    )

    with pytest.raises(application_agent.ApplicationAPIError) as exc_info:
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )

    assert "test-key" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Malformed JSON
# ---------------------------------------------------------------------------


def test_malformed_json_raises_response_error(monkeypatch):
    _mock_gemini(monkeypatch, "not valid json {{")

    with pytest.raises(application_agent.ApplicationResponseError, match="valid JSON"):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


# ---------------------------------------------------------------------------
# Non-object JSON (array instead of object)
# ---------------------------------------------------------------------------


def test_non_object_json_raises_response_error(monkeypatch):
    _mock_gemini(monkeypatch, json.dumps(["cover letter text"]))

    with pytest.raises(application_agent.ApplicationResponseError, match="not a JSON object"):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


# ---------------------------------------------------------------------------
# Missing required output fields
# ---------------------------------------------------------------------------


def test_missing_required_field_raises_validation_error(monkeypatch):
    payload = _make_package_payload()
    del payload["cover_letter"]
    _mock_gemini(monkeypatch, json.dumps(payload))

    with pytest.raises(application_agent.ApplicationValidationError, match="missing required fields"):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


def test_multiple_missing_fields_raises_validation_error(monkeypatch):
    payload = _make_package_payload()
    del payload["sop"]
    del payload["checklist"]
    _mock_gemini(monkeypatch, json.dumps(payload))

    with pytest.raises(application_agent.ApplicationValidationError, match="missing required fields"):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


# ---------------------------------------------------------------------------
# Prompt contains all four source inputs
# ---------------------------------------------------------------------------


def test_prompt_contains_candidate_profile(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))
    candidate = _make_candidate(name="Unique Candidate Name")

    application_agent.generate_application_package(
        candidate, _make_opportunity(), _make_optimized_resume(), _make_ats_report()
    )

    assert "Unique Candidate Name" in captured["request"]["contents"]


def test_prompt_contains_opportunity(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))
    opportunity = _make_opportunity(title="Unique Opportunity Title XYZ")

    application_agent.generate_application_package(
        _make_candidate(), opportunity, _make_optimized_resume(), _make_ats_report()
    )

    assert "Unique Opportunity Title XYZ" in captured["request"]["contents"]


def test_prompt_contains_optimized_resume(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))
    resume = _make_optimized_resume(optimized_resume="Unique resume content ABCDEF")

    application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), resume, _make_ats_report()
    )

    assert "Unique resume content ABCDEF" in captured["request"]["contents"]


def test_prompt_contains_ats_report(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))
    ats = _make_ats_report(ats_score=42.0)

    application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), _make_optimized_resume(), ats
    )

    assert "42.0" in captured["request"]["contents"]


# ---------------------------------------------------------------------------
# Grounded / factual prompt constraints
# ---------------------------------------------------------------------------


def test_prompt_forbids_fabrication(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))

    application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
    )

    prompt = captured["request"]["contents"]
    assert "NEVER invent" in prompt or "fabricate" in prompt
    assert "web search" in prompt or "web\nsearch" in prompt


# ---------------------------------------------------------------------------
# No live Gemini API calls (all tests use mocks — verified by absence of real key)
# ---------------------------------------------------------------------------


def test_no_live_api_calls_without_mock(monkeypatch):
    """Confirms that without a real key and without a mock, a config error is raised."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(application_agent.ApplicationConfigurationError):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


# ---------------------------------------------------------------------------
# Type validation on inputs
# ---------------------------------------------------------------------------


def test_invalid_candidate_type_raises_type_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(TypeError, match="CandidateProfile"):
        application_agent.generate_application_package(
            {}, _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


def test_invalid_opportunity_type_raises_type_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(TypeError, match="Opportunity"):
        application_agent.generate_application_package(
            _make_candidate(), {}, _make_optimized_resume(), _make_ats_report()
        )


def test_invalid_optimized_resume_type_raises_type_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(TypeError, match="ResumeOptimizationResult"):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), {}, _make_ats_report()
        )


def test_invalid_ats_report_type_raises_type_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(TypeError, match="ATSReport"):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), {}
        )


# ---------------------------------------------------------------------------
# Empty Gemini response
# ---------------------------------------------------------------------------


def test_empty_gemini_response_raises_response_error(monkeypatch):
    _mock_gemini(monkeypatch, "")

    with pytest.raises(application_agent.ApplicationResponseError):
        application_agent.generate_application_package(
            _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
        )


# ---------------------------------------------------------------------------
# GEMINI_MODEL env var
# ---------------------------------------------------------------------------


def test_gemini_model_env_var_is_respected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))

    application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
    )

    assert captured["request"]["model"] == "gemini-custom-model"


def test_gemini_model_defaults_to_flash_when_env_absent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    captured = _mock_gemini(monkeypatch, json.dumps(_make_package_payload()))

    application_agent.generate_application_package(
        _make_candidate(), _make_opportunity(), _make_optimized_resume(), _make_ats_report()
    )

    assert captured["request"]["model"] == "gemini-3.6-flash"
