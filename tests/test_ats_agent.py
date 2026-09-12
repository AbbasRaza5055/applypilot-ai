"""Unit tests for AP-004 ATS Intelligence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents import ats_agent
from models import ATSReport, CandidateProfile, Opportunity


def _make_candidate(**kwargs) -> CandidateProfile:
    defaults = dict(
        name="Jane Doe",
        education=["BS Computer Science"],
        skills=["Python", "SQL"],
        experience=["Built data pipelines with Python"],
        projects=["Analytics dashboard using SQL"],
        certifications=["Python Certificate"],
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


def _make_report(**kwargs) -> dict:
    defaults = dict(
        ats_score=82.0,
        strengths=["The resume demonstrates Python and SQL experience."],
        issues=["Tableau is not mentioned in the resume or profile."],
        matched_keywords=["Python", "SQL"],
        missing_keywords=["Tableau"],
        recommendations=["Add Tableau work only if you have relevant experience."],
    )
    defaults.update(kwargs)
    return defaults


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
    monkeypatch.setattr(ats_agent, "_get_gemini_client", lambda api_key: (client, types))
    return captured


def test_analyze_resume_ats_returns_validated_report(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_report()))

    result = ats_agent.analyze_resume_ats(
        _make_candidate(), _make_opportunity(), "Python and SQL data analyst resume"
    )

    assert isinstance(result, ATSReport)
    assert result.ats_score == 82.0
    assert result.matched_keywords == ["Python", "SQL"]
    assert captured["config"] == {
        "response_mime_type": "application/json",
        "temperature": 0,
        "max_output_tokens": ats_agent.MAX_OUTPUT_TOKENS,
    }


@pytest.mark.parametrize("score", [0, 100])
def test_score_boundaries_are_accepted(monkeypatch, score):
    _mock_gemini(monkeypatch, json.dumps(_make_report(ats_score=score)))

    result = ats_agent.analyze_resume_ats(
        _make_candidate(), _make_opportunity(), "Python and SQL"
    )

    assert result.ats_score == score


@pytest.mark.parametrize("score", [-1, 100.1])
def test_invalid_score_is_rejected_without_clamping(monkeypatch, score):
    _mock_gemini(monkeypatch, json.dumps(_make_report(ats_score=score)))

    with pytest.raises(ats_agent.ATSValidationError, match="0–100"):
        ats_agent.analyze_resume_ats(_make_candidate(), _make_opportunity(), "Python")


def test_malformed_json_raises_response_error(monkeypatch):
    _mock_gemini(monkeypatch, "not json")

    with pytest.raises(ats_agent.ATSResponseError, match="valid JSON"):
        ats_agent.analyze_resume_ats(_make_candidate(), _make_opportunity(), "Python")


def test_invalid_report_structure_raises_validation_error(monkeypatch):
    report = _make_report()
    del report["recommendations"]
    _mock_gemini(monkeypatch, json.dumps(report))

    with pytest.raises(ats_agent.ATSValidationError, match="missing required fields"):
        ats_agent.analyze_resume_ats(_make_candidate(), _make_opportunity(), "Python")


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ats_agent.ATSConfigurationError, match="GEMINI_API_KEY"):
        ats_agent.analyze_resume_ats(_make_candidate(), _make_opportunity(), "Python")


def test_api_failure_raises_controlled_error(monkeypatch):
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("credential test-key failed")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        ats_agent,
        "_get_gemini_client",
        lambda api_key: (SimpleNamespace(models=BrokenModels()), SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)),
    )

    with pytest.raises(ats_agent.ATSAPIError, match="ATS analysis") as exc_info:
        ats_agent.analyze_resume_ats(_make_candidate(), _make_opportunity(), "Python")

    assert "test-key" not in str(exc_info.value)


def test_empty_resume_is_rejected_before_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="empty"):
        ats_agent.analyze_resume_ats(_make_candidate(), _make_opportunity(), "  ")


def test_invalid_candidate_and_opportunity_inputs_are_rejected():
    with pytest.raises(TypeError, match="CandidateProfile"):
        ats_agent.analyze_resume_ats({}, _make_opportunity(), "Python")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Opportunity"):
        ats_agent.analyze_resume_ats(_make_candidate(), {}, "Python")  # type: ignore[arg-type]


def test_prompt_contains_all_sources_and_requires_grounding(monkeypatch):
    captured = _mock_gemini(monkeypatch, json.dumps(_make_report()))
    candidate = _make_candidate(projects=["Fraud detector"])
    opportunity = _make_opportunity(title="Fraud Data Analyst")

    ats_agent.analyze_resume_ats(candidate, opportunity, "Jane built a fraud detector in Python.")

    prompt = captured["request"]["contents"]
    assert "Fraud detector" in prompt
    assert "Fraud Data Analyst" in prompt
    assert "Jane built a fraud detector" in prompt
    assert "web\nsearch" in prompt


def test_exactly_one_mocked_gemini_call(monkeypatch):
    calls = {"count": 0}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class CountingModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(text=json.dumps(_make_report()))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        ats_agent,
        "_get_gemini_client",
        lambda api_key: (SimpleNamespace(models=CountingModels()), SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)),
    )

    ats_agent.analyze_resume_ats(_make_candidate(), _make_opportunity(), "Python and SQL")

    assert calls["count"] == 1
