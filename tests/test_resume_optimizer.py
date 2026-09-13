import json
from types import SimpleNamespace

import pytest

from agents import resume_optimizer
from models.ats import ATSReport
from models.opportunity import Opportunity
from models.profile import CandidateProfile
from models.resume_optimization import ResumeOptimizationResult


def _make_candidate() -> CandidateProfile:
    return CandidateProfile(
        name="Test User",
        email="test@example.com",
        skills=["Python", "Machine Learning"],
        education=["BS Computer Science"],
        experience=["AI/ML Intern"],
        projects=["Resume Screening Assistant"],
    )


def _make_opportunity() -> Opportunity:
    return Opportunity(
        id="opp_001",
        title="AI/ML Intern",
        organization="Example Company",
        type="internship",
        description="Build machine learning applications.",
        requirements=["Python", "Machine Learning"],
        url="https://example.com/opportunity",
    )


def _make_ats_report() -> ATSReport:
    return ATSReport(
        ats_score=78.0,
        strengths=["Strong Python background"],
        issues=["Missing explicit ML project keywords"],
        matched_keywords=["Python", "Machine Learning"],
        missing_keywords=["TensorFlow"],
        recommendations=["Highlight machine learning project work"],
    )


def _make_resume_text() -> str:
    return """
    Test User
    BS Computer Science

    Skills:
    Python, Machine Learning

    Experience:
    AI/ML Intern

    Projects:
    Resume Screening Assistant
    """


def _make_gemini_response(payload: dict):
    return SimpleNamespace(text=json.dumps(payload))


def _make_result_payload() -> dict:
    return {
        "optimized_resume": (
            "Test User\n\n"
            "BS Computer Science\n\n"
            "Skills: Python, Machine Learning\n\n"
            "Experience: AI/ML Intern\n\n"
            "Projects: Resume Screening Assistant"
        ),
        "changed_sections": ["Skills", "Projects"],
        "added_keywords": ["machine learning"],
        "removed_or_weak_content": [],
        "recommendations": ["Keep project details concise"],
    }


def test_optimize_resume_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return _make_gemini_response(_make_result_payload())

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(
        resume_optimizer,
        "_get_gemini_client",
        lambda api_key: (
            FakeClient(),
            SimpleNamespace(
                GenerateContentConfig=lambda **kwargs: kwargs
            ),
        ),
    )

    result = resume_optimizer.optimize_resume(
        _make_candidate(),
        _make_opportunity(),
        _make_resume_text(),
        _make_ats_report(),
    )

    assert isinstance(result, ResumeOptimizationResult)
    assert result.optimized_resume
    assert result.changed_sections == ["Skills", "Projects"]
    assert len(calls) == 1


def test_optimize_resume_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(resume_optimizer.ResumeOptimizerConfigurationError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            _make_resume_text(),
            _make_ats_report(),
        )


def test_optimize_resume_empty_resume():
    with pytest.raises(resume_optimizer.ResumeOptimizerResponseError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            "",
            _make_ats_report(),
        )


def test_optimize_resume_whitespace_resume():
    with pytest.raises(resume_optimizer.ResumeOptimizerResponseError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            "   ",
            _make_ats_report(),
        )


def test_optimize_resume_invalid_candidate():
    with pytest.raises(TypeError):
        resume_optimizer.optimize_resume(
            "not-a-profile",
            _make_opportunity(),
            _make_resume_text(),
            _make_ats_report(),
        )


def test_optimize_resume_invalid_opportunity():
    with pytest.raises(TypeError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            "not-an-opportunity",
            _make_resume_text(),
            _make_ats_report(),
        )


def test_optimize_resume_invalid_ats_report():
    with pytest.raises(TypeError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            _make_resume_text(),
            "not-an-ats-report",
        )


def test_optimize_resume_api_failure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def failing_client(_api_key):
        raise RuntimeError("provider failure")

    monkeypatch.setattr(
        resume_optimizer,
        "_get_gemini_client",
        failing_client,
    )

    with pytest.raises(resume_optimizer.ResumeOptimizerAPIError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            _make_resume_text(),
            _make_ats_report(),
        )


def test_optimize_resume_empty_gemini_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(
        resume_optimizer,
        "_get_gemini_client",
        lambda api_key: (
            FakeClient(),
            SimpleNamespace(
                GenerateContentConfig=lambda **kwargs: kwargs
            ),
        ),
    )

    with pytest.raises(resume_optimizer.ResumeOptimizerResponseError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            _make_resume_text(),
            _make_ats_report(),
        )


def test_optimize_resume_malformed_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="{invalid-json")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(
        resume_optimizer,
        "_get_gemini_client",
        lambda api_key: (
            FakeClient(),
            SimpleNamespace(
                GenerateContentConfig=lambda **kwargs: kwargs
            ),
        ),
    )

    with pytest.raises(resume_optimizer.ResumeOptimizerResponseError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            _make_resume_text(),
            _make_ats_report(),
        )


def test_optimize_resume_invalid_output_structure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    invalid_payload = {
        "optimized_resume": "text"
    }

    class FakeModels:
        def generate_content(self, **kwargs):
            return _make_gemini_response(invalid_payload)

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(
        resume_optimizer,
        "_get_gemini_client",
        lambda api_key: (
            FakeClient(),
            SimpleNamespace(
                GenerateContentConfig=lambda **kwargs: kwargs
            ),
        ),
    )

    with pytest.raises(resume_optimizer.ResumeOptimizerResponseError):
        resume_optimizer.optimize_resume(
            _make_candidate(),
            _make_opportunity(),
            _make_resume_text(),
            _make_ats_report(),
        )


def test_optimize_resume_prompt_contains_required_sources(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return _make_gemini_response(_make_result_payload())

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(
        resume_optimizer,
        "_get_gemini_client",
        lambda api_key: (
            FakeClient(),
            SimpleNamespace(
                GenerateContentConfig=lambda **kwargs: kwargs
            ),
        ),
    )

    candidate = _make_candidate()
    opportunity = _make_opportunity()
    resume = _make_resume_text()
    ats_report = _make_ats_report()

    resume_optimizer.optimize_resume(
        candidate,
        opportunity,
        resume,
        ats_report,
    )

    prompt = calls[0]["contents"]

    assert candidate.name in prompt
    assert opportunity.title in prompt
    assert resume.strip() in prompt
    assert str(ats_report.ats_score) in prompt


def test_optimize_resume_uses_json_mode_and_temperature(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured_config = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured_config.update(kwargs["config"])
            return _make_gemini_response(_make_result_payload())

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(
        resume_optimizer,
        "_get_gemini_client",
        lambda api_key: (
            FakeClient(),
            SimpleNamespace(
                GenerateContentConfig=lambda **kwargs: kwargs
            ),
        ),
    )

    resume_optimizer.optimize_resume(
        _make_candidate(),
        _make_opportunity(),
        _make_resume_text(),
        _make_ats_report(),
    )

    assert captured_config["response_mime_type"] == "application/json"
    assert captured_config["temperature"] == 0
    assert captured_config["max_output_tokens"] == 8192