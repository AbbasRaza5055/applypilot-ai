"""Unit tests for AP-002 Opportunity Discovery Engine."""

from __future__ import annotations

import builtins
import hashlib
import json
from types import SimpleNamespace

import pytest

from agents import discovery_agent
from models import CandidateProfile, Opportunity


def _candidate(**overrides) -> CandidateProfile:
    value = dict(
        name="Jane Doe",
        location="Pakistan",
        skills=["Python", "Machine Learning"],
        education=["BS Computer Science"],
        interests=["AI", "open source"],
        experience=["Software intern at XYZ"],
    )
    value.update(overrides)
    return CandidateProfile(**value)


def _raw(**overrides) -> dict:
    value = dict(
        id="",
        title="AI/ML Intern",
        organization="Example Corp",
        type="internship",
        description="An AI/ML internship.",
        requirements=["Python"],
        deadline=None,
        location="Remote",
        url="https://example.com/jobs/ai-intern",
        source="example.com",
    )
    value.update(overrides)
    return value


def _stable_id(url: str) -> str:
    normalised = url.strip().lower().rstrip("/")
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def _mock_groq(monkeypatch, research="Research prose.", stage2=None):
    calls = []
    responses = [research, json.dumps([]) if stage2 is None else stage2]

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            text = responses[len(calls) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(discovery_agent, "_get_groq_client", lambda api_key: client)
    return calls


def test_valid_candidate_returns_list_of_opportunities(monkeypatch):
    calls = _mock_groq(monkeypatch, stage2=json.dumps([_raw()]))
    results = discovery_agent.discover_opportunities(_candidate())
    assert isinstance(results, list)
    assert isinstance(results[0], Opportunity)
    assert len(calls) == 2


def test_invalid_urls_and_duplicate_urls_are_filtered(monkeypatch):
    extracted = [
        _raw(url="not-a-url"),
        _raw(url="ftp://bad.example/job"),
        _raw(url="https://example.com/jobs/ai-intern/"),
        _raw(url="https://example.com/jobs/ai-intern", title="Duplicate"),
        _raw(url="https://example.com/valid"),
    ]
    _mock_groq(monkeypatch, stage2=json.dumps(extracted))
    results = discovery_agent.discover_opportunities(_candidate())
    assert [result.url for result in results] == [
        "https://example.com/jobs/ai-intern/", "https://example.com/valid"
    ]


@pytest.mark.parametrize("raw_type, expected", [
    ("internship", "internship"),
    ("Full Time Employment", "job"),
    ("grant", "scholarship"),
    ("competition", "hackathon"),
    ("graduate", "fellowship"),
    ("co-op", "internship"),
    ("unknown_type", "job"),
])
def test_type_normalization(raw_type, expected):
    assert discovery_agent._normalize_type(raw_type) == expected


def test_deterministic_id_type_normalization_and_optional_fields(monkeypatch):
    url = "https://example.com/jobs/ai-intern"
    _mock_groq(monkeypatch, stage2=json.dumps([_raw(
        url=url,
        type="Full Time Employment",
        deadline="2026-12-01",
        location=None,
        source=None,
        requirements=[],
    )]))
    result = discovery_agent.discover_opportunities(_candidate())[0]
    assert result.id == _stable_id(url)
    assert result.type == "job"
    assert result.deadline == "2026-12-01"
    assert result.location is None
    assert result.source is None
    assert result.requirements == []


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(discovery_agent.DiscoveryConfigurationError, match="GROQ_API_KEY"):
        discovery_agent.discover_opportunities(_candidate())


def test_missing_groq_sdk_raises_configuration_error(monkeypatch):
    original_import = builtins.__import__

    def missing_groq(name, *args, **kwargs):
        if name == "groq":
            raise ImportError("missing groq")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_groq)
    with pytest.raises(discovery_agent.DiscoveryConfigurationError, match="groq"):
        discovery_agent._get_groq_client("test-key")


def test_stage1_api_failure_raises_api_error(monkeypatch):
    class Broken:
        def create(self, **kwargs):
            raise RuntimeError("search quota exceeded")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Broken()))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(discovery_agent, "_get_groq_client", lambda key: client)
    with pytest.raises(discovery_agent.DiscoveryAPIError, match="search quota exceeded"):
        discovery_agent.discover_opportunities(_candidate())


def test_stage2_api_failure_raises_api_error(monkeypatch):
    calls = []

    class FailsSecond:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError("extraction failed")
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Research prose.")
            )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailsSecond()))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(discovery_agent, "_get_groq_client", lambda key: client)
    with pytest.raises(discovery_agent.DiscoveryAPIError, match="extraction failed"):
        discovery_agent.discover_opportunities(_candidate())


@pytest.mark.parametrize("stage1, stage2", [
    ("Research prose.", "not json"),
    ("Research prose.", json.dumps({"key": "value"})),
    ("", json.dumps([])),
    ("Research prose.", ""),
])
def test_malformed_non_array_or_empty_responses_raise_response_error(
    monkeypatch, stage1, stage2
):
    _mock_groq(monkeypatch, research=stage1, stage2=stage2)
    with pytest.raises(discovery_agent.DiscoveryResponseError):
        discovery_agent.discover_opportunities(_candidate())


def test_malformed_items_are_skipped_and_max_results_respected(monkeypatch):
    extracted = [{"not": "an opportunity"}] + [
        _raw(url=f"https://example.com/job/{index}", title=f"Job {index}")
        for index in range(5)
    ]
    _mock_groq(monkeypatch, stage2=json.dumps(extracted))
    results = discovery_agent.discover_opportunities(_candidate(), max_results=3)
    assert len(results) == 3


def test_search_queries_use_candidate_fields():
    queries = discovery_agent._build_search_queries(_candidate())
    assert any("Python" in query for query in queries)
    assert any("Pakistan" in query for query in queries)


def test_invalid_candidate_type_raises_type_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(TypeError, match="CandidateProfile"):
        discovery_agent.discover_opportunities({"name": "Jane"})  # type: ignore


def test_groq_request_contract_and_research_forwarding(monkeypatch):
    research = "Research prose with a real source."
    calls = _mock_groq(monkeypatch, research=research, stage2=json.dumps([_raw()]))
    monkeypatch.setenv("GROQ_MODEL", "custom/model")
    discovery_agent.discover_opportunities(_candidate())
    assert calls[0]["model"] == calls[1]["model"] == "custom/model"
    assert calls[0]["tools"] == [{"type": "browser_search"}]
    assert calls[0]["tool_choice"] == "required"
    assert "response_format" not in calls[0]
    assert "tools" not in calls[1]
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert research in calls[1]["messages"][0]["content"]
    assert calls[0]["max_completion_tokens"] == 4096
    assert calls[1]["max_completion_tokens"] == 8192


def test_default_model_is_openai_gpt_oss_120b(monkeypatch):
    calls = _mock_groq(monkeypatch)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    discovery_agent.discover_opportunities(_candidate())
    assert all(call["model"] == "openai/gpt-oss-120b" for call in calls)


def test_no_gemini_api_key_is_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _mock_groq(monkeypatch)
    assert discovery_agent.discover_opportunities(_candidate()) == []
