"""Unit tests for AP-002 Opportunity Discovery Engine."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from agents import discovery_agent
from models import CandidateProfile, Opportunity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(**kwargs) -> CandidateProfile:
    defaults = dict(
        name="Jane Doe",
        location="Pakistan",
        skills=["Python", "Machine Learning"],
        education=["BS Computer Science"],
        interests=["AI", "open source"],
        experience=["Software intern at XYZ"],
    )
    defaults.update(kwargs)
    return CandidateProfile(**defaults)


def _make_raw_opportunity(**kwargs) -> dict:
    base = dict(
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
    base.update(kwargs)
    return base


def _stable_id(url: str) -> str:
    normalised = url.strip().lower().rstrip("/")
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def _mock_gemini_two_stage(
    monkeypatch,
    research_text: str,
    structured_json: str,
):
    """Patch _get_gemini_client to simulate the two-stage Gemini flow.

    Call 1 (grounded research) → returns research_text via response.text
    Call 2 (structured extraction) → returns structured_json via response.text
    """
    captured = {"calls": [], "configs": []}
    call_responses = [research_text, structured_json]

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured["configs"].append(kwargs)

    class FakeTool:
        def __init__(self, **kwargs):
            pass

    class FakeGoogleSearch:
        pass

    class FakeModels:
        def generate_content(self, **kwargs):
            idx = len(captured["calls"])
            captured["calls"].append(kwargs)
            text = call_responses[idx] if idx < len(call_responses) else ""
            return SimpleNamespace(text=text)

    client = SimpleNamespace(models=FakeModels())
    types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        discovery_agent, "_get_gemini_client", lambda api_key: (client, types)
    )
    return captured


# ---------------------------------------------------------------------------
# 1. Valid CandidateProfile returns Opportunity objects
# ---------------------------------------------------------------------------


def test_discover_returns_opportunity_objects(monkeypatch):
    raw = [_make_raw_opportunity()]
    _mock_gemini_two_stage(monkeypatch, "Research prose text.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert len(results) == 1
    assert isinstance(results[0], Opportunity)


# ---------------------------------------------------------------------------
# 2. Output is list[Opportunity]
# ---------------------------------------------------------------------------


def test_discover_returns_list(monkeypatch):
    raw = [
        _make_raw_opportunity(),
        _make_raw_opportunity(url="https://example.com/job2", title="Job 2"),
    ]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert isinstance(results, list)
    assert all(isinstance(r, Opportunity) for r in results)


# ---------------------------------------------------------------------------
# 3. Invalid / missing URL is filtered out
# ---------------------------------------------------------------------------


def test_invalid_url_is_filtered(monkeypatch):
    raw = [
        _make_raw_opportunity(url="not-a-url"),
        _make_raw_opportunity(url=""),
        _make_raw_opportunity(url="ftp://bad-scheme.com/job"),
        _make_raw_opportunity(url="https://example.com/valid"),
    ]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert len(results) == 1
    assert results[0].url == "https://example.com/valid"


# ---------------------------------------------------------------------------
# 4. Duplicate URLs are removed
# ---------------------------------------------------------------------------


def test_duplicate_urls_removed(monkeypatch):
    url = "https://example.com/jobs/ai-intern"
    raw = [
        _make_raw_opportunity(url=url),
        _make_raw_opportunity(url=url, title="Duplicate"),
    ]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert len(results) == 1


def test_duplicate_urls_trailing_slash_normalised(monkeypatch):
    raw = [
        _make_raw_opportunity(url="https://example.com/job/"),
        _make_raw_opportunity(url="https://example.com/job", title="Same without slash"),
    ]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert len(results) == 1


# ---------------------------------------------------------------------------
# 5. Opportunity IDs are deterministic
# ---------------------------------------------------------------------------


def test_opportunity_id_is_deterministic(monkeypatch):
    url = "https://example.com/jobs/ai-intern"
    raw = [_make_raw_opportunity(url=url)]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert results[0].id == _stable_id(url)


def test_same_url_same_id_across_calls(monkeypatch):
    url = "https://example.com/jobs/ai-intern"
    raw = [_make_raw_opportunity(url=url)]

    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))
    first = discovery_agent.discover_opportunities(_make_candidate())

    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))
    second = discovery_agent.discover_opportunities(_make_candidate())

    assert first[0].id == second[0].id


# ---------------------------------------------------------------------------
# 6. Opportunity type is normalized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw_type,expected", [
    ("internship", "internship"),
    ("Internship", "internship"),
    ("full-time", "job"),
    ("Full Time Employment", "job"),
    ("scholarship", "scholarship"),
    ("grant", "scholarship"),
    ("hackathon", "hackathon"),
    ("competition", "hackathon"),
    ("fellowship", "fellowship"),
    ("graduate", "fellowship"),
    ("co-op", "internship"),
    ("unknown_type", "job"),
])
def test_type_normalization(raw_type, expected):
    assert discovery_agent._normalize_type(raw_type) == expected


def test_normalized_type_in_returned_opportunity(monkeypatch):
    raw = [_make_raw_opportunity(type="Full Time Employment")]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert results[0].type == "job"


# ---------------------------------------------------------------------------
# 7. Missing optional fields are handled correctly
# ---------------------------------------------------------------------------


def test_optional_fields_can_be_none(monkeypatch):
    raw = [_make_raw_opportunity(deadline=None, location=None, source=None)]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert results[0].deadline is None
    assert results[0].location is None
    assert results[0].source is None


def test_empty_requirements_list_accepted(monkeypatch):
    raw = [_make_raw_opportunity(requirements=[])]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert results[0].requirements == []


# ---------------------------------------------------------------------------
# 8. API / search failure is handled
# ---------------------------------------------------------------------------


def test_api_failure_raises_discovery_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("network error")

    client = SimpleNamespace(models=BrokenModels())

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): pass

    class FakeTool:
        def __init__(self, **kwargs): pass

    class FakeGoogleSearch:
        pass

    types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
    )
    monkeypatch.setattr(
        discovery_agent, "_get_gemini_client", lambda api_key: (client, types)
    )

    with pytest.raises(discovery_agent.DiscoveryAPIError, match="network error"):
        discovery_agent.discover_opportunities(_make_candidate())


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(discovery_agent.DiscoveryConfigurationError, match="GEMINI_API_KEY"):
        discovery_agent.discover_opportunities(_make_candidate())


def test_invalid_json_response_raises_response_error(monkeypatch):
    _mock_gemini_two_stage(monkeypatch, "Research prose.", "this is not json")

    with pytest.raises(discovery_agent.DiscoveryResponseError, match="not valid JSON"):
        discovery_agent.discover_opportunities(_make_candidate())


def test_non_array_json_raises_response_error(monkeypatch):
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps({"key": "value"}))

    with pytest.raises(discovery_agent.DiscoveryResponseError, match="not a JSON array"):
        discovery_agent.discover_opportunities(_make_candidate())


def test_empty_stage2_response_raises_response_error(monkeypatch):
    _mock_gemini_two_stage(monkeypatch, "Research prose.", "")

    with pytest.raises(discovery_agent.DiscoveryResponseError, match="empty response"):
        discovery_agent.discover_opportunities(_make_candidate())


# ---------------------------------------------------------------------------
# 9. Zero relevant results returns empty list safely
# ---------------------------------------------------------------------------


def test_all_invalid_urls_returns_empty_list(monkeypatch):
    raw = [_make_raw_opportunity(url="bad-url"), _make_raw_opportunity(url="")]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert results == []


def test_empty_array_response_returns_empty_list(monkeypatch):
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps([]))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert results == []


# ---------------------------------------------------------------------------
# 10. max_results is respected
# ---------------------------------------------------------------------------


def test_max_results_is_respected(monkeypatch):
    raw = [
        _make_raw_opportunity(url=f"https://example.com/job/{i}", title=f"Job {i}")
        for i in range(8)
    ]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate(), max_results=3)

    assert len(results) <= 3


# ---------------------------------------------------------------------------
# 11. Non-CandidateProfile input raises TypeError
# ---------------------------------------------------------------------------


def test_non_profile_input_raises_type_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(TypeError, match="CandidateProfile"):
        discovery_agent.discover_opportunities({"name": "Jane"})  # type: ignore


# ---------------------------------------------------------------------------
# 12. Malformed items in array are skipped, valid ones returned
# ---------------------------------------------------------------------------


def test_malformed_items_skipped(monkeypatch):
    raw = [
        {"not": "an opportunity"},
        _make_raw_opportunity(url="https://example.com/valid"),
    ]
    _mock_gemini_two_stage(monkeypatch, "Research prose.", json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert len(results) == 1
    assert results[0].url == "https://example.com/valid"


# ---------------------------------------------------------------------------
# 13. _build_search_queries uses profile fields
# ---------------------------------------------------------------------------


def test_build_search_queries_uses_skills():
    candidate = _make_candidate(skills=["Python", "ML"], location="Pakistan")
    queries = discovery_agent._build_search_queries(candidate)

    assert any("Python" in q for q in queries)
    assert any("Pakistan" in q for q in queries)


def test_build_search_queries_fallback_when_no_skills():
    candidate = _make_candidate(skills=[], interests=[], education=[])
    queries = discovery_agent._build_search_queries(candidate)

    assert len(queries) >= 1
    assert all(isinstance(q, str) for q in queries)


# ---------------------------------------------------------------------------
# 14. Two-stage flow — grounded research call is made first
# ---------------------------------------------------------------------------


def test_two_stage_research_call_is_made_first(monkeypatch):
    raw = [_make_raw_opportunity()]
    captured = _mock_gemini_two_stage(
        monkeypatch, "Research prose with citations [1].", json.dumps(raw)
    )

    discovery_agent.discover_opportunities(_make_candidate())

    assert len(captured["calls"]) == 2


def test_stage1_uses_google_search_tool(monkeypatch):
    """Stage 1 config must include a Tool (google_search), not response_mime_type."""
    raw = [_make_raw_opportunity()]
    captured = _mock_gemini_two_stage(
        monkeypatch, "Research prose.", json.dumps(raw)
    )

    discovery_agent.discover_opportunities(_make_candidate())

    stage1_config = captured["configs"][0]
    assert "tools" in stage1_config
    assert "response_mime_type" not in stage1_config


def test_stage2_uses_json_mime_type(monkeypatch):
    """Stage 2 config must use response_mime_type=application/json, no tools."""
    raw = [_make_raw_opportunity()]
    captured = _mock_gemini_two_stage(
        monkeypatch, "Research prose.", json.dumps(raw)
    )

    discovery_agent.discover_opportunities(_make_candidate())

    stage2_config = captured["configs"][1]
    assert stage2_config.get("response_mime_type") == "application/json"
    assert "tools" not in stage2_config


def test_stage1_has_max_output_tokens(monkeypatch):
    raw = [_make_raw_opportunity()]
    captured = _mock_gemini_two_stage(
        monkeypatch, "Research prose.", json.dumps(raw)
    )

    discovery_agent.discover_opportunities(_make_candidate())

    assert captured["configs"][0].get("max_output_tokens") == 4096


def test_stage2_has_max_output_tokens(monkeypatch):
    raw = [_make_raw_opportunity()]
    captured = _mock_gemini_two_stage(
        monkeypatch, "Research prose.", json.dumps(raw)
    )

    discovery_agent.discover_opportunities(_make_candidate())

    assert captured["configs"][1].get("max_output_tokens") == 8192


def test_stage1_prose_with_citations_does_not_crash(monkeypatch):
    """Stage 1 may return prose with citation markers — must not be parsed as JSON."""
    prose = (
        "Found several opportunities. Google internship [1] at "
        "https://careers.google.com/jobs/intern. "
        "MLH Fellowship [2] at https://fellowship.mlh.io. "
        "Sources: [1] careers.google.com [2] fellowship.mlh.io"
    )
    raw = [
        _make_raw_opportunity(
            url="https://careers.google.com/jobs/intern",
            title="Google Intern",
            organization="Google",
        )
    ]
    _mock_gemini_two_stage(monkeypatch, prose, json.dumps(raw))

    results = discovery_agent.discover_opportunities(_make_candidate())

    assert len(results) == 1
    assert results[0].organization == "Google"


def test_stage1_failure_raises_discovery_api_error(monkeypatch):
    """If Stage 1 (grounded research) fails, DiscoveryAPIError is raised."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    call_count = {"n": 0}

    class FailOnFirstCall:
        def generate_content(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("search quota exceeded")
            return SimpleNamespace(text=json.dumps([]))

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): pass

    class FakeTool:
        def __init__(self, **kwargs): pass

    class FakeGoogleSearch:
        pass

    client = SimpleNamespace(models=FailOnFirstCall())
    types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
    )
    monkeypatch.setattr(
        discovery_agent, "_get_gemini_client", lambda api_key: (client, types)
    )

    with pytest.raises(discovery_agent.DiscoveryAPIError, match="search quota exceeded"):
        discovery_agent.discover_opportunities(_make_candidate())


def test_stage2_failure_raises_discovery_api_error(monkeypatch):
    """If Stage 2 (structured extraction) fails, DiscoveryAPIError is raised."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    call_count = {"n": 0}

    class FailOnSecondCall:
        def generate_content(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return SimpleNamespace(text="Research prose.")
            raise RuntimeError("extraction failed")

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): pass

    class FakeTool:
        def __init__(self, **kwargs): pass

    class FakeGoogleSearch:
        pass

    client = SimpleNamespace(models=FailOnSecondCall())
    types = SimpleNamespace(
        GenerateContentConfig=FakeGenerateContentConfig,
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
    )
    monkeypatch.setattr(
        discovery_agent, "_get_gemini_client", lambda api_key: (client, types)
    )

    with pytest.raises(discovery_agent.DiscoveryAPIError, match="extraction failed"):
        discovery_agent.discover_opportunities(_make_candidate())


def test_stage1_empty_response_raises_response_error(monkeypatch):
    """Empty Stage 1 response raises DiscoveryResponseError before Stage 2 runs."""
    _mock_gemini_two_stage(monkeypatch, "", json.dumps([]))

    with pytest.raises(discovery_agent.DiscoveryResponseError, match="empty response"):
        discovery_agent.discover_opportunities(_make_candidate())
