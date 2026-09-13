"""Unit tests for AP-002 Opportunity Discovery Engine."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

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
        description="Currently open AI/ML internship.",
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


def _mock_search(monkeypatch, payload=None):
    calls = []

    def search(query, api_url, api_key):
        calls.append((query, api_url, api_key))
        return {"organic": [] if payload is None else payload}

    monkeypatch.setenv("DISCOVERY_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(discovery_agent, "_search_provider", search)
    return calls


def test_valid_candidate_returns_list_of_opportunities(monkeypatch):
    calls = _mock_search(monkeypatch, [_raw()])
    results = discovery_agent.discover_opportunities(_candidate())
    assert isinstance(results, list)
    assert isinstance(results[0], Opportunity)
    assert len(calls) == len(discovery_agent._build_search_queries(_candidate()))


def test_invalid_urls_and_duplicate_urls_are_filtered(monkeypatch):
    extracted = [
        _raw(url="not-a-url"),
        _raw(url="ftp://bad.example/job"),
        _raw(url="https://www.google.com/search?q=internships"),
        _raw(url="https://example.com/search?q=internships"),
        _raw(url="https://example.com/jobs/ai-intern/"),
        _raw(url="https://example.com/jobs/ai-intern", title="Duplicate"),
        _raw(url="https://example.com/valid"),
    ]
    _mock_search(monkeypatch, extracted)
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
    extracted = [_raw(
        url=url,
        type="Full Time Employment",
        deadline=(date.today() + timedelta(days=30)).isoformat(),
        location=None,
        source=None,
        requirements=[],
    )]
    _mock_search(monkeypatch, extracted)
    result = discovery_agent.discover_opportunities(_candidate())[0]
    assert result.id == _stable_id(url)
    assert result.type == "job"
    assert result.deadline == (date.today() + timedelta(days=30)).isoformat()
    assert result.location is None
    assert result.source is None
    assert result.requirements == []


def test_missing_search_configuration_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("DISCOVERY_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(discovery_agent.DiscoveryConfigurationError, match="SEARCH_API_KEY"):
        discovery_agent.discover_opportunities(_candidate())


def test_search_provider_error_raises_api_error(monkeypatch):
    def broken_search(query, api_url, api_key):
        raise RuntimeError("search provider unavailable")

    monkeypatch.setenv("DISCOVERY_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(discovery_agent, "_search_provider", broken_search)
    with pytest.raises(discovery_agent.DiscoveryAPIError, match="unavailable"):
        discovery_agent.discover_opportunities(_candidate())


def test_malformed_search_results_raise_response_error(monkeypatch):
    monkeypatch.setenv("DISCOVERY_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(
        discovery_agent, "_search_provider", lambda query, api_url, api_key: {}
    )
    with pytest.raises(discovery_agent.DiscoveryResponseError):
        discovery_agent.discover_opportunities(_candidate())


def test_malformed_items_are_skipped_and_max_results_respected(monkeypatch):
    extracted = [{"not": "an opportunity"}] + [
        _raw(url=f"https://example.com/job/{index}", title=f"Job {index}")
        for index in range(5)
    ]
    _mock_search(monkeypatch, extracted)
    results = discovery_agent.discover_opportunities(_candidate(), max_results=3)
    assert len(results) == 3


@pytest.mark.parametrize("deadline, expected", [
    ((date.today() - timedelta(days=1)).isoformat(), False),
    (date.today().isoformat(), True),
    ((date.today() + timedelta(days=1)).isoformat(), True),
])
def test_deadline_freshness_filter(monkeypatch, deadline, expected):
    _mock_search(monkeypatch, [_raw(deadline=deadline)])
    results = discovery_agent.discover_opportunities(_candidate())
    assert bool(results) is expected


@pytest.mark.parametrize("deadline", [
    "2026-09-14T23:59:00Z",
    "14-09-2026",
    "09-14-2026",
    "14/09/2026",
    "09/14/2026",
])
def test_common_future_deadline_formats_are_accepted(monkeypatch, deadline):
    _mock_search(monkeypatch, [_raw(deadline=deadline)])
    assert discovery_agent.discover_opportunities(_candidate())


def test_expired_2026_deadline_is_rejected(monkeypatch):
    _mock_search(monkeypatch, [_raw(deadline="2026-01-01")])
    assert discovery_agent.discover_opportunities(_candidate()) == []


def test_unparseable_deadline_requires_explicit_open_status(monkeypatch):
    _mock_search(monkeypatch, [
        _raw(url="https://example.com/open", deadline="late September"),
        _raw(
            url="https://example.com/unknown",
            deadline="deadline to be announced",
            description="A listing without current status.",
        ),
    ])
    results = discovery_agent.discover_opportunities(_candidate())
    assert [result.url for result in results] == ["https://example.com/open"]


def test_ten_valid_candidates_can_be_returned(monkeypatch):
    extracted = [
        _raw(
            url=f"https://example.com/opportunity/{index}",
            title=f"Python Opportunity {index}",
            deadline=(date.today() + timedelta(days=index + 1)).isoformat(),
        )
        for index in range(10)
    ]
    _mock_search(monkeypatch, extracted)
    results = discovery_agent.discover_opportunities(_candidate(), max_results=10)
    assert len(results) == 10


def test_all_target_categories_are_searched(monkeypatch):
    calls = _mock_search(monkeypatch)
    with pytest.raises(discovery_agent.DiscoveryResponseError):
        discovery_agent.discover_opportunities(_candidate())
    queries = [call[0] for call in calls]
    assert len(queries) == 5
    assert any("jobs" in query for query in queries)
    assert any("internships" in query for query in queries)
    assert any("scholarships" in query for query in queries)
    assert any("hackathons" in query for query in queries)
    assert any("fellowships" in query for query in queries)


def test_repeated_discovery_calls_refresh_provider(monkeypatch):
    calls = _mock_search(monkeypatch, [_raw()])
    discovery_agent.discover_opportunities(_candidate())
    first_count = len(calls)
    discovery_agent.discover_opportunities(_candidate())
    assert len(calls) == first_count * 2


def test_missing_deadline_requires_explicitly_open_description(monkeypatch):
    _mock_search(monkeypatch, [
        _raw(url="https://example.com/open", deadline=None),
        _raw(
            url="https://example.com/unknown",
            deadline=None,
            description="An opportunity with no stated status.",
        ),
    ])
    results = discovery_agent.discover_opportunities(_candidate())
    assert [result.url for result in results] == ["https://example.com/open"]


def test_unparseable_deadline_current_status_is_accepted(monkeypatch):
    _mock_search(monkeypatch, [
        _raw(
            deadline="date announced on official page",
            description="Current opportunity with active applications.",
        )
    ])
    assert discovery_agent.discover_opportunities(_candidate())


def test_filtering_stage_counts_are_consistent(monkeypatch):
    payload = [
        _raw(url="https://example.com/valid"),
        _raw(url="https://example.com/valid/", title="Duplicate"),
        _raw(url="not-a-url"),
        _raw(
            url="https://example.com/expired",
            deadline=(date.today() - timedelta(days=1)).isoformat(),
        ),
        {"title": "Malformed result"},
    ]
    _mock_search(monkeypatch, payload)
    results = discovery_agent.discover_opportunities(_candidate(), max_results=10)
    counts = discovery_agent._last_filter_counts
    assert counts["raw_search_results"] >= counts["parsed_opportunity_candidates"]
    assert counts["invalid_url_candidates"] > 0
    assert counts["expired_candidates"] > 0
    assert counts["duplicate_candidates"] > 0
    assert counts["malformed_candidates"] > 0
    assert counts["final_valid_opportunities"] == len(results)
    assert all(isinstance(value, int) for value in counts.values())


def test_search_queries_use_candidate_fields():
    queries = discovery_agent._build_search_queries(_candidate())
    assert any("Python" in query for query in queries)
    assert any("Pakistan" in query for query in queries)


def test_invalid_candidate_type_raises_type_error(monkeypatch):
    monkeypatch.setenv("DISCOVERY_SEARCH_API_KEY", "test-key")
    with pytest.raises(TypeError, match="CandidateProfile"):
        discovery_agent.discover_opportunities({"name": "Jane"})  # type: ignore


def test_search_provider_receives_profile_queries_and_configuration(monkeypatch):
    calls = _mock_search(monkeypatch, [_raw()])
    monkeypatch.setenv("DISCOVERY_SEARCH_API_URL", "https://search.test/api")
    discovery_agent.discover_opportunities(_candidate())
    assert calls
    assert all(call[1:] == ("https://search.test/api", "test-key") for call in calls)
    assert any("Python" in call[0] for call in calls)


def test_no_gemini_api_key_is_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _mock_search(monkeypatch)
    with pytest.raises(discovery_agent.DiscoveryResponseError, match="no usable"):
        discovery_agent.discover_opportunities(_candidate())