"""Unit tests for AP-003 Eligibility & Matching Engine."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents import matching_agent
from models import CandidateProfile, MatchResult, Opportunity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(**kwargs) -> CandidateProfile:
    defaults = dict(
        name="Jane Doe",
        location="Pakistan",
        skills=["Python", "Machine Learning"],
        education=["BS Computer Science"],
        experience=["Software intern at XYZ"],
        interests=["AI"],
        years_of_experience=1.0,
    )
    defaults.update(kwargs)
    return CandidateProfile(**defaults)


def _make_opportunity(**kwargs) -> Opportunity:
    defaults = dict(
        id="opp_001",
        title="AI/ML Intern",
        organization="Example Corp",
        type="internship",
        description="An AI/ML internship for CS students.",
        requirements=["Python", "Machine Learning"],
        location="Remote",
        url="https://example.com/jobs/ai-intern",
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


def _make_raw_result(**kwargs) -> dict:
    base = dict(
        opportunity_id="opp_001",
        match_score=85.0,
        eligible=True,
        strengths=["Python matches required skill"],
        gaps=[],
        reasons=["Strong skills match"],
    )
    base.update(kwargs)
    return base


def _mock_gemini(monkeypatch, json_response: str):
    """Patch _get_gemini_client to return a fake single-call client."""
    captured = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(text=json_response)

    client = SimpleNamespace(models=FakeModels())
    types = SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        matching_agent, "_get_gemini_client", lambda api_key: (client, types)
    )
    return captured


# ---------------------------------------------------------------------------
# 1. Valid candidate + multiple opportunities returns list[MatchResult]
# ---------------------------------------------------------------------------


def test_evaluate_returns_match_results(monkeypatch):
    opps = [
        _make_opportunity(id="opp_001"),
        _make_opportunity(id="opp_002", title="Data Analyst"),
    ]
    raw = [
        _make_raw_result(opportunity_id="opp_001", match_score=85.0),
        _make_raw_result(opportunity_id="opp_002", match_score=70.0),
    ]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), opps)

    assert len(results) == 2
    assert all(isinstance(r, MatchResult) for r in results)


# ---------------------------------------------------------------------------
# 2. Output is list[MatchResult]
# ---------------------------------------------------------------------------


def test_output_type_is_list_of_match_result(monkeypatch):
    opps = [_make_opportunity()]
    raw = [_make_raw_result()]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), opps)

    assert isinstance(results, list)
    assert isinstance(results[0], MatchResult)


# ---------------------------------------------------------------------------
# 3. One Gemini call is made for multiple opportunities
# ---------------------------------------------------------------------------


def test_single_gemini_call_for_multiple_opportunities(monkeypatch):
    opps = [
        _make_opportunity(id="opp_001"),
        _make_opportunity(id="opp_002", title="Job 2"),
        _make_opportunity(id="opp_003", title="Job 3"),
    ]
    raw = [
        _make_raw_result(opportunity_id="opp_001"),
        _make_raw_result(opportunity_id="opp_002"),
        _make_raw_result(opportunity_id="opp_003"),
    ]
    call_count = {"n": 0}

    class CountingModels:
        def generate_content(self, **kwargs):
            call_count["n"] += 1
            return SimpleNamespace(text=json.dumps(raw))

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): pass

    client = SimpleNamespace(models=CountingModels())
    types = SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        matching_agent, "_get_gemini_client", lambda api_key: (client, types)
    )

    matching_agent.evaluate_opportunities(_make_candidate(), opps)

    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# 4. Empty opportunity list returns [] without calling Gemini
# ---------------------------------------------------------------------------


def test_empty_opportunities_returns_empty_list(monkeypatch):
    call_count = {"n": 0}

    class ShouldNotBeCalled:
        def generate_content(self, **kwargs):
            call_count["n"] += 1
            return SimpleNamespace(text="[]")

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): pass

    client = SimpleNamespace(models=ShouldNotBeCalled())
    types = SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        matching_agent, "_get_gemini_client", lambda api_key: (client, types)
    )

    results = matching_agent.evaluate_opportunities(_make_candidate(), [])

    assert results == []
    assert call_count["n"] == 0


# ---------------------------------------------------------------------------
# 5. Candidate input validation
# ---------------------------------------------------------------------------


def test_non_candidate_profile_raises_type_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(TypeError, match="CandidateProfile"):
        matching_agent.evaluate_opportunities({"name": "Jane"}, [])  # type: ignore


# ---------------------------------------------------------------------------
# 6. Opportunity input validation
# ---------------------------------------------------------------------------


def test_non_list_opportunities_raises_type_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(TypeError, match="list"):
        matching_agent.evaluate_opportunities(_make_candidate(), "not a list")  # type: ignore


def test_list_with_non_opportunity_raises_type_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with pytest.raises(TypeError, match="Opportunity"):
        matching_agent.evaluate_opportunities(
            _make_candidate(), [{"id": "x"}]  # type: ignore
        )


# ---------------------------------------------------------------------------
# 7. opportunity_id correctly maps to input Opportunity.id
# ---------------------------------------------------------------------------


def test_opportunity_id_matches_input(monkeypatch):
    opp = _make_opportunity(id="exact_id_abc")
    raw = [_make_raw_result(opportunity_id="exact_id_abc")]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), [opp])

    assert results[0].opportunity_id == "exact_id_abc"


# ---------------------------------------------------------------------------
# 8. match_score within 0–100 is accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", [0.0, 1.0, 50.0, 99.9, 100.0])
def test_valid_match_score_accepted(monkeypatch, score):
    opp = _make_opportunity()
    raw = [_make_raw_result(match_score=score)]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), [opp])

    assert results[0].match_score == score


# ---------------------------------------------------------------------------
# 9. match_score outside 0–100 is rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_score", [-1.0, -0.1, 100.1, 200.0])
def test_invalid_match_score_raises_validation_error(monkeypatch, bad_score):
    opp = _make_opportunity()
    raw = [_make_raw_result(match_score=bad_score)]
    _mock_gemini(monkeypatch, json.dumps(raw))

    with pytest.raises(matching_agent.MatchingValidationError, match="0–100"):
        matching_agent.evaluate_opportunities(_make_candidate(), [opp])


# ---------------------------------------------------------------------------
# 10. Unknown opportunity_id is rejected
# ---------------------------------------------------------------------------


def test_unknown_opportunity_id_raises_validation_error(monkeypatch):
    opp = _make_opportunity(id="real_id")
    raw = [_make_raw_result(opportunity_id="invented_id")]
    _mock_gemini(monkeypatch, json.dumps(raw))

    with pytest.raises(matching_agent.MatchingValidationError, match="unknown opportunity_id"):
        matching_agent.evaluate_opportunities(_make_candidate(), [opp])


# ---------------------------------------------------------------------------
# 11. eligible=True when hard requirements are satisfied
# ---------------------------------------------------------------------------


def test_eligible_true_when_requirements_satisfied(monkeypatch):
    opp = _make_opportunity(requirements=["Python"])
    raw = [_make_raw_result(eligible=True, strengths=["Python matches required skill"])]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), [opp])

    assert results[0].eligible is True


# ---------------------------------------------------------------------------
# 12. eligible=False when a hard requirement is violated
# ---------------------------------------------------------------------------


def test_eligible_false_when_hard_requirement_violated(monkeypatch):
    opp = _make_opportunity(requirements=["5 years experience required"])
    raw = [_make_raw_result(
        eligible=False,
        gaps=["Opportunity requires 5 years experience; candidate has 1 year"],
        reasons=["Hard requirement not met"],
    )]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), [opp])

    assert results[0].eligible is False
    assert len(results[0].gaps) > 0


# ---------------------------------------------------------------------------
# 13. Missing optional information handled safely
# ---------------------------------------------------------------------------


def test_empty_strengths_gaps_reasons_accepted(monkeypatch):
    opp = _make_opportunity()
    raw = [_make_raw_result(strengths=[], gaps=[], reasons=[])]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), [opp])

    assert results[0].strengths == []
    assert results[0].gaps == []
    assert results[0].reasons == []


def test_candidate_with_no_skills_handled(monkeypatch):
    candidate = _make_candidate(skills=[], experience=[], education=[])
    opp = _make_opportunity()
    raw = [_make_raw_result(
        match_score=10.0,
        eligible=False,
        gaps=["No skills listed in candidate profile"],
    )]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(candidate, [opp])

    assert results[0].match_score == 10.0
    assert results[0].eligible is False


def test_opportunity_with_no_requirements_handled(monkeypatch):
    opp = _make_opportunity(requirements=[])
    raw = [_make_raw_result(
        eligible=True,
        reasons=["No disqualifying hard requirement identified"],
    )]
    _mock_gemini(monkeypatch, json.dumps(raw))

    results = matching_agent.evaluate_opportunities(_make_candidate(), [opp])

    assert results[0].eligible is True


# ---------------------------------------------------------------------------
# 14. Malformed JSON handled
# ---------------------------------------------------------------------------


def test_invalid_json_raises_response_error(monkeypatch):
    _mock_gemini(monkeypatch, "this is not json")

    with pytest.raises(matching_agent.MatchingResponseError, match="not valid JSON"):
        matching_agent.evaluate_opportunities(_make_candidate(), [_make_opportunity()])


def test_non_array_json_raises_response_error(monkeypatch):
    _mock_gemini(monkeypatch, json.dumps({"key": "value"}))

    with pytest.raises(matching_agent.MatchingResponseError, match="not a JSON array"):
        matching_agent.evaluate_opportunities(_make_candidate(), [_make_opportunity()])


def test_empty_response_raises_response_error(monkeypatch):
    _mock_gemini(monkeypatch, "")

    with pytest.raises(matching_agent.MatchingResponseError, match="empty response"):
        matching_agent.evaluate_opportunities(_make_candidate(), [_make_opportunity()])


def test_non_dict_item_in_array_raises_validation_error(monkeypatch):
    _mock_gemini(monkeypatch, json.dumps(["not a dict"]))

    with pytest.raises(matching_agent.MatchingValidationError):
        matching_agent.evaluate_opportunities(_make_candidate(), [_make_opportunity()])


# ---------------------------------------------------------------------------
# 15. Gemini / API failure handled
# ---------------------------------------------------------------------------


def test_api_failure_raises_matching_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("connection timeout")

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs): pass

    client = SimpleNamespace(models=BrokenModels())
    types = SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)
    monkeypatch.setattr(
        matching_agent, "_get_gemini_client", lambda api_key: (client, types)
    )

    with pytest.raises(matching_agent.MatchingAPIError, match="connection timeout"):
        matching_agent.evaluate_opportunities(_make_candidate(), [_make_opportunity()])


def test_missing_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(matching_agent.MatchingConfigurationError, match="GEMINI_API_KEY"):
        matching_agent.evaluate_opportunities(_make_candidate(), [_make_opportunity()])


# ---------------------------------------------------------------------------
# 16. No real Gemini API calls — config verification
# ---------------------------------------------------------------------------


def test_gemini_config_uses_json_mime_type_and_bounded_tokens(monkeypatch):
    opp = _make_opportunity()
    raw = [_make_raw_result()]
    captured = _mock_gemini(monkeypatch, json.dumps(raw))

    matching_agent.evaluate_opportunities(_make_candidate(), [opp])

    assert captured["config"]["response_mime_type"] == "application/json"
    assert captured["config"]["temperature"] == 0
    assert captured["config"]["max_output_tokens"] == 8192


def test_gemini_not_called_when_opportunities_empty(monkeypatch):
    """Confirm _get_gemini_client is never reached for empty input."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # No API key set — if Gemini were called, MatchingConfigurationError would raise.
    # Empty list must return [] before reaching the API key check.
    results = matching_agent.evaluate_opportunities(_make_candidate(), [])

    assert results == []


# ---------------------------------------------------------------------------
# Additional: prompt contains all opportunity IDs
# ---------------------------------------------------------------------------


def test_prompt_contains_all_opportunity_ids(monkeypatch):
    opps = [
        _make_opportunity(id="alpha_id"),
        _make_opportunity(id="beta_id", title="Beta Job"),
    ]
    raw = [
        _make_raw_result(opportunity_id="alpha_id"),
        _make_raw_result(opportunity_id="beta_id"),
    ]
    captured = _mock_gemini(monkeypatch, json.dumps(raw))

    matching_agent.evaluate_opportunities(_make_candidate(), opps)

    prompt_contents = captured["request"]["contents"]
    assert "alpha_id" in prompt_contents
    assert "beta_id" in prompt_contents
