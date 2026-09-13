"""Configurable web-search opportunity discovery from a CandidateProfile."""

from __future__ import annotations

import hashlib
import json
import os
import re
from calendar import monthrange
from datetime import date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from pydantic import ValidationError

from models import CandidateProfile, Opportunity

VALID_TYPES = {"job", "internship", "scholarship", "hackathon", "fellowship"}

_FILTER_COUNT_KEYS = (
    "raw_search_results",
    "parsed_opportunity_candidates",
    "invalid_url_candidates",
    "expired_candidates",
    "duplicate_candidates",
    "malformed_candidates",
    "final_valid_opportunities",
    "irrelevant_candidates",
    "unverified_freshness_candidates",
)
_last_filter_counts: dict[str, int] = {
    key: 0 for key in _FILTER_COUNT_KEYS
}

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
    """Raised when the search provider cannot be configured."""


class DiscoveryAPIError(DiscoveryAgentError):
    """Raised when the search provider call fails."""


class DiscoveryResponseError(DiscoveryAgentError):
    """Raised when the search provider returns an unusable response."""


class DiscoveryValidationError(DiscoveryAgentError):
    """Raised when a parsed opportunity violates the Opportunity schema."""


# ---------------------------------------------------------------------------
# Search provider
# ---------------------------------------------------------------------------


DEFAULT_SEARCH_API_URL = "https://google.serper.dev/search"


def _search_configuration() -> tuple[str, str]:
    api_url = os.getenv("DISCOVERY_SEARCH_API_URL", DEFAULT_SEARCH_API_URL)
    api_key = os.getenv("DISCOVERY_SEARCH_API_KEY") or os.getenv("SERPER_API_KEY")
    if not api_key:
        raise DiscoveryConfigurationError(
            "DISCOVERY_SEARCH_API_KEY or SERPER_API_KEY is not set."
        )
    return api_url, api_key


def _search_provider(query: str, api_url: str, api_key: str) -> Any:
    """Retrieve search hits through a Serper-compatible JSON search endpoint."""
    payload = json.dumps({"q": query, "num": 10}).encode("utf-8")
    request = Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise DiscoveryAPIError(
            f"Search provider request failed: {exc}"
        ) from exc


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
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False

        host = parsed.netloc.lower().split("@")[-1].split(":")[0]
        path = parsed.path.lower().rstrip("/")
        search_hosts = {
            "google.com", "www.google.com", "bing.com", "www.bing.com",
            "yahoo.com", "www.yahoo.com", "duckduckgo.com", "www.duckduckgo.com",
        }
        query_keys = set(parse_qs(parsed.query).keys())
        if host in search_hosts and (
            path == "/search" or path.startswith("/search/")
            or bool(query_keys & {"q", "query", "search"})
        ):
            return False
        if path == "/search" or path.startswith("/search/"):
            return False
        return True
    except Exception:
        return False


def _parse_deadline(value: Any) -> date | None:
    """Parse common deadline formats without rejecting unknown text."""
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass

    iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group())
        except ValueError:
            return None

    year_match = re.fullmatch(r"\d{4}", text)
    if year_match:
        return date(int(text), 12, 31)

    month_year_match = re.fullmatch(
        r"([A-Za-z]+)\s+(\d{4})", text
    )
    if month_year_match:
        for format_string in ("%B %Y", "%b %Y"):
            try:
                parsed = datetime.strptime(text, format_string)
                last_day = monthrange(parsed.year, parsed.month)[1]
                return date(parsed.year, parsed.month, last_day)
            except ValueError:
                continue

    for format_string in (
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
        "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(text, format_string).date()
        except ValueError:
            continue
    return None


def _is_explicitly_open(item: dict[str, Any]) -> bool:
    """Return whether the model explicitly describes an opportunity as open."""
    text_parts = [
        item.get("title"),
        item.get("description"),
        item.get("status"),
        item.get("source_status"),
    ]
    requirements = item.get("requirements")
    if isinstance(requirements, list):
        text_parts.extend(requirements)
    text = " ".join(
        part for part in text_parts if isinstance(part, str)
    ).lower()
    return any(phrase in text for phrase in (
        "currently open", "open now", "currently active", "active now",
        "current opportunity", "currently available", "active opportunity",
        "currently accepting", "accepting applications", "applications open",
        "applications are open", "open for applications",
        "currently accepting applications", "ongoing",
        "rolling basis", "rolling applications", "open registration",
    ))


def _has_clear_closed_evidence(item: dict[str, Any]) -> bool:
    text_parts = [
        item.get("title"), item.get("description"), item.get("status"),
        item.get("source_status"),
    ]
    text = " ".join(
        part for part in text_parts if isinstance(part, str)
    ).lower()
    return any(phrase in text for phrase in (
        "expired", "closed", "no longer accepting", "applications closed",
        "archived", "historical", "past opportunity",
    ))


def _is_current_opportunity(item: dict[str, Any], today: date) -> bool:
    """Keep dated current opportunities or explicitly open undated ones."""
    deadline = _parse_deadline(item.get("deadline"))
    if deadline is not None:
        return deadline >= today
    return _is_explicitly_open(item) and not _has_clear_closed_evidence(item)


def _is_relevant_to_candidate(
    item: dict[str, Any], candidate: CandidateProfile, query: str = ""
) -> bool:
    """Require a candidate profile signal in the retrieved opportunity text."""
    signals = [
        *candidate.skills,
        *candidate.interests,
        *candidate.education,
    ]
    if candidate.location:
        signals.append(candidate.location)
    signals = [signal.strip().lower() for signal in signals if signal.strip()]
    if not signals:
        return True

    text_parts = [item.get("title"), item.get("organization"), item.get("description")]
    requirements = item.get("requirements")
    if isinstance(requirements, list):
        text_parts.extend(requirements)
    searchable = " ".join(
        part for part in text_parts if isinstance(part, str)
    ).lower() + " " + query.lower()
    return any(signal in searchable for signal in signals)


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
    """Generate one fresh, profile-driven query for each target category."""
    location_hint = f" in {candidate.location}" if candidate.location else ""
    skills = ", ".join(candidate.skills[:3]) or "early career"
    education = candidate.education[0] if candidate.education else "students"
    interest = candidate.interests[0] if candidate.interests else skills

    return [
        f"{skills} jobs entry level official application current{location_hint}",
        f"{skills} internships official application current{location_hint}",
        f"scholarships for {education} official application current{location_hint}",
        f"{skills} hackathons official registration current{location_hint}",
        f"{interest} fellowships official application current{location_hint}",
    ]


# ---------------------------------------------------------------------------
# Search query and result shaping
# ---------------------------------------------------------------------------


def _infer_type(query: str, result: dict[str, Any]) -> str:
    raw_type = result.get("type") or result.get("opportunity_type")
    if isinstance(raw_type, str) and raw_type.strip():
        return raw_type
    return query


def _deadline_from_text(text: str) -> str | None:
    patterns = (
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group()
    return None


def _result_to_item(result: Any, query: str) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None

    url = result.get("url") or result.get("link")
    if not isinstance(url, str) or not url.strip():
        return None

    parsed = urlparse(url)
    source = result["source"] if "source" in result else parsed.netloc
    if source is not None and not isinstance(source, str):
        source = parsed.netloc
    description = result.get("description") or result.get("snippet") or ""
    deadline = result.get("deadline")
    if deadline is None and isinstance(description, str):
        deadline = _deadline_from_text(description)

    requirements = result.get("requirements", [])
    if not isinstance(requirements, list):
        requirements = []

    title = result.get("title")
    if not isinstance(title, str) or not title.strip():
        return None

    organization = result.get("organization") or result.get("company")
    if not isinstance(organization, str) or not organization.strip():
        organization = source

    location = result.get("location")
    item = {
        "id": "",
        "title": title,
        "organization": organization,
        "type": _infer_type(query, result),
        "description": description if isinstance(description, str) else "",
        "requirements": requirements,
        "deadline": deadline if isinstance(deadline, str) else None,
        "location": location if isinstance(location, str) else None,
        "url": url,
        "source": source,
    }
    for status_key in ("status", "source_status"):
        if isinstance(result.get(status_key), str):
            item[status_key] = result[status_key]
    return item


def _search_results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("opportunities", "organic", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise DiscoveryResponseError(
        "Search provider response did not contain a results list."
    )


def _retrieve_opportunities(
    candidate: CandidateProfile,
    max_results: int,
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Run profile-driven searches and return raw, locally verifiable records."""
    api_url, api_key = _search_configuration()
    records: list[dict[str, Any]] = []
    for query in _build_search_queries(candidate):
        search_query = (
            f"{query} currently open no expired or archived listings "
            "verified deadline direct official page"
        )
        try:
            payload = _search_provider(search_query, api_url, api_key)
        except DiscoveryAgentError:
            raise
        except Exception as exc:
            raise DiscoveryAPIError(
                f"Search provider request failed: {exc}"
            ) from exc
        results = _search_results(payload)
        counts["raw_search_results"] += len(results)
        for result in results:
            item = _result_to_item(result, query)
            if item is None:
                counts["malformed_candidates"] += 1
            else:
                counts["parsed_opportunity_candidates"] += 1
                if _is_relevant_to_candidate(item, candidate, query):
                    records.append(item)
                else:
                    counts["irrelevant_candidates"] += 1
            if len(records) >= max_results * 10:
                break
        if len(records) >= max_results * 10:
            break
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_opportunities(
    candidate: CandidateProfile,
    max_results: int = 10,
) -> list[Opportunity]:
    """Discover current opportunities relevant to the candidate profile.

    A configurable search provider supplies web results, which are then
    validated locally.
    """
    if not isinstance(candidate, CandidateProfile):
        raise TypeError("candidate must be a CandidateProfile instance.")

    global _last_filter_counts
    _last_filter_counts = {key: 0 for key in _FILTER_COUNT_KEYS}
    today = date.today()
    raw_list = _retrieve_opportunities(
        candidate, max_results, _last_filter_counts
    )
    if not _last_filter_counts["parsed_opportunity_candidates"]:
        raise DiscoveryResponseError(
            "Search provider returned no usable opportunity candidates. "
            f"Filtering counts: {_last_filter_counts}"
        )

    opportunities: list[Opportunity] = []
    seen_urls: set[str] = set()
    for item in raw_list:
        if not isinstance(item, dict):
            _last_filter_counts["malformed_candidates"] += 1
            continue

        url = item.get("url", "") or ""
        if not _is_valid_url(url):
            _last_filter_counts["invalid_url_candidates"] += 1
            continue
        deadline = _parse_deadline(item.get("deadline"))
        if deadline is not None and deadline < today:
            _last_filter_counts["expired_candidates"] += 1
            continue
        if not _is_current_opportunity(item, today):
            _last_filter_counts["unverified_freshness_candidates"] += 1
            continue

        item["type"] = _normalize_type(item.get("type", ""))
        item["id"] = _stable_id(url)

        try:
            opp = Opportunity.model_validate(item)
        except ValidationError:
            _last_filter_counts["malformed_candidates"] += 1
            continue

        normalised_url = opp.url.strip().lower().rstrip("/")
        if normalised_url in seen_urls:
            _last_filter_counts["duplicate_candidates"] += 1
            continue
        seen_urls.add(normalised_url)
        opportunities.append(opp)

    opportunities = opportunities[:max_results]
    _last_filter_counts["final_valid_opportunities"] = len(opportunities)
    return opportunities