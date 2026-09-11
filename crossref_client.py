from __future__ import annotations

import email.utils
import threading
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import requests

import core

# Crossref's list endpoints are rate-limited. Paper Radar monitors many journals,
# so requests are deliberately paced even when multiple worker threads are used.
#
# The anonymous pool is kept below 1 request/second. When a mailto address is
# supplied, Paper Radar uses a conservative 2 requests/second and at most three
# concurrent workers.
_ANON_RPS = 0.75
_POLITE_RPS = 2.0
_MAX_RETRIES = 4
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_rate_lock = threading.Lock()
_next_request_at = 0.0

_original_fetch_catalog = core.fetch_catalog


def _wait_for_slot(mailto: str) -> None:
    """Globally pace Crossref requests across all worker threads."""
    global _next_request_at

    rps = _POLITE_RPS if (mailto or "").strip() else _ANON_RPS
    interval = 1.0 / max(0.1, rps)

    with _rate_lock:
        now = time.monotonic()
        scheduled = max(now, _next_request_at)
        _next_request_at = scheduled + interval

    delay = scheduled - now
    if delay > 0:
        time.sleep(delay)


def _retry_after_seconds(response: requests.Response) -> float:
    """Parse Retry-After as seconds or an HTTP date."""
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return 0.0

    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass

    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 0.0


def fetch_crossref_journal(
    issn: str,
    days: int = 30,
    rows: int = 40,
    mailto: str = "",
    timeout: int = 25,
):
    """Fetch one journal with pacing and automatic retry/backoff.

    Retryable failures include HTTP 429 and common transient 5xx responses.
    When Crossref sends Retry-After, that value is respected; otherwise an
    exponential 2/4/8/16 second backoff is used.
    """
    issn = core.normalize_issn(issn)
    if not issn:
        return []

    start = (date.today() - timedelta(days=max(1, int(days)))).isoformat()
    params = {
        "filter": f"from-pub-date:{start},type:journal-article",
        "sort": "published",
        "order": "desc",
        "rows": max(1, min(int(rows), 1000)),
        "select": (
            "DOI,title,author,container-title,published-online,published-print,"
            "published,issued,created,URL,abstract,ISSN,type"
        ),
    }

    mailto = (mailto or "").strip()
    if mailto:
        params["mailto"] = mailto

    user_agent = "PaperRadar/0.7 (scholarly metadata discovery"
    if mailto:
        user_agent += f"; mailto:{mailto}"
    user_agent += ")"
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    url = f"{core.CROSSREF_BASE}/journals/{quote(issn)}/works"
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        _wait_for_slot(mailto)

        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= _MAX_RETRIES:
                raise
            time.sleep(min(16.0, 2.0 ** (attempt + 1)))
            continue

        if response.ok:
            return ((response.json() or {}).get("message") or {}).get("items") or []

        if response.status_code not in _RETRYABLE_STATUS:
            response.raise_for_status()

        last_error = requests.HTTPError(
            f"{response.status_code} {response.reason} for url: {response.url}",
            response=response,
        )
        if attempt >= _MAX_RETRIES:
            response.raise_for_status()

        retry_after = _retry_after_seconds(response)
        fallback = min(16.0, 2.0 ** (attempt + 1))
        # Respect server guidance, while ensuring a meaningful cooldown even
        # when Retry-After is absent or unusually small.
        time.sleep(max(retry_after, fallback))

    if last_error:
        raise last_error
    return []


def fetch_catalog(
    journal_df,
    days: int,
    rows_per_journal: int,
    mailto: str = "",
    max_workers: int = 8,
):
    """Call the existing catalog assembler with safe Crossref concurrency."""
    safe_cap = 3 if (mailto or "").strip() else 1
    safe_workers = max(1, min(int(max_workers), safe_cap))
    return _original_fetch_catalog(
        journal_df,
        days=days,
        rows_per_journal=rows_per_journal,
        mailto=mailto,
        max_workers=safe_workers,
    )


def install() -> None:
    """Patch core before paper_radar_app imports fetch_catalog from it."""
    core.fetch_crossref_journal = fetch_crossref_journal
    core.fetch_catalog = fetch_catalog
