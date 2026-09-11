from __future__ import annotations

import os
import re
from typing import Dict, Tuple

import pandas as pd
import requests

OPENALEX_BASE = "https://api.openalex.org"


def _normalize_doi(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    return value.strip().lower()


def _reconstruct_abstract(inverted_index) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        if not isinstance(idxs, list):
            continue
        for idx in idxs:
            try:
                positions.append((int(idx), str(word)))
            except Exception:
                continue
    if not positions:
        return ""
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions).strip()


def enrich_missing_abstracts(
    df: pd.DataFrame,
    timeout: int = 25,
    batch_size: int = 80,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Fill missing Crossref abstracts from OpenAlex using DOI batch lookup.

    OpenAlex exposes abstracts as an inverted index. We reconstruct the plain-text
    abstract locally. An API key is optional; if OPENALEX_API_KEY is present it is
    used automatically.
    """
    if df.empty:
        return df.copy(), {"before": 0, "after": 0, "added": 0, "looked_up": 0}

    out = df.copy()
    if "abstract" not in out.columns:
        out["abstract"] = ""
    out["abstract"] = out["abstract"].fillna("").astype(str)
    if "abstract_source" not in out.columns:
        out["abstract_source"] = ""
    out["abstract_source"] = out["abstract_source"].fillna("").astype(str)
    out.loc[out["abstract"].str.strip().astype(bool) & ~out["abstract_source"].str.strip().astype(bool), "abstract_source"] = "Crossref"

    before = int(out["abstract"].str.strip().astype(bool).sum())
    missing_mask = ~out["abstract"].str.strip().astype(bool)
    doi_map = {}
    for idx, value in out.loc[missing_mask, "doi"].items():
        doi = _normalize_doi(str(value))
        if doi:
            doi_map.setdefault(doi, []).append(idx)

    dois = list(doi_map)
    if not dois:
        out.loc[~out["abstract"].str.strip().astype(bool), "abstract_source"] = "未获取"
        return out, {"before": before, "after": before, "added": 0, "looked_up": 0}

    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    headers = {"User-Agent": "PaperRadar/0.6 (abstract enrichment for scholarly discovery)"}
    added = 0

    for start in range(0, len(dois), max(1, min(batch_size, 100))):
        batch = dois[start : start + max(1, min(batch_size, 100))]
        doi_values = "|".join(f"https://doi.org/{doi}" for doi in batch)
        params = {
            "filter": f"doi:{doi_values}",
            "per-page": 100,
            "select": "id,doi,abstract_inverted_index",
        }
        if api_key:
            params["api_key"] = api_key
        try:
            response = requests.get(f"{OPENALEX_BASE}/works", params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            items = (response.json() or {}).get("results") or []
        except Exception:
            continue

        for item in items:
            doi = _normalize_doi(str(item.get("doi") or ""))
            abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
            if not doi or not abstract:
                continue
            for idx in doi_map.get(doi, []):
                if not str(out.at[idx, "abstract"]).strip():
                    out.at[idx, "abstract"] = abstract
                    out.at[idx, "abstract_source"] = "OpenAlex"
                    added += 1

    out.loc[~out["abstract"].str.strip().astype(bool), "abstract_source"] = "未获取"
    after = int(out["abstract"].str.strip().astype(bool).sum())
    return out, {"before": before, "after": after, "added": added, "looked_up": len(dois)}
