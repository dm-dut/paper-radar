from __future__ import annotations

import concurrent.futures as cf
import html
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests

CROSSREF_BASE = "https://api.crossref.org"


@dataclass
class Keyword:
    term: str
    weight: float = 1.0


def strip_jats(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_issn(value: str) -> str:
    value = (value or "").strip().upper()
    value = re.sub(r"[^0-9X]", "", value)
    if len(value) == 8:
        return f"{value[:4]}-{value[4:]}"
    return value


def parse_keywords(text: str) -> List[Keyword]:
    """Parse one keyword per line. Supported: `term`, `term|2`, `term,2`."""
    out: List[Keyword] = []
    for raw in (text or "").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        term, weight = raw, 1.0
        m = re.match(r"^(.*?)[|,，]\s*([0-9.]+)\s*$", raw)
        if m:
            term = m.group(1).strip()
            try:
                weight = float(m.group(2))
            except ValueError:
                weight = 1.0
        if term:
            out.append(Keyword(term=term, weight=max(0.0, weight)))
    return out


def profile_keywords(profile: dict) -> List[Keyword]:
    """Flatten a structured research-interest profile into weighted keywords."""
    out: List[Keyword] = []
    seen: Dict[str, float] = {}
    originals: Dict[str, str] = {}
    for theme in profile.get("themes") or []:
        theme_weight = float(theme.get("weight", 1.0) or 1.0)
        for item in theme.get("keywords") or []:
            if isinstance(item, str):
                term, item_weight = item.strip(), 1.0
            else:
                term = str(item.get("term", "")).strip()
                item_weight = float(item.get("weight", 1.0) or 1.0)
            if not term:
                continue
            effective = theme_weight * item_weight
            key = term.casefold()
            seen[key] = max(seen.get(key, 0.0), effective)
            originals.setdefault(key, term)
    return [Keyword(term=originals[k], weight=w) for k, w in seen.items()]


def profile_keywords_text(profile: dict) -> str:
    return "\n".join(f"{kw.term}|{kw.weight:.2f}" for kw in profile_keywords(profile))


def _date_parts(item: dict) -> Optional[date]:
    for key in ("published-online", "published-print", "published", "issued", "created"):
        value = item.get(key) or {}
        if key == "created" and value.get("date-time"):
            try:
                return datetime.fromisoformat(value["date-time"].replace("Z", "+00:00")).date()
            except Exception:
                pass
        parts = value.get("date-parts") or []
        if parts and parts[0]:
            p = parts[0]
            try:
                y = int(p[0])
                m = int(p[1]) if len(p) > 1 else 1
                d = int(p[2]) if len(p) > 2 else 1
                return date(y, m, d)
            except Exception:
                continue
    return None


def _authors(item: dict) -> str:
    names = []
    for a in item.get("author") or []:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        name = " ".join(x for x in (given, family) if x)
        if name:
            names.append(name)
    return ", ".join(names)


def _first(xs) -> str:
    if isinstance(xs, list) and xs:
        return str(xs[0])
    return str(xs or "")


def fetch_crossref_journal(
    issn: str,
    days: int = 30,
    rows: int = 40,
    mailto: str = "",
    timeout: int = 20,
) -> List[dict]:
    issn = normalize_issn(issn)
    if not issn:
        return []
    start = (date.today() - timedelta(days=max(1, days))).isoformat()
    params = {
        "filter": f"from-pub-date:{start},type:journal-article",
        "sort": "published",
        "order": "desc",
        "rows": max(1, min(int(rows), 1000)),
        "select": "DOI,title,author,container-title,published-online,published-print,published,issued,created,URL,abstract,ISSN,type",
    }
    if mailto.strip():
        params["mailto"] = mailto.strip()
    headers = {
        "User-Agent": "PaperRecommender/0.3 (scholarly metadata discovery; contact via configured mailto)"
    }
    url = f"{CROSSREF_BASE}/journals/{quote(issn)}/works"
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return ((r.json() or {}).get("message") or {}).get("items") or []


def fetch_catalog(
    journal_df: pd.DataFrame,
    days: int,
    rows_per_journal: int,
    mailto: str = "",
    max_workers: int = 8,
) -> Tuple[pd.DataFrame, List[str]]:
    errors: List[str] = []
    records: List[dict] = []
    prepared = []
    for _, row in journal_df.iterrows():
        issn = normalize_issn(str(row.get("issn", "")))
        if not issn:
            continue
        prepared.append(
            {
                "journal": str(row.get("journal", "")).strip(),
                "issn": issn,
                "catalog": str(row.get("catalog", "")).strip(),
                "journal_weight": float(row.get("weight", 1.0) or 1.0),
            }
        )

    def worker(j):
        try:
            return j, fetch_crossref_journal(
                j["issn"], days=days, rows=rows_per_journal, mailto=mailto
            ), None
        except Exception as e:
            return j, [], str(e)

    with cf.ThreadPoolExecutor(max_workers=max(1, min(max_workers, 12))) as ex:
        futures = [ex.submit(worker, j) for j in prepared]
        for fut in cf.as_completed(futures):
            j, items, err = fut.result()
            if err:
                errors.append(f"{j['journal'] or j['issn']}: {err}")
                continue
            for item in items:
                pubdate = _date_parts(item)
                records.append(
                    {
                        "title": _first(item.get("title")),
                        "authors": _authors(item),
                        "journal": _first(item.get("container-title")) or j["journal"],
                        "catalog": j["catalog"],
                        "issn": j["issn"],
                        "date": pubdate.isoformat() if pubdate else "",
                        "doi": item.get("DOI") or "",
                        "url": item.get("URL") or (f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else ""),
                        "abstract": strip_jats(item.get("abstract") or ""),
                        "journal_weight": j["journal_weight"],
                    }
                )

    if not records:
        return pd.DataFrame(), errors
    df = pd.DataFrame(records)
    df["_rich"] = df["abstract"].str.len()
    df = df.sort_values(["_rich", "date"], ascending=[False, False])
    if df["doi"].astype(bool).any():
        with_doi = df[df["doi"].astype(bool)].drop_duplicates("doi", keep="first")
        without_doi = df[~df["doi"].astype(bool)].drop_duplicates(["title", "journal"], keep="first")
        df = pd.concat([with_doi, without_doi], ignore_index=True)
    else:
        df = df.drop_duplicates(["title", "journal"], keep="first")
    return df.drop(columns=["_rich"], errors="ignore"), errors


def _count_phrase(text: str, term: str) -> int:
    if not text or not term:
        return 0
    if re.fullmatch(r"[\w\s]+", term, flags=re.UNICODE):
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        return len(re.findall(pattern, text, flags=re.IGNORECASE))
    return text.casefold().count(term.casefold())


def _profile_hits(title: str, abstract: str, profile: Optional[dict]) -> List[dict]:
    if not profile:
        return []
    hits = []
    for theme in profile.get("themes") or []:
        theme_name = str(theme.get("name", "")).strip()
        theme_weight = float(theme.get("weight", 1.0) or 1.0)
        terms = []
        score = 0.0
        for item in theme.get("keywords") or []:
            if isinstance(item, str):
                term, item_weight = item.strip(), 1.0
            else:
                term = str(item.get("term", "")).strip()
                item_weight = float(item.get("weight", 1.0) or 1.0)
            if not term:
                continue
            tc = _count_phrase(title, term)
            ac = _count_phrase(abstract, term)
            if tc or ac:
                contribution = theme_weight * item_weight * (3.0 * min(tc, 2) + min(ac, 4))
                score += contribution
                terms.append((term, contribution))
        if terms:
            terms = sorted(terms, key=lambda x: x[1], reverse=True)
            hits.append(
                {
                    "name": theme_name,
                    "score": score,
                    "terms": [x[0] for x in terms],
                    "description": str(theme.get("description", "")).strip(),
                }
            )
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _theme_document(theme: dict) -> str:
    parts = [
        str(theme.get("name", "")),
        str(theme.get("description", "")),
        " ".join(str(x) for x in (theme.get("semantic_queries") or [])),
    ]
    for item in theme.get("keywords") or []:
        parts.append(str(item if isinstance(item, str) else item.get("term", "")))
    return " ".join(x for x in parts if x).strip()


def semantic_theme_matches(
    df: pd.DataFrame,
    profile: Optional[dict],
    threshold: float = 0.055,
) -> Dict[int, List[dict]]:
    """Lightweight semantic matching using theme descriptions + concept expansion + TF-IDF similarity.

    This is intentionally local and API-free. It combines word n-grams and character n-grams,
    so related formulations such as ordinal classification / preference learning or consensus
    building / consensus reaching can still receive a signal when the exact core phrase differs.
    """
    if df.empty or not profile or not (profile.get("themes") or []):
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return {}

    themes = profile.get("themes") or []
    theme_docs = [_theme_document(t) for t in themes]
    paper_docs = []
    for _, r in df.iterrows():
        title = safe_text(r.get("title"))
        abstract = safe_text(r.get("abstract"))
        journal = safe_text(r.get("journal"))
        # Title is repeated to emphasize problem framing while still using the abstract context.
        paper_docs.append(f"{title}. {title}. {abstract[:6000]} {journal}")

    corpus = theme_docs + paper_docs
    if not any(x.strip() for x in corpus):
        return {}

    try:
        word_vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_features=20000,
        )
        word_matrix = word_vec.fit_transform(corpus)
        word_sim = cosine_similarity(word_matrix[: len(themes)], word_matrix[len(themes) :])
    except Exception:
        word_sim = None

    try:
        char_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
            max_features=30000,
        )
        char_matrix = char_vec.fit_transform(corpus)
        char_sim = cosine_similarity(char_matrix[: len(themes)], char_matrix[len(themes) :])
    except Exception:
        char_sim = None

    if word_sim is None and char_sim is None:
        return {}
    if word_sim is None:
        combined = char_sim
    elif char_sim is None:
        combined = word_sim
    else:
        combined = 0.75 * word_sim + 0.25 * char_sim

    result: Dict[int, List[dict]] = {}
    df_indices = list(df.index)
    for paper_pos, df_idx in enumerate(df_indices):
        hits = []
        for theme_pos, theme in enumerate(themes):
            sim = float(combined[theme_pos, paper_pos])
            if sim >= threshold:
                hits.append(
                    {
                        "name": str(theme.get("name", "")),
                        "similarity": sim,
                        "theme_weight": float(theme.get("weight", 1.0) or 1.0),
                    }
                )
        result[df_idx] = sorted(hits, key=lambda x: x["similarity"] * x["theme_weight"], reverse=True)
    return result


def _profile_reason(
    title: str,
    abstract: str,
    profile: Optional[dict],
    fallback_terms: List[str],
    age: int,
    semantic_hits: Optional[List[dict]] = None,
) -> Tuple[str, str, float]:
    exact_hits = _profile_hits(title, abstract, profile)
    semantic_hits = semantic_hits or []
    theme_names: List[str] = []
    for h in exact_hits:
        if h["name"] not in theme_names:
            theme_names.append(h["name"])
    for h in semantic_hits:
        if h["name"] not in theme_names:
            theme_names.append(h["name"])

    semantic_top = float(semantic_hits[0]["similarity"]) if semantic_hits else 0.0

    if exact_hits:
        primary = exact_hits[0]
        terms = []
        for h in exact_hits[:2]:
            terms.extend(h["terms"])
        deduped = []
        seen = set()
        for t in terms:
            k = t.casefold()
            if k not in seen:
                deduped.append(t)
                seen.add(k)
        strength = "高度契合" if primary["score"] >= 12 or len(deduped) >= 3 else "较为契合"
        reason = f"与你的“{primary['name']}”研究兴趣{strength}，直接命中：{'、'.join(deduped[:5])}"
        if len(exact_hits) >= 2:
            reason += f"；同时与“{exact_hits[1]['name']}”形成交叉"
        sem_extra = next((h for h in semantic_hits if h["name"] not in [x["name"] for x in exact_hits[:2]]), None)
        if sem_extra:
            reason += f"；语义上还与“{sem_extra['name']}”接近"
        if not abstract.strip():
            reason += "。当前判断主要基于标题与期刊元数据"
        else:
            reason += "，说明其研究问题或方法与你的核心方向存在直接联系"
    elif semantic_hits:
        primary = semantic_hits[0]
        reason = (
            f"未直接出现你的核心关键词，但标题/摘要的整体表述与“{primary['name']}”主题具有语义关联"
        )
        if len(semantic_hits) >= 2:
            reason += f"，并与“{semantic_hits[1]['name']}”存在一定交叉"
        reason += "；建议作为潜在相关论文浏览，以避免仅依赖关键词造成漏检"
    elif fallback_terms:
        reason = f"命中你的临时关注关键词：{'、'.join(fallback_terms[:5])}"
    else:
        reason = "来自目标期刊目录且较新，但与当前研究画像的直接或语义匹配较弱"

    if age <= 14:
        reason += "；且属于近两周新论文，建议优先浏览"
    elif age <= 30:
        reason += "；且属于近一个月的新论文"
    return reason + "。", "、".join(theme_names[:3]), semantic_top


def rank_papers(
    df: pd.DataFrame,
    positive: List[Keyword],
    negative: List[Keyword],
    recency_half_life: float = 30.0,
    min_score: float = 0.0,
    profile: Optional[dict] = None,
    use_semantic: bool = True,
    semantic_weight: float = 10.0,
    semantic_threshold: float = 0.055,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    semantic_map = semantic_theme_matches(df, profile, threshold=semantic_threshold) if use_semantic else {}
    today = date.today()
    rows = []
    for idx, r in df.iterrows():
        title = str(r.get("title", ""))
        abstract = str(r.get("abstract", ""))
        matched: List[Tuple[str, float]] = []
        pos_score = 0.0
        neg_score = 0.0

        for kw in positive:
            tc = _count_phrase(title, kw.term)
            ac = _count_phrase(abstract, kw.term)
            if tc or ac:
                contribution = kw.weight * (3.0 * min(tc, 2) + 1.0 * min(ac, 4))
                pos_score += contribution
                matched.append((kw.term, contribution))
        for kw in negative:
            tc = _count_phrase(title, kw.term)
            ac = _count_phrase(abstract, kw.term)
            if tc or ac:
                neg_score += kw.weight * (4.0 * min(tc, 2) + 1.5 * min(ac, 4))

        try:
            pub = date.fromisoformat(str(r.get("date", "")))
            age = max(0, (today - pub).days)
        except Exception:
            age = 365
        recency = math.exp(-math.log(2) * age / max(1.0, recency_half_life))
        journal_weight = float(r.get("journal_weight", 1.0) or 1.0)
        coverage = len(matched) / max(1, len(positive))

        semantic_hits = semantic_map.get(idx, [])
        semantic_component = 0.0
        if semantic_hits:
            primary = semantic_hits[0]
            semantic_component += semantic_weight * primary["similarity"] * primary["theme_weight"]
            if len(semantic_hits) > 1:
                secondary = semantic_hits[1]
                semantic_component += 0.35 * semantic_weight * secondary["similarity"] * secondary["theme_weight"]

        score = (pos_score - neg_score) * journal_weight + semantic_component + 1.2 * recency + 0.8 * coverage
        matched = sorted(matched, key=lambda x: x[1], reverse=True)
        matched_terms = [x[0] for x in matched]
        reason, profile_match, semantic_top = _profile_reason(
            title, abstract, profile, matched_terms, age, semantic_hits=semantic_hits
        )

        out = dict(r)
        out.update(
            {
                "score": round(score, 3),
                "keyword_score": round(pos_score - neg_score, 3),
                "semantic_score": round(semantic_component, 3),
                "semantic_similarity": round(semantic_top, 4),
                "matched_keywords": ", ".join(matched_terms),
                "profile_match": profile_match,
                "recommendation_reason": reason,
                "age_days": age,
            }
        )
        if score >= min_score:
            rows.append(out)

    extra_cols = [
        "score", "keyword_score", "semantic_score", "semantic_similarity",
        "matched_keywords", "profile_match", "recommendation_reason", "age_days"
    ]
    if not rows:
        return pd.DataFrame(columns=list(df.columns) + extra_cols)
    out = pd.DataFrame(rows)
    return out.sort_values(["score", "date"], ascending=[False, False]).reset_index(drop=True)


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)
