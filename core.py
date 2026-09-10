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
            key = term.casefold()
            effective = theme_weight * item_weight
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
                return date(int(p[0]), int(p[1]) if len(p) > 1 else 1, int(p[2]) if len(p) > 2 else 1)
            except Exception:
                continue
    return None


def _authors(item: dict) -> str:
    names = []
    for a in item.get("author") or []:
        name = " ".join(x for x in ((a.get("given") or "").strip(), (a.get("family") or "").strip()) if x)
        if name:
            names.append(name)
    return ", ".join(names)


def _first(xs) -> str:
    if isinstance(xs, list) and xs:
        return str(xs[0])
    return str(xs or "")


def fetch_crossref_journal(issn: str, days: int = 30, rows: int = 40, mailto: str = "", timeout: int = 20) -> List[dict]:
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
    headers = {"User-Agent": "PaperRecommender/0.4 (scholarly metadata discovery; contact via configured mailto)"}
    r = requests.get(f"{CROSSREF_BASE}/journals/{quote(issn)}/works", params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return ((r.json() or {}).get("message") or {}).get("items") or []


def fetch_catalog(journal_df: pd.DataFrame, days: int, rows_per_journal: int, mailto: str = "", max_workers: int = 8) -> Tuple[pd.DataFrame, List[str]]:
    errors: List[str] = []
    records: List[dict] = []
    prepared = []
    for _, row in journal_df.iterrows():
        issn = normalize_issn(str(row.get("issn", "")))
        if issn:
            prepared.append({
                "journal": str(row.get("journal", "")).strip(),
                "issn": issn,
                "catalog": str(row.get("catalog", "")).strip(),
                "journal_weight": float(row.get("weight", 1.0) or 1.0),
            })

    def worker(j):
        try:
            return j, fetch_crossref_journal(j["issn"], days=days, rows=rows_per_journal, mailto=mailto), None
        except Exception as e:
            return j, [], str(e)

    with cf.ThreadPoolExecutor(max_workers=max(1, min(max_workers, 12))) as ex:
        for fut in cf.as_completed([ex.submit(worker, j) for j in prepared]):
            j, items, err = fut.result()
            if err:
                errors.append(f"{j['journal'] or j['issn']}: {err}")
                continue
            for item in items:
                pubdate = _date_parts(item)
                records.append({
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
                })

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
        return len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.IGNORECASE))
    return text.casefold().count(term.casefold())


def _profile_hits(title: str, abstract: str, profile: Optional[dict]) -> List[dict]:
    hits = []
    if not profile:
        return hits
    for theme in profile.get("themes") or []:
        theme_weight = float(theme.get("weight", 1.0) or 1.0)
        terms, score = [], 0.0
        for item in theme.get("keywords") or []:
            if isinstance(item, str):
                term, item_weight = item.strip(), 1.0
            else:
                term = str(item.get("term", "")).strip()
                item_weight = float(item.get("weight", 1.0) or 1.0)
            if not term:
                continue
            tc, ac = _count_phrase(title, term), _count_phrase(abstract, term)
            if tc or ac:
                contribution = theme_weight * item_weight * (3.0 * min(tc, 2) + min(ac, 4))
                score += contribution
                terms.append((term, contribution))
        if terms:
            terms.sort(key=lambda x: x[1], reverse=True)
            hits.append({
                "id": str(theme.get("id", "")),
                "name": str(theme.get("name", "")),
                "tier": str(theme.get("tier", "cross")),
                "weight": theme_weight,
                "score": score,
                "terms": [x[0] for x in terms],
            })
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _theme_document(theme: dict) -> str:
    parts = [str(theme.get("name", "")), str(theme.get("description", "")), " ".join(map(str, theme.get("semantic_queries") or []))]
    for item in theme.get("keywords") or []:
        parts.append(str(item if isinstance(item, str) else item.get("term", "")))
    return " ".join(x for x in parts if x).strip()


def semantic_theme_matches(df: pd.DataFrame, profile: Optional[dict], threshold: float = 0.055) -> Dict[int, List[dict]]:
    if df.empty or not profile or not (profile.get("themes") or []):
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return {}

    themes = profile.get("themes") or []
    theme_docs = [_theme_document(t) for t in themes]
    paper_docs = [
        f"{safe_text(r.get('title'))}. {safe_text(r.get('title'))}. {safe_text(r.get('abstract'))[:6000]} {safe_text(r.get('journal'))}"
        for _, r in df.iterrows()
    ]
    corpus = theme_docs + paper_docs
    try:
        word_vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=24000)
        wm = word_vec.fit_transform(corpus)
        word_sim = cosine_similarity(wm[:len(themes)], wm[len(themes):])
    except Exception:
        word_sim = None
    try:
        char_vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1, max_features=32000)
        cm = char_vec.fit_transform(corpus)
        char_sim = cosine_similarity(cm[:len(themes)], cm[len(themes):])
    except Exception:
        char_sim = None
    if word_sim is None and char_sim is None:
        return {}
    combined = char_sim if word_sim is None else word_sim if char_sim is None else 0.78 * word_sim + 0.22 * char_sim

    result: Dict[int, List[dict]] = {}
    for paper_pos, df_idx in enumerate(df.index):
        phits = []
        for theme_pos, theme in enumerate(themes):
            sim = float(combined[theme_pos, paper_pos])
            if sim >= threshold:
                phits.append({
                    "id": str(theme.get("id", "")),
                    "name": str(theme.get("name", "")),
                    "tier": str(theme.get("tier", "cross")),
                    "similarity": sim,
                    "theme_weight": float(theme.get("weight", 1.0) or 1.0),
                })
        result[df_idx] = sorted(phits, key=lambda x: x["similarity"] * x["theme_weight"], reverse=True)
    return result


def _matched_theme_evidence(exact_hits: List[dict], semantic_hits: List[dict]) -> Dict[str, float]:
    evidence: Dict[str, float] = {}
    for h in exact_hits:
        if h.get("id"):
            evidence[h["id"]] = max(evidence.get(h["id"], 0.0), min(1.0, h["score"] / 18.0))
    for h in semantic_hits:
        if h.get("id"):
            evidence[h["id"]] = max(evidence.get(h["id"], 0.0), min(1.0, h["similarity"] / 0.18))
    return evidence


def _intersection_signal(profile: Optional[dict], evidence: Dict[str, float], intersection_weight: float = 2.4, cross_theme_bonus: float = 0.8) -> Tuple[float, List[str]]:
    if not profile or len(evidence) < 2:
        return 0.0, []
    labels, score = [], 0.0
    for rel in profile.get("relations") or []:
        a, b = str(rel.get("source", "")), str(rel.get("target", ""))
        if a in evidence and b in evidence:
            rel_w = float(rel.get("weight", 0.0) or 0.0)
            strength = math.sqrt(evidence[a] * evidence[b])
            score += intersection_weight * rel_w * strength
            if rel.get("label"):
                labels.append((str(rel["label"]), rel_w * strength))
    if len(evidence) >= 3:
        score += cross_theme_bonus * min(3, len(evidence) - 2)
    labels = [x[0] for x in sorted(labels, key=lambda x: x[1], reverse=True)]
    return score, labels[:3]


def _profile_reason(title: str, abstract: str, profile: Optional[dict], fallback_terms: List[str], age: int, semantic_hits: Optional[List[dict]] = None, intersection_labels: Optional[List[str]] = None) -> Tuple[str, str, float, str]:
    exact_hits = _profile_hits(title, abstract, profile)
    semantic_hits = semantic_hits or []
    intersection_labels = intersection_labels or []
    theme_names: List[str] = []
    for h in exact_hits + semantic_hits:
        if h.get("name") and h["name"] not in theme_names:
            theme_names.append(h["name"])
    semantic_top = float(semantic_hits[0]["similarity"]) if semantic_hits else 0.0

    if exact_hits:
        primary = exact_hits[0]
        terms = []
        for h in exact_hits[:2]:
            terms.extend(h["terms"])
        deduped = list(dict.fromkeys(terms))
        reason = f"与你的“{primary['name']}”研究兴趣{'高度契合' if primary['score'] >= 12 or len(deduped) >= 3 else '较为契合'}，直接命中：{'、'.join(deduped[:5])}"
        if len(theme_names) >= 2:
            reason += f"；同时涉及“{theme_names[1]}”"
        if intersection_labels:
            reason += f"；属于你重点关注的交叉方向“{intersection_labels[0]}”"
        if abstract.strip():
            reason += "，研究问题或方法与你的既有研究主线存在直接联系"
        else:
            reason += "。当前判断主要基于标题与期刊元数据"
    elif semantic_hits:
        primary = semantic_hits[0]
        reason = f"虽未直接出现核心关键词，但标题/摘要整体表述与“{primary['name']}”具有语义关联"
        if len(semantic_hits) >= 2:
            reason += f"，并与“{semantic_hits[1]['name']}”交叉"
        if intersection_labels:
            reason += f"；该组合对应你的交叉兴趣“{intersection_labels[0]}”"
        reason += "，可减少单纯关键词检索造成的漏检"
    elif fallback_terms:
        reason = f"命中你的临时关注关键词：{'、'.join(fallback_terms[:5])}"
    else:
        reason = "来自目标期刊目录且较新，但与当前研究画像的直接或语义匹配较弱"

    if intersection_labels and len(theme_names) >= 2:
        recommendation_level = "强推荐"
    elif exact_hits and exact_hits[0]["score"] >= 12:
        recommendation_level = "强推荐"
    elif exact_hits or semantic_top >= 0.09:
        recommendation_level = "推荐"
    else:
        recommendation_level = "拓展阅读"

    if age <= 14:
        reason += "；且属于近两周新论文，建议优先浏览"
    elif age <= 30:
        reason += "；且属于近一个月的新论文"
    return reason + "。", "、".join(theme_names[:3]), semantic_top, recommendation_level


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
    intersection_weight: float = 2.4,
    cross_theme_bonus: float = 0.8,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    semantic_map = semantic_theme_matches(df, profile, threshold=semantic_threshold) if use_semantic else {}
    today = date.today()
    rows = []
    for idx, r in df.iterrows():
        title, abstract = str(r.get("title", "")), str(r.get("abstract", ""))
        matched: List[Tuple[str, float]] = []
        pos_score = neg_score = 0.0
        for kw in positive:
            tc, ac = _count_phrase(title, kw.term), _count_phrase(abstract, kw.term)
            if tc or ac:
                contribution = kw.weight * (3.0 * min(tc, 2) + min(ac, 4))
                pos_score += contribution
                matched.append((kw.term, contribution))
        for kw in negative:
            tc, ac = _count_phrase(title, kw.term), _count_phrase(abstract, kw.term)
            if tc or ac:
                neg_score += kw.weight * (4.0 * min(tc, 2) + 1.5 * min(ac, 4))

        try:
            age = max(0, (today - date.fromisoformat(str(r.get("date", "")))).days)
        except Exception:
            age = 365
        recency = math.exp(-math.log(2) * age / max(1.0, recency_half_life))
        journal_weight = float(r.get("journal_weight", 1.0) or 1.0)
        coverage = len(matched) / max(1, len(positive))

        semantic_hits = semantic_map.get(idx, [])
        semantic_component = 0.0
        if semantic_hits:
            semantic_component += semantic_weight * semantic_hits[0]["similarity"] * semantic_hits[0]["theme_weight"]
            if len(semantic_hits) > 1:
                semantic_component += 0.35 * semantic_weight * semantic_hits[1]["similarity"] * semantic_hits[1]["theme_weight"]

        exact_hits = _profile_hits(title, abstract, profile)
        evidence = _matched_theme_evidence(exact_hits, semantic_hits)
        intersection_component, intersection_labels = _intersection_signal(profile, evidence, intersection_weight, cross_theme_bonus)
        score = (pos_score - neg_score) * journal_weight + semantic_component + intersection_component + 1.2 * recency + 0.8 * coverage

        matched.sort(key=lambda x: x[1], reverse=True)
        matched_terms = [x[0] for x in matched]
        reason, profile_match, semantic_top, rec_level = _profile_reason(title, abstract, profile, matched_terms, age, semantic_hits, intersection_labels)
        out = dict(r)
        out.update({
            "score": round(score, 3),
            "keyword_score": round(pos_score - neg_score, 3),
            "semantic_score": round(semantic_component, 3),
            "intersection_score": round(intersection_component, 3),
            "semantic_similarity": round(semantic_top, 4),
            "matched_keywords": ", ".join(matched_terms),
            "profile_match": profile_match,
            "intersection_match": "、".join(intersection_labels),
            "recommendation_level": rec_level,
            "recommendation_reason": reason,
            "age_days": age,
        })
        if score >= min_score:
            rows.append(out)

    extra = ["score", "keyword_score", "semantic_score", "intersection_score", "semantic_similarity", "matched_keywords", "profile_match", "intersection_match", "recommendation_level", "recommendation_reason", "age_days"]
    if not rows:
        return pd.DataFrame(columns=list(df.columns) + extra)
    return pd.DataFrame(rows).sort_values(["score", "date"], ascending=[False, False]).reset_index(drop=True)


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)
