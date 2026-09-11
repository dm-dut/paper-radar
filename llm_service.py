from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional

import pandas as pd
import requests

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def openai_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def configured_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _headers() -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _paper_key(row: dict, position: int = 0) -> str:
    doi = str(row.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    basis = "|".join(
        [
            str(row.get("title") or "").strip(),
            str(row.get("journal") or "").strip(),
            str(row.get("date") or "").strip(),
            str(position),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return f"paper:{digest}"


def compact_profile(profile: dict) -> dict:
    themes = []
    for theme in profile.get("themes") or []:
        kws = []
        for item in theme.get("keywords") or []:
            if isinstance(item, str):
                kws.append(item)
            else:
                term = str(item.get("term", "")).strip()
                if term:
                    kws.append(term)
        themes.append(
            {
                "id": theme.get("id", ""),
                "name": theme.get("name", ""),
                "tier": theme.get("tier", ""),
                "weight": theme.get("weight", 1.0),
                "description": theme.get("description", ""),
                "semantic_queries": (theme.get("semantic_queries") or [])[:6],
                "keywords": kws[:18],
            }
        )
    relations = []
    for rel in profile.get("relations") or []:
        relations.append(
            {
                "themes": rel.get("themes") or rel.get("theme_ids") or rel.get("between") or [],
                "label": rel.get("label") or rel.get("name") or rel.get("description") or "",
                "weight": rel.get("weight", 1.0),
            }
        )
    return {
        "profile_name": profile.get("profile_name", "研究兴趣画像"),
        "description": profile.get("description", ""),
        "themes": themes,
        "relations": relations[:20],
    }


def prepare_papers(df: pd.DataFrame, limit: int = 10) -> List[dict]:
    papers: List[dict] = []
    for pos, (_, row) in enumerate(df.head(max(1, int(limit))).iterrows()):
        raw = row.to_dict()
        abstract = str(raw.get("abstract") or "").strip()
        papers.append(
            {
                "id": _paper_key(raw, pos),
                "title": str(raw.get("title") or "").strip(),
                "authors": str(raw.get("authors") or "").strip(),
                "journal": str(raw.get("journal") or "").strip(),
                "catalog": str(raw.get("catalog") or "").strip(),
                "date": str(raw.get("date") or "").strip(),
                "doi": str(raw.get("doi") or "").strip(),
                "abstract": abstract[:4500],
                "base_score": float(raw.get("score", 0.0) or 0.0),
                "profile_match": str(raw.get("profile_match") or "").strip(),
                "intersection_match": str(raw.get("intersection_match") or "").strip(),
                "matched_keywords": str(raw.get("matched_keywords") or "").strip(),
            }
        )
    return papers


def _extract_output_text(data: dict) -> str:
    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    chunks: List[str] = []
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def analyze_papers_with_llm(
    papers: List[dict],
    profile: dict,
    model: Optional[str] = None,
    timeout: int = 120,
) -> List[dict]:
    if not papers:
        return []
    model = (model or configured_model()).strip()

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "papers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "priority": {"type": "string", "enum": ["必读", "高", "中", "低"]},
                        "matched_themes": {"type": "array", "items": {"type": "string"}},
                        "chinese_summary": {"type": "string"},
                        "recommendation_reason": {"type": "string"},
                        "research_connection": {"type": "string"},
                        "novelty_or_value": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["高", "中", "低"]},
                    },
                    "required": [
                        "id",
                        "relevance_score",
                        "priority",
                        "matched_themes",
                        "chinese_summary",
                        "recommendation_reason",
                        "research_connection",
                        "novelty_or_value",
                        "confidence",
                    ],
                },
            }
        },
        "required": ["papers"],
    }

    instructions = (
        "你是一名管理科学、决策分析与人工智能领域的资深学术文献筛选助手。"
        "请严格根据给定研究兴趣画像和论文元数据判断相关性，不要因为论文来自高水平期刊就夸大相关性。"
        "重点识别：群体决策与共识、语言决策、多准则决策与分类/排序、偏好学习与数据驱动决策、"
        "粒计算/粒球计算、个性化语义、社会网络与意见演化、双边匹配，以及这些方向与AI方法的交叉。"
        "如果摘要缺失，只能基于标题、期刊和已有匹配信号谨慎判断，并降低confidence。"
        "中文摘要控制在80-160字，推荐理由控制在60-140字；research_connection说明它与用户已有研究主线可能如何衔接，"
        "novelty_or_value说明值得阅读的潜在方法、问题或应用价值。不要编造论文中没有提供的信息。"
    )

    input_payload = {
        "research_profile": compact_profile(profile),
        "papers": papers,
    }

    payload = {
        "model": model,
        "instructions": instructions,
        "input": json.dumps(input_payload, ensure_ascii=False),
        "reasoning": {"effort": "low"},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "paper_recommendation_analysis",
                "schema": schema,
                "strict": True,
            },
        },
        "max_output_tokens": 6500,
        "store": False,
        "truncation": "auto",
    }

    response = requests.post(
        f"{OPENAI_BASE_URL}/responses",
        headers=_headers(),
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        detail = response.text[:1000]
        raise RuntimeError(f"OpenAI API request failed ({response.status_code}): {detail}")
    data = response.json()
    output_text = _extract_output_text(data)
    if not output_text:
        raise RuntimeError("OpenAI API returned no output text")
    parsed = json.loads(output_text)
    return parsed.get("papers") or []


def merge_ai_analysis(
    ranked_df: pd.DataFrame,
    ai_results: List[dict],
    analyzed_count: int,
    ai_weight: float = 0.35,
) -> pd.DataFrame:
    if ranked_df.empty or not ai_results:
        return ranked_df.copy()

    out = ranked_df.copy().reset_index(drop=True)
    ai_map = {str(x.get("id", "")): x for x in ai_results if x.get("id")}
    top_n = min(max(1, int(analyzed_count)), len(out))

    ids: List[str] = []
    for pos in range(top_n):
        ids.append(_paper_key(out.iloc[pos].to_dict(), pos))

    ai_columns = {
        "ai_relevance_score": None,
        "ai_priority": "",
        "ai_matched_themes": "",
        "ai_summary": "",
        "ai_recommendation_reason": "",
        "ai_research_connection": "",
        "ai_novelty_or_value": "",
        "ai_confidence": "",
        "hybrid_score": None,
    }
    for col, default in ai_columns.items():
        out[col] = default

    base_values = out.loc[: top_n - 1, "score"].astype(float)
    bmin = float(base_values.min())
    bmax = float(base_values.max())
    if bmax > bmin:
        base_norm = (base_values - bmin) / (bmax - bmin) * 100.0
    else:
        base_norm = pd.Series([70.0] * top_n, index=base_values.index)

    weight = min(0.75, max(0.0, float(ai_weight)))
    analyzed_positions: List[int] = []
    for pos, paper_id in enumerate(ids):
        item = ai_map.get(paper_id)
        if not item:
            continue
        ai_score = float(item.get("relevance_score", 0) or 0)
        out.at[pos, "ai_relevance_score"] = ai_score
        out.at[pos, "ai_priority"] = str(item.get("priority") or "")
        out.at[pos, "ai_matched_themes"] = "、".join(item.get("matched_themes") or [])
        out.at[pos, "ai_summary"] = str(item.get("chinese_summary") or "")
        out.at[pos, "ai_recommendation_reason"] = str(item.get("recommendation_reason") or "")
        out.at[pos, "ai_research_connection"] = str(item.get("research_connection") or "")
        out.at[pos, "ai_novelty_or_value"] = str(item.get("novelty_or_value") or "")
        out.at[pos, "ai_confidence"] = str(item.get("confidence") or "")
        out.at[pos, "hybrid_score"] = round((1.0 - weight) * float(base_norm.iloc[pos]) + weight * ai_score, 3)
        analyzed_positions.append(pos)

    if not analyzed_positions:
        return out

    analyzed = out.iloc[:top_n].copy()
    analyzed["_hybrid_sort"] = pd.to_numeric(analyzed["hybrid_score"], errors="coerce").fillna(-1)
    analyzed = analyzed.sort_values(["_hybrid_sort", "score"], ascending=[False, False]).drop(columns=["_hybrid_sort"])
    remaining = out.iloc[top_n:].copy()
    return pd.concat([analyzed, remaining], ignore_index=True)
