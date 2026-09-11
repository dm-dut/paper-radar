from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Optional

import pandas as pd
import requests

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def configured_provider() -> str:
    """Return the configured LLM provider.

    Priority: explicit LLM_PROVIDER -> provider-specific keys/settings.
    Generic mode is intended for school/lab OpenAI-compatible gateways.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("DEEPSEEK_API_KEY", "").strip():
        return "deepseek"
    if os.getenv("OPENAI_API_KEY", "").strip():
        legacy_base = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).lower()
        return "deepseek" if "deepseek" in legacy_base else "openai"
    if os.getenv("LLM_BASE_URL", "").strip() and os.getenv("LLM_MODEL", "").strip():
        return "generic"
    return "none"


def _provider_config() -> dict:
    provider = configured_provider()
    if provider == "deepseek":
        api_key = (
            os.getenv("DEEPSEEK_API_KEY", "").strip()
            or os.getenv("LLM_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        base_url = (
            os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or os.getenv("LLM_BASE_URL", "").strip()
            or DEFAULT_DEEPSEEK_BASE_URL
        )
        model = (
            os.getenv("DEEPSEEK_MODEL", "").strip()
            or os.getenv("LLM_MODEL", "").strip()
            or DEFAULT_DEEPSEEK_MODEL
        )
        api_mode = (
            os.getenv("DEEPSEEK_API_MODE", "").strip().lower()
            or os.getenv("LLM_API_MODE", "").strip().lower()
            or "chat_completions"
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        base_url = (
            os.getenv("OPENAI_BASE_URL", "").strip()
            or os.getenv("LLM_BASE_URL", "").strip()
            or DEFAULT_OPENAI_BASE_URL
        )
        model = (
            os.getenv("OPENAI_MODEL", "").strip()
            or os.getenv("LLM_MODEL", "").strip()
            or DEFAULT_OPENAI_MODEL
        )
        api_mode = os.getenv("LLM_API_MODE", "responses").strip().lower() or "responses"
    elif provider == "generic":
        api_key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        api_mode = os.getenv("LLM_API_MODE", "chat_completions").strip().lower() or "chat_completions"
    else:
        api_key = ""
        base_url = ""
        model = ""
        api_mode = ""

    aliases = {
        "chat": "chat_completions",
        "chat-completions": "chat_completions",
        "chat/completions": "chat_completions",
        "completions": "chat_completions",
        "response": "responses",
    }
    api_mode = aliases.get(api_mode, api_mode)
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_mode": api_mode,
    }


def llm_configured() -> bool:
    cfg = _provider_config()
    if not cfg["base_url"] or not cfg["model"]:
        return False
    # Generic lab gateways may authenticate by campus network/IP and not use a token.
    if cfg["provider"] == "generic":
        return True
    return bool(cfg["api_key"])


def openai_configured() -> bool:
    """Backward-compatible alias used by the current Streamlit UI."""
    return llm_configured()


def configured_model() -> str:
    return _provider_config()["model"]


def configured_api_mode() -> str:
    return _provider_config()["api_mode"]


def configured_provider_name() -> str:
    provider = configured_provider()
    return {
        "deepseek": "DeepSeek",
        "openai": "OpenAI",
        "generic": "学校/实验室模型服务",
        "none": "未配置",
    }.get(provider, provider or "未配置")


def model_options() -> List[str]:
    provider = configured_provider()
    current = configured_model()
    if provider == "deepseek":
        options = [current, "deepseek-chat", "deepseek-reasoner"]
    elif provider == "openai":
        options = [current, "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    else:
        options = [current] if current else []
    result: List[str] = []
    for item in options:
        if item and item not in result:
            result.append(item)
    return result


def _headers() -> dict:
    cfg = _provider_config()
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    return headers


def _endpoint(base_url: str, api_mode: str) -> str:
    base = base_url.rstrip("/")
    if api_mode == "responses":
        return base if base.endswith("/responses") else f"{base}/responses"
    if api_mode == "chat_completions":
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    raise RuntimeError(f"Unsupported LLM_API_MODE: {api_mode}")


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


def _analysis_schema() -> dict:
    return {
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


def _instructions() -> str:
    return (
        "你是一名管理科学、决策分析与人工智能领域的资深学术文献筛选助手。"
        "请严格根据给定研究兴趣画像和论文元数据判断相关性，不要因为论文来自高水平期刊就夸大相关性。"
        "重点识别：群体决策与共识、语言决策、多准则决策与分类/排序、偏好学习与数据驱动决策、"
        "粒计算/粒球计算、个性化语义、社会网络与意见演化、双边匹配，以及这些方向与AI方法的交叉。"
        "如果摘要缺失，只能基于标题、期刊和已有匹配信号谨慎判断，并降低confidence。"
        "中文摘要控制在80-160字，推荐理由控制在60-140字；research_connection说明它与用户已有研究主线可能如何衔接，"
        "novelty_or_value说明值得阅读的潜在方法、问题或应用价值。不要编造论文中没有提供的信息。"
        "必须只输出一个合法JSON对象，不要输出Markdown代码块或JSON之外的解释。"
    )


def _extract_responses_text(data: dict) -> str:
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


def _extract_chat_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("text"):
                chunks.append(str(item.get("text")))
        return "\n".join(chunks).strip()
    return ""


def _parse_json_object(text: str) -> dict:
    if not text:
        raise RuntimeError("模型未返回文本")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _call_responses_api(cfg: dict, model: str, input_payload: dict, timeout: int) -> dict:
    output_format = {
        "type": "json_schema",
        "name": "paper_recommendation_analysis",
        "schema": _analysis_schema(),
    }
    if cfg["provider"] == "openai":
        output_format["strict"] = True
    payload = {
        "model": model,
        "instructions": _instructions(),
        "input": json.dumps(input_payload, ensure_ascii=False),
        "text": {"format": output_format},
        "max_output_tokens": 6500,
    }
    if cfg["provider"] == "openai":
        payload["reasoning"] = {"effort": "low"}
        payload["store"] = False
        payload["truncation"] = "auto"
    response = requests.post(
        _endpoint(cfg["base_url"], "responses"), headers=_headers(), json=payload, timeout=timeout
    )
    if not response.ok:
        raise RuntimeError(
            f"{configured_provider_name()} Responses API request failed "
            f"({response.status_code}): {response.text[:1200]}"
        )
    return _parse_json_object(_extract_responses_text(response.json()))


def _call_chat_completions(cfg: dict, model: str, input_payload: dict, timeout: int) -> dict:
    user_prompt = (
        "请根据下面的研究兴趣画像分析每篇论文。输出必须是JSON对象，顶层只有papers数组。"
        "每个数组元素必须包含：id, relevance_score(0-100整数), priority(必读/高/中/低), "
        "matched_themes(字符串数组), chinese_summary, recommendation_reason, research_connection, "
        "novelty_or_value, confidence(高/中/低)。\n\n"
        + json.dumps(input_payload, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _instructions()},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 6500,
    }
    json_mode = os.getenv("LLM_JSON_MODE", "auto").strip().lower() or "auto"
    if json_mode in {"auto", "json_object", "on", "true", "1"}:
        payload["response_format"] = {"type": "json_object"}

    endpoint = _endpoint(cfg["base_url"], "chat_completions")
    response = requests.post(endpoint, headers=_headers(), json=payload, timeout=timeout)
    if (
        not response.ok
        and "response_format" in payload
        and json_mode == "auto"
        and response.status_code in {400, 404, 415, 422}
    ):
        payload.pop("response_format", None)
        response = requests.post(endpoint, headers=_headers(), json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(
            f"{configured_provider_name()} Chat Completions request failed "
            f"({response.status_code}): {response.text[:1200]}"
        )
    return _parse_json_object(_extract_chat_text(response.json()))


def analyze_papers_with_llm(
    papers: List[dict],
    profile: dict,
    model: Optional[str] = None,
    timeout: int = 180,
) -> List[dict]:
    if not papers:
        return []
    cfg = _provider_config()
    if not llm_configured():
        raise RuntimeError("大模型 API 尚未配置")
    model = (model or cfg["model"]).strip()
    input_payload = {"research_profile": compact_profile(profile), "papers": papers}

    if cfg["api_mode"] == "responses":
        parsed = _call_responses_api(cfg, model, input_payload, timeout)
    elif cfg["api_mode"] == "chat_completions":
        parsed = _call_chat_completions(cfg, model, input_payload, timeout)
    else:
        raise RuntimeError(f"不支持的 LLM_API_MODE：{cfg['api_mode']}")
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
    ids = [_paper_key(out.iloc[pos].to_dict(), pos) for pos in range(top_n)]

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
        out.at[pos, "hybrid_score"] = round(
            (1.0 - weight) * float(base_norm.iloc[pos]) + weight * ai_score, 3
        )
        analyzed_positions.append(pos)

    if not analyzed_positions:
        return out
    analyzed = out.iloc[:top_n].copy()
    analyzed["_hybrid_sort"] = pd.to_numeric(analyzed["hybrid_score"], errors="coerce").fillna(-1)
    analyzed = analyzed.sort_values(["_hybrid_sort", "score"], ascending=[False, False]).drop(columns=["_hybrid_sort"])
    remaining = out.iloc[top_n:].copy()
    return pd.concat([analyzed, remaining], ignore_index=True)
