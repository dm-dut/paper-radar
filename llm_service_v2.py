from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Optional

import pandas as pd
import requests

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
PRIMARY_TIERS = {"core", "emerging", "cross"}


def configured_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.getenv("LLM_BASE_URL", "").strip() and os.getenv("LLM_MODEL", "").strip():
        return "generic"
    return "none"


def _provider_config() -> dict:
    provider = configured_provider()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_OPENAI_BASE_URL
        model = os.getenv("OPENAI_MODEL", "").strip() or os.getenv("LLM_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
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
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_mode": aliases.get(api_mode, api_mode),
    }


def llm_configured() -> bool:
    cfg = _provider_config()
    return bool(cfg["base_url"] and cfg["model"] and cfg["api_key"])


def openai_configured() -> bool:
    return llm_configured()


def configured_model() -> str:
    return _provider_config()["model"]


def configured_provider_name() -> str:
    provider = configured_provider()
    return {"openai": "GPT / OpenAI兼容服务", "generic": "兼容模型服务", "none": "未配置"}.get(provider, provider or "未配置")


def model_options() -> List[str]:
    current = configured_model()
    provider = configured_provider()
    options = [current]
    if provider == "openai":
        options += ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    out = []
    for item in options:
        if item and item not in out:
            out.append(item)
    return out


def _headers() -> dict:
    cfg = _provider_config()
    return {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}


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
    basis = "|".join([str(row.get("title") or "").strip(), str(row.get("journal") or "").strip(), str(row.get("date") or "").strip(), str(position)])
    return f"paper:{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16]}"


def compact_profile(profile: dict) -> dict:
    themes = []
    for theme in profile.get("themes") or []:
        tier = str(theme.get("tier", "cross"))
        kws = []
        for item in theme.get("keywords") or []:
            term = str(item if isinstance(item, str) else item.get("term", "")).strip()
            if term:
                kws.append(term)
        themes.append({
            "id": theme.get("id", ""),
            "name": theme.get("name", ""),
            "tier": tier,
            "role": "primary_research_interest" if tier in PRIMARY_TIERS else "context_only",
            "description": theme.get("description", ""),
            "semantic_queries": (theme.get("semantic_queries") or [])[:5],
            "keywords": kws[:16],
        })
    return {
        "profile_name": profile.get("profile_name", "研究兴趣画像"),
        "description": profile.get("description", ""),
        "scoring_policy": {
            "primary_tiers": ["core", "emerging", "cross"],
            "context_only_tiers": ["method", "application"],
        },
        "themes": themes,
    }


def prepare_papers(df: pd.DataFrame, limit: int = 10) -> List[dict]:
    papers = []
    for pos, (_, row) in enumerate(df.head(max(1, int(limit))).iterrows()):
        raw = row.to_dict()
        papers.append({
            "id": _paper_key(raw, pos),
            "title": str(raw.get("title") or "").strip(),
            "authors": str(raw.get("authors") or "").strip(),
            "journal": str(raw.get("journal") or "").strip(),
            "date": str(raw.get("date") or "").strip(),
            "doi": str(raw.get("doi") or "").strip(),
            "abstract": str(raw.get("abstract") or "").strip()[:6000],
            "abstract_source": str(raw.get("abstract_source") or "").strip(),
            "base_score": float(raw.get("score", 0.0) or 0.0),
            "profile_match": str(raw.get("profile_match") or "").strip(),
            "intersection_match": str(raw.get("intersection_match") or "").strip(),
            "matched_keywords": str(raw.get("matched_keywords") or "").strip(),
        })
    return papers


def _instructions() -> str:
    return (
        "你是一个严格的学术论文相关性筛选器，而不是泛化的推荐助手。"
        "目标是高精度筛选与用户当前研究主线直接相关的论文，宁可漏掉边缘论文，也不要把泛泛相关的方法论文推荐进来。"
        "研究画像中 tier=core、emerging、cross 且 role=primary_research_interest 的主题才是主要相关性依据；"
        "tier=method 或 application 且 role=context_only 的内容只用于解释交叉点，不能单独证明论文相关。"
        "尤其是 machine learning、neural network、graph neural network、attention、LLM、robust optimization、uncertainty、"
        "forecasting、clustering、recommendation、medical application 等宽泛方法或应用，如果没有同时解决用户的群体决策/共识、"
        "语言决策、多准则决策与分类排序、偏好学习、粒计算、个性化语义、意见演化或双边匹配等主要问题，相关度不得超过40。"
        "相关度评分标准：90-100=研究问题和方法均直接契合；75-89=与主要研究主线高度相关；60-74=有明确可借鉴联系；"
        "45-59=边缘相关，仅适合作为拓展阅读；0-44=不相关或过于宽泛，应被过滤。"
        "若摘要缺失，除非标题明确命中主要研究主题，否则相关度一般不应超过60，并降低confidence。"
        "不要因为期刊水平高、方法流行或使用AI就提高相关度。不要编造摘要中不存在的信息。"
        "中文摘要应概括研究问题、主要方法和结论/价值，80-160字。推荐理由要说明为什么与用户研究直接相关或为什么不够相关。"
        "必须只输出合法JSON对象，不要输出Markdown代码块或额外解释。"
    )


def _schema() -> dict:
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
                    "required": ["id", "relevance_score", "priority", "matched_themes", "chinese_summary", "recommendation_reason", "research_connection", "novelty_or_value", "confidence"],
                },
            }
        },
        "required": ["papers"],
    }


def _extract_json(text: str) -> dict:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise RuntimeError("模型返回内容不是合法JSON")


def _extract_chat(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    content = ((choices[0] or {}).get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x.get("text") or "") for x in content if isinstance(x, dict))
    return ""


def _extract_responses(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks)


def analyze_papers_with_llm(papers: List[dict], profile: dict, model: Optional[str] = None, timeout: int = 180) -> List[dict]:
    if not papers:
        return []
    cfg = _provider_config()
    if not llm_configured():
        raise RuntimeError("大模型 API 尚未配置")
    model = (model or cfg["model"]).strip()
    input_payload = {"research_profile": compact_profile(profile), "papers": papers}

    if cfg["api_mode"] == "chat_completions":
        prompt = "请逐篇进行严格相关性筛选，并按要求返回JSON。\n\n" + json.dumps(input_payload, ensure_ascii=False)
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": _instructions()}, {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 7000,
        }
        json_mode = os.getenv("LLM_JSON_MODE", "auto").strip().lower() or "auto"
        if json_mode in {"auto", "json_object", "on", "true", "1"}:
            payload["response_format"] = {"type": "json_object"}
        endpoint = _endpoint(cfg["base_url"], "chat_completions")
        response = requests.post(endpoint, headers=_headers(), json=payload, timeout=timeout)
        if not response.ok and "response_format" in payload and json_mode == "auto" and response.status_code in {400, 404, 415, 422}:
            payload.pop("response_format", None)
            response = requests.post(endpoint, headers=_headers(), json=payload, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"AI Chat Completions request failed ({response.status_code}): {response.text[:1000]}")
        parsed = _extract_json(_extract_chat(response.json()))
    else:
        fmt = {"type": "json_schema", "name": "paper_recommendation_analysis", "schema": _schema()}
        if cfg["provider"] == "openai":
            fmt["strict"] = True
        payload = {
            "model": model,
            "instructions": _instructions(),
            "input": json.dumps(input_payload, ensure_ascii=False),
            "text": {"format": fmt},
            "max_output_tokens": 7000,
        }
        response = requests.post(_endpoint(cfg["base_url"], "responses"), headers=_headers(), json=payload, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"AI Responses request failed ({response.status_code}): {response.text[:1000]}")
        parsed = _extract_json(_extract_responses(response.json()))
    return parsed.get("papers") or []


def merge_ai_analysis(ranked_df: pd.DataFrame, ai_results: List[dict], analyzed_count: int, ai_weight: float = 0.45) -> pd.DataFrame:
    if ranked_df.empty or not ai_results:
        return ranked_df.copy()
    out = ranked_df.copy().reset_index(drop=True)
    top_n = min(max(1, int(analyzed_count)), len(out))
    ai_map = {str(x.get("id", "")): x for x in ai_results if x.get("id")}
    ids = [_paper_key(out.iloc[pos].to_dict(), pos) for pos in range(top_n)]
    for col, default in {
        "ai_relevance_score": None,
        "ai_priority": "",
        "ai_matched_themes": "",
        "ai_summary": "",
        "ai_recommendation_reason": "",
        "ai_research_connection": "",
        "ai_novelty_or_value": "",
        "ai_confidence": "",
        "hybrid_score": None,
    }.items():
        out[col] = default

    base = out.loc[: top_n - 1, "score"].astype(float)
    if float(base.max()) > float(base.min()):
        base_norm = (base - float(base.min())) / (float(base.max()) - float(base.min())) * 100.0
    else:
        base_norm = pd.Series([70.0] * top_n, index=base.index)
    weight = min(0.75, max(0.0, float(ai_weight)))

    for pos, paper_id in enumerate(ids):
        item = ai_map.get(paper_id)
        if not item:
            continue
        ai_score = float(item.get("relevance_score", 0) or 0)
        priority = "必读" if ai_score >= 90 else "高" if ai_score >= 75 else "中" if ai_score >= 60 else "低"
        out.at[pos, "ai_relevance_score"] = ai_score
        out.at[pos, "ai_priority"] = priority
        out.at[pos, "ai_matched_themes"] = "、".join(item.get("matched_themes") or [])
        out.at[pos, "ai_summary"] = str(item.get("chinese_summary") or "")
        out.at[pos, "ai_recommendation_reason"] = str(item.get("recommendation_reason") or "")
        out.at[pos, "ai_research_connection"] = str(item.get("research_connection") or "")
        out.at[pos, "ai_novelty_or_value"] = str(item.get("novelty_or_value") or "")
        out.at[pos, "ai_confidence"] = str(item.get("confidence") or "")
        out.at[pos, "hybrid_score"] = round((1.0 - weight) * float(base_norm.iloc[pos]) + weight * ai_score, 3)

    analyzed = out.iloc[:top_n].copy()
    analyzed["_sort"] = pd.to_numeric(analyzed["hybrid_score"], errors="coerce").fillna(-1)
    analyzed = analyzed.sort_values(["_sort", "score"], ascending=[False, False]).drop(columns=["_sort"])
    return pd.concat([analyzed, out.iloc[top_n:].copy()], ignore_index=True)
