from __future__ import annotations

import html
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from abstract_enrichment import enrich_missing_abstracts
from core import fetch_catalog, parse_keywords, profile_keywords, profile_keywords_text, rank_papers, safe_text
from llm_service_v2 import (
    analyze_papers_with_llm,
    configured_model,
    configured_provider_name,
    merge_ai_analysis,
    model_options,
    openai_configured,
    prepare_papers,
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CATALOG = APP_DIR / "journal_catalog.sample.csv"
PROFILE_PATH = APP_DIR / "config" / "research_profile.json"


def _esc(value) -> str:
    return html.escape(safe_text(value), quote=True)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch(catalog_json: str, days: int, rows_per_journal: int, mailto: str, enrich_abstracts: bool):
    catalog = pd.read_json(catalog_json, orient="split")
    raw, errors = fetch_catalog(catalog, days=days, rows_per_journal=rows_per_journal, mailto=mailto)
    stats = {"before": 0, "after": 0, "added": 0, "looked_up": 0}
    if raw.empty:
        return raw, errors, stats
    if enrich_abstracts:
        raw, stats = enrich_missing_abstracts(raw)
    else:
        raw = raw.copy()
        raw["abstract_source"] = raw["abstract"].fillna("").astype(str).map(lambda x: "Crossref" if x.strip() else "未获取")
        count = int(raw["abstract"].fillna("").astype(str).str.strip().astype(bool).sum())
        stats = {"before": count, "after": count, "added": 0, "looked_up": 0}
    return raw, errors, stats


@st.cache_data(ttl=86400, show_spinner=False)
def cached_ai_analyze(papers_json: str, profile_json: str, model: str):
    return analyze_papers_with_llm(json.loads(papers_json), json.loads(profile_json), model=model)


def load_default_catalog() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CATALOG)


def read_uploaded_catalog(uploaded) -> pd.DataFrame:
    data = uploaded.getvalue()
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法识别 CSV 编码：{last_error}")


def load_profile() -> dict:
    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"profile_name": "默认研究兴趣画像", "description": "", "themes": [], "relations": [], "negative_keywords": [], "settings": {}}


def render_card(i: int, row: pd.Series):
    title = _esc(row.get("title"))
    authors = _esc(row.get("authors")) or "作者信息暂缺"
    journal = _esc(row.get("journal")) or "期刊信息暂缺"
    pubdate = _esc(row.get("date")) or "日期暂缺"
    catalog = _esc(row.get("catalog"))
    doi = _esc(row.get("doi"))
    url = safe_text(row.get("url"))

    score = float(row.get("score", 0.0) or 0.0)
    semantic_score = float(row.get("semantic_score", 0.0) or 0.0)
    intersection_score = float(row.get("intersection_score", 0.0) or 0.0)
    rec_level = _esc(row.get("recommendation_level")) or "推荐"

    ai_score = None
    try:
        if pd.notna(row.get("ai_relevance_score")):
            ai_score = float(row.get("ai_relevance_score"))
    except Exception:
        pass
    ai_priority = _esc(row.get("ai_priority"))

    badges = [f'<span class="score-badge">规则 {score:.1f}</span>']
    if semantic_score > 0:
        badges.append(f'<span class="semantic-badge">语义 {semantic_score:.1f}</span>')
    if intersection_score > 0:
        badges.append(f'<span class="intersection-badge">交叉 {intersection_score:.1f}</span>')
    badges.append(f'<span class="level-badge">{rec_level}</span>')
    if ai_score is not None:
        badges.append(f'<span class="ai-badge">AI相关度 {ai_score:.0f}</span>')
    if ai_priority:
        badges.append(f'<span class="priority-badge">{ai_priority}</span>')

    st.markdown(
        f"""
<div class="paper-card">
  <div class="paper-title">{i}. {title}{''.join(badges)}</div>
  <div class="paper-meta">✍️ {authors} &nbsp;|&nbsp; 📖 {journal}{(' &nbsp;|&nbsp; ' + catalog) if catalog else ''}<br>
  📅 {pubdate}{(' &nbsp;|&nbsp; DOI: ' + doi) if doi else ''}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    abstract = safe_text(row.get("abstract")).strip()
    abstract_source = safe_text(row.get("abstract_source")).strip() or "未知来源"
    if abstract:
        st.markdown(f"**摘要 · {abstract_source}：** {abstract}")
    else:
        st.caption("暂未从 Crossref 或 OpenAlex 获取到摘要。AI 判断将主要依据标题和元数据，置信度会相应降低。")

    ai_summary = safe_text(row.get("ai_summary")).strip()
    if ai_summary:
        st.markdown(f"**🤖 AI 中文摘要：** {ai_summary}")

    profile_match = safe_text(row.get("profile_match")).strip()
    intersection_match = safe_text(row.get("intersection_match")).strip()
    if profile_match:
        st.caption(f"研究画像匹配：{profile_match}")
    if intersection_match:
        st.caption(f"交叉研究信号：{intersection_match}")

    reason = safe_text(row.get("recommendation_reason")).strip()
    if reason:
        st.markdown(f"**🎯 规则推荐理由：** {reason}")

    ai_reason = safe_text(row.get("ai_recommendation_reason")).strip()
    ai_connection = safe_text(row.get("ai_research_connection")).strip()
    ai_value = safe_text(row.get("ai_novelty_or_value")).strip()
    ai_confidence = safe_text(row.get("ai_confidence")).strip()
    if ai_reason:
        st.markdown(f"**🧠 AI 推荐判断：** {ai_reason}")
    if ai_connection:
        st.markdown(f"**与你研究的衔接：** {ai_connection}")
    if ai_value:
        st.markdown(f"**潜在价值：** {ai_value}")
    if ai_confidence:
        st.caption(f"AI 判断置信度：{ai_confidence}")

    if url:
        st.link_button("打开论文页面", url, use_container_width=False)
    st.markdown("---")


def run():
    st.set_page_config(page_title="Paper Radar · 论文推荐", page_icon="📚", layout="wide")
    st.markdown(
        """
<style>
.block-container {max-width:1220px;padding-top:2rem;padding-bottom:4rem;}
.paper-card {padding:1rem 1.2rem .75rem;border:1px solid #e7ecef;border-radius:14px;background:#fff;box-shadow:0 2px 10px rgba(31,45,61,.045);margin-bottom:.8rem;}
.paper-title {font-size:1.18rem;font-weight:750;color:#26384d;line-height:1.5;margin-bottom:.45rem;}
.paper-meta {font-size:.92rem;color:#71808a;line-height:1.75;}
.score-badge,.semantic-badge,.intersection-badge,.level-badge,.ai-badge,.priority-badge {display:inline-block;padding:.13rem .5rem;border-radius:999px;font-size:.78rem;font-weight:700;margin-left:.32rem;}
.score-badge {background:#eef7f2;color:#16784b}.semantic-badge{background:#f4f1fb;color:#67538c}.intersection-badge{background:#fff4e8;color:#9a6418}.level-badge{background:#eef3f7;color:#465c6c}.ai-badge{background:#edf3ff;color:#315f9b}.priority-badge{background:#fff0f0;color:#a34343}
div[data-testid="stSidebar"] {background:#f8fafb;}
</style>
""",
        unsafe_allow_html=True,
    )

    profile = load_profile()
    settings = profile.get("settings") or {}
    profile_kws = profile_keywords(profile)
    tier_labels = profile.get("tier_labels") or {}
    ai_ready = openai_configured()

    st.sidebar.title("⚙️ 推荐设置")
    st.sidebar.caption("高精度模式：核心研究画像召回 → 摘要补全 → AI严格筛选。")

    with st.sidebar.expander("🧭 我的研究兴趣画像", expanded=True):
        st.markdown(f"**{profile.get('profile_name', '研究兴趣画像')}**")
        st.caption(profile.get("description", ""))
        grouped = defaultdict(list)
        for theme in profile.get("themes") or []:
            grouped[str(theme.get("tier", "cross"))].append(theme)
        for tier in ["core", "emerging", "cross", "method", "application"]:
            themes = grouped.get(tier) or []
            if themes:
                st.markdown(f"**{tier_labels.get(tier, tier)}**")
                for theme in themes:
                    st.markdown(f"• {theme.get('name','')}")
        st.caption(f"当前用于主动召回的加权关键词：{len(profile_kws)} 个。方法/应用层主题仅作为上下文，不再单独召回论文。")

    use_profile = st.sidebar.toggle("使用研究兴趣画像", value=True)
    use_semantic = st.sidebar.toggle("启用本地语义匹配", value=True)
    default_semantic = max(0.09, float(settings.get("semantic_threshold", 0.09) or 0.09))
    semantic_threshold = st.sidebar.slider("本地语义最低相似度", 0.05, 0.20, min(default_semantic, 0.20), 0.005, help="数值越高越严格。已提高默认阈值，以减少弱相关论文。")
    enrich_abstracts = st.sidebar.toggle("用 OpenAlex 补全缺失摘要", value=True, help="Crossref 没有摘要时，根据 DOI 批量从 OpenAlex 补全。")

    uploaded = st.sidebar.file_uploader("期刊目录 CSV", type=["csv"])
    if uploaded is not None:
        try:
            catalog_df = read_uploaded_catalog(uploaded)
        except Exception as exc:
            st.sidebar.error(f"CSV读取失败：{exc}")
            catalog_df = load_default_catalog()
    else:
        catalog_df = load_default_catalog()

    if not {"journal", "issn"}.issubset(catalog_df.columns):
        st.sidebar.error("期刊目录至少需要 journal 和 issn 两列，已退回默认目录。")
        catalog_df = load_default_catalog()
    if "catalog" not in catalog_df.columns:
        catalog_df["catalog"] = "自定义目录"
    if "weight" not in catalog_df.columns:
        catalog_df["weight"] = 1.0
    catalog_df["catalog"] = catalog_df["catalog"].fillna("未分类").astype(str)

    available_catalogs = catalog_df["catalog"].drop_duplicates().tolist()
    selected_source_catalogs = st.sidebar.multiselect("本次监测目录", available_catalogs, default=available_catalogs)
    working_catalog_df = catalog_df[catalog_df["catalog"].isin(selected_source_catalogs)].copy()

    with st.sidebar.expander(f"目标期刊（{len(working_catalog_df)}/{len(catalog_df)}）", expanded=False):
        if not working_catalog_df.empty:
            st.dataframe(working_catalog_df[["journal", "issn", "catalog", "weight"]], hide_index=True, use_container_width=True)

    extra_positive_text = st.sidebar.text_area("临时增加关注关键词（可选）", value="", height=90, help="每行一个，可写 keyword|权重。")
    negative_text = st.sidebar.text_area("降权关键词（可选）", value="", height=70)

    days = st.sidebar.slider("检索最近多少天", 7, 180, max(7, min(180, int(settings.get("default_days", 60) or 60))), 1)
    rows_per_journal = st.sidebar.slider("每刊最多获取论文数", 5, 100, max(5, min(100, int(settings.get("rows_per_journal", 20) or 20))), 5)
    max_results = st.sidebar.slider("首页最多展示", 5, 100, 30, 5)
    min_score = st.sidebar.number_input("最低规则推荐分", min_value=-20.0, max_value=100.0, value=max(2.0, float(settings.get("default_min_score", 1.0) or 1.0)), step=0.5)
    mailto = st.sidebar.text_input("Crossref 联系邮箱（可选）", value="", placeholder="name@university.edu")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI 严格筛选")
    if ai_ready:
        st.sidebar.success(f"{configured_provider_name()} 已配置")
    else:
        st.sidebar.info("请在 Streamlit Secrets 中配置大模型 API。")
    use_ai = st.sidebar.toggle("启用 AI 深度分析", value=ai_ready, disabled=not ai_ready)
    options = model_options() or [configured_model() or "gpt-5.6-terra"]
    default_model = configured_model() or options[0]
    if default_model not in options:
        options.insert(0, default_model)
    ai_model = st.sidebar.selectbox("AI 模型", options, index=options.index(default_model), disabled=not use_ai)
    ai_candidate_count = st.sidebar.slider("AI 深度分析候选数", 5, 20, max(5, min(20, int(settings.get("ai_candidate_count", 15) or 15))), 1, disabled=not use_ai)
    ai_weight = st.sidebar.slider("AI 重排序权重", 0.10, 0.70, 0.45, 0.05, disabled=not use_ai)
    ai_min_relevance = st.sidebar.slider("AI最低相关度", 0, 100, 45, 5, disabled=not use_ai, help="低于该分数的AI已分析论文默认隐藏。45以下视为不相关。")
    strict_ai_filter = st.sidebar.toggle("只显示通过AI筛选的论文", value=True, disabled=not use_ai, help="开启后，只显示AI已分析且达到最低相关度的论文。")

    fetch_now = st.sidebar.button("🔄 获取并生成推荐", type="primary", use_container_width=True)

    st.title("📚 Paper Radar · 论文推荐")
    st.caption("高精度推荐：核心兴趣召回 + OpenAlex摘要补全 + GPT严格相关性判别。")

    active_positive = []
    if use_profile:
        active_positive.extend(profile_kws)
    active_positive.extend(parse_keywords(extra_positive_text))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("目标期刊", len(working_catalog_df))
    c2.metric("目录", len(selected_source_catalogs))
    c3.metric("主动画像关键词", len(profile_kws) if use_profile else 0)
    c4.metric("语义阈值", f"{semantic_threshold:.3f}" if use_semantic else "关闭")
    c5.metric("摘要补全", "开启" if enrich_abstracts else "关闭")
    c6.metric("AI筛选", "开启" if use_ai else "关闭")

    if use_profile:
        with st.expander("查看主动召回关键词", expanded=False):
            st.code(profile_keywords_text(profile), language="text")

    if "results" not in st.session_state:
        st.session_state.results = pd.DataFrame()
        st.session_state.errors = []
        st.session_state.ai_error = ""
        st.session_state.last_run = None
        st.session_state.ai_model_used = ""
        st.session_state.abstract_stats = {"before": 0, "after": 0, "added": 0, "looked_up": 0}

    if fetch_now:
        negative = parse_keywords(negative_text)
        if working_catalog_df.empty:
            st.warning("请至少选择一个监测目录。")
        elif not active_positive:
            st.warning("请启用研究兴趣画像，或至少填写一个临时关注关键词。")
        else:
            with st.spinner("第一阶段：获取论文、补全摘要并计算高精度规则/语义匹配…"):
                raw_df, errors, abstract_stats = cached_fetch(
                    working_catalog_df.to_json(orient="split"), days, rows_per_journal, mailto, enrich_abstracts
                )
                ranked = rank_papers(
                    raw_df,
                    active_positive,
                    negative,
                    recency_half_life=float(settings.get("recency_half_life_days", 30) or 30),
                    min_score=min_score,
                    profile=profile if use_profile else None,
                    use_semantic=bool(use_profile and use_semantic),
                    semantic_weight=min(8.0, float(settings.get("semantic_weight", 8.0) or 8.0)),
                    semantic_threshold=semantic_threshold,
                    intersection_weight=float(settings.get("intersection_weight", 2.4) or 2.4),
                    cross_theme_bonus=float(settings.get("cross_theme_bonus", 0.8) or 0.8),
                )

            ai_error = ""
            ai_model_used = ""
            if use_ai and not ranked.empty:
                with st.spinner(f"第二阶段：{ai_model} 正在严格筛选前 {min(ai_candidate_count, len(ranked))} 篇候选论文…"):
                    try:
                        papers = prepare_papers(ranked, limit=ai_candidate_count)
                        ai_results = cached_ai_analyze(
                            json.dumps(papers, ensure_ascii=False, sort_keys=True),
                            json.dumps(profile, ensure_ascii=False, sort_keys=True),
                            ai_model,
                        )
                        ranked = merge_ai_analysis(ranked, ai_results, analyzed_count=ai_candidate_count, ai_weight=ai_weight)
                        ai_model_used = ai_model
                    except Exception as exc:
                        ai_error = str(exc)

            st.session_state.results = ranked
            st.session_state.errors = errors
            st.session_state.ai_error = ai_error
            st.session_state.ai_model_used = ai_model_used
            st.session_state.abstract_stats = abstract_stats
            st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")

    results = st.session_state.results
    errors = st.session_state.errors
    stats = st.session_state.abstract_stats
    if st.session_state.last_run:
        ai_note = f" · AI: {st.session_state.ai_model_used}" if st.session_state.ai_model_used else ""
        st.caption(
            f"最近刷新：{st.session_state.last_run}{ai_note} · 摘要覆盖 {stats.get('after',0)} 篇，OpenAlex 新增 {stats.get('added',0)} 篇。"
        )
    if errors:
        with st.expander(f"⚠️ {len(errors)} 个期刊请求未成功", expanded=False):
            st.code("\n".join(errors[:40]))
    if st.session_state.ai_error:
        st.warning(f"规则推荐已完成，但 AI 深度分析未成功：{st.session_state.ai_error}")

    if not results.empty:
        view = results.copy()
        catalogs = [x for x in view["catalog"].dropna().astype(str).unique().tolist() if x]
        f1, f2 = st.columns([2, 1])
        selected_catalogs = f1.multiselect("按目录筛选结果", catalogs, default=catalogs)
        keyword_filter = f2.text_input("结果内搜索", placeholder="标题/作者/期刊")
        if selected_catalogs:
            view = view[view["catalog"].astype(str).isin(selected_catalogs)]
        if keyword_filter.strip():
            q = keyword_filter.strip().casefold()
            mask = (
                view["title"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
                | view["authors"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
                | view["journal"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
            )
            view = view[mask]

        screened_out = 0
        if st.session_state.ai_model_used and "ai_relevance_score" in view.columns:
            analyzed_mask = view["ai_relevance_score"].notna()
            if strict_ai_filter:
                screened_out = int((~analyzed_mask | (pd.to_numeric(view["ai_relevance_score"], errors="coerce").fillna(-1) < ai_min_relevance)).sum())
                view = view[analyzed_mask]
                view = view[pd.to_numeric(view["ai_relevance_score"], errors="coerce").fillna(-1) >= ai_min_relevance]

        if view.empty:
            st.info(f"当前没有论文通过 AI ≥ {ai_min_relevance} 的严格筛选。可以降低阈值，或增加 AI 深度分析候选数。")
        else:
            suffix = f" · 已隐藏 {screened_out} 篇低相关/未分析论文" if screened_out else ""
            st.subheader(f"为你推荐 · {min(len(view), max_results)} 篇{suffix}")
            for i, (_, row) in enumerate(view.head(max_results).iterrows(), 1):
                render_card(i, row)

            export_cols = [
                "title", "authors", "journal", "catalog", "date", "doi", "url", "abstract", "abstract_source",
                "score", "keyword_score", "semantic_score", "intersection_score", "semantic_similarity", "profile_match",
                "intersection_match", "recommendation_level", "matched_keywords", "recommendation_reason", "hybrid_score",
                "ai_relevance_score", "ai_priority", "ai_matched_themes", "ai_summary", "ai_recommendation_reason",
                "ai_research_connection", "ai_novelty_or_value", "ai_confidence",
            ]
            csv_bytes = view[[c for c in export_cols if c in view.columns]].to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 导出当前推荐 CSV", csv_bytes, file_name="paper_recommendations.csv", mime="text/csv")
    else:
        st.info("点击左侧“获取并生成推荐”开始。")
        st.markdown(
            "系统现在采用三步流程：**核心研究画像高精度召回 → Crossref/OpenAlex 摘要补全 → GPT 严格相关性筛选**。"
            "宽泛的机器学习、预测、医疗等方法/应用词不会再单独召回论文。"
        )
