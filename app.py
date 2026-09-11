from __future__ import annotations

import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core import fetch_catalog, parse_keywords, profile_keywords, profile_keywords_text, rank_papers, safe_text
from llm_service import (
    analyze_papers_with_llm,
    configured_model,
    merge_ai_analysis,
    openai_configured,
    prepare_papers,
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CATALOG = APP_DIR / "journal_catalog.sample.csv"
PROFILE_PATH = APP_DIR / "config" / "research_profile.json"

st.set_page_config(page_title="Paper Radar · 论文推荐", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1220px; padding-top: 2rem; padding-bottom: 4rem;}
.paper-card {padding:1.15rem 1.3rem 1rem;border:1px solid #e7ecef;border-radius:14px;background:#fff;box-shadow:0 2px 10px rgba(31,45,61,.045);margin-bottom:1rem;}
.paper-title {font-size:1.22rem;font-weight:750;color:#26384d;line-height:1.45;margin-bottom:.5rem;}
.paper-meta {font-size:.93rem;color:#7d8b8e;line-height:1.8;margin-bottom:.55rem;}
.paper-abstract {font-size:1.0rem;color:#3d536b;line-height:1.85;margin:.55rem 0;}
.paper-ai-summary {font-size:1.0rem;color:#32465a;line-height:1.85;background:#f7fafc;border-left:3px solid #7a91a8;padding:.65rem .8rem;margin:.7rem 0;border-radius:0 8px 8px 0;}
.paper-reason {font-size:.98rem;color:#08a857;font-weight:650;line-height:1.8;margin-top:.75rem;}
.paper-ai-reason {font-size:.96rem;color:#5b4b84;line-height:1.8;margin-top:.5rem;}
.paper-connection {font-size:.93rem;color:#657786;line-height:1.75;margin-top:.45rem;}
.paper-profile {font-size:.90rem;color:#5a7184;margin-top:.45rem;}
.score-badge,.semantic-badge,.intersection-badge,.level-badge,.theme-badge,.ai-badge,.priority-badge {display:inline-block;padding:.13rem .55rem;border-radius:999px;font-size:.8rem;font-weight:700;margin-left:.35rem;}
.score-badge {background:#eef7f2;color:#16784b;}
.semantic-badge {background:#f4f1fb;color:#67538c;}
.intersection-badge {background:#fff4e8;color:#9a6418;}
.level-badge {background:#eef3f7;color:#465c6c;}
.ai-badge {background:#edf3ff;color:#315f9b;}
.priority-badge {background:#fff0f0;color:#a34343;}
.theme-badge {background:#f3f6f8;color:#536b7b;font-weight:500;margin-left:0;margin-right:.25rem;}
div[data-testid="stSidebar"] {background:#f8fafb;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch(catalog_json: str, days: int, rows_per_journal: int, mailto: str):
    df = pd.read_json(catalog_json, orient="split")
    return fetch_catalog(df, days=days, rows_per_journal=rows_per_journal, mailto=mailto)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_ai_analyze(papers_json: str, profile_json: str, model: str):
    papers = json.loads(papers_json)
    profile = json.loads(profile_json)
    return analyze_papers_with_llm(papers, profile, model=model)


def load_default_catalog() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CATALOG)


def read_uploaded_catalog(uploaded) -> pd.DataFrame:
    data = uploaded.getvalue()
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except Exception as e:
            last_error = e
    raise ValueError(f"无法识别 CSV 编码：{last_error}")


def load_profile() -> dict:
    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "profile_name": "默认研究兴趣画像",
            "description": "当前未能读取研究兴趣配置。",
            "themes": [],
            "relations": [],
            "negative_keywords": [],
            "settings": {},
        }


def html_escape(s: str) -> str:
    import html

    return html.escape(safe_text(s), quote=True)


def render_card(i: int, row: pd.Series):
    title = html_escape(row.get("title"))
    authors = html_escape(row.get("authors"))
    journal = html_escape(row.get("journal"))
    pubdate = html_escape(row.get("date"))
    abstract = safe_text(row.get("abstract"))
    abstract_display = abstract if abstract else "Crossref 暂未提供摘要；当前规则推荐主要依据标题、期刊与发布时间。"
    if len(abstract_display) > 900:
        abstract_display = abstract_display[:900].rstrip() + "…"
    abstract_display = html_escape(abstract_display)

    reason = html_escape(row.get("recommendation_reason"))
    profile_match = safe_text(row.get("profile_match"))
    intersection_match = safe_text(row.get("intersection_match"))
    rec_level = safe_text(row.get("recommendation_level")) or "推荐"
    score = float(row.get("score", 0.0) or 0.0)
    semantic_score = float(row.get("semantic_score", 0.0) or 0.0)
    intersection_score = float(row.get("intersection_score", 0.0) or 0.0)

    ai_score_raw = row.get("ai_relevance_score")
    try:
        ai_score = float(ai_score_raw) if pd.notna(ai_score_raw) else None
    except Exception:
        ai_score = None
    ai_priority = safe_text(row.get("ai_priority"))
    ai_summary = html_escape(row.get("ai_summary"))
    ai_reason = html_escape(row.get("ai_recommendation_reason"))
    ai_connection = html_escape(row.get("ai_research_connection"))
    ai_value = html_escape(row.get("ai_novelty_or_value"))
    ai_confidence = safe_text(row.get("ai_confidence"))

    doi = safe_text(row.get("doi"))
    url = safe_text(row.get("url"))
    catalog = safe_text(row.get("catalog"))
    catalog_piece = f" &nbsp;|&nbsp; {html_escape(catalog)}" if catalog else ""

    theme_html = ""
    if profile_match:
        badges = "".join(
            f'<span class="theme-badge">{html_escape(x.strip())}</span>'
            for x in profile_match.split("、")
            if x.strip()
        )
        theme_html = f'<div class="paper-profile">研究画像匹配：{badges}</div>'

    intersection_html = ""
    if intersection_match:
        ibadges = "".join(
            f'<span class="theme-badge">{html_escape(x.strip())}</span>'
            for x in intersection_match.split("、")
            if x.strip()
        )
        intersection_html = f'<div class="paper-profile">交叉研究信号：{ibadges}</div>'

    sem_badge = f'<span class="semantic-badge">语义 {semantic_score:.1f}</span>' if semantic_score > 0 else ""
    int_badge = f'<span class="intersection-badge">交叉 {intersection_score:.1f}</span>' if intersection_score > 0 else ""
    level_badge = f'<span class="level-badge">{html_escape(rec_level)}</span>'
    ai_badge = f'<span class="ai-badge">AI相关度 {ai_score:.0f}</span>' if ai_score is not None else ""
    priority_badge = f'<span class="priority-badge">AI优先级 {html_escape(ai_priority)}</span>' if ai_priority else ""

    ai_summary_html = f'<div class="paper-ai-summary"><b>🤖 AI中文摘要：</b>{ai_summary}</div>' if ai_summary else ""
    ai_reason_html = f'<div class="paper-ai-reason"><b>🧠 AI推荐判断：</b>{ai_reason}</div>' if ai_reason else ""
    connection_parts = []
    if ai_connection:
        connection_parts.append(f"<b>与你研究的衔接：</b>{ai_connection}")
    if ai_value:
        connection_parts.append(f"<b>潜在价值：</b>{ai_value}")
    if ai_confidence:
        connection_parts.append(f"<b>判断置信度：</b>{html_escape(ai_confidence)}")
    connection_html = (
        '<div class="paper-connection">' + "<br>".join(connection_parts) + "</div>"
        if connection_parts
        else ""
    )

    st.markdown(
        f"""
<div class="paper-card">
  <div class="paper-title">{i}. {title}<span class="score-badge">规则综合 {score:.1f}</span>{sem_badge}{int_badge}{level_badge}{ai_badge}{priority_badge}</div>
  <div class="paper-meta">✍️ {authors or '作者信息暂缺'} &nbsp;|&nbsp; 📖 {journal or '期刊信息暂缺'}{catalog_piece}<br>📅 {pubdate or '日期暂缺'} {('&nbsp;|&nbsp; DOI: ' + html_escape(doi)) if doi else ''}</div>
  {ai_summary_html}
  <div class="paper-abstract"><b>原始摘要：</b>{abstract_display}</div>
  {theme_html}
  {intersection_html}
  <div class="paper-reason">🎯 规则推荐理由：{reason}</div>
  {ai_reason_html}
  {connection_html}
</div>
""",
        unsafe_allow_html=True,
    )
    if url:
        st.link_button("打开论文页面", url, use_container_width=False)


profile = load_profile()
profile_settings = profile.get("settings") or {}
profile_kws = profile_keywords(profile)
tier_labels = profile.get("tier_labels") or {}
ai_ready = openai_configured()

st.sidebar.title("⚙️ 推荐设置")
st.sidebar.caption("系统使用分层研究兴趣画像，并结合关键词、语义、交叉主题关系与可选的大模型深度分析进行推荐。")

with st.sidebar.expander("🧭 我的研究兴趣画像", expanded=True):
    st.markdown(f"**{profile.get('profile_name', '研究兴趣画像')}**")
    st.caption(profile.get("description", ""))
    grouped = defaultdict(list)
    for theme in profile.get("themes") or []:
        grouped[str(theme.get("tier", "cross"))].append(theme)
    for tier in ["core", "emerging", "cross", "method", "application"]:
        themes = grouped.get(tier) or []
        if not themes:
            continue
        st.markdown(f"**{tier_labels.get(tier, tier)}**")
        for theme in themes:
            st.markdown(f"• {theme.get('name','')} · 权重 {float(theme.get('weight',1.0) or 1.0):g}")
    st.caption(
        f"当前共 {len(profile.get('themes') or [])} 个主题、{len(profile.get('relations') or [])} 条主题关联、{len(profile_kws)} 个加权关键词。"
    )

use_profile = st.sidebar.toggle("使用研究兴趣画像", value=True)
use_semantic = st.sidebar.toggle("启用本地语义匹配", value=True, help="识别概念相近但措辞不同的论文。")

uploaded = st.sidebar.file_uploader("期刊目录 CSV", type=["csv"], help="列：journal, issn, catalog, weight")
if uploaded is not None:
    try:
        catalog_df = read_uploaded_catalog(uploaded)
    except Exception as e:
        st.sidebar.error(f"CSV 读取失败：{e}")
        catalog_df = load_default_catalog()
else:
    catalog_df = load_default_catalog()

required = {"journal", "issn"}
if not required.issubset(set(catalog_df.columns)):
    st.sidebar.error("期刊目录至少需要 journal 和 issn 两列。已退回默认目录。")
    catalog_df = load_default_catalog()
if "catalog" not in catalog_df.columns:
    catalog_df["catalog"] = "自定义目录"
if "weight" not in catalog_df.columns:
    catalog_df["weight"] = 1.0

catalog_df["catalog"] = catalog_df["catalog"].fillna("未分类").astype(str)
available_catalogs = catalog_df["catalog"].drop_duplicates().tolist()
selected_source_catalogs = st.sidebar.multiselect(
    "本次监测目录",
    available_catalogs,
    default=available_catalogs,
    help="期刊较多时可只选择部分目录，减少 Crossref 请求量。",
)
working_catalog_df = catalog_df[catalog_df["catalog"].isin(selected_source_catalogs)].copy()

with st.sidebar.expander(f"目标期刊（{len(working_catalog_df)}/{len(catalog_df)}）", expanded=False):
    if not working_catalog_df.empty:
        st.dataframe(
            working_catalog_df[["journal", "issn", "catalog", "weight"]],
            hide_index=True,
            use_container_width=True,
        )

extra_positive_text = st.sidebar.text_area(
    "临时增加关注关键词（可选）",
    value="",
    height=110,
    help="每行一个，可写 keyword|权重。内容会叠加在研究兴趣画像上。",
)
configured_negative = "\n".join(
    f"{x.get('term','')}|{x.get('weight',1.0)}" if isinstance(x, dict) else str(x)
    for x in (profile.get("negative_keywords") or [])
)
negative_text = st.sidebar.text_area("降权关键词（可选）", value=configured_negative, height=80)

default_days = max(7, min(180, int(profile_settings.get("default_days", 60) or 60)))
days = st.sidebar.slider("检索最近多少天", 7, 180, default_days, 1)
default_rows = max(5, min(100, int(profile_settings.get("rows_per_journal", 20) or 20)))
rows_per_journal = st.sidebar.slider("每刊最多获取论文数", 5, 100, default_rows, 5)
max_results = st.sidebar.slider("首页最多展示", 5, 100, 30, 5)
default_min_score = float(profile_settings.get("default_min_score", 1.0) or 1.0)
min_score = st.sidebar.number_input("最低规则推荐分", min_value=-20.0, max_value=100.0, value=default_min_score, step=0.5)
mailto = st.sidebar.text_input("Crossref 联系邮箱（建议填写）", value="", placeholder="name@university.edu")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 大模型深度分析")
if ai_ready:
    st.sidebar.success("OpenAI API 已配置，可生成中文摘要、深度推荐理由并进行 AI 重排序。")
else:
    st.sidebar.info("大模型代码已接入；在 Railway Variables 中配置 OPENAI_API_KEY 后即可启用。")

use_ai = st.sidebar.toggle("启用 AI 深度分析", value=ai_ready, disabled=not ai_ready)
model_options = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
default_model = configured_model()
if default_model not in model_options:
    model_options.insert(0, default_model)
ai_model = st.sidebar.selectbox(
    "AI 模型",
    model_options,
    index=model_options.index(default_model),
    disabled=not use_ai,
    help="Luna 适合高频筛选；Terra 平衡质量与成本；Sol 适合少量高价值论文深度判断。",
)
ai_candidate_count = st.sidebar.slider(
    "AI 深度分析候选数",
    5,
    20,
    int(profile_settings.get("ai_candidate_count", 10) or 10),
    1,
    disabled=not use_ai,
)
ai_weight = st.sidebar.slider(
    "AI 重排序权重",
    0.10,
    0.60,
    float(profile_settings.get("ai_rerank_weight", 0.35) or 0.35),
    0.05,
    disabled=not use_ai,
    help="仅对进入 AI 深度分析的候选论文重新排序。",
)

fetch_now = st.sidebar.button("🔄 获取并生成推荐", type="primary", use_container_width=True)

st.title("📚 Paper Radar · 论文推荐")
st.caption("基于分层研究画像进行两阶段筛选：规则/语义召回 → 可选大模型深度分析与重排序。")

active_positive = []
if use_profile:
    active_positive.extend(profile_kws)
active_positive.extend(parse_keywords(extra_positive_text))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("目标期刊", len(working_catalog_df))
c2.metric("期刊目录", len(selected_source_catalogs))
c3.metric("画像主题", len(profile.get("themes") or []) if use_profile else 0)
c4.metric("主题关联", len(profile.get("relations") or []) if use_profile else 0)
c5.metric("本地语义", "开启" if (use_profile and use_semantic) else "关闭")
c6.metric("AI分析", "开启" if use_ai else "关闭")

if len(working_catalog_df) >= 80 and rows_per_journal >= 30:
    st.warning("当前一次会请求较多期刊和论文。建议日常使用时按目录筛选，或将“每刊最多获取论文数”设为 10–20，以减少等待和 Crossref 限流风险。")

if use_profile:
    with st.expander("查看分层画像与关键词", expanded=False):
        st.markdown("**主题层级**")
        for tier in ["core", "emerging", "cross", "method", "application"]:
            names = [t.get("name", "") for t in profile.get("themes") or [] if t.get("tier") == tier]
            if names:
                st.markdown(f"**{tier_labels.get(tier, tier)}：** {'；'.join(names)}")
        st.markdown("**加权关键词**")
        st.code(profile_keywords_text(profile), language="text")

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []
    st.session_state.ai_error = ""
    st.session_state.last_run = None
    st.session_state.ai_model_used = ""

if fetch_now:
    negative = parse_keywords(negative_text)
    if working_catalog_df.empty:
        st.warning("请至少选择一个监测目录。")
    elif not active_positive:
        st.warning("请启用研究兴趣画像，或至少填写一个临时关注关键词。")
    else:
        with st.spinner("第一阶段：正在从期刊目录获取论文并计算规则、语义和交叉主题匹配…"):
            raw_df, errors = cached_fetch(
                working_catalog_df.to_json(orient="split"),
                days,
                rows_per_journal,
                mailto,
            )
            ranked = rank_papers(
                raw_df,
                active_positive,
                negative,
                recency_half_life=float(profile_settings.get("recency_half_life_days", 30) or 30),
                min_score=min_score,
                profile=profile if use_profile else None,
                use_semantic=bool(use_profile and use_semantic),
                semantic_weight=float(profile_settings.get("semantic_weight", 10.0) or 10.0),
                semantic_threshold=float(profile_settings.get("semantic_threshold", 0.055) or 0.055),
                intersection_weight=float(profile_settings.get("intersection_weight", 2.4) or 2.4),
                cross_theme_bonus=float(profile_settings.get("cross_theme_bonus", 0.8) or 0.8),
            )

        ai_error = ""
        if use_ai and not ranked.empty:
            with st.spinner(f"第二阶段：正在用 {ai_model} 深度分析前 {min(ai_candidate_count, len(ranked))} 篇候选论文…"):
                try:
                    papers = prepare_papers(ranked, limit=ai_candidate_count)
                    ai_results = cached_ai_analyze(
                        json.dumps(papers, ensure_ascii=False, sort_keys=True),
                        json.dumps(profile, ensure_ascii=False, sort_keys=True),
                        ai_model,
                    )
                    ranked = merge_ai_analysis(
                        ranked,
                        ai_results,
                        analyzed_count=ai_candidate_count,
                        ai_weight=ai_weight,
                    )
                    st.session_state.ai_model_used = ai_model
                except Exception as e:
                    ai_error = str(e)
                    st.session_state.ai_model_used = ""

        st.session_state.results = ranked
        st.session_state.errors = errors
        st.session_state.ai_error = ai_error
        st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")

results = st.session_state.results
errors = st.session_state.errors
if st.session_state.last_run:
    ai_note = f" · AI: {st.session_state.ai_model_used}" if st.session_state.ai_model_used else ""
    st.caption(f"最近刷新：{st.session_state.last_run}{ai_note} · Crossref 数据缓存 30 分钟；AI分析缓存 24 小时。")
if errors:
    with st.expander(f"⚠️ {len(errors)} 个期刊请求未成功", expanded=False):
        st.code("\n".join(errors[:40]))
if st.session_state.ai_error:
    st.warning(f"本次规则推荐已完成，但 AI 深度分析未成功：{st.session_state.ai_error}")

if not results.empty:
    catalogs = [x for x in results["catalog"].dropna().astype(str).unique().tolist() if x]
    f1, f2 = st.columns([2, 1])
    selected_catalogs = f1.multiselect("按目录筛选结果", catalogs, default=catalogs)
    keyword_filter = f2.text_input("结果内搜索", placeholder="标题/作者/期刊")
    view = results.copy()
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

    if "ai_relevance_score" in view.columns and view["ai_relevance_score"].notna().any():
        ai_done = int(view["ai_relevance_score"].notna().sum())
        st.subheader(f"为你推荐 · {min(len(view), max_results)} 篇（其中 {ai_done} 篇已完成 AI 深度分析）")
    else:
        st.subheader(f"为你推荐 · {min(len(view), max_results)} 篇")

    for i, (_, row) in enumerate(view.head(max_results).iterrows(), 1):
        render_card(i, row)

    export_cols = [
        "title",
        "authors",
        "journal",
        "catalog",
        "date",
        "doi",
        "url",
        "abstract",
        "score",
        "keyword_score",
        "semantic_score",
        "intersection_score",
        "semantic_similarity",
        "profile_match",
        "intersection_match",
        "recommendation_level",
        "matched_keywords",
        "recommendation_reason",
        "hybrid_score",
        "ai_relevance_score",
        "ai_priority",
        "ai_matched_themes",
        "ai_summary",
        "ai_recommendation_reason",
        "ai_research_connection",
        "ai_novelty_or_value",
        "ai_confidence",
    ]
    csv_bytes = view[[c for c in export_cols if c in view.columns]].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 导出当前推荐 CSV",
        csv_bytes,
        file_name="paper_recommendations.csv",
        mime="text/csv",
    )
else:
    st.info("点击左侧“获取并生成推荐”开始。默认期刊库已更新为你的完整期刊清单。")
    st.markdown(
        """
当前系统采用两阶段推荐。第一阶段使用你的**分层研究兴趣画像 + 精确关键词 + 本地语义匹配 + 主题交叉关系**快速召回；第二阶段在配置 OpenAI API 后，对高分候选论文进一步生成**中文摘要、AI相关度、阅读优先级、深度推荐理由、与你现有研究的衔接点和潜在研究价值**，并对候选论文进行重排序。

这样既避免把全部论文都交给大模型造成不必要的成本，也比单纯关键词匹配更容易发现“措辞不同但研究问题真正相关”的论文。
"""
    )
