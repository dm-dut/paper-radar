from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core import fetch_catalog, parse_keywords, profile_keywords, profile_keywords_text, rank_papers, safe_text

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CATALOG = APP_DIR / "journal_catalog.sample.csv"
PROFILE_PATH = APP_DIR / "config" / "research_profile.json"

st.set_page_config(page_title="Paper Radar · 论文推荐", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
.paper-card {padding:1.15rem 1.3rem 1rem;border:1px solid #e7ecef;border-radius:14px;background:#fff;box-shadow:0 2px 10px rgba(31,45,61,.045);margin-bottom:1rem;}
.paper-title {font-size:1.22rem;font-weight:750;color:#26384d;line-height:1.45;margin-bottom:.5rem;}
.paper-meta {font-size:.93rem;color:#7d8b8e;line-height:1.8;margin-bottom:.55rem;}
.paper-abstract {font-size:1.01rem;color:#3d536b;line-height:1.9;margin:.55rem 0;}
.paper-reason {font-size:.98rem;color:#08a857;font-weight:650;line-height:1.8;margin-top:.75rem;}
.paper-profile {font-size:.90rem;color:#5a7184;margin-top:.45rem;}
.score-badge,.semantic-badge,.intersection-badge,.level-badge,.theme-badge {display:inline-block;padding:.13rem .55rem;border-radius:999px;font-size:.8rem;font-weight:700;margin-left:.35rem;}
.score-badge {background:#eef7f2;color:#16784b;}
.semantic-badge {background:#f4f1fb;color:#67538c;}
.intersection-badge {background:#fff4e8;color:#9a6418;}
.level-badge {background:#eef3f7;color:#465c6c;}
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


def load_default_catalog() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CATALOG)


def load_profile() -> dict:
    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"profile_name": "默认研究兴趣画像", "description": "当前未能读取研究兴趣配置。", "themes": [], "relations": [], "negative_keywords": [], "settings": {}}


def html_escape(s: str) -> str:
    import html
    return html.escape(safe_text(s), quote=True)


def render_card(i: int, row: pd.Series):
    title = html_escape(row.get("title"))
    authors = html_escape(row.get("authors"))
    journal = html_escape(row.get("journal"))
    pubdate = html_escape(row.get("date"))
    abstract = safe_text(row.get("abstract"))
    abstract_display = abstract if abstract else "Crossref 暂未提供摘要；当前推荐主要依据标题、期刊与发布时间。"
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
    doi = safe_text(row.get("doi"))
    url = safe_text(row.get("url"))
    catalog = safe_text(row.get("catalog"))
    catalog_piece = f" &nbsp;|&nbsp; {html_escape(catalog)}" if catalog else ""

    theme_html = ""
    if profile_match:
        badges = "".join(f'<span class="theme-badge">{html_escape(x.strip())}</span>' for x in profile_match.split("、") if x.strip())
        theme_html = f'<div class="paper-profile">研究画像匹配：{badges}</div>'
    intersection_html = ""
    if intersection_match:
        ibadges = "".join(f'<span class="theme-badge">{html_escape(x.strip())}</span>' for x in intersection_match.split("、") if x.strip())
        intersection_html = f'<div class="paper-profile">交叉研究信号：{ibadges}</div>'

    sem_badge = f'<span class="semantic-badge">语义 {semantic_score:.1f}</span>' if semantic_score > 0 else ""
    int_badge = f'<span class="intersection-badge">交叉 {intersection_score:.1f}</span>' if intersection_score > 0 else ""
    level_badge = f'<span class="level-badge">{html_escape(rec_level)}</span>'

    st.markdown(
        f"""
<div class="paper-card">
  <div class="paper-title">{i}. {title}<span class="score-badge">综合 {score:.1f}</span>{sem_badge}{int_badge}{level_badge}</div>
  <div class="paper-meta">✍️ {authors or '作者信息暂缺'} &nbsp;|&nbsp; 📖 {journal or '期刊信息暂缺'}{catalog_piece}<br>📅 {pubdate or '日期暂缺'} {('&nbsp;|&nbsp; DOI: ' + html_escape(doi)) if doi else ''}</div>
  <div class="paper-abstract">{abstract_display}</div>
  {theme_html}
  {intersection_html}
  <div class="paper-reason">🎯 推荐理由：{reason}</div>
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

st.sidebar.title("⚙️ 推荐设置")
st.sidebar.caption("系统使用分层研究兴趣画像，并结合关键词、语义和交叉主题关系进行推荐。")

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
    st.caption(f"当前共 {len(profile.get('themes') or [])} 个主题、{len(profile.get('relations') or [])} 条主题关联、{len(profile_kws)} 个加权关键词。")

use_profile = st.sidebar.toggle("使用研究兴趣画像", value=True)
use_semantic = st.sidebar.toggle("启用语义匹配", value=True, help="识别概念相近但措辞不同的论文。")

uploaded = st.sidebar.file_uploader("期刊目录 CSV", type=["csv"], help="列：journal, issn, catalog, weight")
if uploaded is not None:
    try:
        catalog_df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"CSV 读取失败：{e}")
        catalog_df = load_default_catalog()
else:
    catalog_df = load_default_catalog()

required = {"journal", "issn"}
if not required.issubset(set(catalog_df.columns)):
    st.sidebar.error("期刊目录至少需要 journal 和 issn 两列。已退回示例目录。")
    catalog_df = load_default_catalog()
if "catalog" not in catalog_df.columns:
    catalog_df["catalog"] = "自定义目录"
if "weight" not in catalog_df.columns:
    catalog_df["weight"] = 1.0

with st.sidebar.expander(f"目标期刊（{len(catalog_df)}）", expanded=False):
    st.dataframe(catalog_df[["journal", "issn", "catalog", "weight"]], hide_index=True, use_container_width=True)

extra_positive_text = st.sidebar.text_area("临时增加关注关键词（可选）", value="", height=120, help="每行一个，可写 keyword|权重。内容会叠加在研究兴趣画像上。")
configured_negative = "\n".join(f"{x.get('term','')}|{x.get('weight',1.0)}" if isinstance(x, dict) else str(x) for x in (profile.get("negative_keywords") or []))
negative_text = st.sidebar.text_area("降权关键词（可选）", value=configured_negative, height=90)

default_days = max(7, min(180, int(profile_settings.get("default_days", 60) or 60)))
days = st.sidebar.slider("检索最近多少天", 7, 180, default_days, 1)
rows_per_journal = st.sidebar.slider("每刊最多获取论文数", 5, 100, 35, 5)
max_results = st.sidebar.slider("首页最多展示", 5, 100, 30, 5)
default_min_score = float(profile_settings.get("default_min_score", 1.0) or 1.0)
min_score = st.sidebar.number_input("最低推荐分", min_value=-20.0, max_value=100.0, value=default_min_score, step=0.5)
mailto = st.sidebar.text_input("Crossref 联系邮箱（建议填写）", value="", placeholder="name@university.edu")
fetch_now = st.sidebar.button("🔄 获取并生成推荐", type="primary", use_container_width=True)

st.title("📚 Paper Radar · 论文推荐")
st.caption("根据分层个人研究画像，通过关键词 + 语义 + 主题关系识别近期论文，特别提升与你多个研究方向同时交叉的论文。")

active_positive = []
if use_profile:
    active_positive.extend(profile_kws)
active_positive.extend(parse_keywords(extra_positive_text))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("目标期刊", len(catalog_df))
c2.metric("画像主题", len(profile.get("themes") or []) if use_profile else 0)
c3.metric("主题关联", len(profile.get("relations") or []) if use_profile else 0)
c4.metric("语义匹配", "开启" if (use_profile and use_semantic) else "关闭")
c5.metric("检索窗口", f"{days} 天")

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
    st.session_state.last_run = None

if fetch_now:
    negative = parse_keywords(negative_text)
    if not active_positive:
        st.warning("请启用研究兴趣画像，或至少填写一个临时关注关键词。")
    else:
        with st.spinner("正在获取论文，并计算关键词、语义和交叉主题匹配…"):
            raw_df, errors = cached_fetch(catalog_df.to_json(orient="split"), days, rows_per_journal, mailto)
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
            st.session_state.results = ranked
            st.session_state.errors = errors
            st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M")

results = st.session_state.results
errors = st.session_state.errors
if st.session_state.last_run:
    st.caption(f"最近刷新：{st.session_state.last_run} · 当前缓存 30 分钟，可手动再次刷新。")
if errors:
    with st.expander(f"⚠️ {len(errors)} 个期刊请求未成功", expanded=False):
        st.code("\n".join(errors[:30]))

if not results.empty:
    catalogs = [x for x in results["catalog"].dropna().astype(str).unique().tolist() if x]
    f1, f2 = st.columns([2, 1])
    selected_catalogs = f1.multiselect("按目录筛选", catalogs, default=catalogs)
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

    st.subheader(f"为你推荐 · {min(len(view), max_results)} 篇")
    for i, (_, row) in enumerate(view.head(max_results).iterrows(), 1):
        render_card(i, row)

    export_cols = [
        "title", "authors", "journal", "catalog", "date", "doi", "url", "abstract",
        "score", "keyword_score", "semantic_score", "intersection_score", "semantic_similarity",
        "profile_match", "intersection_match", "recommendation_level", "matched_keywords", "recommendation_reason"
    ]
    csv_bytes = view[[c for c in export_cols if c in view.columns]].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 导出当前推荐 CSV", csv_bytes, file_name="paper_recommendations.csv", mime="text/csv")
else:
    st.info("点击左侧“获取并生成推荐”开始。系统已启用分层研究兴趣画像。")
    st.markdown(
        """
当前画像采用五级结构：**核心方向、新核心方向、重要交叉方向、方法交叉方向、应用方向**。系统不仅判断论文属于哪个主题，还会检查主题之间的关联。

例如一篇论文同时涉及 **粒球计算 + 偏好学习 + 多准则分类**，或者 **语言偏好 + 群体共识 + 个性化语义**，将获得额外的“交叉研究信号”并优先推荐。这比单纯把所有关键词平铺在一起更能反映你的真实研究脉络。
"""
    )
