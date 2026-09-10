from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core import fetch_catalog, parse_keywords, rank_papers, safe_text

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CATALOG = APP_DIR / "journal_catalog.sample.csv"

st.set_page_config(page_title="Paper Radar · 论文推荐", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    .paper-card {
        padding: 1.15rem 1.3rem 1rem 1.3rem;
        border: 1px solid #e7ecef;
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(31, 45, 61, 0.045);
        margin-bottom: 1rem;
    }
    .paper-title {font-size: 1.22rem; font-weight: 750; color: #26384d; line-height: 1.45; margin-bottom: .5rem;}
    .paper-meta {font-size: .93rem; color: #7d8b8e; line-height: 1.8; margin-bottom: .55rem;}
    .paper-abstract {font-size: 1.01rem; color: #3d536b; line-height: 1.9; margin: .55rem 0;}
    .paper-reason {font-size: .98rem; color: #08a857; font-weight: 650; line-height: 1.8; margin-top: .75rem;}
    .score-badge {display:inline-block; background:#eef7f2; color:#16784b; padding:.13rem .55rem; border-radius:999px; font-size:.8rem; font-weight:700; margin-left:.35rem;}
    .hint {color:#7b8794;font-size:.9rem;}
    div[data-testid="stSidebar"] {background: #f8fafb;}
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
    score = float(row.get("score", 0.0) or 0.0)
    doi = safe_text(row.get("doi"))
    url = safe_text(row.get("url"))
    catalog = safe_text(row.get("catalog"))
    catalog_piece = f" &nbsp;|&nbsp; {html_escape(catalog)}" if catalog else ""

    st.markdown(
        f"""
<div class="paper-card">
  <div class="paper-title">{i}. {title}<span class="score-badge">匹配度 {score:.1f}</span></div>
  <div class="paper-meta">✍️ {authors or '作者信息暂缺'} &nbsp;|&nbsp; 📖 {journal or '期刊信息暂缺'}{catalog_piece}<br>📅 {pubdate or '日期暂缺'} {('&nbsp;|&nbsp; DOI: ' + html_escape(doi)) if doi else ''}</div>
  <div class="paper-abstract">{abstract_display}</div>
  <div class="paper-reason">🎯 推荐理由：{reason}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if url:
        st.link_button("打开论文页面", url, use_container_width=False)


st.sidebar.title("⚙️ 推荐设置")
st.sidebar.caption("维护目标期刊目录 + 关键词画像，系统从 Crossref 拉取近期论文并排序。")

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

positive_text = st.sidebar.text_area(
    "关注关键词",
    value="preference learning|3\ngroup decision making|2.5\nconsensus|2.5\nmultiple criteria sorting|3\nMCDA|2\n人工智能|1.5\n风险管理|1.5",
    height=190,
    help="每行一个。可写 keyword|权重，例如 preference learning|3。中英文都支持。",
)
negative_text = st.sidebar.text_area(
    "降权关键词（可选）", value="", height=90, help="不感兴趣的主题，每行一个，可带权重。"
)

days = st.sidebar.slider("检索最近多少天", 7, 180, 45, 1)
rows_per_journal = st.sidebar.slider("每刊最多获取论文数", 5, 100, 35, 5)
max_results = st.sidebar.slider("首页最多展示", 5, 100, 30, 5)
min_score = st.sidebar.number_input("最低推荐分", min_value=-20.0, max_value=50.0, value=1.0, step=0.5)
mailto = st.sidebar.text_input("Crossref 联系邮箱（建议填写）", value="", placeholder="name@university.edu")

fetch_now = st.sidebar.button("🔄 获取并生成推荐", type="primary", use_container_width=True)

st.title("📚 Paper Radar · 论文推荐")
st.caption("面向自定义期刊目录的近期论文发现与关键词推荐。界面采用“论文信息 + 摘要 + 推荐理由”的阅读流。")

c1, c2, c3 = st.columns(3)
c1.metric("目标期刊", len(catalog_df))
c2.metric("关注关键词", len(parse_keywords(positive_text)))
c3.metric("检索窗口", f"{days} 天")

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []
    st.session_state.last_run = None

if fetch_now:
    positive = parse_keywords(positive_text)
    negative = parse_keywords(negative_text)
    if not positive:
        st.warning("请至少填写一个关注关键词。")
    else:
        with st.spinner("正在从目标期刊获取近期论文并计算推荐分…"):
            raw_df, errors = cached_fetch(
                catalog_df.to_json(orient="split"), days, rows_per_journal, mailto
            )
            ranked = rank_papers(raw_df, positive, negative, min_score=min_score)
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
        "score", "matched_keywords", "recommendation_reason"
    ]
    csv_bytes = view[[c for c in export_cols if c in view.columns]].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 导出当前推荐 CSV", csv_bytes, file_name="paper_recommendations.csv", mime="text/csv")
else:
    st.info("点击左侧“获取并生成推荐”开始。项目已预置少量示例期刊；正式使用时建议上传你自己的目标期刊目录 CSV。")
    st.markdown(
        """
**推荐逻辑（MVP）**：标题命中权重大于摘要命中；关键词可设置不同权重；同时考虑期刊权重与论文新近度。这样即使暂时不接入大模型，也可以稳定、可解释地工作。

**下一步可升级**：接入 OpenAlex/Semantic Scholar 做摘要补全与引用指标，增加“已读/收藏/不感兴趣”反馈学习，以及每天自动刷新并邮件/微信推送。
"""
    )
