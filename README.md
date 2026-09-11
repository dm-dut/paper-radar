# Paper Radar

Paper Radar is a Streamlit app for discovering recent papers from a personalized journal catalog and ranking them against a structured research-interest profile.

## Current features

- Default journal catalog with 100+ journals across AI, OR/MCDA, OM, industrial engineering, computational intelligence, tourism, transportation, information systems/marketing, systems science and general journals.
- Custom CSV upload with `journal`, `issn`, `catalog`, and `weight` columns.
- Crossref retrieval by ISSN with journal/category filtering and configurable retrieval windows.
- Hierarchical research-interest profile covering core, emerging, cross-disciplinary, method and application themes.
- Weighted keyword matching, local semantic similarity and cross-theme relationship bonuses.
- Optional OpenAI-powered second-stage analysis for the highest-ranked papers.
- AI-generated Chinese summaries, relevance scores, reading priority, recommendation reasons, links to the user's research lines, and potential research value.
- Hybrid reranking that combines rule/semantic ranking with AI relevance judgments.
- CSV export of both rule-based and AI-enhanced recommendation fields.

## Recommendation pipeline

1. Fetch recent metadata from the selected journal categories through Crossref.
2. Use the saved research profile for keyword matching, lightweight semantic matching, recency and journal weighting.
3. Reward papers that connect multiple related research themes.
4. Optionally send only the top candidate papers to an OpenAI model for deeper analysis and reranking.

This two-stage design keeps the high-volume discovery step inexpensive while reserving the large model for the small set of papers most likely to be relevant.

## OpenAI setup

The app works without an API key. To enable AI deep analysis, configure these environment variables:

```text
OPENAI_API_KEY=your_secret_key
OPENAI_MODEL=gpt-5.6-luna
OPENAI_BASE_URL=https://api.openai.com/v1
```

For Railway, add `OPENAI_API_KEY` under the service's Variables/Secrets settings rather than committing it to GitHub. The app also lets you choose `gpt-5.6-luna`, `gpt-5.6-terra`, or `gpt-5.6-sol` from the sidebar after the key is configured.

AI responses are requested with structured JSON output and `store=false`. Results are cached by the Streamlit app for 24 hours for identical paper/profile/model inputs.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Copy `.env.example` only as a reference; set environment variables in your shell or deployment platform. Never commit a real API key.

## Railway deployment

The service is deployed from this GitHub repository. Streamlit binds to Railway's assigned port. After code changes, deploy the latest commit from the `main` branch.

## Main files

- `app.py`: Streamlit interface and two-stage workflow.
- `core.py`: Crossref retrieval and rule/semantic ranking.
- `llm_service.py`: OpenAI Responses API integration and AI reranking.
- `config/research_profile.json`: saved hierarchical research-interest profile.
- `journal_catalog.sample.csv`: default journal catalog.
