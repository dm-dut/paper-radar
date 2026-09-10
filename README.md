# Paper Radar

A lightweight Streamlit web app for discovering recent papers from a custom journal catalog and ranking them against a personal keyword profile.

## Features

- Upload a journal catalog CSV with `journal`, `issn`, `catalog`, and `weight` columns.
- Fetch recent journal articles from Crossref by ISSN.
- Rank papers with weighted positive and negative keywords.
- Read paper metadata, abstracts, matching reasons, and DOI links in a card-style feed.
- Filter and export recommendations to CSV.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Railway deployment

This repository includes a Dockerfile. Railway should deploy it as a Docker service. The container starts Streamlit on Railway's dynamic `$PORT` and binds to `0.0.0.0`.

## Journal catalog format

See `journal_catalog.sample.csv`.
