# AI Health Symptoms Assistant

A Streamlit app that gives **general, educational** information about common symptoms, answered only from a curated set of reference documents, with citations.

**It is not a doctor.** It cannot diagnose, prescribe, or give medicine doses.

## Scope and limitations

- Intended region: **India** (emergency numbers 112 / ambulance 108 / Tele-MANAS 14416). Verify numbers before deploying and override them in `.env` if the region changes.
- Audience: adults and children **12 months and older**. Infants under 12 months get a "contact a paediatrician" message instead of an answer.
- Language: English only.
- Emergency screening (`safety.py`) is a reviewed pattern list with negation handling, **not a complete triage system**. It does not understand tense or who a symptom is about, and it errs toward escalating.
- Answers are limited to documents in `knowledge/`. If nothing relevant is retrieved, the app says the question is not covered and does not use model knowledge.
- Retrieval is TF-IDF cosine similarity, so it matches words, not meaning.
- The model can still make mistakes; output is checked for required sections, citations and dose-like numbers, and falls back to the retrieved passages if it fails.

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` (create one at console.groq.com). Never commit `.env`.

## Add reference documents

Put approved sources in `knowledge/` with publisher, URL, date and version. See `knowledge/README.md`. The folder ships with six NHS-based reference summaries (cough, fever, diarrhoea and vomiting, headache, common cold, sore throat). They are UK sources, so consider adding Indian sources to match the deployment region.

## Run

```
streamlit run app.py
```

Open http://localhost:8501. Emergency screening and the disclaimer work even with no API key.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

Covers emergency phrases, paraphrases, negation, punctuation, missing keys, blank outputs, API failures, invalid credentials, follow-up context, retrieval and loading.

## Evaluation

```
python evaluation/run_eval.py
```

Runs the 12 cases in `evaluation/cases.json` and writes `evaluation/results.md` with retrieved sources and observed outputs. Run it after adding documents and your API key, and submit the generated file.

## Deploy (Streamlit Community Cloud)

1. Push the repo to GitHub (`.env` is git-ignored).
2. At share.streamlit.io choose the repo, branch `main`, main file `app.py`.
3. In **Advanced settings → Secrets** add `GROQ_API_KEY = "your_key"` (and optionally `MODEL_NAME`).
4. Deploy. The `knowledge/` folder must be committed so the deployed app can use it.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit interface only |
| `pipeline.py` | One question end to end: safety, scope, retrieval, generation, validation |
| `safety.py` | Local emergency screening |
| `document_loader.py` | Loads and chunks `knowledge/` documents, enforces provenance |
| `retriever.py` | TF-IDF vector index and search |
| `llm_service.py` | Groq calls, error classes, output validation |
| `config.py` | Settings, region numbers, disclaimer |
| `knowledge/` | Approved reference documents |
| `tests/` | Automated tests |
| `evaluation/` | Evaluation cases and runner |
