# insurance-copilot

A multi-agent system that turns Brazilian home-insurance policy documents into a queryable knowledge base. It harvests *condições gerais* (general terms) registered with SUSEP, extracts their structure into Postgres, and answers coverage-comparison questions in natural language.

> **Status: work in progress.** The data pipeline (SUSEP harvester + extraction schema) and the service skeleton (FastAPI + Postgres) are in place. The agent layer is under construction. See [Roadmap](#roadmap).

## Why

Comparing home-insurance products in Brazil means reading dozens of 50–90 page PDFs to find what each one actually covers — which perils, which exclusions, how the deductible (POS) works. This project automates that comparison.

**Scope of the data:** the corpus is made of *general terms* (which describe the **product**), not individual policies. So the system answers questions about **coverage structure** — "which insurers cover windstorm without a deductible?", "which perils does insurer A cover that B doesn't?" — and not about **prices**, which live in each customer's individual policy.

## Architecture

A supervisor agent routes each question to specialized workers (canonical hub-and-spoke):

- **extraction** — turns a policy PDF into structured rows (insurer, product, coverages, perils, exclusions).
- **SQL** — aggregates over the structured tables (coverage comparison, deductible structure, exclusion patterns).
- **RAG** — retrieves and explains raw clause text (pgvector).

Each LLM call is cost-attributed per agent (one row per call: request id, agent, model, tokens, cost). Surface: WhatsApp (Meta Cloud API), with HMAC-verified webhooks.

## Data: SUSEP corpus harvester

`scripts/susep_harvest.py` builds the corpus of residential general terms (ramo `01 — Compreensivo Residencial`) from three public, login-free SUSEP endpoints:

1. **Index** — SUSEP's OData service (Olinda), resource `DadosProdutos`: every registered product with `{tipoproduto, entnome, cnpj, numeroprocesso, ramo, subramo}`. Filtered to residential. *Gotcha:* the service only accepts `$format=json` (any `$top`/`$filter`/`$select` returns 500), so the full dataset is fetched and cached.
2. **Resolve** — `POST Produto.aspx/Consultar` (field `numeroProcesso`) returns HTML with the version table; each version exposes a `DownloadConsultaPublica/{id}` link, filename and commercialization dates. *Gotcha:* a ~14-query-per-session quota then returns HTTP 200 with an empty page (not 429); the harvester rotates the session (a fresh cookie resets the quota) while keeping 1 req/s and an identifiable User-Agent.
3. **Download** — `GET DownloadConsultaPublica/{id}` → PDF.

Output: `data/corpus/susep_{id}.pdf` + `corpus_manifest.json` (per-version provenance: process, id, insurer, CNPJ, file, dates, ramo, url, sha256, has_text, downloaded_at). Defaults to the in-force version of each process; `--all-versions` fetches full history. Resumable (skips ids already downloaded).

```bash
python scripts/susep_harvest.py                # in-force version per process
python scripts/susep_harvest.py --all-versions # full history
python scripts/susep_harvest.py --limit 5      # smoke test
```

**The PDFs are not committed to this repository.** `data/corpus/*.pdf` is gitignored — the documents are public and large, so versioning them would bloat the repo. Only `corpus_manifest.json` is committed, which records the provenance of every file (process, id, insurer, url, sha256, dates). To reproduce the corpus, clone the repo and run the harvester above: it re-downloads the PDFs straight from SUSEP, and the manifest lets you verify you got the same documents (by sha256). This keeps the repo small and the corpus reproducible from its source of record.

## Extraction schema

The extraction grain is **(insurer × coverage)**, not the insurer — a policy has deductible rules *per coverage*, not one. Key modeling decision: coverages are normalized **by peril**, not by commercial name, because insurers bundle perils differently (one calls it "windstorm+hail", another "windstorm+hail+smoke+vehicle impact"). This makes `coverage ↔ peril` a many-to-many relationship.

| table | grain |
|---|---|
| `policy_document` | one document (insurer, product, SUSEP process, version, property type, provenance) |
| `coverage` | one coverage (basic/additional, deductible type/rule) |
| `peril` | one canonical peril |
| `coverage_peril` | join (which perils a coverage includes) |
| `exclusion` | one exclusion (general or per-coverage) |

Categorical columns feed the SQL worker; raw-text columns feed the RAG worker.

## Tech stack

FastAPI · Postgres (+ pgvector) · SQLAlchemy 2.0 (async) · Alembic · Docker Compose · pytest + testcontainers · deployed on Railway.

## Getting started

**Prerequisites:** Docker (or Colima on macOS) and Python 3.11+ (the code uses `str | None` annotations, which require 3.10+).

```bash
# 1. clone and create a virtual environment
git clone https://github.com/luanmoura-aimanager/insurance-copilot.git
cd insurance-copilot
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure environment
cp .env.example .env        # then edit if needed

# 4. start Postgres
docker compose up -d        # serves Postgres on localhost:5433

# 5. create the tables
alembic upgrade head

# 6. run the API
uvicorn app.main:app --reload
```

Verify it's up:

```bash
curl localhost:8000/health      # {"status":"ok"}
curl localhost:8000/health/db   # {"db":"ok"}  — API ↔ Postgres OK
```

## Project structure

```
insurance-copilot/
├── app/
│   ├── main.py             # FastAPI app + health endpoints
│   ├── db.py               # async engine + session (SQLAlchemy 2.0)
│   └── models.py           # ORM models: PolicyDocument, Coverage, Peril, CoveragePeril, Exclusion
├── alembic/
│   └── versions/           # migrations (alembic upgrade head)
├── docs/
│   └── schema.html         # visual schema diagram (open in browser)
├── scripts/
│   └── susep_harvest.py    # SUSEP corpus harvester
├── data/
│   └── corpus/             # downloaded PDFs (gitignored) + corpus_manifest.json
├── docker-compose.yml      # local Postgres
├── requirements.txt
└── .env.example
```

## Roadmap

- [x] Corpus harvester (SUSEP, residential)
- [x] Extraction schema validated against real policies
- [x] Service skeleton — FastAPI + Postgres, health checks
- [x] ORM models — all 5 tables: `policy_document`, `coverage`, `peril`, `coverage_peril`, `exclusion`
- [x] Alembic migrations
- [ ] Test suite (testcontainers)
- [ ] Production extraction (LLM → tables)
- [ ] Agent layer (supervisor + SQL/RAG/extraction workers)
- [ ] WhatsApp surface + cost attribution
- [ ] Deploy to Railway

## License

MIT
