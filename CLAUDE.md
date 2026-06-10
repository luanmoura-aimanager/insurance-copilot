# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent system that turns Brazilian home-insurance *condições gerais* (general terms registered with SUSEP) into a queryable knowledge base. It answers **coverage-structure** questions ("which insurers cover windstorm without a deductible?"), not pricing — the corpus describes products, and prices live in individual customer policies which are out of scope.

The project is mid-build: the data pipeline (harvester + validated extraction schema) and a FastAPI/Postgres skeleton exist; the LLM extraction pipeline, ORM models/migrations, agent layer, and WhatsApp surface do not yet. See the Roadmap in `README.md` for current state before assuming a component exists.

## Commands

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Local Postgres (host port 5433, not 5432 — see DATABASE_URL note below)
docker compose up -d

# Run the API
uvicorn app.main:app --reload
curl localhost:8000/health      # {"status":"ok"}
curl localhost:8000/health/db   # {"db":"ok"} — verifies API ↔ Postgres
```

There is **no test suite, linter, or migration tooling wired up yet** despite `README.md` listing pytest/testcontainers/Alembic in the intended stack — they are roadmap items. Don't invent commands for them.

## Architecture & key decisions

**Async-only data layer.** `app/db.py` uses SQLAlchemy 2.0 async (`asyncpg` driver). The `DATABASE_URL` **must** use the `postgresql+asyncpg://` scheme, not plain `postgresql://`. The async route (`/health/db`) injects a session via FastAPI `Depends(get_session)`.

**Port 5433.** docker-compose maps Postgres to host `5433` (not the default 5432) to avoid colliding with a local Postgres; `.env.example` reflects this.

**The corpus is general terms, not policies.** This is the central scoping decision and it shapes the schema: a document describes a *product*. The extraction grain is therefore **(insurer × coverage)**, and coverages are normalized **by peril, not by commercial name**, because insurers bundle perils differently. That makes `coverage ↔ peril` many-to-many. Intended tables: `policy_document`, `coverage`, `peril`, `coverage_peril` (join), `exclusion` (scope = general or per-coverage). Categorical columns feed the future SQL worker; raw-text columns feed the future RAG worker.

**Deductible terminology.** In residential CGs the term is **POS (Participação Obrigatória do Segurado)**, treated as a synonym for "franquia". The dominant pattern is "valor ou percentual definido na apólice" — the CG fixes the *structure*, the number lives in the customer policy. The validated `franquia_tipo` enum is `{sem_franquia, percentual, valor_fixo, definido_na_apolice}`. See `data/pilot_findings.md` for the reasoning behind every schema choice — read it before modeling tables.

**Intended agent layer (not built):** supervisor hub-and-spoke routing to `extraction` / `SQL` / `RAG` workers, with per-call cost attribution and a HMAC-verified WhatsApp (Meta Cloud API) surface.

## SUSEP corpus pipeline (`scripts/`)

The harvester builds the corpus from three login-free public SUSEP endpoints. Critical gotchas baked into `susep_harvest.py`:
- The Olinda OData index (`DadosProdutos`) **only accepts `$format=json`** — any `$top`/`$filter`/`$select` returns HTTP 500, so the full dataset is fetched and cached locally.
- The version-resolution endpoint enforces a **~14-query-per-session quota**, after which it returns HTTP 200 with an *empty page* (not 429). The harvester detects this and rotates the session cookie to reset the quota, keeping ~1 req/s.

```bash
python scripts/susep_harvest.py                # in-force version per process
python scripts/susep_harvest.py --all-versions # full version history
python scripts/susep_harvest.py --limit 5      # smoke test
```

**Script defaults are anchored to the repo root** via `Path(__file__).resolve().parent.parent`, so they read/write `data/corpus/` correctly regardless of CWD (run them from anywhere; override with `--out`/`--index-cache`).

**The PDFs are gitignored** (`data/corpus/*.pdf`) — large and public. Only `corpus_manifest.json` is committed; it records per-version provenance (process, internal id, insurer, CNPJ, url, sha256, dates, `has_text`) and lets anyone re-download and verify the corpus by hash. The download endpoint keys on an **internal numeric id**, while the index keys on **process number** (`15414.NNNNNN/AAAA-DD`) — bridging the two is what the resolve step does.

The other scripts are one-off, not production: `susep_probe.py` (blind-sampling viability probe that established corpus volume) and `pilot_extraction.py` (manual hand-read extraction of 2 CGs that validated the schema — produces `data/pilot_extraction.json`, summarized in `data/pilot_findings.md`).

## Conventions

Code comments and the pilot/brief docs are written in **Portuguese (pt-BR)**; domain terms (POS, CG, ramo, vendaval) stay in Portuguese. Match this when editing existing files.
