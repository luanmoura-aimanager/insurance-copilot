# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent system that turns Brazilian home-insurance *condições gerais* (general terms registered with SUSEP) into a queryable knowledge base. It answers **coverage-structure** questions ("which insurers cover windstorm without a deductible?"), not pricing — the corpus describes products, and prices live in individual customer policies which are out of scope.

The project is mid-build. What exists: the data pipeline (harvester + validated extraction schema + `app/extraction/` LLM pipeline), a FastAPI/Postgres skeleton, the ORM models + Alembic migrations (`app/models.py`), per-call cost attribution (`app/cost.py` + `pricing.json`), a standalone Postgres MCP SQL server (`mcp_servers/`), a first real agent graph (`app/agents/graph.py`: LLM supervisor + single-pass SQL worker — see below), and a testcontainers-backed test suite. What does *not* exist yet: the extraction/RAG workers, a ReAct refinement loop in the SQL worker, and the WhatsApp surface. See the Roadmap in `README.md` for current state before assuming a component exists.

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

# Migrations (Alembic; env.py reads DATABASE_URL and swaps asyncpg→sync)
alembic upgrade head
alembic revision --autogenerate -m "message"

# Tests (boots a throwaway Postgres via testcontainers — Docker must be running)
pytest -q

# Postgres MCP SQL server (stdio; sync psycopg3, not asyncpg). PYTHONPATH=. is
# required — the script's own dir goes on sys.path, so `app` isn't importable otherwise.
PYTHONPATH=. python mcp_servers/postgres_mcp_server.py   # exposes get_schema() / run_query(sql)
# Standalone verification (NOT pytest — no test_ functions, pytest collects 0):
PYTHONPATH=. python mcp_servers/test_postgres_mcp.py                       # schema + SELECT + injection rejections
RO_ROLE_PASSWORD=... PYTHONPATH=. python mcp_servers/test_readonly_role.py # proves insurance_ro can't write
```

`pytest` and `python -m pytest` both work (`pytest.ini` sets `pythonpath = .`). There is **no linter wired up yet** — that's still a roadmap item; don't invent commands for it.

## Architecture & key decisions

**Async-only data layer.** `app/db.py` uses SQLAlchemy 2.0 async (`asyncpg` driver). The `DATABASE_URL` **must** use the `postgresql+asyncpg://` scheme, not plain `postgresql://`. The async route (`/health/db`) injects a session via FastAPI `Depends(get_session)`.

**Port 5433.** docker-compose maps Postgres to host `5433` (not the default 5432) to avoid colliding with a local Postgres; `.env.example` reflects this.

**The corpus is general terms, not policies.** This is the central scoping decision and it shapes the schema: a document describes a *product*. The extraction grain is therefore **(insurer × coverage)**, and coverages are normalized **by peril, not by commercial name**, because insurers bundle perils differently. That makes `coverage ↔ peril` many-to-many. The tables are built in `app/models.py`: `policy_document`, `coverage` (includes a nullable `plan` free-text commercial tier), `peril`, `coverage_peril` (join), `exclusion` (scope = general or per-coverage). Categorical columns feed the future SQL worker; raw-text columns feed the future RAG worker.

**Deductible terminology.** In residential CGs the term is **POS (Participação Obrigatória do Segurado)**, treated as a synonym for "franquia". The dominant pattern is "valor ou percentual definido na apólice" — the CG fixes the *structure*, the number lives in the customer policy. The validated `franquia_tipo` enum is `{sem_franquia, percentual, valor_fixo, definido_na_apolice}`. See `data/pilot_findings.md` for the reasoning behind every schema choice — read it before modeling tables.

**Three Postgres drivers, one URL normalizer.** The app runs on **asyncpg** (async), Alembic on **psycopg2** (sync), and the MCP SQL server on **psycopg3** (`psycopg`, sync — one fresh connection per call, no pool). `app/db_url.py::normalize_url(url, driver)` is the single point that rewrites any provider-supplied `DATABASE_URL` (`postgres://` vs `postgresql://`, and stripping libpq-only query params like `sslmode` that asyncpg rejects) into `postgresql+<driver>://`. The MCP server then strips `+psycopg` back off for the libpq conninfo.

**Cost attribution is built.** `app/cost.py` writes one `cost_event` row per LLM call. Money uses `Decimal`, never float; prices live in `app/pricing.json` (config, not code — Sonnet 5's promo pricing expires 2026-08-31) and the Batch API's 50% discount is applied via `BATCH_MULTIPLIER`. Unknown model → **fail loud** rather than record a wrong cost.

**SQL worker boundary + defense in depth.** `mcp_servers/postgres_mcp_server.py` is the SQL worker's window onto the data: a stdio FastMCP server exposing `get_schema()` and `run_query(sql)` scoped to the 5 domain tables. `get_schema`/`run_query` are plain module-level functions (the core logic) that are *both* registered as FastMCP tools (`mcp.tool()(fn)`) **and** imported directly by the in-process SQL worker — one implementation, two access paths. Writes are blocked twice: (1) a text filter in `run_query` (reject stacked statements → require a leading `SELECT` → auto-append `LIMIT 100`), and (2) the physically read-only `insurance_ro` Postgres role (Alembic migration `4b285ffad59b`, `GRANT SELECT` only, password from `RO_ROLE_PASSWORD`) — so a write is impossible even if the text filter is bypassed. Connections prefer `DATABASE_URL_RO` (the RO role) when set, falling back to the admin `DATABASE_URL`. `run_query` never raises: SQL *and* connection errors come back as text so the caller doesn't crash. The folder is `mcp_servers/` (not `mcp/`) so it doesn't shadow the installed FastMCP `mcp` package; the filename uses an underscore (not a hyphen) so it's importable.

**Agent layer: real supervisor + single-pass SQL worker.** `app/agents/graph.py` is a LangGraph hub-and-spoke. The **supervisor** is a real LLM call (`claude-haiku-4-5`) that decides the next hop via forced structured output — `SupervisorDecision.next` is an enum (`Literal["sql_worker","END"]`), the belt that stops the model routing to a worker that doesn't exist. `route()` fails closed (unknown `next` → `END`) and keeps a mechanical iteration circuit-breaker (`MAX_ITERATIONS`). The **SQL worker** is single-pass: question → `get_schema()` → LLM emits one `SELECT` (structured output `{"sql": ...}`) → `run_query()` through the RO role; query/connection errors return as message text so the graph never crashes. All Anthropic calls go through the `app/llm.py::get_client()` factory (the one place to later add retry/timeout/cost tracking). Importing the module no longer runs the graph — the demo `invoke` is behind `if __name__ == "__main__"` (`python -m app.agents.graph`). *Not yet:* a ReAct refinement loop in the worker, and `extraction`/`RAG` workers.

**Intended (not built):** the extraction and RAG workers, a ReAct loop in the SQL worker, and a HMAC-verified WhatsApp (Meta Cloud API) surface.

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

The extraction logic lives in **`app/extraction/`** (importable modules: `pdf` → `extract` → `schema` → `persist`, plus `manifest`, `batch`, `sample`, `evaluate`); the `scripts/` files are thin CLI wrappers over it. The split is deliberate — persistence can be re-run without paying for the LLM again:
- `scripts/run_extraction.py <pdf> --out <json>` — pdfplumber → Anthropic forced tool-use (single tool, `input_schema` = the Pydantic tree, `tool_choice` forced) → validated **nested** output, cross-checked against the manifest. Needs `ANTHROPIC_API_KEY`.
- `scripts/persist_extraction.py <json> --pdf <pdf>` — deterministic flatten of the nested tree into the 5 tables; the LLM returns a tree because it doesn't know DB ids, and this layer assigns ids/FKs. Provenance comes from the manifest, not the LLM read. Needs `DATABASE_URL`.
- `scripts/run_batch.py` / `app/extraction/batch.py` — mass extraction via the **Batch API at 50% price** (offline, nobody waiting); `sample.py` picks the corpus sample; `evaluate.py` scores a cheaper model against a hand-checked golden (`data/golden/`). `rescue_batch.py` re-reads an already-paid batch's results (they persist ~29 days) so a local *parse* failure costs nothing to retry.

The extracted JSON under `data/extractions/` is gitignored derived output, like the PDFs.

The other scripts are one-off, not production: `susep_probe.py` (blind-sampling viability probe that established corpus volume) and `pilot_extraction.py` (manual hand-read extraction of 2 CGs that validated the schema — produces `data/pilot_extraction.json`, summarized in `data/pilot_findings.md`).

## Conventions

Code comments and the pilot/brief docs are written in **Portuguese (pt-BR)**; domain terms (POS, CG, ramo, vendaval) stay in Portuguese. Match this when editing existing files.
