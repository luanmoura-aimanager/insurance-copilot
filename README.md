# insurance-copilot

A multi-agent system that turns Brazilian home-insurance policy documents into a queryable knowledge base. It harvests *condições gerais* (general terms) registered with SUSEP, extracts their structure into Postgres, and answers coverage-comparison questions in natural language.

> **Status: work in progress.** The data pipeline (SUSEP harvester + extraction schema) and the service skeleton (FastAPI + Postgres) are in place. The agent layer has its first real slice — an LLM supervisor routing to a single-pass SQL worker over the Postgres MCP server, with a synthesizer node turning the query result into a natural-language answer, exposed at `POST /ask`. The RAG worker has its vector storage (pgvector + `clause_chunk`), its chunk materialization and its embedding pass (`app/rag/`, Voyage `voyage-4-lite`); retrieval is not wired yet. See [Roadmap](#roadmap).

## Why

Comparing home-insurance products in Brazil means reading dozens of 50–90 page PDFs to find what each one actually covers — which perils, which exclusions, how the deductible (POS) works. This project automates that comparison.

**Scope of the data:** the corpus is made of *general terms* (which describe the **product**), not individual policies. So the system answers questions about **coverage structure** — "which insurers cover windstorm without a deductible?", "which perils does insurer A cover that B doesn't?" — and not about **prices**, which live in each customer's individual policy.

## Architecture

A supervisor agent routes each question to specialized workers (canonical hub-and-spoke):

- **extraction** — turns a policy PDF into structured rows (insurer, product, coverages, perils, exclusions).
- **SQL** — aggregates over the structured tables (coverage comparison, deductible structure, exclusion patterns).
- **RAG** — retrieves and explains raw clause text (pgvector). *Storage, chunking and embedding are in place (`clause_chunk`, `app/rag/`); retrieval is not.*

A **synthesizer** node closes every path: it turns the worker's raw output into a single natural-language sentence, so the API answers in prose rather than in tuples. When no worker ran (the question is out of scope), it returns a fixed sentence *without* calling the model — there is nothing to synthesize, and paying for a call to say "I don't know" is wasted money.

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

### Retrieval storage — `clause_chunk` (pgvector)

Postgres runs the `pgvector/pgvector:pg16` image (the official `postgres:16` plus the `vector` extension compiled in), and `clause_chunk` stores one **chunk** of clause text per row, with its embedding.

- **The grain is the chunk, not the extracted row.** A long clause becomes several pieces, each with its own vector. Putting the embedding on `exclusion` / `coverage` would force one chunk per row — bad retrieval — and tie the domain schema to whichever embedding model is current.
- **The origin is an exclusive arc**: `exclusion_id` **or** `coverage_id` — exactly one, enforced by a check constraint. Real foreign keys, so the "cite where this came from" guarantee has referential integrity behind it. The FKs are **composite with `document_id`** (`(document_id, exclusion_id) → exclusion(document_id, id)`), which is what forbids a chunk whose `document_id` and whose origin belong to different documents — such a row would slip past the by-`document_id` delete and then break the FK on the way out, leaving a half-deleted document. `MATCH SIMPLE` is why this composes with the arc: a FK with any `NULL` column isn't checked, so only the arm in use is validated. Partial unique indexes on `(exclusion_id, chunk_index)` and `(coverage_id, chunk_index)` make re-indexing idempotent at half the index size (by design, half the rows are `NULL` in each arm).
- Embeddings are **voyage-4-lite, 1024 dimensions** — multilingual (the corpus is pt-BR), cheap, and the smaller dimension keeps the index light. The dimension lives in the column type (`vector(1024)`) because Postgres needs it to index, so changing models is a migration, not configuration.
- `embedding` / `embedding_model` are nullable: chunking and embedding are separate steps, so `NULL` means "not indexed yet".
- The index is `USING hnsw (embedding vector_cosine_ops)`, matching the cosine distance the search will use. It is written as raw SQL in the migration (Alembic emits neither `CREATE EXTENSION` nor an operator class) *and* declared on the model, so a later `--autogenerate` doesn't propose dropping it. `document_id` gets a plain btree index too — Postgres does not index FK child columns on its own, and this is by design the largest table in the schema.

The first version modelled the origin as a polymorphic pointer (`source` text + `source_id` with no FK). It was replaced because neither promise held: a `source_id` with no foreign key goes dangling when its origin row is deleted (and re-extraction reuses ids, so the chunk starts citing a *different* clause), and the unique constraint enforced nothing at all, since `source_id` was nullable and Postgres treats `NULL`s as distinct in a unique index.

**`clause_chunk` is not reachable by the SQL worker**, enforced in two independent layers: a **table allowlist in `run_query`** (every `FROM`/`JOIN` target must be one of the 5 domain tables or a local CTE), *and* a `REVOKE SELECT` on the table for `insurance_ro`. Being absent from `TABLES` was never protection on its own — the text filter only inspects the first token, so a `SELECT * FROM clause_chunk` written by the model would pass and the automatic `LIMIT 100` would dump 100 vectors of 1024 floats into the next call's context. Neither layer subsumes the other: the revoke does nothing when `DATABASE_URL_RO` is unset and connections fall back to the admin URL, and the allowlist would not stop a writable role if the `SELECT` filter were bypassed. The revoke is surgical — the default privileges for future tables stay in place, and the domain tables stay readable. Similarity search belongs to the RAG worker.

### Chunking — one source row, one chunk (`app/rag/`)

`app/rag/` fills `clause_chunk` from the text extraction already persisted: `chunking.py` (pure functions, no database) builds each chunk's text and `index.py::index_document(session, document_id)` writes the rows, leaving `embedding`/`embedding_model` `NULL`. This pass costs nothing; the vectors are filled by a separate pass ([below](#embedding--the-pass-that-costs-money-apprag)).

**There is no splitter, and that is a measured decision.** Across the 30 persisted documents, the 4,176 exclusions have a **median length of 101 characters**, a p99 of 456 and a maximum of 907; the 210 deductible rules average 180. Nothing comes near 1,400 characters — far below any embedding model's limit. So **one source row = one chunk**, and `chunk_index` is always `0` (the largest chunk produced over the real corpus is 934 characters: the 907-character clause plus its header). Splitting by size and overlapping pieces would be machinery with no text to exercise it; when the corpus does bring long clauses, `chunk_index` is already in the schema and only the chunking layer changes. Current corpus: 4,176 + 210 = **4,386 chunks**.

**The chunk text carries meaning, not identity.** A header exists because `"Danos causados por ato doloso"` on its own doesn't say what it is an exclusion *of* — an exclusion of the Fire coverage and a policy-wide exclusion are different facts, and the vector has to know which. The three formats are deliberately short (`Exclusão da cobertura {name}: …`, `Exclusão geral da apólice: …`, `Regra de franquia da cobertura {name}: …`): with the median clause at 101 characters, a long prefix would dominate the vector and search would start ranking by header. **Insurer, product and SUSEP process stay out** — that is *identity*: it already lives in `policy_document` and is answered by a SQL filter (`WHERE document_id ...`) at query time, exactly and for free. Inside the text it would only pull every chunk from the same insurer closer together, noise that drags similarity away from what the question actually asks.

**Re-indexing is idempotent, and new text invalidates the vector.** The upsert conflicts on the two partial unique indexes — which is why `index_where` must accompany `index_elements` (without it Postgres cannot infer the arbiter), and why there are two upserts, one per arm of the arc. On conflict the text is always refreshed, while `embedding`/`embedding_model` go through a `CASE`: **preserved when the text is identical, reset to `NULL` when it changed**. An old vector over new text is an index that *lies* — search keeps matching the old wording and confidently cites a clause that no longer exists, and `NULL` simply means "not indexed yet". Both halves matter: always clearing the vector would still pass the invalidation test and force a full re-embed of the corpus on every pass, which is real money.

**The pass is authoritative — it also prunes.** An upsert alone is idempotent only for sources that were *added or changed*, never for sources that were *removed*: clearing `coverage.deductible_rule_text` leaves the old chunk alive with its paid-for vector, and retrieval starts confidently citing a deductible rule the document no longer contains. So `index_document` deletes the document's chunks that this pass did not produce, in the same transaction. Blank source text (`""`, `"   "`) is treated as no text for the same reason — it passes the `IS NOT NULL` filter but yields a chunk that is nothing but its header: zero content, a paid embedding, and one more near-identical noise vector competing inside the HNSW index.

```bash
python scripts/index_chunks.py                  # every document; costs nothing (no LLM, no embedding API)
python scripts/index_chunks.py --document-id 7  # just one
```

### Embedding — the pass that costs money (`app/rag/`)

`app/rag/embedding.py` is the only place that talks to Voyage; `app/rag/embed.py::embed_pending` walks the rows where `embedding IS NULL` and fills them in batches. Needs `VOYAGE_API_KEY`.

```bash
python scripts/embed_chunks.py --dry-run   # how many are pending, and what it would cost
python scripts/embed_chunks.py             # embed everything pending
python scripts/embed_chunks.py --limit 50  # a small slice first
python scripts/embed_chunks.py --remodel   # also re-embed rows carrying another model's vector
```

**Changing the model is not allowed to be a silent no-op.** Already-embedded rows never return to the `embedding IS NULL` filter, so pointing `EMBED_MODEL` at a same-dimension successor and running would fill only the `NULL`s — leaving the HNSW index holding vectors from **two models**, whose cosine distances are not comparable, and retrieval returning wrong neighbours with no error at all. So the pass refuses to run (before the first paid call) when any row carries another model's vector; `--remodel` includes those rows and restores consistency. It is opt-in because re-embedding the corpus is spending, and spending shouldn't happen as a side effect.

**Each batch is claimed with `FOR UPDATE SKIP LOCKED`.** Two concurrent passes would otherwise read the same ids, both pay for the same text, and both write a `cost_event` — double spend inside the ledger this project exists to keep trustworthy. The lock lasts the *batch*, matching the per-batch commit; the accepted cost is that the transaction stays open across the HTTP call and any backoff, which is where a Postgres with `idle_in_transaction_session_timeout` enabled would bill you (stock Postgres ships it off). The dry-run counts without locking.

**`input_type="document"` is not decoration.** Voyage trains the model *asymmetrically*: documents and queries are embedded with different prefixes, and the space is optimized so a **question** lands near the **documents that answer it** — not near other questions. Indexing always uses `input_type="document"`; retrieval (R3) will query with `input_type="query"`. Getting the pair wrong raises no error at all — it silently degrades recall — so the contract is pinned by a test.

**Only rate limits are retried.** The batch is resent with exponential backoff plus jitter (6 attempts, 20s base, factor 2, 120s ceiling) on `RateLimitError` alone — it is the only failure that fixes itself by waiting, because it is a window that reopens. An invalid key, an oversized text or an API outage propagate immediately: retrying a real error only makes the backfill take minutes to produce the same message. Every wait is logged at **WARNING** with the attempt number and the seconds, because a 20-minute backfill that goes quiet is indistinguishable from a hung one. When the attempts run out the exception propagates — the per-batch commit has already preserved everything paid for, and a re-run resumes on its own.

**The recorded price is the list price, not the effective one.** `voyage-4-lite` sits in `pricing.json` at `$0.02` per 1M input tokens and `0` output (an embedding model has no output tokens by nature). Voyage's first 200M tokens are a **credit against the invoice, not a tariff**: recording zero would destroy the corpus cost projection, because "what does re-embedding everything cost?" is a question about the tariff, and the credit runs out. The current corpus is 4,386 chunks / ~199k estimated tokens ≈ US$ 0.004 at list price.

**`embed_pending` commits per batch — the deliberate exception to the transaction contract.** `persist_document` and `index_document` do *not* commit: there the caller owns the transaction and all-or-nothing is right, because repeating the pass is free. Here each batch is a paid API call, so the transaction boundary follows the **cost of repeating**: a failure on batch 15 must not throw away the 14 batches already paid for. Each batch is committed together with its own `cost_event` row (`agent_name="embedder"`, one row per *call*, `request_id`/`client` `NULL` because the pass is offline), so a re-run resumes exactly where it stopped — committed rows no longer match `embedding IS NULL`. A second pass over a fully indexed corpus makes **zero** API calls, which is what the test asserts (counting calls, not comparing the database: a version that re-embedded everything and rewrote identical vectors would leave the database unchanged and the invoice larger).

Retrieval is still pending, and the agent graph is unchanged — see [Roadmap](#roadmap).

## Tech stack

FastAPI · Postgres (+ pgvector) · SQLAlchemy 2.0 (async) · Alembic · LangGraph · Anthropic (agents/extraction) · Voyage (embeddings) · Docker Compose · pytest + testcontainers · GitHub Actions · deployed on Railway.

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
docker compose up -d        # serves Postgres (pgvector/pgvector:pg16) on localhost:5433

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

Ask the agent graph a question (needs `ANTHROPIC_API_KEY`, a populated database, and a token from `API_TOKENS` — this spends money: one LLM call per supervisor hop, per worker, and one for the synthesizer):

```bash
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $YOUR_TOKEN" \
  -d '{"question":"Quantos perigos existem na base?"}'
# {"answer":"Existem 7 perigos cadastrados na base.","iterations":2}
```

### Tests and CI

```bash
pytest -q       # needs Docker running (Colima on macOS); no environment variables
```

The suite boots its own throwaway Postgres with testcontainers and applies the real
migrations to it, so **no environment variable is required to run it** — not even
`DATABASE_URL`. That works because the database engine is **lazy**: `app/db.py` reads
`DATABASE_URL` on the first `SessionLocal()` call (behind an `lru_cache`, so there is
exactly one engine per process), not at import time. Importing the app used to blow up
with `KeyError: 'DATABASE_URL'` in any environment without a database configured —
including pytest's own collection phase, before the container existed.
`tests/test_lazy_db.py` pins that: it imports `app.main` in a subprocess with a
scrubbed environment and asserts it exits 0.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every pull request
and on pushes to `main`: Python 3.11, `pip install -r requirements.txt`, `pytest -q`.
It deliberately sets no `DATABASE_URL` and writes no `.env` — the lazy engine is what
makes a database-less checkout able to import the app and let testcontainers do the
rest. No secrets are needed either: every test uses fakes, so nothing calls the
Anthropic API.

### Auth and rate limiting on `/ask`

`POST /ask` is the only paid endpoint (each supervisor hop is an Anthropic call), so it is closed by default. `/health` and `/health/db` stay open — they are the Railway healthcheck, which sends no `Authorization` header.

| env | example | meaning |
| --- | --- | --- |
| `API_TOKENS` | `whatsapp:abc…,dev:def…` | `name:token` pairs. The **name** identifies the caller; the token is the secret. Generate with `openssl rand -hex 32`. |
| `ASK_RATE_LIMIT_CLIENT` | `30/hour` | Spend quota, keyed by the token's name. Per-route (`/ask`). |
| `ASK_RATE_LIMIT_IP` | `10/minute` | Burst ceiling, keyed by IP. Applies to every non-exempt route, and counts rejected requests too. |

Tokens carry an **identity** rather than being one shared secret, which is what makes per-client limits (and revoking one caller without touching the others) possible. Comparison uses `secrets.compare_digest` over *all* tokens with no early exit, so neither the token prefix nor its position in the list leaks by timing. An unset or empty `API_TOKENS` returns **500**, not 401 — a broken deploy should not look like a bad client credential.

**The two limits are applied at different layers, and that split is the point.**

The **per-IP ceiling** is a `default_limits` enforced by `SlowAPIMiddleware`, which runs before routing and before dependencies. It therefore counts requests that never reach the endpoint — including the **401s**. As a route decorator it would never be reached at all: `require_client` is a dependency, FastAPI resolves dependencies before invoking the endpoint, so an invalid token raised 401 and the limiter was never consulted. Anyone could hammer `/ask` with wrong tokens for free.

It applies to every route that isn't exempt, but it is *counted per route path*: slowapi's default `key_style` is `"url"`, so the bucket key is `(IP, path)`. `10/minute` means ten requests per minute **per path**, not ten across the whole API — worth knowing before adding a second paid route.

The **per-client quota** stays a route decorator on `/ask`, because it keys on `request.state.client_name`, which only exists after authentication has run.

Each request is counted **once per limit**, never twice: slowapi only folds the global default into the route check when every route limit has `override_defaults=False`, and `.limit()` sets it to `True`. So the middleware pass evaluates only the IP ceiling and the decorator pass only the client quota. Exceeding either returns **429**.

`/health` and `/health/db` are `@limiter.exempt`, so they are outside the global ceiling too. Railway's healthcheck polls them in a loop; if that consumed quota, the platform would start receiving 429s and tear down a perfectly healthy service.

A limit string that doesn't parse falls back to the default with a logged warning — slowapi's own behaviour is to swallow the parse error and apply *no* limit, so a typo in the env would silently remove the spend ceiling.

**Deploying behind a proxy** (Railway, and any other edge that terminates TLS): run uvicorn with

```bash
uvicorn app.main:app --proxy-headers --forwarded-allow-ips='*'
```

Without it every request arrives with the proxy's IP as `request.client.host`, so the per-IP limit counts the whole internet as a single caller and the first burst locks out everyone. Prefer the proxy's actual address over `'*'` when you know it — `--forwarded-allow-ips` decides who is trusted to set `X-Forwarded-For`, and trusting everyone lets a client forge its own IP (which, here, only lets it *dodge* the IP limit; the per-client limit is unaffected).

## Project structure

```
insurance-copilot/
├── .github/
│   └── workflows/ci.yml    # CI: pytest on every PR and push to main
├── app/
│   ├── main.py             # FastAPI app + health endpoints + POST /ask
│   ├── auth.py             # Bearer auth with identity (API_TOKENS: name -> token)
│   ├── limits.py           # slowapi limiter: per-client + per-IP keys
│   ├── db.py               # lazy async engine + session factory (SQLAlchemy 2.0)
│   └── models.py           # ORM models: PolicyDocument, Coverage, Peril, CoveragePeril, Exclusion, ClauseChunk, CostEvent
├── alembic/
│   └── versions/           # migrations (alembic upgrade head)
├── docs/
│   └── schema.html         # visual schema diagram (open in browser)
├── scripts/
│   ├── susep_harvest.py    # SUSEP corpus harvester
│   └── eval/               # F4 extraction eval harness (see "Extraction eval")
├── data/
│   └── corpus/             # downloaded PDFs (gitignored) + corpus_manifest.json
├── docker-compose.yml      # local Postgres
├── requirements.txt
└── .env.example
```

## Extraction eval (F4)

A zero-cost harness that checks whether the LLM extraction is faithful to the source CGs —
no re-extraction, **no Anthropic API calls**. It runs on the Claude Code subscription: the
scripts only read Postgres + the source PDFs, and the judging is done by Claude reading each
CG's text against the persisted coverages.

Two parts: a deterministic floor (integrity + a term-density vs coverage-count check via
`scripts/diagnose_extraction.py`) and a per-document judge (MISSING / HALLUCINATED, with a
verbatim source quote required for every finding).

```bash
docker compose up -d && source .venv/bin/activate
# dump source text + persisted coverages per doc into <out_dir> (one bundle per document)
python scripts/eval/dump_judge_bundles.py <out_dir>
# per-doc comparison sheet: extracted coverages vs candidate headers found in the source text
python scripts/eval/judge_sheet.py <out_dir> <slug>
```

**Result** ([`scripts/eval/f4_extraction_eval.md`](scripts/eval/f4_extraction_eval.md)):
30 insurers judged against their source text — **29 PASS · 1 MINOR · 0 FAIL · 0 hallucination**
(the MINOR is one accessory Assistência-24h coverage HDI skipped). Extraction is faithful at the
coverage grain; extrapolated to the 138 eligible residential CGs, this is the confidence baseline.

Open items: (1) non-deterministic scope for Assistência 24h — sometimes extracted as a coverage,
sometimes dropped (ADR candidate); (2) provenance mismatch on XS3 (`susep_process` in DB vs the
PDF footer).

## Roadmap

- [x] Corpus harvester (SUSEP, residential)
- [x] Extraction schema validated against real policies
- [x] Service skeleton — FastAPI + Postgres, health checks
- [x] ORM models — all 5 tables: `policy_document`, `coverage`, `peril`, `coverage_peril`, `exclusion`
- [x] Alembic migrations
- [x] Test suite (testcontainers)
- [x] CI on GitHub Actions — `pytest -q` on every PR and push to `main`, with no database configured (lazy engine)
- [ ] Production extraction (LLM → tables)
- [x] Postgres MCP SQL server + read-only `insurance_ro` role
- [~] Agent layer — async LLM supervisor (structured output) + single-pass SQL worker + synthesizer (natural-language answer), served at `POST /ask`; RAG/extraction workers + a ReAct refinement loop pending
- [x] `POST /ask` hardening — Bearer auth with identity + per-client and per-IP rate limits
- [x] Cost attribution in the agent graph — one `cost_event` per LLM call, tagged with a per-request id and the calling client
- [~] RAG worker — vector storage ready (pgvector extension, `clause_chunk` with an exclusive-arc origin, HNSW/cosine index, revoked from the SQL worker's role), chunks materialized from the extracted text (one source row = one chunk, idempotent re-indexing that invalidates the vector when the text changes) and embeddings filled by a resumable, cost-attributed pass (Voyage `voyage-4-lite`, `input_type="document"`, one `cost_event` per batch); retrieval pending
- [ ] WhatsApp surface
- [ ] Deploy to Railway

## License

MIT
