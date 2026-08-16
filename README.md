# insurance-copilot

A multi-agent system that turns Brazilian home-insurance policy documents into a queryable knowledge base. It harvests *condições gerais* (general terms) registered with SUSEP, extracts their structure into Postgres, and answers coverage-comparison questions in natural language.

> **Status: work in progress.** The data pipeline (SUSEP harvester + extraction schema) and the service skeleton (FastAPI + Postgres) are in place. The agent layer routes each question three ways — a single-pass SQL worker over the Postgres MCP server, a RAG worker over the pgvector index, or an out-of-scope refusal — with a synthesizer node turning the worker's output into a natural-language answer, exposed at `POST /ask`. The RAG side is complete end to end (`clause_chunk` storage, chunking, the Voyage embedding pass, and similarity search), except for the relevance threshold, which is decided but not yet measured. See [Roadmap](#roadmap).

## Why

Comparing home-insurance products in Brazil means reading dozens of 50–90 page PDFs to find what each one actually covers — which perils, which exclusions, how the deductible (POS) works. This project automates that comparison.

**Scope of the data:** the corpus is made of *general terms* (which describe the **product**), not individual policies. So the system answers questions about **coverage structure** — "which insurers cover windstorm without a deductible?", "which perils does insurer A cover that B doesn't?" — and not about **prices**, which live in each customer's individual policy.

## Architecture

A supervisor agent routes each question to specialized workers (hub-and-spoke, **single-hop** — see below):

- **extraction** — turns a policy PDF into structured rows (insurer, product, coverages, perils, exclusions). *Offline pipeline only; no node in the graph yet.*
- **SQL** — aggregates over the structured tables (coverage comparison, deductible structure, exclusion patterns).
- **RAG** — retrieves clause text by similarity over pgvector and hands it to the synthesizer with its origin ids attached.

**The supervisor picks between three destinations**, and the split is *structure vs. wording*: counting, filtering or comparing categorical fields ("how many insurers cover windstorm?") is the SQL worker; anything answered by reading a clause ("in what situations is theft not covered?") is the RAG worker; a question about another line of business, a price, or another subject entirely is `unsupported` and never reaches a worker.

Two rules in the supervisor's prompt exist because both failures are silent and expensive. First, **"not covered" is an answer, not an out-of-scope question**: "is flooding covered?" is answered by the flood *exclusion* clause, which is in the corpus — classifying it as out of scope would refuse a question the system can answer. Second, the **tie-break is `rag_worker`**: refusing by mistake costs a legitimate user the right answer, while searching in vain costs fractions of a cent and comes back with "nothing found".

**The graph is single-hop, and termination is structural.** The supervisor classifies the question **once, before any work**; each worker edges straight to the synthesizer and never back. That is a correction with a measured price tag: with the worker→supervisor edges in place, the supervisor re-classified the same question on every lap, re-routed to the same worker, and only stopped at the `MAX_ITERATIONS` circuit breaker — **10 LLM calls where 3 sufficed**, the supervisor burning **8,790 tokens against the worker's 4,204**, and on the RAG route **4 paid embeddings, 3 of them wasted**. A prompt instruction ("if a worker already answered, choose END") had been tried first: it lowers the *odds* of the loop, not the *cost of the bad case*. The circuit breaker stays as a guard even though nothing can reach it today — one edge pointed back at the supervisor is all it takes to make the cycle possible again.

What this gives up is multi-hop: a compound question ("how many insurers cover windstorm **and** what does the clause say about the deductible?") is answered by a single worker today. That is deliberately a slice of its own — multi-hop needs an explicit stopping rule (who decides it's enough, and on what basis), and inheriting the cycle for free is exactly how the 10 calls happened.

A **synthesizer** node closes every path and turns the worker's raw output into a single natural-language sentence, so the API answers in prose rather than tuples. It reads the result of the **most recent** worker, whichever one ran. Four of its outcomes are fixed sentences returned *without* calling the model — there is nothing to synthesize, and paying for a call to write a sentence that already exists is wasted money:

| Sentence | Meaning | What happened | Traceback in the log? | Declares its base? |
|---|---|---|---|---|
| `NO_ANSWER` | "I should have been able to, and failed" | No worker produced a result **and nothing broke** — under single-hop, only if the supervisor returns `END` or an invalid value (the circuit breaker is unreachable today). A **routing** failure: ours to fix, worth retrying. | no | no — nothing was consulted |
| `NADA_RELEVANTE` | "I searched and the corpus has nothing" | The search ran and every neighbour was past the relevance threshold. A fact about the **corpus**. | no | **yes**, the *searchable* base — without it the sentence reads as "that doesn't exist" instead of "I didn't find it in the ones I have" |
| `FORA_DE_ESCOPO` | "I don't handle that subject" | The supervisor classified the question as another domain, before spending anything. A fact about the **system**; rewording won't help. | no | no — the claim is about scope, and nothing was opened |
| `FALHA_INTERNA` | "something broke on our side" | A worker raised (database down, embedding API refusing, malformed payload). A fact about the **infrastructure** — nothing to do with the question. | **yes** | no — the query never ran, and a base line would lend an air of completeness to the one sentence that means the opposite |

Swapping any of them for another does not break anything — it just lies plausibly — so each has its own test. That last column is what separates `FALHA_INTERNA` from `NO_ANSWER`, which from the outside are both "our problem": one always has an incident to investigate, the other has none, so confusing them sends the operator hunting a traceback that doesn't exist — or ignoring one that does.

None of the four is reachable while there is a real result to present, and inside that case the order runs from the most specific claim to the least: `FALHA_INTERNA` → `NADA_RELEVANTE` → `FORA_DE_ESCOPO` → `NO_ANSWER`. A crashed worker comes first because "the corpus has nothing" and "I don't handle that" are claims **nobody verified** when the query never ran, and `FORA_DE_ESCOPO` would be the worst of them: it sends the user away over a database being down. The out-of-scope check sits *after* the results rather than before, which outlives single-hop on purpose — it is what multi-hop will need, since a supervisor that decides again after each worker can answer `unsupported` on a later hop by reading the RAG worker's own "found nothing", and checking that first would refuse a legitimate question after the search had already been paid for. Likewise `NADA_RELEVANTE` and `FALHA_INTERNA` are final only when there is no other result: if a SQL worker answered earlier, that answer is the one to synthesize. These guards are covered by calling the synthesizer directly with the state multi-hop reintroduces — through `/ask` they are unreachable today, and a test routed through it would pass without exercising anything.

**Every answer declares the base it stands on, and that footer is assembled in code — the synthesizer's prompt never mentions it.** The suffix is deterministic: a number counted in Python glued to a fixed string, joined to the sentence by the one function that is allowed to do it. Asking the model to declare the base would be asking it to *state how many documents we consulted* — precisely the kind of number it invents confidently when it isn't in the payload, and rewrites when it is. A wrong coverage claim is **worse than none at all**: it hands the reader a false reason to trust the answer. So the sentence comes from the model and the footer comes from the code, and both halves are pinned by tests.

There are **three** bases, because three different things get claimed. The SQL route aggregates over whole tables, so its base *is* the corpus: `Base: {n} apólice(s) analisada(s).`, counted from `policy_document`. The RAG route answers by reading `k` clauses from `d` documents, and that is what it covers: `Base: {k} cláusula(s) de {d} apólice(s).` — `k` and `d` counted **after** the relevance cut, never on what the search returned, since counting before inflates both halves and makes the answer look better supported than it is. And `NADA_RELEVANTE` declares the **searchable** base: `Base: {n} apólice(s) pesquisada(s).` That third one exists because semantic search only reaches chunks carrying a vector of the *current* model, so a document that is extracted but not yet embedded — the normal gap until `scripts/embed_chunks.py` runs, or half a corpus after an interrupted `--remodel` — is in the database and out of reach. Using the corpus count there would claim "I searched 40" when 30 were searched, inflating the one sentence whose entire job is to say "I didn't find it *in the ones I have*". The count lives next to the search in `app/rag/search.py` and shares its predicate, so the two cannot drift apart.

Both counts count **products** (`distinct susep_process`), not table rows: `policy_document`'s grain is `(susep_process, version)` and the harvester's `--all-versions` exists to fetch every version of a process, so three versions of one product would otherwise read as three insurers.

There is deliberately **no denominator**. An earlier version closed with "de 138 elegíveis", but that figure lived only in this README's prose and is not derivable from the manifest (147 processes, 143 with a residential version, 141 currently in force) — an unverifiable denominator is the worst possible content for a footer whose only value is being trustworthy, and nothing in the repo would notice it going stale.

**The worker declares the base, in its own message, and no declaration means no footer.** The worker is the one that consulted, so it is the one that knows *whether* it consulted and *what* the query covered — the synthesizer only reads the declaration and renders it. That closes two holes at once. A `run_query` that came back with an error string (invalid SQL is returned as *text*, by contract) declares nothing, so a paraphrased error never gets stamped "N policies analysed" over a SELECT that read none. And the default is fail-*closed*: a future worker that forgets the key gets no footer instead of silently claiming the whole corpus.

Building the footer is **best effort** end to end, like cost attribution: a failed count, or a malformed declaration from another node, drops the footer (warning in the log) rather than blowing up the last node of a request that already has its answer. Inventing a number would be worse — the footer is only worth having if it is trustworthy.

**A worker's exception never reaches the user, and `/ask` still answers 200.** Both workers run their whole body inside a `try`; on failure they log `logger.exception` (traceback + `request_id` + `client`, the same keys as `cost_event`, so the incident links back to the cost rows and the question) and return a message *marked* by `name="worker_error"` whose content is not derived from the exception. Before this, the workers returned `f"SQL error: {exc}"` as if it were a result: that text went into the synthesizer's prompt, and on its degradation path it became *literally* the API's answer — a Postgres `OperationalError` carries host, port, user and database, and a Voyage auth error names an API key. Infrastructure detail is the **operator's** information, and the operator's channel is the log. The status code stays 200 because the surface is a conversation (today `/ask`, tomorrow WhatsApp): a 5xx hands the user an error page or an empty bubble — no answer, and no idea whether retrying helps — while "something broke on my side, try again shortly" is actionable. What the status would buy (alerts, dashboards, investigation) is an operator need, and the log serves it with strictly more than an opaque 500 would carry.

The same split applies one layer down, in the MCP server: `run_query` returns a **query** error as text (the model wrote invalid SQL — that is a result the worker must be able to report) but lets a **connection** error propagate, so it becomes `FALHA_INTERNA` like any other infrastructure failure. That is also what makes the two database calls consistent: `get_schema()` never caught anything, so the same `OperationalError` used to be handled two different ways depending on which call hit it. Still uncovered, and deliberately left for its own slice: the **supervisor** node has no `try`, so an Anthropic outage there still surfaces as a 5xx.

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

**Only rate limits are retried.** The batch is resent with exponential backoff plus jitter (6 attempts, 20s base, factor 2, 120s ceiling) on `RateLimitError` alone — it is the only failure that fixes itself by waiting, because it is a window that reopens. An invalid key or an oversized text propagates immediately: retrying a real error only makes the backfill take minutes to produce the same message. `Timeout` is deliberately excluded too, for a different reason — a 429 is refused *before* processing, so retrying costs nothing twice, but a timed-out request may already have been processed and billed, and resending it would be an undetectable double charge inside the ledger. Every wait is logged at **WARNING** with the attempt number and the seconds, because a 20-minute backfill that goes quiet is indistinguishable from a hung one. When the attempts run out the exception propagates — the per-batch commit has already preserved everything paid for, and a re-run resumes on its own.

**`truncation=False`, against the SDK default.** With truncation on, a text longer than the model's context is silently cut and embedded half-way: `clause_chunk.text` holds the whole clause, the vector covers only its beginning, and `embedding_model` reads `voyage-4-lite` like any correct row — so retrieval matches on a prefix and cites the entire clause. Since chunking deliberately ships without a splitter, this flag is what decides, on the day the corpus brings a long clause, between a loud error and a quietly rotten index. The HTTP timeout is set explicitly (60s) rather than inherited (600s), because the batch's rows are held under `FOR UPDATE` for the duration of the call.

**The recorded price is the list price, not the effective one.** `voyage-4-lite` sits in `pricing.json` at `$0.02` per 1M input tokens and `0` output (an embedding model has no output tokens by nature). Voyage's first 200M tokens are a **credit against the invoice, not a tariff**: recording zero would destroy the corpus cost projection, because "what does re-embedding everything cost?" is a question about the tariff, and the credit runs out. The current corpus is 4,386 chunks / ~199k estimated tokens ≈ US$ 0.004 at list price.

**`embed_pending` commits per batch — the deliberate exception to the transaction contract.** `persist_document` and `index_document` do *not* commit: there the caller owns the transaction and all-or-nothing is right, because repeating the pass is free. Here each batch is a paid API call, so the transaction boundary follows the **cost of repeating**: a failure on batch 15 must not throw away the 14 batches already paid for. Each batch is committed together with its own `cost_event` row (`agent_name="embedder"`, one row per *call*, `request_id`/`client` `NULL` because the pass is offline), so a re-run resumes exactly where it stopped — committed rows no longer match `embedding IS NULL`. A second pass over a fully indexed corpus makes **zero** API calls, which is what the test asserts (counting calls, not comparing the database: a version that re-embedded everything and rewrote identical vectors would leave the database unchanged and the invoice larger).

### Similarity search — `search_clauses` (`app/rag/search.py`)

`search_clauses(session, question, *, k=5, max_distance=None, document_ids=None)` embeds the question with `input_type="query"` and returns the nearest chunks by cosine distance as `Hit(chunk_id, document_id, exclusion_id, coverage_id, text, distance)`. The origin ids travel with the hit because a recovered clause without its provenance is just a sentence.

**The graph reaches it through the `rag_worker` node** (`app/agents/graph.py`), which opens its own `SessionLocal`, calls `search_clauses(k=5)` and formats each hit as one line carrying `chunk_id`, `document_id` and the distance — the ids are what make a citation possible downstream. Unlike every other node it makes no Anthropic call: the only paid call is the question's embedding, inside the search. The threshold **is** applied, and is not optional: without it a question that slipped past the supervisor comes back with five residential clauses and the synthesizer turns them into a confident wrong answer. It is applied **in the node rather than as `max_distance`** so the discarded distances stay visible — the same cut either way, but those numbers are what tell you whether `MAX_DISTANCE_PADRAO` sits in the right place, and when the cut empties the list the best distance is logged. An empty result becomes an explicit "nothing relevant found" message, never an empty string — a blank message in the history is indistinguishable from "the worker never ran", and the answer would degrade from "I searched and found nothing" into "I failed". Hit text is `.strip()`ed: the message becomes the final `assistant` turn of the next supervisor call, and the API rejects one ending in whitespace with a 400 — PDF-derived clause text ends in `\n` easily, and the R2a chunker only rejects *blank* text.

**The vector search always returns `k` results — there is no "not found".** `ORDER BY embedding <=> :q LIMIT k` hands back the k least distant chunks in the corpus even when the closest one is on the other side of the space: a question about travel-insurance cancellation comes back with the five *least unrelated* home-insurance clauses, and a confident synthesizer turns that into a wrong answer with a citation. So an empty result is produced **by the threshold** (`max_distance`), not by "nothing found" — which is what the test named after it pins. The only other way to get fewer than `k` is the model filter described below, and it is deliberate.

**`hnsw.ef_search` is a silent ceiling, so each search sets it per transaction.** pgvector's default scans 40 candidates: with the index in use, a `LIMIT 100` comes back with **40 rows and no error** — verified on the real 4,386-chunk corpus by forcing `enable_seqscan=off`, where `SET LOCAL hnsw.ef_search = 100` restores all 100. At today's size the planner still prefers a seq scan and results are exact, which is precisely the trap: the "k results" guarantee would be resting on a plan choice that flips on its own as the table grows. So `search_clauses` sets `hnsw.ef_search = min(max(4k, 40), 1000)` before the select: it tracks `k`, floored at the default and capped at pgvector's maximum.

**`ef_search` alone does not cover the filtered case, and the filtered case is the one that breaks.** The `WHERE` (`document_id`, `embedding_model`, `IS NOT NULL`) is applied *after* the approximate scan, so of ~40 global candidates only the ones passing the filter survive. Measured, forcing the HNSW plan on the real corpus: `k=5` restricted to one document returned **1 row**; with `hnsw.iterative_scan` on, 5 of 5. Stretching `ef_search` by a fixed factor does not fix it either — the dilution is the filter's *selectivity* (29 documents ⇒ ~29×), not a multiple of `k`; at the default `k=5` a factor of 4 yields exactly pgvector's own default, i.e. nothing. `hnsw.iterative_scan = strict_order` is built for this: the index returns further batches until `LIMIT` rows survive the filter. `strict_order` rather than `relaxed_order` because here the distance ordering *is* the result — the threshold cuts from the top, so "roughly ordered" means discarding a hit by accident. It is on for every search, not just filtered ones: the model filter dilutes just as much on a half-remodelled corpus, and with no filter the first batch already satisfies the `LIMIT`, so it costs nothing. Both parameters go through `set_config(..., is_local => true)` — `SET LOCAL` as a function: same lifetime (dies with the transaction, never leaks to other connections), one round trip, and it accepts bind parameters, which `SET` syntax does not.

**Only vectors from the current model are ranked.** `embed_pending` commits per batch by design, so a `--remodel` interrupted halfway leaves the corpus split across two models — and cosine distances from different models are not comparable. Without a filter the search would order both spaces together and return wrong neighbours with no error: the same "index that lies" the write side already refuses to create, entering through the read path. Hence `WHERE embedding_model = EMBED_MODEL`, with a deliberate consequence recorded in the docstring — a half-remodelled corpus returns fewer results, or none, instead of returning wrong ones. Empty is recoverable by finishing the `--remodel`; a confident wrong ranking is not.

**Waiting is bounded at both ends here, and the timeout is the term that matters.** Retries are on a shorter budget than the backfill's (2 attempts starting at 2s, against 6 starting at 20s), because the offline pass can afford to sleep for minutes while a search holds an `asyncio.to_thread` worker inside a request. But attempt count alone bounds nothing: the client's `timeout` is what caps a single call, and the backfill's 60s was chosen for a 200-text batch held under `FOR UPDATE`. So search uses its own cached client (`get_query_client`, 10s) — the timeout is a property of the client, not the call, so two paths with opposite patience need two clients. A few concurrent searches under the old arrangement would exhaust the default executor, which also stalls the graph's synchronous Postgres calls.

**The threshold is cut after the `LIMIT`, not in the `WHERE`.** It is a relevance cut, not pagination: filtering in SQL would make the query dig deeper looking for k rows that qualify — more work for the same answer — and would hide the discarded distances from the tests and from calibration.

**`MAX_DISTANCE_PADRAO` is `0.60` — a decided number, not yet a measured one — and it is not applied by default** (`max_distance=None` means no cut; callers who want one pass it explicitly, because silently dropping results on an unvalidated threshold is worse than returning them). To validate it, `scripts/calibrate_search.py` runs the search *without* a threshold over labelled questions (`data/eval/search_questions.txt`: 10 in-scope, 5 out-of-scope) and prints, per question, the 1st/3rd/5th distances plus the mean first-hit distance per group and — the point of the exercise — the **largest `+` distance against the smallest `-` distance**. If they separate, any value in between is the threshold; if they overlap, no distance cut separates the two groups and that is a result too. Calibrating (and then fixing the constant) is still open.

The out-of-scope half is deliberately two kinds: entirely off-domain (a carrot cake recipe, filing income tax) and **insurance from another line of business** (life, travel, cosmetic surgery). Only the second kind is a real test — a threshold that merely rejects the cake is separating vocabulary, not subject matter, and everyone who writes to this system writes in insurance language. On the other side, a bicycle kept inside the home and a car parked in the house's garage are labelled in-scope on purpose: residential general terms do speak to both, so they are the questions that sound peripheral and are not.

```bash
python scripts/calibrate_search.py             # 1 embedding call per question (15 by default); needs VOYAGE_API_KEY + an embedded corpus
python scripts/calibrate_search.py -k 10
```

`document_ids` filters **before** ranking, in SQL: narrowing to an insurer or product is *identity*, which chunking deliberately kept out of the chunk text precisely because a `WHERE` answers it exactly and for free. An empty list means "no documents" and short-circuits to `[]` without paying for the embedding. Rows with `embedding IS NULL` (chunked by R2a, not yet embedded) never take part — they carry no distance and are not in the HNSW index.

The question's embedding is a paid call like any other, so it writes a `cost_event` with `agent_name="rag_search"` — and since search runs inside a request, `request_id`/`client` come from the ContextVars, unlike the offline `embedder` rows. The write is best effort (a failed ledger row must not 500 a request that already paid). One caveat worth knowing: a single question is ~6 tokens, which at $0.02/1M rounds to `0.000000` in `Numeric(12, 6)` — the truthful, aggregatable number for retrieval is `input_tokens`, not the per-row `cost_usd`.

**Verifying the plan.** Cosine (`<=>`) is not a preference, it is the operator class the index was built with (`USING hnsw (embedding vector_cosine_ops)`); using any other distance still returns *results*, just computed by a full scan. Check it against the real corpus:

```bash
docker compose exec postgres psql -U insurance -d insurance -c "
EXPLAIN ANALYZE
SELECT id, document_id, embedding <=> (SELECT embedding FROM clause_chunk ORDER BY id LIMIT 1) AS distance
FROM clause_chunk
WHERE embedding IS NOT NULL
ORDER BY embedding <=> (SELECT embedding FROM clause_chunk ORDER BY id LIMIT 1)
LIMIT 5;"
```

What to look for: `Index Scan using ix_clause_chunk_embedding` with an `Order By: (embedding <=> $0)` line — the index is doing the ordering. What means it is *not*: a `Seq Scan on clause_chunk` followed by `Sort`, which is what the same query produces if you swap `<=>` for `<->` (L2) — verified on the 4,386-chunk corpus, where the cosine version runs the HNSW scan and the L2 version falls back to seq scan + sort. On a small corpus a seq scan is still fast, so the plan — not the wall clock — is the thing to check.

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

Ask the agent graph a question (needs `ANTHROPIC_API_KEY`, a populated database, and a token from `API_TOKENS` — this spends money: 3 LLM calls on a worker route — supervisor, worker, synthesizer — and 1 on the out-of-scope route. A question routed to RAG also needs `VOYAGE_API_KEY` and an embedded corpus, and pays one embedding call):

```bash
# structure question -> sql_worker
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $YOUR_TOKEN" \
  -d '{"question":"Quantos perigos existem na base?"}'
# {"answer":"Existem 7 perigos cadastrados na base.","iterations":1}

# wording question -> rag_worker
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $YOUR_TOKEN" \
  -d '{"question":"Em que situações o roubo não é coberto?"}'

# another subject -> unsupported (one supervisor call and nothing else)
curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $YOUR_TOKEN" \
  -d '{"question":"Seguro de vida cobre suicídio?"}'
# {"answer":"Só respondo perguntas sobre as condições gerais de seguro residencial...","iterations":1}
```

If a worker breaks (database down, embedding API refusing), the call still returns **200** with `{"answer":"Tive uma falha interna ao consultar as condições gerais. Tente de novo em alguns instantes.", ...}` — the diagnosis is in the server log, not in the response. See the sentence table under [Architecture](#architecture).

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

One trap, if you write a test that asserts on log output — or if you add another
in-process `command.upgrade` call: the migrations run **inside** the pytest process, and
configuring logging is a process-global act. `alembic/env.py` therefore skips `fileConfig`
entirely when the caller passes `configure_logger=False` in the Config's `attributes`,
which is what `tests/conftest.py` does. Both of `fileConfig`'s defaults are hostile here:
it disables every logger created so far, **and** it replaces the root handlers — where
pytest's `caplog` handler lives. Either one makes a log assertion silently see nothing
and pass. (`env.py` also passes `disable_existing_loggers=False` as a net for callers that
don't know about the flag, but that alone does not cover the root-handler half.)
`tests/test_migration.py` pins both properties.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every pull request
and on pushes to `main`: Python 3.11, `pip install -r requirements.txt`, `pytest -q`.
It deliberately sets no `DATABASE_URL` and writes no `.env` — the lazy engine is what
makes a database-less checkout able to import the app and let testcontainers do the
rest. No secrets are needed either: every test uses fakes, so nothing calls the
Anthropic API.

### Auth and rate limiting on `/ask`

`POST /ask` is the only paid endpoint (up to three Anthropic calls plus, on the RAG route, one embedding), so it is closed by default. `/health` and `/health/db` stay open — they are the Railway healthcheck, which sends no `Authorization` header.

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
│   ├── rag/                # chunking + embedding + similarity search over clause_chunk
│   └── models.py           # ORM models: PolicyDocument, Coverage, Peril, CoveragePeril, Exclusion, ClauseChunk, CostEvent
├── alembic/
│   └── versions/           # migrations (alembic upgrade head)
├── docs/
│   └── schema.html         # visual schema diagram (open in browser)
├── scripts/
│   ├── susep_harvest.py    # SUSEP corpus harvester
│   ├── calibrate_search.py # runs the search unthresholded to find the relevance cut
│   └── eval/               # F4 extraction eval harness (see "Extraction eval")
├── data/
│   ├── corpus/             # downloaded PDFs (gitignored) + corpus_manifest.json
│   └── eval/               # labelled questions for the search threshold calibration
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
- [~] Agent layer — async LLM supervisor (structured output) classifying each question once into a single-pass SQL worker, a RAG worker or an out-of-scope refusal, closed by a synthesizer (natural-language answer), served at `POST /ask`; **single-hop by design** (measured: the cyclic version cost 10 LLM calls where 3 sufficed). Pending: multi-hop with an explicit stopping rule, the extraction worker, and a ReAct refinement loop
- [x] `POST /ask` hardening — Bearer auth with identity + per-client and per-IP rate limits; a worker's exception never reaches the caller (marked failure message → `FALHA_INTERNA`, traceback with `request_id`/`client` in the log); every answer declares the base it stands on — the corpus count on the SQL route, the retrieved-clause count on the RAG route, the *searchable* count on "found nothing" — declared by the worker that consulted and assembled in code, never by the model
- [x] Cost attribution in the agent graph — one `cost_event` per LLM call, tagged with a per-request id and the calling client
- [~] RAG worker — vector storage ready (pgvector extension, `clause_chunk` with an exclusive-arc origin, HNSW/cosine index, revoked from the SQL worker's role), chunks materialized from the extracted text (one source row = one chunk, idempotent re-indexing that invalidates the vector when the text changes), embeddings filled by a resumable, cost-attributed pass (Voyage `voyage-4-lite`, `input_type="document"`, one `cost_event` per batch), similarity search as a testable function (`search_clauses`, `input_type="query"`, relevance threshold, per-document filter), and a `rag_worker` node wired into the graph behind a three-way supervisor; **pending:** calibrating the threshold against the labelled question set
- [ ] WhatsApp surface
- [ ] Deploy to Railway

## License

MIT
