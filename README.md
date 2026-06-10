# insurance-copilot

Capstone II (Cap 4) — sistema multi-agent que extrai informação de condições gerais de **seguro residencial** (PDFs da SUSEP) para Postgres e responde perguntas de análise via WhatsApp.

Brief completo, schema e fonte de corpus em `../AI Eng Journey/`:
- `insurance-copilot-brief.md`
- `insurance-copilot-extraction-schema.md`
- `susep-corpus-source.md`

## Estrutura

```
insurance-copilot/
├── scripts/
│   ├── susep_probe.py          # sonda de viabilidade (amostragem cega de IDs)
│   ├── susep_harvest.py        # harvester índice→resolve→download (entregável)
│   └── CLAUDE_CODE_BRIEF.md     # brief original do harvester
├── data/
│   ├── index/                  # cache do índice OData (gitignored)
│   └── corpus/                 # PDFs (gitignored) + corpus_manifest.json (commit)
└── (skeleton da app — ver abaixo)
```

## Corpus SUSEP (residencial)

`scripts/susep_harvest.py` monta o corpus de condições gerais de **seguro
residencial** (ramo `01 | COMPREENSIVO RESIDENCIAL`) usando 3 endpoints
públicos, sem login:

1. **Índice** — OData da SUSEP (Olinda), recurso `DadosProdutos`. Lista todos
   os produtos com `{tipoproduto, entnome, cnpj, numeroprocesso, ramo, subramo}`.
   Filtramos residencial. *Gotcha:* o serviço só aceita `$format=json` (nada de
   `$top`/`$filter`/`$select` — devolve 500); baixa o dataset inteiro e cacheia.
2. **Resolve** — `POST Produto.aspx/Consultar` (campo `numeroProcesso`) devolve
   HTML com a tabela de versões; cada versão tem `DownloadConsultaPublica/{id}`,
   nome do arquivo e datas de comercialização. *Gotcha:* cota de ~14 consultas
   por sessão — depois devolve 200 com página vazia (não 429). O harvester
   rotaciona a sessão (novo cookie zera a cota), mantendo 1 req/s e UA identificável.
3. **Download** — `GET DownloadConsultaPublica/{id}` → PDF.

Saída: `data/corpus/susep_{id}.pdf` + `corpus_manifest.json` (proveniência por
versão: processo, id, seguradora, cnpj, arquivo, datas, ramo, url, sha256,
tem_texto, baixado_em). Por padrão baixa só a versão vigente de cada processo;
`--all-versions` baixa o histórico completo. Resumível (pula ids já baixados).

```
python susep_harvest.py                # versão vigente por processo
python susep_harvest.py --all-versions # histórico completo
python susep_harvest.py --limit 5      # smoke test
```

## Skeleton da app — EXERCÍCIO DE MEMÓRIA (não preenchido de propósito)

Esta pasta foi criada vazia de app de propósito. O skeleton (FastAPI + Postgres + Alembic + testcontainers + docker-compose) é um **exercício de recriação de memória** do stack do Cap 3 (interleaving) — você escreve, o Claude corrige com pista, só no fim compara com o `sql-agent`. Não pedir o código pronto antes de tentar de memória.

Checklist do que recriar de memória (sessão B):
- [ ] `pyproject.toml` / `requirements.txt` + venv
- [ ] `app/main.py` — FastAPI, `GET /health`
- [ ] `app/db.py` — engine SQLAlchemy 2.0 async + session
- [ ] `alembic/` — init + primeira migration (tabelas `policy_document`, `coverage`)
- [ ] `docker-compose.yml` — Postgres (+ pgvector)
- [ ] `tests/` — testcontainers + 1 teste de health + 1 de migration
```
