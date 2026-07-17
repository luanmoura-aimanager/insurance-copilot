# F4 — Extraction eval (juiz)

- **Data:** 17/07/2026
- **Juiz:** Opus 4.8 via Claude Code — leitura do texto-fonte de cada CG contra as coberturas
  persistidas no Postgres. **Custo zero de API** (nenhuma chamada à Anthropic; sem re-extração).
- **Placar:** **29 PASS · 1 MINOR · 0 FAIL · 0 alucinação.** MISSING total = 1 (acessório).

Método: para cada doc, isolar a seção de enumeração de garantias no texto-fonte (cabeçalhos
`COBERTURA ADICIONAL Nº`, `CLÁUSULA N - COBERTURA ACESSÓRIA DE`, `CONDIÇÕES ESPECIAIS DA
COBERTURA DE`, `COBERTURA 01..N`…) e comparar 1-a-1 com as coberturas do banco. Drills manuais
nos casos cegos/suspeitos (CHUBB, SIMPLE2U, XS3, Gente, ALLSEG, BANESTES, Tokio, HDI).
Reproduzir com `dump_judge_bundles.py` + `judge_sheet.py` (ver README → Extraction eval).

| doc_id | seguradora | n_extraído | n_missing | n_hallucinated | veredicto | evidência curta |
|---|---|---:|---:|---:|---|---|
| 1 | PORTO SEGURO | 8 | 0 | 0 | PASS | 2.1 básica + 2.2.1–2.2.6 adicionais batem 1:1 |
| 2 | XS3 SEGUROS | 8 | 0 | 0 | PASS | DFI/consórcio; riscos cobertos cl.8.3 a)–h) todos capturados |
| 3 | Excelsior | 10 | 0 | 0 | PASS | CLÁUSULA 3–11 "COBERTURA ACESSÓRIA DE …" 1:1 |
| 5 | ANGELUS | 20 | 0 | 0 | PASS | "COBERTURA ADICIONAL Nº 1–15" + básica + 4 extras, todas presentes |
| 6 | GENERALI | 14 | 0 | 0 | PASS | 14 cabeçalhos CAIXA-ALTA == 14 extraídas |
| 7 | SOMBRERO | 6 | 0 | 0 | PASS | cláusulas 34–39 (básica + 5 adicionais) 1:1 |
| 8 | DAYCOVAL | 11 | 0 | 0 | PASS | 10 "CONDIÇÕES ESPECIAIS DA COBERTURA ADICIONAL DE …" + básica |
| 9 | ZURICH MINAS | 17 | 0 | 0 | PASS | lista 8.2.1–8.2.11 confere com as 17 |
| 10 | BP SEGURADORA | 6 | 0 | 0 | PASS | básica + 5 adicionais (Alagamento…Quebra Vidros) 1:1 |
| 12 | 180 SEGUROS | 30 | 0 | 0 | PASS | 19 coberturas + 10 Assistência 24h + básica, todas com cabeçalho no PDF |
| 13 | Gente Seguradora | 10 | 0 | 0 | PASS | "COBERTURA 1–12"; roubo 6&7 (habitual) e 8&9 (veraneio) consolidados corretamente |
| 14 | Berkley | 9 | 0 | 0 | PASS | 9 "CLÁUSULA ESPECIAL DE …" 1:1 |
| 15 | ZURICH SANTANDER | 13 | 0 | 0 | PASS | seções 29.1–29.12 batem |
| 16 | COBUCCIO | 17 | 0 | 0 | PASS | 16 "CONDIÇÕES ESPECIAIS – COBERTURA ADICIONAL …" + básica |
| 17 | HDI SEGUROS | 33 | 1 | 0 | **MINOR** | falta "IV. COBERTURAS ADICIONAIS DE ASSISTÊNCIA 24 HORAS" (l.4125); 180 e Mitsui extraíram a análoga |
| 18 | ALLSEG | 17 | 0 | 0 | PASS | inclui "COBERTURA BÁSICA II - RESIDENCIAL IMOBILIÁRIAS" (l.47/1816) — real, não alucinação |
| 19 | BANESTES | 6 | 0 | 0 | PASS | "Cobertura de …" 32.1–32.6 == 6 extraídas |
| 20 | PO SEGURADORA | 18 | 0 | 0 | PASS | 18 "CONDIÇÕES ESPECIAIS - COBERTURA DE …" 1:1 (texto curto/condensado) |
| 21 | PREVISUL | 21 | 0 | 0 | PASS | "COBERTURA 01–21" 1:1 |
| 22 | SANCOR | 19 | 0 | 0 | PASS | básica + adicionais A–P + Desentulho + Todos os Riscos |
| 23 | TAAMIN | 20 | 0 | 0 | PASS | "COBERTURA ADICIONAL DE …" (18) + básica + RC extras |
| 24 | MITSUI SUMITOMO | 25 | 0 | 0 | PASS | CLÁUSULA 100–122 + 4 planos Assistência 24h, todos presentes |
| 25 | TOKIO MARINE | 16 | 0 | 0 | PASS | cláusulas particulares 001–006 + especiais; todos os perils principais cobertos |
| 26 | ALIANÇA DA BAHIA | 35 | 0 | 0 | PASS | dezenas de "COBERTURA ADICIONAL DE …" batem com as 35 |
| 27 | Aliança do Brasil | 18 | 0 | 0 | PASS | inclui "Pagamento de Franquia p/ Seguro de Automóvel" (l.223 o) — real |
| 28 | COMPREV | 7 | 0 | 0 | PASS | "COBERTURA I–VI" + básica == 7 |
| 29 | ITAU | 16 | 0 | 0 | PASS | 29.1 + 30.1–30.14 batem |
| 30 | CHUBB | 1 | 0 | 0 | PASS | produto de cobertura única (cl.5.1); demais perils só em glossário+exclusões |
| 31 | SIMPLE2U | 2 | 0 | 0 | PASS | bilhete modular: só "1. DANOS ELÉTRICOS" e "2. ROUBO/FURTO QUALIFICADO" |
| 32 | VOLTS | 6 | 0 | 0 | PASS | básica + 5 adicionais 1:1 |

`doc_id` = `policy_document.id`. Faltam 4 e 11 na sequência (re-runs de docs, sem efeito no eval).

## A única falha (HDI, MINOR)

HDI tem seção substantiva "IV. COBERTURAS ADICIONAIS DE ASSISTÊNCIA 24 HORAS" (l.4125) e
**nenhum** item de assistência entre as 33 coberturas extraídas — inconsistente com 180 SEGUROS
(10 itens) e MITSUI (4 planos), que extraíram a análoga. Cobertura **acessória** de serviço →
MINOR, não FAIL.

## Itens em aberto

1. **Escopo não-determinístico de Assistência 24h** — a extração ora conta assistência como
   cobertura (180, Mitsui), ora ignora (HDI). Candidato a ADR: definir se é cobertura ou anexo
   de serviço, e aplicar de forma consistente.
2. **Divergência de proveniência no XS3** — `susep_process` no banco = `15414.602022/2021-57`,
   mas o rodapé do PDF diz `15414.617294/2020-71`. Não é cobertura; investigar à parte.
