# Extração-piloto — achados contra o schema v0

Piloto manual em 2 CGs residenciais vigentes de seguradoras distintas:
- **Porto Seguro Residência** (id 482868, processo 15414.100639/2004-31, 49p)
- **Allianz Coletivo Patrimonial – Residência** (id 502775, processo 15414.002283/2007-14, 61p)

Saída estruturada: `data/pilot_extraction.json` (gerada por `scripts/pilot_extraction.py`).
Objetivo: responder o "Aberto pra validar" do schema **antes** de construir o pipeline.

## 1. `franquia_tipo` cobre os casos reais? — Parcial, ajustar

- O termo nas CGs residenciais é **"Participação Obrigatória do Segurado (POS)"**, não "franquia".
  O extrator precisa tratar **POS ≡ franquia** como sinônimos.
- O padrão dominante: *"valor **ou** percentual definido na apólice"*. Ou seja, a CG fixa a
  **estrutura**; o número vive na apólice do cliente (confirma a nota do schema de que valor
  numérico não entra na `coverage`).
- Isso não casa bem com `percentual`/`valor_fixo` (que implicam um valor concreto). As categorias
  reais observadas: `sem_franquia` (RC, perda de aluguel) e algo como **`definido_na_apolice`**
  (valor ou %, conforme apólice). `misto` ficou sobrecarregado.
- **Recomendação:** redefinir `franquia_tipo` para `{sem_franquia, percentual, valor_fixo,
  definido_na_apolice}` e documentar POS como sinônimo de franquia.

## 2. Exclusões: tabela própria ou campo-texto? — Tabela própria

- Porto Seguro tem ~20 blocos de exclusão: **exclusões gerais** (valem pra todas as coberturas)
  **+ exclusões por cobertura**. Achatar num `exclusoes_principais` por linha duplicaria as gerais
  em toda `coverage` ou perderia as específicas.
- **Recomendação:** grão próprio `exclusion` { id, document_id FK, coverage_id FK (nullable),
  escopo: `geral`/`cobertura`, texto }. Manter `coverage.exclusoes_principais` só como resumo
  opcional pra RAG.

## 3. Lista canônica de `nome_cobertura` — Necessária; normalizar por PERIGO, não por nome

Mesmas proteções com nomes e **agrupamentos** diferentes entre seguradoras:

| Perigo | Porto Seguro | Allianz |
|---|---|---|
| Roubo/furto | "Subtração de Bens" | "Roubo e/ou furto qualificado" |
| RC | "RC Familiar (danos a terceiros)" | "Responsabilidade Civil Familiar" |
| Vento/granizo | "Vendaval, furacão, ciclone, tornado, **granizo**" | "Vendaval, **granizo**, **fumaça**, **impacto de veículos**" |

O agrupamento de perigos numa "cobertura" **difere** entre seguradoras → uma lista canônica de
*nomes de cobertura* não resolve. **Recomendação:** canonizar no nível de **perigo** e mapear
cada `nome_cobertura` (texto cru) para um conjunto de perigos canônicos.

Lista canônica inicial de perigos (das 2 CGs): `incendio_explosao`, `danos_eletricos`,
`vendaval`, `granizo`, `fumaca`, `impacto_veiculos`, `roubo_furto_qualificado`, `rc_familiar`,
`perda_aluguel`, `quebra_vidros`, `desmoronamento`, `alagamento`.

## Notas operacionais pro pipeline

- `tipo_imovel` é extraível da CG (Porto cobre habitual + veraneio). OK.
- `coverage.tipo` (`basica`/`adicional`) está explícito na CG. OK.
- Docs de 49–94 páginas, dados espalhados por "Condições Gerais" + "Condições Especiais" →
  extrator precisa de long-context ou chunking por seção.
- 98% do corpus tem texto; 2% são scans (manifesto: `tem_texto=false`) → OCR antes de extrair.
- Caso de borda já visto: processo Aruana cuja versão vigente virou "Garantia" — filtrar pela
  versão residencial, não cegamente pela vigente.
