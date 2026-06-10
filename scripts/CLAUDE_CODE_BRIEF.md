# Brief pro Claude Code — harvester de corpus SUSEP (residencial)

Rodar no Mac do Luan (precisa de rede pra `susep.gov.br`, que o ambiente do Claude/Cowork não tem). O Claude Code consegue iterar contra o site vivo, inspecionar HTML real e debugar — por isso esta parte foi delegada.

## Contexto

Projeto `insurance-copilot`: corpus = condições gerais de **seguro residencial** baixadas da SUSEP. Já confirmado:

- **Endpoint de download** (funciona): `https://www2.susep.gov.br/safe/menumercado/REP2/Produto.aspx/DownloadConsultaPublica/{id}` — `{id}` é um id numérico INTERNO (ex.: 417717 = CG Youse Auto). Devolve `application/pdf`.
- **Índice oficial**: dataset "Consulta de Produtos" no Portal de Dados Abertos (https://dados.gov.br/dataset/consulta-de-produtos), organização SUSEP. Tem colunas tipo/empresa/CNPJ/**nº de processo**/**ramo**. A página é client-rendered.

## O gap a resolver (a tarefa)

O download usa o **id interno**, mas o índice traz o **número de processo** (formato `15414.NNNNNN/AAAA-DD`). Falta o mapeamento processo → id(s) de versão. Provável caminho: a página de consulta pública por processo (`Produto.aspx`) lista as versões, e cada link "baixar" carrega um `DownloadConsultaPublica/{id}`. Confirmar inspecionando o HTML/Network real.

## Passos sugeridos

1. **Pegar o índice**: baixar o CSV do dataset "Consulta de Produtos" (achar a URL do recurso na página JS — DevTools/Network, ou a API do dados.gov.br). Carregar num DataFrame.
2. **Filtrar ramo residencial**: identificar a coluna de ramo/grupo e os códigos que correspondem a residencial (provavelmente "Compreensivo Residencial" / grupo Patrimonial). Validar olhando alguns nomes de produto.
3. **Resolver processo → id de download**: pra uma amostra de processos residenciais, navegar a página de consulta e extrair os `DownloadConsultaPublica/{id}` de cada versão. Automatizar.
4. **Baixar**: com rate limit educado (~1 req/s), User-Agent identificável, retry com backoff. Salvar em `data/corpus/susep_{id}.pdf`.
5. **Manifesto de proveniência**: `data/corpus/corpus_manifest.json` com {processo, id, seguradora, produto, versão, ramo, url, sha256, baixado_em}. (Proveniência = material de governança do projeto.)
6. **Medir**: total de docs residenciais, % texto extraível vs. scan (rodar `susep_probe.py` lógica de classificação), tamanho.

## Cuidados

- Respeitar o servidor: rate limit, sem paralelismo agressivo, parar se vier 429/403.
- Alguns PDFs antigos podem ser escaneados → marcar no manifesto (`tem_texto: false`) pra orçar OCR depois.
- Não commitar os PDFs no git (são grandes e públicos) — só o manifesto. Adicionar `data/corpus/*.pdf` no `.gitignore`.

## Entregável

`data/corpus/` populado + `corpus_manifest.json` + um número final: **quantas condições gerais residenciais únicas** o corpus tem. Esse número fecha o gate de volumetria.
