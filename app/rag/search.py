"""Busca por similaridade sobre `clause_chunk` (R3a) — função pura de recuperação.

Embedda a pergunta com `input_type="query"` e devolve os chunks mais próximos por
distância de cosseno, que é a mesma métrica do índice HNSW (`vector_cosine_ops`) — se as
duas divergirem o índice deixa de ser usado e o Postgres cai em seq scan em silêncio.

**Ainda NÃO existe nó de RAG no grafo.** Esta fatia entrega a função e os testes; a
ligação com `app/agents/` (e a decisão de quando o supervisor roteia pra cá) é a R3b.

O custo é gravado como qualquer outra chamada paga (`agent_name="rag_search"`), e como
esta é a primeira coisa da R3 que roda **dentro de um request**, `request_id`/`client`
vêm dos ContextVars — mesmo caminho dos nós do grafo, e o oposto da passada de embedding
(`agent_name="embedder"`), que é offline e grava os dois como `NULL`.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cost import record_call_cost
from app.models import ClauseChunk

from .embedding import EMBED_MODEL, embed_query_with_tokens

logger = logging.getLogger(__name__)

# Limiar de relevância. Ele existe porque a busca vetorial **sempre** devolve `k`
# vizinhos: sem corte, uma pergunta sobre seguro de viagem ou imposto de renda — assunto
# que o corpus não tem — volta com 5 cláusulas de seguro residencial, e um sintetizador
# confiante transforma isso numa resposta errada com citação. O limiar é o que permite
# dizer "não sei".
#
# 0.60 é a escolha atual (voyage-4-lite, distância de cosseno). É um número **decidido,
# não medido**: `scripts/calibrate_search.py` — que roda a busca sem limiar sobre
# `data/eval/search_questions.txt` e mostra onde as distribuições de '+' e '-' se
# separam — ainda não foi rodado contra o corpus. Quando for, é este valor que ele
# confirma ou corrige.
#
# `search_clauses` NÃO o aplica por padrão (`max_distance=None` = sem corte): quem quer
# cortar passa o valor explicitamente. Enquanto o número não estiver validado, um corte
# silencioso significaria descartar resultado sem que o chamador tenha pedido.
MAX_DISTANCE_PADRAO = 0.60

# `hnsw.ef_search` é quantos candidatos o índice percorre antes de responder, e o default
# do pgvector é **40**. Ele é um teto silencioso: com o índice em uso, um `LIMIT 100`
# volta com 40 linhas e nenhum erro — verificado no corpus real (4.386 chunks) forçando
# `enable_seqscan=off`. Hoje o planner ainda prefere seq scan nesse tamanho e o resultado
# sai exato, o que quer dizer que a garantia de "`k` resultados" desta função depende de
# uma escolha de plano que muda sozinha quando a tabela cresce.
#
# O piso é o próprio default (não faz sentido pedir menos) e o teto é 1000, o máximo que
# o pgvector aceita para o parâmetro.
EF_SEARCH_FATOR = 4
EF_SEARCH_MIN = 40
EF_SEARCH_MAX = 1000

# **`ef_search` sozinho NÃO resolve o caso filtrado, e é o caso filtrado que quebra.**
# O `WHERE` (document_id, embedding_model, embedding IS NOT NULL) é aplicado **depois**
# da varredura aproximada, então dos ~40 candidatos globais sobram só os que passam pelo
# filtro. Medido no corpus real, forçando o plano do HNSW: `k=5` com `document_ids` de um
# documento devolveu **1 linha** — a função mentindo sobre o próprio contrato. Esticar o
# `ef_search` por um fator fixo não conserta isso, porque a diluição é a seletividade do
# filtro (29 documentos ⇒ ~29×) e não um múltiplo do `k`; o fator 4 anterior era, no `k`
# padrão, literalmente o default do pgvector — ou seja, não fazia nada.
#
# `hnsw.iterative_scan` existe pra exatamente isto: o índice devolve mais lotes até
# juntar `LIMIT` linhas que sobrevivam ao filtro. Mesma medição, com ele ligado: 5 de 5.
# `strict_order` (e não `relaxed_order`) porque a ordem por distância aqui é o resultado
# — o limiar corta pelas primeiras posições, e "quase ordenado" viraria hit descartado
# por engano. Ligado SEMPRE, não só quando há `document_ids`: o filtro por
# `embedding_model` dilui do mesmo jeito num corpus meio remodelado, e sem filtro
# nenhum o primeiro lote já satisfaz o `LIMIT` e o parâmetro não custa nada.
ITERATIVE_SCAN = "strict_order"


@dataclass(frozen=True)
class Hit:
    """Um chunk recuperado, com a origem que ele cita e a distância que o trouxe.

    `frozen` porque é resultado de leitura: nada rio abaixo deveria reescrever a
    distância ou trocar a origem de um hit já recuperado.

    `exclusion_id`/`coverage_id` são o arco exclusivo de `clause_chunk` — exatamente um
    vem preenchido — e viajam junto porque a citação da resposta ("de onde saiu isso")
    depende deles; sem a origem, o texto recuperado é só uma frase sem procedência.
    """

    chunk_id: int
    document_id: int
    exclusion_id: int | None
    coverage_id: int | None
    text: str
    distance: float


async def search_clauses(
    session: AsyncSession,
    question: str,
    *,
    k: int = 5,
    max_distance: float | None = None,
    document_ids: Sequence[int] | None = None,
    client=None,
) -> list[Hit]:
    """Os `k` chunks mais próximos da pergunta, do mais próximo ao mais distante.

    **A busca vetorial sempre devolveria `k` resultados.** Não existe "não encontrei":
    o `ORDER BY ... LIMIT k` devolve os k vizinhos mais próximos do corpus inteiro,
    ainda que o mais próximo esteja do outro lado do espaço — uma pergunta sobre
    bicicleta volta com as 5 cláusulas de seguro residencial menos distantes dela. A
    lista vazia devolvida por esta função é produzida **pelo limiar** (`max_distance`),
    não por "não achei": sem `max_distance`, o retorno tem `k` elementos sempre que
    houver `k` chunks embeddados **pelo modelo atual** dentro do recorte.

    Essa ressalva é a única outra forma de vir menos que `k`, e é deliberada: um corpus
    meio re-embeddado (`--remodel` interrompido) devolve pouco ou nada em vez de devolver
    uma ordenação que mistura dois modelos. Vazio e honesto é recuperável; ranking errado
    com cara de certo, não. O outro teto possível — o índice parar no primeiro lote de
    candidatos e devolver menos que `k` depois dos filtros — é neutralizado pelos dois
    parâmetros de sessão configurados abaixo (`hnsw.ef_search` e `hnsw.iterative_scan`).

    O corte acontece **depois** do `LIMIT`, não como `WHERE distance <= :max`: o limiar é
    de relevância, não de paginação. Filtrar no SQL faria a query varrer mais fundo em
    busca de k linhas que passassem — trabalho a mais pra chegar no mesmo lugar — e
    esconderia dos testes (e da calibração) as distâncias que foram descartadas.
    Ver `MAX_DISTANCE_PADRAO`.

    `document_ids` filtra **antes** do ranking, no SQL: restringir a um documento é
    identidade (seguradora, produto, processo SUSEP), que por decisão da R2a fica fora do
    texto do chunk justamente porque um `WHERE` responde isso de graça e com precisão
    exata. Uma lista vazia significa "nenhum documento" e devolve `[]` sem chamar a API —
    não há o que buscar, e a chamada seria paga por nada.

    `embedding IS NULL` (chunk materializado pela R2a e ainda não embeddado pela R2b)
    nunca entra: `NULL` não tem distância e a linha não está no índice HNSW.
    """
    if k < 1:
        raise ValueError(f"k tem que ser >= 1, veio {k}")
    if document_ids is not None and len(document_ids) == 0:
        # Curto-circuito ANTES de embeddar: o recorte é vazio, então a resposta é vazia
        # sem depender do vetor — e uma chamada à Voyage custa dinheiro.
        return []

    # O SDK da Voyage é síncrono; to_thread mantém o event loop livre (mesmo arranjo do
    # psycopg3 nos nós do grafo e do `embed_documents` na passada de indexação).
    vetor, tokens = await asyncio.to_thread(embed_query_with_tokens, question, client)

    # Gravado ANTES da consulta, e best effort, pelos dois motivos que o grafo já aplica:
    # quando chegamos aqui a chamada JÁ FOI PAGA (a linha tem que existir mesmo que o
    # SELECT abaixo estoure), e perder uma linha de observabilidade é bem mais barato do
    # que derrubar um request que já gastou. `record_call_cost` abre transação própria.
    try:
        await record_call_cost(
            agent_name="rag_search",
            model=EMBED_MODEL,
            input_tokens=tokens,
            output_tokens=0,   # embedding não tem token de saída — ver pricing.json
        )
    except Exception:
        logger.warning("falha ao gravar cost_event da busca RAG", exc_info=True)

    # `cosine_distance` compila pro operador `<=>`, o mesmo do índice
    # (`USING hnsw (embedding vector_cosine_ops)`). Aparecer nos dois lugares — na
    # projeção e no ORDER BY — é o que deixa o planner usar o índice pra ordenar.
    # Sem estes dois o índice pararia no primeiro lote de candidatos e devolveria menos
    # que `k` — sem erro nenhum. Ver as constantes: `ef_search` cobre o `k` grande,
    # `iterative_scan` cobre o filtro. `set_config(..., is_local => true)` é `SET LOCAL`
    # como função: vale até o fim da transação da session (não vaza pra outras conexões
    # nem sobrevive ao commit) e aceita bind param, coisa que a sintaxe `SET` não aceita.
    ef_search = min(max(k * EF_SEARCH_FATOR, EF_SEARCH_MIN), EF_SEARCH_MAX)
    await session.execute(
        select(
            func.set_config("hnsw.ef_search", str(ef_search), True),
            func.set_config("hnsw.iterative_scan", ITERATIVE_SCAN, True),
        )
    )

    distancia = ClauseChunk.embedding.cosine_distance(vetor).label("distance")
    q = (
        select(
            ClauseChunk.id,
            ClauseChunk.document_id,
            ClauseChunk.exclusion_id,
            ClauseChunk.coverage_id,
            ClauseChunk.text,
            distancia,
        )
        .where(ClauseChunk.embedding.is_not(None))
        # Só vetores do modelo ATUAL entram no ranking. `embed_pending` commita por lote
        # (de propósito), então um `--remodel` interrompido deixa metade do corpus no
        # modelo novo e metade no antigo — distâncias de cosseno de modelos diferentes
        # não são comparáveis, e sem este filtro a busca ordenaria os dois juntos e
        # devolveria vizinhos errados sem erro nenhum. `contar_desatualizados` já guarda
        # esse estado do lado da ESCRITA; aqui é a mesma guarda do lado da leitura.
        .where(ClauseChunk.embedding_model == EMBED_MODEL)
        .order_by(distancia)
        .limit(k)
    )
    if document_ids is not None:
        q = q.where(ClauseChunk.document_id.in_(list(document_ids)))

    hits = [
        Hit(
            chunk_id=row.id,
            document_id=row.document_id,
            exclusion_id=row.exclusion_id,
            coverage_id=row.coverage_id,
            text=row.text,
            distance=float(row.distance),
        )
        for row in (await session.execute(q)).all()
    ]
    if max_distance is None:
        return hits
    return [h for h in hits if h.distance <= max_distance]
