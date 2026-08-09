"""Preenche `clause_chunk.embedding` — a passada que custa dinheiro (R2b).

Lê as linhas com `embedding IS NULL` (é isso que "ainda não indexado" significa desde a
R2a), manda cada lote pra Voyage e grava vetor + `embedding_model`. Uma linha de
`cost_event` por lote, `agent_name="embedder"`.

`request_id`/`client` ficam `NULL`: esta é uma passada **offline**, não há request HTTP
nem cliente autenticado por trás — mesma convenção da extração em lote.
"""

import asyncio
from decimal import Decimal
from math import ceil

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.cost import cost_usd, record_cost_event
from app.models import ClauseChunk

from .embedding import EMBED_MODEL, MAX_BATCH, embed_documents

# Estimativa grosseira só pro `--dry-run`: quantos caracteres, em média, cabem num token
# de texto em português. Serve pra dar ordem de grandeza do custo ANTES de gastar; o
# número que vai pro `cost_event` é sempre o `total_tokens` que a API cobrou de verdade.
CHARS_PER_TOKEN = 4


def _validar_limit(limit: int | None) -> None:
    """`limit` menor que 1 é sempre erro, nas duas funções públicas que o aceitam.

    Em `embed_pending`, `limit=0` sai pelo `while` na primeira volta e devolve um
    relatório de zeros; em `pending_stats`, vira `LIMIT 0` e devolve contagem zero. Os
    dois resultados são indistinguíveis de "já estava tudo indexado" — a mentira
    plausível que a validação do CLI existe pra evitar, e a pior saída possível num
    script que gasta dinheiro. O CLI já barra, mas quem chama de fora dele (teste,
    notebook, a ferramenta da R3) não passa pelo argparse.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit tem que ser >= 1 quando informado, veio {limit}")


def _desatualizado():
    """Predicado das linhas já embeddadas por um modelo DIFERENTE do configurado.

    `embedding_model` existe exatamente pra esta pergunta ser respondível. Sem ela,
    trocar `EMBED_MODEL` por um sucessor de mesma dimensão seria um no-op silencioso: as
    linhas antigas nunca voltam ao filtro `IS NULL`, mas todo chunk novo (ou com texto
    reescrito, que a R2a zera) sai com o vetor do modelo novo. O HNSW passaria a guardar
    vetores de dois modelos, cujas distâncias de cosseno não são comparáveis — a busca
    devolveria vizinhos errados sem erro nenhum.
    """
    return and_(
        ClauseChunk.embedding.is_not(None),
        ClauseChunk.embedding_model.is_distinct_from(EMBED_MODEL),
    )


def _a_fazer(remodel: bool = False):
    """Predicado do que esta passada deve embeddar."""
    nulo = ClauseChunk.embedding.is_(None)
    return or_(nulo, _desatualizado()) if remodel else nulo


def _pendentes(limit: int | None = None, remodel: bool = False):
    """Lote a processar, em ordem estável de id e **com as linhas travadas**.

    `FOR UPDATE SKIP LOCKED` é o que impede duas passadas simultâneas (dois operadores,
    ou uma re-execução disparada antes da primeira terminar) de lerem os MESMOS ids,
    pagarem os dois pelo mesmo texto e gravarem duas linhas de `cost_event` — dinheiro em
    dobro num livro-caixa que o projeto inteiro existe pra manter confiável. Com
    `SKIP_LOCKED` a segunda passada simplesmente pega o lote seguinte.

    O trava-e-solta acompanha o commit por lote: a trava dura o lote, não a passada. Isso
    significa que a transação fica aberta durante a chamada HTTP (e durante o backoff de
    rate limit) — é o preço de reivindicar o lote, e a alternativa seria uma coluna de
    "claim" commitada antes da chamada, ou seja, migration. Num Postgres com
    `idle_in_transaction_session_timeout` ligado (o Postgres puro vem com ele desligado,
    alguns gerenciados não) é aí que se paga essa conta.

    Efeito colateral consciente do `SKIP LOCKED`: se todas as linhas restantes estiverem
    travadas por outro processo, o lote volta vazio e a passada termina cedo. Está certo
    — o outro processo está cuidando delas —, e o dicionário devolvido reflete só o que
    esta passada fez.
    """
    q = (
        select(ClauseChunk)
        # `embedding` é a única coluna que esta passada nunca LÊ — ela só escreve por
        # cima. No caminho normal isso não custaria nada (as pendentes têm NULL ali), mas
        # `--remodel` é justamente o caminho que varre o corpus INTEIRO com vetor
        # preenchido: seriam 1024 floats por linha vindos do banco só pra serem
        # descartados (~18 MB no corpus atual). `defer` não impede a escrita — atribuir a
        # um atributo adiado não dispara carga, só marca a coluna como suja.
        .options(defer(ClauseChunk.embedding))
        .where(_a_fazer(remodel))
        .order_by(ClauseChunk.id)
        .with_for_update(skip_locked=True)
    )
    return q if limit is None else q.limit(limit)


async def contar_desatualizados(session: AsyncSession) -> int:
    """Quantas linhas carregam o vetor de outro modelo. Ver `_desatualizado`."""
    return await session.scalar(
        select(func.count()).select_from(ClauseChunk).where(_desatualizado())
    )


async def pending_stats(
    session: AsyncSession, limit: int | None = None, remodel: bool = False
) -> dict:
    """Quantos chunks faltam e quanto custaria, sem chamar a API. Alimenta o `--dry-run`."""
    _validar_limit(limit)
    alvo = select(ClauseChunk.text).where(_a_fazer(remodel)).order_by(ClauseChunk.id)
    if limit is not None:
        alvo = alvo.limit(limit)
    alvo = alvo.subquery()
    chunks, chars = (
        await session.execute(
            select(func.count(), func.coalesce(func.sum(func.length(alvo.c.text)), 0))
            .select_from(alvo)
        )
    ).one()
    tokens = ceil(chars / CHARS_PER_TOKEN)
    return {
        "chunks": chunks,
        "chars": chars,
        "tokens_estimados": tokens,
        "custo_estimado": cost_usd(EMBED_MODEL, tokens, 0),
        "desatualizados": await contar_desatualizados(session),
    }


async def embed_pending(
    session: AsyncSession,
    limit: int | None = None,
    batch_size: int = MAX_BATCH,
    client=None,
    remodel: bool = False,
) -> dict:
    """Embedda os chunks pendentes, em lotes. Devolve `{chunks, batches, tokens, cost_usd}`.

    **Esta função COMMITA — e é a exceção consciente ao contrato do resto do projeto.**
    `persist_document` e `index_document` não commitam: lá o chamador é dono da transação
    e tudo-ou-nada é o desenho certo, porque repetir a passada é de graça (é só CPU lendo
    o texto que já está no banco). Aqui não: cada lote é uma chamada paga à Voyage. Uma
    falha no lote 15 não pode jogar fora os 14 lotes já pagos — a fronteira da transação
    acompanha o custo de repetir, não o gosto por atomicidade. Por isso o commit é **por
    lote**, junto com a linha de custo dele: o que já foi gasto fica gravado, e a próxima
    passada retoma exatamente de onde parou (as linhas commitadas não voltam no
    `embedding IS NULL`).

    O lote é buscado a cada volta em vez de fatiar uma lista carregada no início: as
    linhas já embeddadas saem sozinhas do filtro depois do commit, então a query "próximas
    N pendentes" *é* o cursor — não há offset pra desalinhar se algo mudar no meio.

    **`remodel=False` se RECUSA a rodar quando existe vetor de outro modelo.** Não é
    zelo: misturar dois modelos no mesmo índice HNSW produz distâncias incomparáveis e a
    busca devolve vizinhos errados sem erro nenhum — o mesmo "índice que mente" que a R2a
    já trata como inaceitável, só que por outra porta. Preencher os `NULL` e ir embora
    seria justamente construir essa mistura. `remodel=True` inclui as linhas do modelo
    antigo na passada e devolve o índice à consistência; é opt-in porque re-embeddar o
    corpus é gasto, e gasto não se faz por efeito colateral.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size tem que ser >= 1, veio {batch_size}")
    _validar_limit(limit)   # ver o porquê em _validar_limit

    if not remodel:
        desatualizados = await contar_desatualizados(session)
        if desatualizados:
            raise ValueError(
                f"{desatualizados} chunk(s) têm vetor de um modelo diferente de "
                f"{EMBED_MODEL!r}. Embeddar só os NULL deixaria dois modelos no mesmo "
                "índice HNSW, com distâncias incomparáveis e busca silenciosamente "
                "errada. Rode com remodel=True (--remodel no CLI) pra re-embeddar tudo."
            )

    restantes = limit
    chunks = batches = tokens = 0
    custo = Decimal("0")

    while restantes is None or restantes > 0:
        n = batch_size if restantes is None else min(batch_size, restantes)
        lote = (await session.scalars(_pendentes(n, remodel))).all()
        if not lote:
            break

        # O SDK da Voyage é síncrono (HTTP bloqueante); to_thread mantém o event loop
        # livre — mesmo arranjo do psycopg3 síncrono nos nós do grafo.
        vetores, total_tokens = await asyncio.to_thread(
            embed_documents, [c.text for c in lote], client
        )
        for chunk, vetor in zip(lote, vetores, strict=True):
            chunk.embedding = vetor
            chunk.embedding_model = EMBED_MODEL

        evento = await record_cost_event(
            session,
            agent_name="embedder",
            model=EMBED_MODEL,
            input_tokens=total_tokens,
            output_tokens=0,   # embedding não tem saída — ver pricing.json
            label=f"chunks {lote[0].id}-{lote[-1].id}",
        )
        custo += evento.cost_usd
        await session.commit()   # ver o docstring: a fronteira é o lote, não a passada

        chunks += len(lote)
        batches += 1
        tokens += total_tokens
        if restantes is not None:
            restantes -= len(lote)

    return {"chunks": chunks, "batches": batches, "tokens": tokens, "cost_usd": custo}
