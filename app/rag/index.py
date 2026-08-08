"""Materializa `clause_chunk` a partir do texto que a extração já gravou.

Não chama modelo nenhum: lê `exclusion.clause_text` e `coverage.deductible_rule_text`,
monta o texto do chunk (`app.rag.chunking`) e grava uma linha por origem, com
`embedding`/`embedding_model` `NULL` — "ainda não indexado". Quem preenche o vetor é a
R2b.

Contrato de transação igual ao de `persist_document`: **não commita**. Dá flush no que
precisar e devolve; o chamador (script, teste, batch) é dono da transação.
"""

from sqlalchemy import case, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import InstrumentedAttribute

from app.models import ClauseChunk, Coverage, Exclusion

from .chunking import build_deductible_text, build_exclusion_text


async def _upsert(
    session: AsyncSession, rows: list[dict], arm: InstrumentedAttribute
) -> int:
    """UPSERT idempotente de um braço do arco (`exclusion_id` ou `coverage_id`).

    Os índices únicos que garantem a idempotência são **parciais**
    (`WHERE exclusion_id IS NOT NULL`), então o `index_where` tem que vir junto do
    `index_elements`: sem ele o Postgres não consegue inferir o árbitro do
    `ON CONFLICT` e responde `there is no unique or exclusion constraint matching`.
    É por isso que são dois UPSERTs, um por braço, e não um só.
    """
    if not rows:
        return 0

    stmt = pg_insert(ClauseChunk).values(rows)
    novo_texto = stmt.excluded["text"]
    # `ClauseChunk.text` aqui é a linha JÁ GRAVADA (no DO UPDATE o Postgres qualifica a
    # tabela alvo com o valor antigo); `excluded.text` é o que esta passada trouxe.
    # Todas as expressões do SET leem a linha antiga — não são sequenciais —, então
    # comparar com o texto antigo depois de atribuí-lo funciona. `=` basta: a coluna é
    # NOT NULL nos dois lados, não há NULL pra exigir IS DISTINCT FROM.
    texto_inalterado = ClauseChunk.text == novo_texto
    stmt = stmt.on_conflict_do_update(
        index_elements=[arm, ClauseChunk.chunk_index],
        index_where=arm.is_not(None),   # o predicado do índice parcial, idêntico
        set_={
            "text": novo_texto,
            # Vetor velho sobre texto novo é um índice que mente: a busca casaria pela
            # frase antiga e citaria, confiante, uma cláusula que não existe mais. Quando
            # o texto muda o vetor volta a NULL ("ainda não indexado"), que é a verdade.
            # Quando não muda ele é preservado — zerar sempre obrigaria a re-embeddar o
            # corpus inteiro a cada reindexação, e isso é dinheiro.
            "embedding": case((texto_inalterado, ClauseChunk.embedding), else_=None),
            "embedding_model": case(
                (texto_inalterado, ClauseChunk.embedding_model), else_=None
            ),
        },
    )
    await session.execute(stmt)
    return len(rows)


async def index_document(session: AsyncSession, document_id: int) -> int:
    """Grava (ou atualiza) os chunks de um documento. Devolve quantos.

    Uma linha de origem = um chunk, `chunk_index=0` — ver a justificativa medida em
    `app.rag.chunking`.
    """
    # Outer join: as exclusões de escopo 'general' têm coverage_id NULL, e são
    # justamente as que precisam do cabeçalho "geral da apólice".
    exclusoes = (
        await session.execute(
            select(
                Exclusion.id,
                Exclusion.clause_text,
                Exclusion.scope,
                Coverage.coverage_name,
            )
            .outerjoin(Coverage, Coverage.id == Exclusion.coverage_id)
            .where(Exclusion.document_id == document_id)
            .order_by(Exclusion.id)
        )
    ).all()

    # Só as coberturas que têm regra de franquia escrita: sem texto não há o que indexar
    # (e `deductible_type` sozinho é coluna categórica, assunto do worker SQL).
    franquias = (
        await session.execute(
            select(Coverage.id, Coverage.coverage_name, Coverage.deductible_rule_text)
            .where(
                Coverage.document_id == document_id,
                Coverage.deductible_rule_text.is_not(None),
            )
            .order_by(Coverage.id)
        )
    ).all()

    linhas_exclusao = [
        {
            "document_id": document_id,
            "exclusion_id": exc_id,
            "coverage_id": None,
            "chunk_index": 0,
            "text": build_exclusion_text(clause_text, scope, coverage_name),
        }
        for exc_id, clause_text, scope, coverage_name in exclusoes
    ]
    linhas_franquia = [
        {
            "document_id": document_id,
            "exclusion_id": None,
            "coverage_id": cov_id,
            "chunk_index": 0,
            "text": build_deductible_text(rule_text, coverage_name),
        }
        for cov_id, coverage_name, rule_text in franquias
    ]

    gravados = await _upsert(session, linhas_exclusao, ClauseChunk.exclusion_id)
    gravados += await _upsert(session, linhas_franquia, ClauseChunk.coverage_id)
    return gravados
