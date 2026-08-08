"""Materializa `clause_chunk` a partir do texto que a extração já gravou.

Não chama modelo nenhum: lê `exclusion.clause_text` e `coverage.deductible_rule_text`,
monta o texto do chunk (`app.rag.chunking`) e grava uma linha por origem, com
`embedding`/`embedding_model` `NULL` — "ainda não indexado". Quem preenche o vetor é a
R2b.

A passada é **autoritativa**: o que está em `clause_chunk` depois dela é exatamente o
que as origens do documento produzem agora — o que sumiu da origem some daqui (ver
`_podar`).

Contrato de transação igual ao de `persist_document`: **não commita**. Os statements
são Core (`INSERT ... ON CONFLICT`, `DELETE`), então já estão no banco quando a função
retorna — não há flush a esperar, mas também não há commit: o chamador (script, teste,
batch) é dono da transação.
"""

from sqlalchemy import and_, case, delete, or_, select, tuple_
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


def _sobras(arm: InstrumentedAttribute, chaves: list[tuple[int, int]]):
    """Predicado das linhas DESTE braço que a passada atual não escreveu.

    Sem lista de chaves (a origem não produz mais texto nenhum) o braço inteiro é
    sobra — `tuple_(...).not_in([])` renderiza um "sempre verdadeiro" no SQLAlchemy,
    mas escrever o caso explicitamente é mais barato de ler do que confiar nisso.
    """
    if not chaves:
        return arm.is_not(None)
    return and_(arm.is_not(None), tuple_(arm, ClauseChunk.chunk_index).not_in(chaves))


async def _podar(
    session: AsyncSession,
    document_id: int,
    linhas_exclusao: list[dict],
    linhas_franquia: list[dict],
) -> None:
    """Apaga os chunks do documento que esta passada não produziu.

    Sem isto o UPSERT sozinho é idempotente só para origem *adicionada ou alterada*,
    nunca para origem *removida*: limpar `coverage.deductible_rule_text` (uma
    re-extração que corrigiu uma leitura errada) deixa o chunk antigo vivo, com o
    vetor pago intacto, e a busca passa a citar com confiança uma regra de franquia
    que o documento não tem mais. O braço das exclusões está acidentalmente protegido
    — a FK composta impede apagar uma `exclusion` citada por um chunk —, mas o das
    coberturas não, porque zerar uma coluna de texto não toca em FK nenhuma.

    A poda inclui o `chunk_index` na chave, e não só o id da origem: hoje o grão é um
    chunk por origem, mas no dia em que o chunking cortar por tamanho, uma cláusula
    que encolhe de 3 pedaços para 2 tem que perder o pedaço nº 2.
    """
    await session.execute(
        delete(ClauseChunk).where(
            ClauseChunk.document_id == document_id,
            or_(
                _sobras(
                    ClauseChunk.exclusion_id,
                    [(r["exclusion_id"], r["chunk_index"]) for r in linhas_exclusao],
                ),
                _sobras(
                    ClauseChunk.coverage_id,
                    [(r["coverage_id"], r["chunk_index"]) for r in linhas_franquia],
                ),
            ),
        )
    )


async def index_document(session: AsyncSession, document_id: int) -> int:
    """Grava (ou atualiza) os chunks de um documento. Devolve quantos.

    A passada é autoritativa: escreve o que as origens produzem agora e **poda** o que
    elas não produzem mais (`_podar`). Uma linha de origem = um chunk, `chunk_index=0`
    — ver a justificativa medida em `app.rag.chunking`.

    Origem com texto em branco (`""`, `"   "`) é tratada como origem sem texto e não
    vira chunk: `ExtractedCoverage.deductible_rule_text` é `str | None` sem
    `min_length`, então um LLM que devolve `""` em vez de `null` é persistido tal e
    qual, e o chunk resultante seria só o cabeçalho — conteúdo zero, embedding pago na
    R2b, e um vetor de ruído quase idêntico a todos os outros disputando espaço no
    índice HNSW.
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
    # (e `deductible_type` sozinho é coluna categórica, assunto do worker SQL). O NULL
    # é barrado aqui; o texto em branco é barrado na montagem das linhas, abaixo.
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
        if clause_text and clause_text.strip()
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
        if rule_text and rule_text.strip()
    ]

    gravados = await _upsert(session, linhas_exclusao, ClauseChunk.exclusion_id)
    gravados += await _upsert(session, linhas_franquia, ClauseChunk.coverage_id)
    await _podar(session, document_id, linhas_exclusao, linhas_franquia)
    return gravados
