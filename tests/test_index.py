"""Materialização de `clause_chunk` (R2a): uma linha de origem → um chunk.

Roda sobre o `db_session` function-scoped (rollback por teste) e reusa o `sample_doc`
de `tests/test_persist.py`, que já é o documento canônico da suíte: 2 coberturas (uma
com regra de franquia, outra sem), 1 exclusão de cobertura e 2 gerais.
"""
from sqlalchemy import func, select, update

from app.extraction.persist import persist_document
from app.models import EMBEDDING_DIM, ClauseChunk, Coverage, Exclusion
from app.rag.index import index_document

from tests.test_persist import MANIFEST_ROW, sample_doc

# sample_doc: 3 exclusões (1 de cobertura + 2 gerais) + 1 cobertura com
# deductible_rule_text ("Incêndio"; "Vendaval" tem None e não vira chunk).
CHUNKS_ESPERADOS = 4


async def _documento_indexado(db_session) -> int:
    pd_id = await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    await index_document(db_session, pd_id)
    return pd_id


async def _chunks(db_session, pd_id: int) -> list[ClauseChunk]:
    db_session.expunge_all()   # o UPSERT é Core: sem isto o ORM devolveria o estado antigo
    return (
        await db_session.scalars(
            select(ClauseChunk).where(ClauseChunk.document_id == pd_id).order_by(ClauseChunk.id)
        )
    ).all()


async def test_um_chunk_por_linha_de_origem(db_session):
    """A contagem é exclusões + coberturas COM regra de franquia — a cobertura sem
    `deductible_rule_text` não vira chunk (não há texto pra indexar)."""
    pd_id = await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    gravados = await index_document(db_session, pd_id)
    assert gravados == CHUNKS_ESPERADOS

    n = await db_session.scalar(
        select(func.count()).select_from(ClauseChunk).where(ClauseChunk.document_id == pd_id)
    )
    assert n == CHUNKS_ESPERADOS


async def test_todo_chunk_tem_exatamente_um_braco_do_arco(db_session):
    """O check do banco já barraria o contrário; o que este teste prova é que o
    indexador preenche o braço *certo* — 3 chunks de exclusão e 1 de franquia, não
    4 de um lado só."""
    pd_id = await _documento_indexado(db_session)
    chunks = await _chunks(db_session, pd_id)

    for c in chunks:
        assert (c.exclusion_id is None) != (c.coverage_id is None)
        assert c.chunk_index == 0            # sem splitter: uma origem = um chunk
        assert c.embedding is None           # o vetor é da R2b
        assert c.embedding_model is None

    assert sum(c.exclusion_id is not None for c in chunks) == 3
    assert sum(c.coverage_id is not None for c in chunks) == 1


async def test_texto_carrega_escopo_e_cobertura(db_session):
    """O cabeçalho é o que separa "exclusão de Incêndio" de "exclusão da apólice"
    depois que o texto vira vetor."""
    pd_id = await _documento_indexado(db_session)
    textos = {c.text for c in await _chunks(db_session, pd_id)}

    assert "Exclusão da cobertura Incêndio: exclusão específica da cobertura" in textos
    assert "Exclusão geral da apólice: exclusão geral 1" in textos
    assert "Regra de franquia da cobertura Incêndio: POS na apólice" in textos


async def test_reindexar_e_idempotente(db_session):
    """Sem o UPSERT por (origem, chunk_index) a segunda passada duplicaria tudo — ou
    estouraria no índice único, dependendo do dia."""
    pd_id = await _documento_indexado(db_session)
    await index_document(db_session, pd_id)

    n = await db_session.scalar(
        select(func.count()).select_from(ClauseChunk).where(ClauseChunk.document_id == pd_id)
    )
    assert n == CHUNKS_ESPERADOS


async def test_texto_novo_invalida_o_embedding(db_session):
    """Vetor velho sobre texto novo é um índice que MENTE: a busca continua casando
    pela frase antiga e devolve, com toda a confiança, uma cláusula que não existe
    mais. Reindexar com texto diferente tem que zerar `embedding`/`embedding_model` —
    `NULL` é "ainda não indexado", que é exatamente a verdade nesse momento.
    """
    pd_id = await _documento_indexado(db_session)
    chunks = await _chunks(db_session, pd_id)
    alvo = next(c for c in chunks if c.exclusion_id is not None)

    # Simula o estado pós-R2b: a linha já tem vetor.
    alvo.embedding = [0.1] * EMBEDDING_DIM
    alvo.embedding_model = "voyage-4-lite"
    await db_session.flush()

    # A cláusula de origem é reescrita (re-extração corrigiu a leitura do PDF).
    await db_session.execute(
        update(Exclusion)
        .where(Exclusion.id == alvo.exclusion_id)
        .values(clause_text="exclusão específica da cobertura, agora corrigida")
    )
    await index_document(db_session, pd_id)

    depois = {c.id: c for c in await _chunks(db_session, pd_id)}
    assert "agora corrigida" in depois[alvo.id].text
    assert depois[alvo.id].embedding is None
    assert depois[alvo.id].embedding_model is None


async def test_poda_chunk_cuja_origem_perdeu_o_texto(db_session):
    """O UPSERT sozinho é idempotente só pra origem adicionada ou alterada.

    Zerar `coverage.deductible_rule_text` (re-extração que corrigiu uma leitura errada)
    não toca em FK nenhuma, então sem a poda o chunk antigo sobrevive com o vetor pago
    intacto e a busca cita, confiante, uma regra de franquia que o documento não tem
    mais. O braço das exclusões está acidentalmente protegido pela FK composta; este
    aqui não estava.
    """
    pd_id = await _documento_indexado(db_session)
    alvo = next(c for c in await _chunks(db_session, pd_id) if c.coverage_id is not None)

    await db_session.execute(
        update(Coverage)
        .where(Coverage.id == alvo.coverage_id)
        .values(deductible_rule_text=None)
    )
    await index_document(db_session, pd_id)

    restantes = await _chunks(db_session, pd_id)
    assert len(restantes) == CHUNKS_ESPERADOS - 1
    assert all(c.coverage_id is None for c in restantes)   # o braço de franquia esvaziou


async def test_texto_em_branco_nao_vira_chunk(db_session):
    """`deductible_rule_text=""` passa pelo filtro `IS NOT NULL` — e o schema da
    extração é `str | None` sem `min_length`, então um LLM que devolve `""` em vez de
    `null` é persistido tal e qual. O chunk resultante seria só o cabeçalho: conteúdo
    zero, embedding pago na R2b, e um vetor de ruído quase idêntico a todos os outros
    disputando espaço no índice HNSW.
    """
    pd_id = await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    await db_session.execute(
        update(Coverage)
        .where(Coverage.document_id == pd_id, Coverage.deductible_rule_text.is_not(None))
        .values(deductible_rule_text="   ")
    )

    gravados = await index_document(db_session, pd_id)
    assert gravados == CHUNKS_ESPERADOS - 1     # só as 3 exclusões
    assert all(c.coverage_id is None for c in await _chunks(db_session, pd_id))


async def test_texto_igual_preserva_o_embedding(db_session):
    """A outra metade da invalidação, e a que custa dinheiro se cair: zerar o vetor
    em toda reindexação passaria no teste acima e obrigaria a re-embeddar o corpus
    inteiro a cada passada. Só o texto que MUDOU perde o vetor.
    """
    pd_id = await _documento_indexado(db_session)
    chunks = await _chunks(db_session, pd_id)
    alvo = next(c for c in chunks if c.exclusion_id is not None)

    alvo.embedding = [0.1] * EMBEDDING_DIM
    alvo.embedding_model = "voyage-4-lite"
    await db_session.flush()

    await index_document(db_session, pd_id)   # nada mudou na origem

    depois = {c.id: c for c in await _chunks(db_session, pd_id)}
    assert depois[alvo.id].embedding is not None
    assert depois[alvo.id].embedding_model == "voyage-4-lite"
