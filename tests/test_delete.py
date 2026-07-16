"""Testa a remoção de um documento — usada pra re-extrair extração incompleta.

Por que testar um DELETE: as FKs recusam apagar um pai com filhos vivos, então a ordem
importa. E um delete pela metade é pior que nenhum — deixaria coberturas órfãs apontando
pra um documento que não existe mais (ou, com FK, simplesmente estouraria no meio).
"""

from sqlalchemy import func, select

from app.extraction.persist import delete_document_by_hash, persist_document
from app.models import Coverage, CoveragePeril, Exclusion, Peril, PolicyDocument
from tests.test_persist import MANIFEST_ROW, sample_doc


async def test_delete_removes_document_and_children(db_session):
    pd_id = await persist_document(db_session, sample_doc(), MANIFEST_ROW)

    deleted = await delete_document_by_hash(db_session, MANIFEST_ROW["sha256"])
    assert deleted == pd_id

    for model in (PolicyDocument, Coverage, Exclusion, CoveragePeril):
        n = await db_session.scalar(select(func.count()).select_from(model))
        assert n == 0, f"{model.__name__} ficou com linha órfã"


async def test_delete_keeps_the_peril_vocabulary(db_session):
    """Perigo é vocabulário COMPARTILHADO: apagar um doc não pode sumir com 'vendaval'
    do banco — outras seguradoras dependem dessas linhas."""
    await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    before = await db_session.scalar(select(func.count()).select_from(Peril))

    await delete_document_by_hash(db_session, MANIFEST_ROW["sha256"])
    after = await db_session.scalar(select(func.count()).select_from(Peril))

    assert before == after == 4


async def test_delete_is_a_noop_for_unknown_hash(db_session):
    assert await delete_document_by_hash(db_session, "hash-que-nao-existe") is None


async def test_reextraction_round_trip(db_session):
    """O fluxo real do --force: apaga e grava de novo, sem violar o UNIQUE."""
    first = await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    await delete_document_by_hash(db_session, MANIFEST_ROW["sha256"])
    second = await persist_document(db_session, sample_doc(), MANIFEST_ROW)

    assert second != first  # id novo
    n = await db_session.scalar(select(func.count()).select_from(PolicyDocument))
    assert n == 1  # e só um documento no fim
