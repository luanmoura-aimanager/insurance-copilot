"""Testa persist_document: achatar a árvore da extração nas 5 tabelas.

Usa o db_session function-scoped (rollback por teste), então cada teste começa com
as tabelas vazias. persist_document só dá flush — o rollback do fixture limpa tudo.
"""

from sqlalchemy import func, select

from app.extraction.persist import document_exists, persist_document
from app.extraction.schema import (
    DeductibleType,
    ExtractedCoverage,
    ExtractedDocument,
    Kind,
    Peril,
)
from app.models import Coverage, CoveragePeril, Exclusion, PolicyDocument
from app.models import Peril as PerilModel

# Proveniência autoritativa (vem do manifesto, não da leitura do LLM).
MANIFEST_ROW = {
    "seguradora": "SEGURADORA AUTORITATIVA S.A.",
    "processo": "15414.999999/2024-99",
    "url": "https://example.com/x.pdf",
    "sha256": "abc123",
}


def sample_doc() -> ExtractedDocument:
    return ExtractedDocument(
        insurer="Nome Curto (leitura do LLM, deve ser ignorada)",
        product="Residência Teste",
        susep_process="15414.999999/2024-99",
        version="v1",
        property_type="habitual",
        coverages=[
            ExtractedCoverage(
                coverage_name="Incêndio",
                plan="Essencial/Fácil",
                kind=Kind.basic,
                deductible_type=DeductibleType.defined_in_policy,
                deductible_rule_text="POS na apólice",
                # incendio_explosao repetido de propósito → get-or-create + dedupe
                perils=[Peril.incendio_explosao, Peril.fumaca, Peril.incendio_explosao],
                exclusions=["exclusão específica da cobertura"],
            ),
            ExtractedCoverage(
                coverage_name="Vendaval",
                plan="Essencial",
                kind=Kind.additional,
                deductible_type=DeductibleType.defined_in_policy,
                deductible_rule_text=None,
                perils=[Peril.vendaval, Peril.granizo],
                exclusions=[],
            ),
        ],
        general_exclusions=["exclusão geral 1", "exclusão geral 2"],
    )


async def test_persist_flattens_tree(db_session):
    pd_id = await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    assert pd_id is not None

    n_cov = await db_session.scalar(
        select(func.count()).select_from(Coverage).where(Coverage.document_id == pd_id)
    )
    assert n_cov == 2

    # dedupe dentro da cobertura: Incêndio tinha incendio_explosao repetido → 2 links, não 3
    inc = (
        await db_session.execute(
            select(Coverage).where(
                Coverage.document_id == pd_id, Coverage.coverage_name == "Incêndio"
            )
        )
    ).scalar_one()
    n_links = await db_session.scalar(
        select(func.count()).select_from(CoveragePeril).where(CoveragePeril.coverage_id == inc.id)
    )
    assert n_links == 2

    # exclusões: escopo geral vs cobertura
    n_general = await db_session.scalar(
        select(func.count())
        .select_from(Exclusion)
        .where(Exclusion.document_id == pd_id, Exclusion.scope == "general")
    )
    n_cov_excl = await db_session.scalar(
        select(func.count())
        .select_from(Exclusion)
        .where(Exclusion.document_id == pd_id, Exclusion.scope == "coverage")
    )
    assert n_general == 2
    assert n_cov_excl == 1
    # a exclusão de cobertura aponta pra uma coverage; a geral não
    cov_excl = (
        await db_session.execute(
            select(Exclusion).where(Exclusion.document_id == pd_id, Exclusion.scope == "coverage")
        )
    ).scalar_one()
    assert cov_excl.coverage_id is not None


async def test_provenance_comes_from_manifest(db_session):
    pd_id = await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    pd = await db_session.get(PolicyDocument, pd_id)
    # o nome curto do LLM é descartado; usa o do manifesto
    assert pd.insurer == "SEGURADORA AUTORITATIVA S.A."
    assert pd.pdf_hash == "abc123"
    assert pd.pdf_url == "https://example.com/x.pdf"


async def test_peril_vocabulary_not_duplicated(db_session):
    """4 perigos distintos no doc (incendio_explosao, fumaca, vendaval, granizo) → 4 linhas,
    mesmo com incendio_explosao repetido dentro de uma cobertura."""
    await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    n_perils = await db_session.scalar(select(func.count()).select_from(PerilModel))
    assert n_perils == 4


async def test_idempotency_guard(db_session):
    doc = sample_doc()
    assert await document_exists(db_session, MANIFEST_ROW["processo"], doc.version) is False
    await persist_document(db_session, doc, MANIFEST_ROW)
    assert await document_exists(db_session, MANIFEST_ROW["processo"], doc.version) is True
