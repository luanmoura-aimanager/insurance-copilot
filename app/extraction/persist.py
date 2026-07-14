"""Persistência: achata a árvore da extração (ExtractedDocument) nas 5 tabelas.

Aqui NÃO há LLM — é código determinístico. O LLM devolveu uma árvore aninhada porque
não conhece os ids do banco; esta camada insere as linhas, coleta os ids gerados pelo
Postgres e monta as FKs (document_id, coverage_id, peril_id) e a junção coverage_peril.

Ordem obrigatória (pai → filho, por causa das FKs):
    policy_document → coverage → (get-or-create peril) → coverage_peril → exclusion

Proveniência (pdf_url, pdf_hash, insurer, susep_process) vem do MANIFESTO, não da leitura
do LLM — o manifesto é o ground truth (o LLM encurtou "Porto Seguro", perdeu "veraneio" etc.).
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Coverage, CoveragePeril, Exclusion, Peril, PolicyDocument

from .schema import ExtractedDocument


async def _get_or_create_peril(session: AsyncSession, name: str) -> Peril:
    """Vocabulário canônico COMPARTILHADO: 'vendaval' é UMA linha, reusada por toda
    cobertura de toda seguradora. Procura antes de criar — sem isso, duplicaria o
    perigo por cobertura e o GROUP BY peril do SQL worker quebraria."""
    existing = await session.scalar(select(Peril).where(Peril.name == name))
    if existing is not None:
        return existing
    peril = Peril(name=name)
    session.add(peril)
    await session.flush()  # gera o peril.id sem fechar a transação
    return peril


async def document_exists(session: AsyncSession, susep_process: str, version: str | None) -> bool:
    """Guard de idempotência: (susep_process, version) é UNIQUE em policy_document."""
    hit = await session.scalar(
        select(PolicyDocument.id).where(
            PolicyDocument.susep_process == susep_process,
            PolicyDocument.version == version,
        )
    )
    return hit is not None


async def persist_document(
    session: AsyncSession, doc: ExtractedDocument, manifest_row: dict
) -> int:
    """Insere uma CG extraída nas 5 tabelas e devolve o policy_document.id.

    manifest_row: a linha do corpus_manifest.json (proveniência autoritativa).
    Commita ao final; assume que o guard de idempotência já foi checado pelo chamador.
    """
    pd = PolicyDocument(
        insurer=manifest_row["seguradora"],       # autoritativo (manifesto), não a leitura do LLM
        product=doc.product,
        susep_process=manifest_row["processo"],
        version=doc.version,
        property_type=doc.property_type,
        pdf_url=manifest_row["url"],
        pdf_hash=manifest_row["sha256"],
    )
    session.add(pd)
    await session.flush()  # pd.id

    for cov in doc.coverages:
        c = Coverage(
            document_id=pd.id,
            coverage_name=cov.coverage_name,
            plan=cov.plan,
            kind=cov.kind.value,
            deductible_type=cov.deductible_type.value if cov.deductible_type else None,
            deductible_rule_text=cov.deductible_rule_text,
        )
        session.add(c)
        await session.flush()  # c.id

        # dedupe: se o LLM repetiu um perigo, a PK composta (coverage_id, peril_id) recusaria
        for peril_name in {p.value for p in cov.perils}:
            peril = await _get_or_create_peril(session, peril_name)
            session.add(CoveragePeril(coverage_id=c.id, peril_id=peril.id))

        for clause in cov.exclusions:
            session.add(
                Exclusion(document_id=pd.id, coverage_id=c.id, scope="coverage", clause_text=clause)
            )

    for clause in doc.general_exclusions:
        session.add(
            Exclusion(document_id=pd.id, coverage_id=None, scope="general", clause_text=clause)
        )

    await session.commit()
    return pd.id
