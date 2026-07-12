from sqlalchemy import select

from app.models import Coverage, PolicyDocument


async def test_round_trip(db_session):
    """Write a document + a linked coverage, then read them back — end to end."""
    doc = PolicyDocument(
        insurer="Porto Seguro",
        product="Residência Habitual",
        susep_process="15414.000000/2024-00",
        pdf_url="https://example.com/doc.pdf",
        pdf_hash="deadbeef",
    )
    db_session.add(doc)
    await db_session.flush()  # assigns doc.id without committing

    coverage = Coverage(
        document_id=doc.id,      # the FK ties the coverage to the document
        coverage_name="Incêndio",
        kind="basic",
    )
    db_session.add(coverage)
    await db_session.flush()

    # read the document back
    fetched = await db_session.get(PolicyDocument, doc.id)
    assert fetched is not None
    assert fetched.insurer == "Porto Seguro"

    # read its coverages back via the FK
    coverages = (
        await db_session.execute(
            select(Coverage).where(Coverage.document_id == doc.id)
        )
    ).scalars().all()
    assert len(coverages) == 1
    assert coverages[0].coverage_name == "Incêndio"
