"""Infraestrutura de vetores (fatia R1 do worker RAG).

Não há embedding real aqui — nada chama modelo. O que estes testes provam é o que
uma fatia de infra tem a provar: que a extensão existe depois das migrations, que
a coluna é de fato um `vector` (e não uma lista de floats que só *parece* um) e
que a dimensão é imposta pelo banco.
"""
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from app.models import EMBEDDING_DIM, ClauseChunk, PolicyDocument


async def _documento(db_session) -> PolicyDocument:
    """clause_chunk.document_id é NOT NULL — todo chunk precisa de um documento."""
    doc = PolicyDocument(
        insurer="Porto Seguro",
        product="Residência Habitual",
        susep_process="15414.900001/2024-00",
        pdf_url="https://example.com/doc.pdf",
        pdf_hash="deadbeef",
    )
    db_session.add(doc)
    await db_session.flush()
    return doc


async def test_extensao_vector_instalada(db_session):
    """`alembic upgrade head` (rodado na fixture do container) criou a extensão."""
    result = await db_session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.scalar() == 1


async def test_round_trip_do_embedding(db_session):
    """Grava 1024 floats, lê de volta, e a dimensão sobrevive à ida e volta."""
    doc = await _documento(db_session)
    chunk = ClauseChunk(
        document_id=doc.id,
        source="exclusion",
        source_id=1,
        text="Danos causados por ato doloso do segurado.",
        embedding=[0.1] * EMBEDDING_DIM,
        embedding_model="voyage-4-lite",
    )
    db_session.add(chunk)
    await db_session.flush()

    chunk_id = chunk.id
    db_session.expunge_all()   # sem isso o get() devolveria o objeto do identity map,
                               # e o teste nunca chegaria a ler do Postgres

    fetched = await db_session.get(ClauseChunk, chunk_id)
    assert fetched is not None
    assert len(fetched.embedding) == EMBEDDING_DIM
    assert float(fetched.embedding[0]) == pytest.approx(0.1)
    assert fetched.embedding_model == "voyage-4-lite"


async def test_ordena_por_distancia_de_cosseno(db_session):
    """Este é o teste que prova que o tipo é `vector`, e não um array de floats.

    `cosine_distance` vira o operador `<=>` do pgvector; num `float[]` comum o
    Postgres nem conheceria o operador, então o ORDER BY abaixo só existe porque a
    coluna é vetorial de verdade.
    """
    doc = await _documento(db_session)
    # Dois vetores ortogonais: o primeiro é idêntico à consulta, o segundo aponta
    # para outro eixo (distância de cosseno máxima).
    eixo_a = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    eixo_b = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    # "b" é inserido primeiro de propósito: se o ORDER BY fosse ignorado, a ordem
    # de inserção devolveria "b" na frente e o assert abaixo quebraria.
    db_session.add_all([
        ClauseChunk(
            document_id=doc.id, source="exclusion", source_id=2,
            text="b", embedding=eixo_b,
        ),
        ClauseChunk(
            document_id=doc.id, source="exclusion", source_id=1,
            text="a", embedding=eixo_a,
        ),
    ])
    await db_session.flush()

    async def por_proximidade(consulta):
        return (
            await db_session.execute(
                select(ClauseChunk.text).order_by(
                    ClauseChunk.embedding.cosine_distance(consulta)
                )
            )
        ).scalars().all()

    # A ordem acompanha o vetor de consulta, nos dois sentidos — é isso que
    # distingue "ordenou por similaridade" de "devolveu numa ordem qualquer".
    assert await por_proximidade(eixo_a) == ["a", "b"]
    assert await por_proximidade(eixo_b) == ["b", "a"]


async def test_dimensao_errada_e_rejeitada_pelo_banco(db_session):
    """A dimensão está no tipo da coluna, então quem barra é o Postgres.

    Importa porque um bug de embedding (modelo trocado, resposta truncada) falha
    no INSERT em vez de contaminar o índice com vetores de tamanho errado.
    """
    doc = await _documento(db_session)
    # O INSERT abortado aborta a transação junto. O savepoint isola a falha, senão
    # ela derrubaria a transação da fixture (que é quem mantém o teste isolado).
    with pytest.raises(DBAPIError):
        async with db_session.begin_nested():
            db_session.add(
                ClauseChunk(
                    document_id=doc.id,
                    source="exclusion",
                    source_id=1,
                    text="vetor curto demais",
                    embedding=[0.1, 0.2, 0.3],   # 3 dimensões, a coluna exige 1024
                )
            )
            await db_session.flush()
