"""Infraestrutura de vetores (fatia R1 do worker RAG).

Não há embedding real aqui — nada chama modelo. O que estes testes provam é o que
uma fatia de infra tem a provar: que a extensão existe depois das migrations, que
a coluna é de fato um `vector` (e não uma lista de floats que só *parece* um), que
a dimensão é imposta pelo banco, que o índice HNSW existe com a operator class
certa, que o arco exclusivo é impositivo e que a tabela está fora do alcance da
role do worker SQL.
"""
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.models import EMBEDDING_DIM, ClauseChunk, Coverage, Exclusion, PolicyDocument


async def _documento(
    db_session, susep_process: str = "15414.900001/2024-00"
) -> PolicyDocument:
    """clause_chunk.document_id é NOT NULL — todo chunk precisa de um documento.

    O processo é parametrizável porque (susep_process, version) é unique e o teste da
    FK composta precisa de dois documentos distintos.
    """
    doc = PolicyDocument(
        insurer="Porto Seguro",
        product="Residência Habitual",
        susep_process=susep_process,
        pdf_url="https://example.com/doc.pdf",
        pdf_hash="deadbeef",
    )
    db_session.add(doc)
    await db_session.flush()
    return doc


async def _exclusao(db_session, doc: PolicyDocument) -> Exclusion:
    """Origem de chunk: o arco exclusivo exige uma linha real do outro lado da FK."""
    exc = Exclusion(
        document_id=doc.id, coverage_id=None, scope="general",
        clause_text="Danos causados por ato doloso do segurado.",
    )
    db_session.add(exc)
    await db_session.flush()
    return exc


async def _cobertura(db_session, doc: PolicyDocument) -> Coverage:
    cov = Coverage(
        document_id=doc.id, coverage_name="Incêndio", kind="basic",
        deductible_type="defined_in_policy",
        deductible_rule_text="POS de 10% dos prejuízos, mínimo definido na apólice.",
    )
    db_session.add(cov)
    await db_session.flush()
    return cov


async def test_extensao_vector_instalada(db_session):
    """`alembic upgrade head` (rodado na fixture do container) criou a extensão."""
    result = await db_session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.scalar() == 1


async def test_round_trip_do_embedding(db_session):
    """Grava 1024 floats, lê de volta, e a dimensão sobrevive à ida e volta."""
    doc = await _documento(db_session)
    exc = await _exclusao(db_session, doc)
    chunk = ClauseChunk(
        document_id=doc.id,
        exclusion_id=exc.id,
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
    exc_a = await _exclusao(db_session, doc)
    exc_b = await _exclusao(db_session, doc)
    # Dois vetores ortogonais: o primeiro é idêntico à consulta, o segundo aponta
    # para outro eixo (distância de cosseno máxima).
    eixo_a = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    eixo_b = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    # "b" é inserido primeiro de propósito: se o ORDER BY fosse ignorado, a ordem
    # de inserção devolveria "b" na frente e o assert abaixo quebraria.
    db_session.add_all([
        ClauseChunk(document_id=doc.id, exclusion_id=exc_b.id, text="b", embedding=eixo_b),
        ClauseChunk(document_id=doc.id, exclusion_id=exc_a.id, text="a", embedding=eixo_a),
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


async def test_indice_hnsw_existe_com_a_operator_class_certa(db_session):
    """O índice é o único artefato que a migration precisa escrever à mão.

    Sem este teste, apagar o `CREATE INDEX ... USING hnsw` da migration — ou trocar
    `vector_cosine_ops` por `vector_l2_ops` — deixaria os outros testes todos verdes:
    com duas linhas o planner ordena por seq scan de qualquer jeito, e uma opclass L2
    simplesmente nunca seria usada por uma consulta de cosseno, sem ninguém notar.
    """
    indexdef = await db_session.scalar(
        text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'clause_chunk' AND indexname = 'ix_clause_chunk_embedding'"
        )
    )
    assert indexdef is not None, "o índice HNSW não existe"
    assert "USING hnsw" in indexdef
    assert "vector_cosine_ops" in indexdef   # casa com o operador <=> usado na busca


async def test_indice_de_fk_em_document_id_existe(db_session):
    """O Postgres não indexa coluna filha de FK sozinho, e sem este índice todo
    DELETE em policy_document varre a maior tabela do schema.

    Checa a *definição*, não só a existência da linha: com `assert indexdef is not
    None` bastava criar o índice na coluna errada (`['coverage_id']`) pro teste passar
    verde com o seq scan intacto — o nome do índice não prova o conteúdo dele.
    """
    indexdef = await db_session.scalar(
        text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'clause_chunk' AND indexname = 'ix_clause_chunk_document_id'"
        )
    )
    assert indexdef is not None, "o índice de FK não existe"
    assert "(document_id)" in indexdef


async def test_indices_de_idempotencia_sao_parciais(db_session):
    """Metade das linhas tem NULL no braço que não usa, por desenho.

    Um índice único total gastaria uma entrada por linha do outro braço — o dobro do
    tamanho, na tabela que também carrega o HNSW. O predicado não afrouxa a garantia
    (as linhas que ele exclui são as que o NULL já tornava não-conflitantes), e é o
    `WHERE` na definição que prova que ele foi aplicado.
    """
    for indexname, coluna in (
        ("uq_clause_chunk_exclusion_chunk", "exclusion_id"),
        ("uq_clause_chunk_coverage_chunk", "coverage_id"),
    ):
        indexdef = await db_session.scalar(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'clause_chunk' AND indexname = :nome"
            ).bindparams(nome=indexname)
        )
        assert indexdef is not None, f"{indexname} não existe"
        assert "UNIQUE" in indexdef
        assert f"WHERE ({coluna} IS NOT NULL)" in indexdef


async def test_arco_exclusivo_rejeita_nenhuma_origem(db_session):
    """Chunk órfão: sem exclusion_id nem coverage_id não há origem pra citar."""
    doc = await _documento(db_session)
    with pytest.raises(IntegrityError, match="ck_clause_chunk_exactly_one_source"):
        async with db_session.begin_nested():
            db_session.add(
                ClauseChunk(document_id=doc.id, text="sem origem nenhuma")
            )
            await db_session.flush()


async def test_arco_exclusivo_rejeita_as_duas_origens(db_session):
    """Os dois braços preenchidos: era exatamente a ambiguidade que o desenho antigo
    (source + source_id + coverage_id denormalizado) deixava passar."""
    doc = await _documento(db_session)
    exc = await _exclusao(db_session, doc)
    cov = await _cobertura(db_session, doc)
    with pytest.raises(IntegrityError, match="ck_clause_chunk_exactly_one_source"):
        async with db_session.begin_nested():
            db_session.add(
                ClauseChunk(
                    document_id=doc.id, exclusion_id=exc.id, coverage_id=cov.id,
                    text="origem ambígua",
                )
            )
            await db_session.flush()


async def test_fk_rejeita_origem_inexistente(db_session):
    """O ponteiro órfão — o furo nº 1 do desenho antigo — agora é impossível.

    Sem este teste a FK inteira podia sumir da migration e a suíte ficava verde: o
    check XOR continua disparando, os uniques continuam disparando, e o teste de
    delete passa porque apaga os chunks primeiro de qualquer jeito. Nada exercitava a
    integridade referencial em si.
    """
    doc = await _documento(db_session)
    with pytest.raises(IntegrityError, match="fk_clause_chunk_exclusion"):
        async with db_session.begin_nested():
            db_session.add(
                ClauseChunk(document_id=doc.id, exclusion_id=999_999, text="origem fantasma")
            )
            await db_session.flush()


async def test_fk_impede_apagar_a_exclusao_citada(db_session):
    """A outra metade da garantia: enquanto um chunk cita a cláusula, ela não some.

    É o que impede a re-extração de reciclar um id e fazer o chunk passar a citar uma
    cláusula *diferente* — o dano real do ponteiro sem FK, que não é o NULL e sim a
    citação silenciosamente errada.
    """
    doc = await _documento(db_session)
    exc = await _exclusao(db_session, doc)
    db_session.add(ClauseChunk(document_id=doc.id, exclusion_id=exc.id, text="pedaço"))
    await db_session.flush()

    with pytest.raises(IntegrityError, match="fk_clause_chunk_exclusion"):
        async with db_session.begin_nested():
            await db_session.execute(
                text("DELETE FROM exclusion WHERE id = :id").bindparams(id=exc.id)
            )


async def test_fk_composta_rejeita_chunk_de_outro_documento(db_session):
    """`document_id` e a origem do arco têm que concordar.

    Com FKs simples este INSERT passava (verificado), e aí o buraco reabria pelo lado:
    `delete_document_by_hash` apaga chunks por `document_id`, não veria este chunk, e
    estouraria ForeignKeyViolation ao apagar a `exclusion` — o documento pela metade
    que a fatia inteira existe pra impedir. Quem barra é a FK composta
    (document_id, exclusion_id) -> exclusion(document_id, id).
    """
    doc_a = await _documento(db_session)
    doc_b = await _documento(db_session, susep_process="15414.900002/2024-11")
    exc_de_a = await _exclusao(db_session, doc_a)

    with pytest.raises(IntegrityError, match="fk_clause_chunk_exclusion"):
        async with db_session.begin_nested():
            db_session.add(
                ClauseChunk(
                    document_id=doc_b.id, exclusion_id=exc_de_a.id,
                    text="origem no documento errado",
                )
            )
            await db_session.flush()


async def test_reindexacao_e_idempotente_por_origem(db_session):
    """O unique (exclusion_id, chunk_index) morde de verdade.

    No desenho antigo o unique era (source, source_id, chunk_index) com source_id
    nullable — e o Postgres trata NULLs como distintos, então duas linhas idênticas
    entravam as duas. Com a FK NOT NULL no braço usado, a duplicata é barrada.
    """
    doc = await _documento(db_session)
    exc = await _exclusao(db_session, doc)
    db_session.add(ClauseChunk(document_id=doc.id, exclusion_id=exc.id, chunk_index=0, text="p1"))
    await db_session.flush()

    with pytest.raises(IntegrityError, match="uq_clause_chunk_exclusion_chunk"):
        async with db_session.begin_nested():
            db_session.add(
                ClauseChunk(document_id=doc.id, exclusion_id=exc.id, chunk_index=0, text="p1 de novo")
            )
            await db_session.flush()


async def test_worker_sql_nao_enxerga_clause_chunk(db_session):
    """Uma das duas camadas que tornam verdadeira a frase "não é exposta ao worker SQL".

    Antes, a tabela só estava *omitida* do get_schema() do MCP server — um
    `SELECT * FROM clause_chunk` escrito pelo LLM passaria pelo filtro textual, e o
    LIMIT 100 automático despejaria 100 vetores de 1024 floats no contexto da próxima
    chamada, num projeto que mede custo por chamada.

    O que este teste NÃO prova: que o worker conecta *como* insurance_ro. Ele afirma o
    privilégio da role, e `_conninfo()` cai silenciosamente pro DATABASE_URL admin
    quando DATABASE_URL_RO não está setada — cenário em que este assert continua verde
    e o REVOKE não participa de nada. Quem cobre esse caso é a allowlist de tabela do
    run_query (`tests/test_sql_allowlist.py`); as duas camadas são independentes de
    propósito.
    """
    pode_ler_chunk = await db_session.scalar(
        text("SELECT has_table_privilege('insurance_ro', 'clause_chunk', 'SELECT')")
    )
    assert pode_ler_chunk is False

    # E o revoke é cirúrgico: as tabelas de domínio continuam legíveis, senão o
    # worker SQL teria parado de funcionar de vez.
    pode_ler_exclusion = await db_session.scalar(
        text("SELECT has_table_privilege('insurance_ro', 'exclusion', 'SELECT')")
    )
    assert pode_ler_exclusion is True


async def test_dimensao_errada_e_rejeitada_pelo_banco(db_session):
    """A dimensão está no tipo da coluna, então quem barra é o Postgres.

    Importa porque um bug de embedding (modelo trocado, resposta truncada) falha
    no INSERT em vez de contaminar o índice com vetores de tamanho errado.

    O match na mensagem não é decoração: `DBAPIError` sozinho seria satisfeito por
    qualquer IntegrityError vizinha (uma FK, um NOT NULL), e o teste passaria verde
    sem nunca exercitar a dimensão.
    """
    doc = await _documento(db_session)
    exc = await _exclusao(db_session, doc)
    # O INSERT abortado aborta a transação junto. O savepoint isola a falha, senão
    # ela derrubaria a transação da fixture (que é quem mantém o teste isolado).
    with pytest.raises(DBAPIError, match="expected 1024 dimensions"):
        async with db_session.begin_nested():
            db_session.add(
                ClauseChunk(
                    document_id=doc.id,
                    exclusion_id=exc.id,
                    text="vetor curto demais",
                    embedding=[0.1, 0.2, 0.3],   # 3 dimensões, a coluna exige 1024
                )
            )
            await db_session.flush()
