"""A passada de embedding (R2b): `app/rag/embed.py::embed_pending`.

Roda sobre o mesmo documento canônico da suíte (`sample_doc` de `tests/test_persist.py`,
já indexado pela R2a) — 4 chunks —, com o cliente falso de `tests/test_embedding.py`.
Nenhuma chamada real: nada aqui gasta dinheiro.
"""

from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.cost import cost_usd
from app.extraction.persist import persist_document
from app.models import EMBEDDING_DIM, ClauseChunk, CostEvent
from app.rag.embed import _pendentes, embed_pending, pending_stats
from app.rag.embedding import EMBED_MODEL
from app.rag.index import index_document

from tests.test_embedding import FakeVoyage
from tests.test_persist import MANIFEST_ROW, sample_doc

CHUNKS = 4   # 3 exclusões + 1 regra de franquia — ver tests/test_index.py


@pytest_asyncio.fixture
async def db_session(engine):
    """Override do `db_session` do conftest: aqui `commit()` COMMITA de verdade.

    O fixture do conftest liga a Session a uma Connection que já está numa transação
    simples, e nesse caso o SQLAlchemy resolve `join_transaction_mode` como
    **"rollback_only"**: `session.commit()` vira no-op (os dados só ficam pendurados na
    transação externa) e `session.rollback()` derruba tudo. Isso é perfeito pra isolar
    testes que não commitam — e inútil justamente aqui, onde o objeto do teste **é** a
    fronteira do commit: sob "rollback_only", commit por lote e commit único no fim são
    literalmente indistinguíveis, e `test_falha_no_segundo_lote_preserva_o_primeiro`
    passaria verde contra as duas versões.

    Abrindo um `begin_nested()` antes, o modo vira **"create_savepoint"**: cada
    `session.commit()` libera um SAVEPOINT — durável dentro da transação externa — e
    `session.rollback()` volta só até o último savepoint, em vez de apagar a passada
    inteira. É a semântica real de um commit, e o rollback externo no fim continua
    limpando tudo, então o isolamento entre testes não muda.
    """
    conn = await engine.connect()
    txn = await conn.begin()
    await conn.begin_nested()
    Session = async_sessionmaker(bind=conn, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        await session.close()
        await txn.rollback()
        await conn.close()


async def _documento_com_chunks(db_session) -> int:
    pd_id = await persist_document(db_session, sample_doc(), MANIFEST_ROW)
    await index_document(db_session, pd_id)
    return pd_id


async def _chunks(db_session) -> list[ClauseChunk]:
    db_session.expunge_all()   # a R2a grava por Core; sem isto o ORM devolveria o estado antigo
    return (await db_session.scalars(select(ClauseChunk).order_by(ClauseChunk.id))).all()


async def _eventos(db_session) -> list[CostEvent]:
    return (
        await db_session.scalars(
            select(CostEvent).where(CostEvent.agent_name == "embedder").order_by(CostEvent.id)
        )
    ).all()


async def test_preenche_os_chunks_pendentes(db_session):
    await _documento_com_chunks(db_session)
    fake = FakeVoyage()

    r = await embed_pending(db_session, client=fake)

    assert r["chunks"] == CHUNKS
    assert r["batches"] == 1              # 4 chunks cabem num lote de 200
    assert r["tokens"] > 0
    for c in await _chunks(db_session):
        assert c.embedding is not None
        assert len(c.embedding) == EMBEDDING_DIM
        assert c.embedding_model == EMBED_MODEL


async def test_so_toca_nas_linhas_com_embedding_nulo(db_session):
    """`NULL` é a definição de "ainda não indexado" desde a R2a. Uma linha que já tem
    vetor **do modelo atual** não pode ser reenviada: seria pagar de novo pelo mesmo
    texto pra gravar o mesmo vetor."""
    await _documento_com_chunks(db_session)
    ja_tem = (await _chunks(db_session))[0]
    ja_tem.embedding = [0.5] * EMBEDDING_DIM
    ja_tem.embedding_model = EMBED_MODEL
    await db_session.flush()

    fake = FakeVoyage()
    r = await embed_pending(db_session, client=fake)

    assert r["chunks"] == CHUNKS - 1
    assert ja_tem.text not in fake.calls[0]["texts"]   # nem chegou a ser enviado

    depois = {c.id: c for c in await _chunks(db_session)}
    assert depois[ja_tem.id].embedding[0] == 0.5   # o vetor plantado, não um novo
    assert all(
        c.embedding_model == EMBED_MODEL for c in depois.values() if c.id != ja_tem.id
    )


async def test_uma_linha_de_custo_por_lote(db_session):
    """O grão do `cost_event` é a CHAMADA — aqui, o lote (uma chamada à Voyage), não o
    chunk e não a passada. Com 4 chunks em lotes de 2, são duas linhas."""
    await _documento_com_chunks(db_session)
    fake = FakeVoyage()

    r = await embed_pending(db_session, batch_size=2, client=fake)

    assert r["batches"] == 2
    eventos = await _eventos(db_session)
    assert len(eventos) == 2

    for ev, call in zip(eventos, fake.calls, strict=True):
        esperado = sum(max(1, len(t) // 4) for t in call["texts"])
        assert ev.model == EMBED_MODEL
        assert ev.input_tokens == esperado
        assert ev.output_tokens == 0                     # embedding não tem saída
        assert ev.cost_usd == cost_usd(EMBED_MODEL, esperado, 0)
        assert ev.batch is False                         # não é a Batch API da Anthropic
        assert ev.label.startswith("chunks ")
        # Passada offline: não há request HTTP nem cliente autenticado por trás.
        assert ev.request_id is None
        assert ev.client is None

    assert r["cost_usd"] == sum((e.cost_usd for e in eventos), Decimal("0"))
    assert r["tokens"] == sum(e.input_tokens for e in eventos)


async def test_segunda_passada_nao_chama_a_api(db_session):
    """O que torna a passada segura de repetir não é o banco ficar igual — é a API não
    ser chamada. Uma versão que re-embeddasse tudo e regravasse o mesmo vetor deixaria o
    banco idêntico e a fatura maior; por isso o assert é sobre a contagem de chamadas."""
    await _documento_com_chunks(db_session)
    await embed_pending(db_session, client=FakeVoyage())

    segundo = FakeVoyage()
    r = await embed_pending(db_session, client=segundo)

    assert segundo.calls == []
    assert r == {"chunks": 0, "batches": 0, "tokens": 0, "cost_usd": Decimal("0")}
    assert len(await _eventos(db_session)) == 1   # nenhuma linha de custo nova


async def test_falha_no_segundo_lote_preserva_o_primeiro(db_session):
    """A FRONTEIRA DA TRANSAÇÃO é o lote, não a passada — a exceção consciente ao
    contrato de `persist_document`/`index_document`.

    Lá, repetir é de graça, então tudo-ou-nada é o desenho certo. Aqui cada lote é uma
    chamada paga: se um commit único no fim fosse usado, a falha no lote 2 desfaria o
    lote 1 e o dinheiro dele viraria prejuízo — e a passada seguinte pagaria de novo
    pelos mesmos textos. Este teste é o que trava isso: com commit único no fim ele fica
    VERMELHO (0 chunks gravados, 0 linhas de custo).
    """
    await _documento_com_chunks(db_session)
    fake = FakeVoyage(falha_no_lote=2)

    try:
        await embed_pending(db_session, batch_size=2, client=fake)
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("o cliente falso deveria ter estourado no segundo lote")

    # O rollback é o que dá sentido ao teste, e não é artifício: é o que o mundo real
    # faz — o `async with SessionLocal()` do script desfaz o que não foi commitado ao
    # sair pela exceção. Sem ele, o que sobrou *pendurado na session* (o flush do lote
    # que morreu, e até o do lote 1 numa versão com commit único no fim) ainda estaria
    # visível na mesma transação, e o teste ficaria verde sem provar nada.
    await db_session.rollback()

    chunks = await _chunks(db_session)
    embeddados = [c for c in chunks if c.embedding is not None]
    assert len(embeddados) == 2                       # o primeiro lote sobreviveu
    assert all(c.embedding_model == EMBED_MODEL for c in embeddados)

    eventos = await _eventos(db_session)
    assert len(eventos) == 1                          # e o custo dele está gravado
    assert eventos[0].input_tokens > 0


async def test_dimensao_errada_nao_grava_nada(db_session):
    """A validação de `embed_documents` roda antes de qualquer escrita: um vetor do
    tamanho errado tem que morrer com mensagem sobre o MODELO, não virar
    `expected 1024 dimensions` no meio de uma transação já paga."""
    await _documento_com_chunks(db_session)
    fake = FakeVoyage(dim=512)

    try:
        await embed_pending(db_session, client=fake)
    except ValueError as e:
        assert str(EMBEDDING_DIM) in str(e)
    else:
        raise AssertionError("dimensão errada deveria ter levantado ValueError")

    assert all(c.embedding is None for c in await _chunks(db_session))
    assert await _eventos(db_session) == []


async def test_limit_para_a_passada(db_session):
    """`--limit` existe pra provar a passada num punhado de chunks antes de soltar o
    corpus inteiro (que é onde o custo mora)."""
    await _documento_com_chunks(db_session)
    fake = FakeVoyage()

    r = await embed_pending(db_session, limit=3, batch_size=2, client=fake)

    assert r["chunks"] == 3
    assert r["batches"] == 2                          # lotes de 2 e 1
    assert [len(c["texts"]) for c in fake.calls] == [2, 1]
    pendentes = await db_session.scalar(
        select(func.count()).select_from(ClauseChunk).where(ClauseChunk.embedding.is_(None))
    )
    assert pendentes == CHUNKS - 3


async def _com_vetor_de_outro_modelo(db_session) -> ClauseChunk:
    """Planta uma linha embeddada por um modelo anterior — o estado pós-troca de modelo."""
    await _documento_com_chunks(db_session)
    velho = (await _chunks(db_session))[0]
    velho.embedding = [0.5] * EMBEDDING_DIM
    velho.embedding_model = "voyage-modelo-antigo"
    await db_session.flush()
    return velho


async def test_vetor_de_outro_modelo_bloqueia_a_passada(db_session):
    """Trocar `EMBED_MODEL` por um sucessor de mesma dimensão não pode ser um no-op
    silencioso.

    As linhas antigas nunca voltam ao filtro `IS NULL`, mas todo chunk novo sairia com o
    vetor do modelo novo: o HNSW acabaria com dois modelos dentro, distâncias de cosseno
    incomparáveis e busca errada **sem erro nenhum** — o mesmo "índice que mente" da R2a,
    por outra porta. Preencher os NULL e ir embora é justamente construir essa mistura,
    então a passada se recusa a rodar. E se recusa ANTES de gastar.
    """
    await _com_vetor_de_outro_modelo(db_session)
    fake = FakeVoyage()

    with pytest.raises(ValueError, match="modelo diferente"):
        await embed_pending(db_session, client=fake)

    assert fake.calls == []                       # recusa antes de qualquer chamada paga
    assert await _eventos(db_session) == []


async def test_remodel_reembedda_as_linhas_do_modelo_antigo(db_session):
    """A saída da recusa acima: `--remodel` inclui as linhas do modelo velho e devolve o
    índice à consistência. É opt-in porque re-embeddar o corpus é gasto, e gasto não se
    faz por efeito colateral."""
    velho = await _com_vetor_de_outro_modelo(db_session)
    fake = FakeVoyage()

    r = await embed_pending(db_session, client=fake, remodel=True)

    assert r["chunks"] == CHUNKS                  # os 3 NULL + o desatualizado
    assert velho.text in fake.calls[0]["texts"]   # o antigo foi de fato reenviado
    depois = {c.id: c for c in await _chunks(db_session)}
    assert depois[velho.id].embedding[0] != 0.5   # vetor novo por cima do velho
    assert all(c.embedding_model == EMBED_MODEL for c in depois.values())


async def test_lote_e_reivindicado_com_for_update_skip_locked(db_session):
    """Duas passadas simultâneas leriam os MESMOS ids, pagariam as duas pelo mesmo texto
    e gravariam duas linhas de `cost_event` — dinheiro em dobro no livro-caixa que o
    projeto existe pra manter confiável.

    O teste é estrutural (a cláusula no SQL compilado), não de concorrência: provar a
    corrida exigiria duas conexões commitando de verdade, e a fixture isola por
    transação. Ele trava o que dá pra travar aqui — que o lote é reivindicado, e que a
    contagem do `--dry-run` NÃO trava nada (ela é leitura pura e travar linhas num
    dry-run seria bloquear a passada de verdade).
    """
    sql = str(_pendentes(10).compile(db_session.bind))
    assert "FOR UPDATE" in sql and "SKIP LOCKED" in sql

    await _documento_com_chunks(db_session)
    assert (await pending_stats(db_session))["chunks"] == CHUNKS   # não trava, só conta


async def test_dry_run_nao_chama_a_api(db_session):
    """`pending_stats` é o que o `--dry-run` do CLI imprime: conta e estima, sem gastar."""
    await _documento_com_chunks(db_session)

    stats = await pending_stats(db_session)
    assert stats["chunks"] == CHUNKS
    assert stats["chars"] > 0
    assert stats["tokens_estimados"] > 0
    assert stats["custo_estimado"] == cost_usd(EMBED_MODEL, stats["tokens_estimados"], 0)

    await embed_pending(db_session, client=FakeVoyage())
    assert (await pending_stats(db_session))["chunks"] == 0
