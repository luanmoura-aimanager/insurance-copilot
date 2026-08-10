"""Busca por similaridade (R3a): `app/rag/search.py::search_clauses`.

Os vetores são **plantados à mão** em vez de virem de um modelo: o que estes testes
provam é a mecânica da recuperação (ordem, corte, filtro, exclusão dos não indexados),
e amarrar isso à saída de um modelo real tornaria o resultado dependente de semântica
que nem sequer roda aqui. O cliente da Voyage é falso e determinístico — **nenhum teste
gasta dinheiro**.

Toda a geometria vive em duas dimensões do vetor de 1024 (o resto é zero), o que deixa
as distâncias de cosseno calculáveis de cabeça e o teste legível:

    pergunta = (1, 0)
    PERTO  = (1, 0)   -> distância 0.0
    MEIO   = (1, 1)   -> distância 1 - 1/raiz(2) ~ 0.293
    LONGE  = (0, 1)   -> distância 1.0
    OPOSTO = (-1, 0)  -> distância 2.0
"""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.cost import cost_usd
from app.extraction.persist import persist_document
from app.models import EMBEDDING_DIM, ClauseChunk, CostEvent
from app.rag.embedding import EMBED_MODEL
from app.rag.index import index_document
from app.rag.search import search_clauses

from tests.test_persist import MANIFEST_ROW, sample_doc

PERGUNTA = "granizo está coberto?"


def _vec(a: float, b: float) -> list[float]:
    return [a, b] + [0.0] * (EMBEDDING_DIM - 2)


CONSULTA = _vec(1.0, 0.0)
PERTO = _vec(1.0, 0.0)
MEIO = _vec(1.0, 1.0)
LONGE = _vec(0.0, 1.0)
OPOSTO = _vec(-1.0, 0.0)


class FakeQueryVoyage:
    """Cliente falso: devolve sempre o mesmo vetor de consulta, e conta as chamadas.

    O vetor é fixo (e não função do texto) de propósito: aqui a pergunta é só o gatilho
    da chamada — quem decide o resultado são os vetores plantados nos chunks.
    """

    def __init__(self, vetor: list[float] | None = None):
        self.vetor = vetor if vetor is not None else CONSULTA
        self.calls: list[dict] = []

    def embed(self, texts, model=None, input_type=None, truncation=True):
        self.calls.append(
            {"texts": list(texts), "model": model, "input_type": input_type}
        )
        return SimpleNamespace(
            embeddings=[list(self.vetor) for _ in texts],
            total_tokens=sum(max(1, len(t) // 4) for t in texts),
        )


@pytest_asyncio.fixture(autouse=True)
async def cost_rows(engine, monkeypatch):
    """Aponta a session de `record_call_cost` pro container e limpa `cost_event`.

    Autouse porque **toda** busca grava uma linha de custo, e essas linhas são
    COMMITADAS numa transação própria — o rollback do `db_session` não as alcança. Sem a
    limpeza elas vazariam pro `tests/test_cost.py`, que conta a tabela inteira.
    """
    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.SessionLocal", Session)

    async def _limpa():
        async with Session() as s:
            await s.execute(delete(CostEvent))
            await s.commit()

    await _limpa()
    yield Session
    await _limpa()


async def _chunks(db_session, pd_id: int) -> list[ClauseChunk]:
    db_session.expunge_all()   # a R2a grava por Core; sem isto o ORM devolveria o estado antigo
    return (
        await db_session.scalars(
            select(ClauseChunk)
            .where(ClauseChunk.document_id == pd_id)
            .order_by(ClauseChunk.id)
        )
    ).all()


async def _indexar(db_session, manifest: dict | None = None, version: str = "v1") -> int:
    doc = sample_doc()
    doc.version = version
    pd_id = await persist_document(db_session, doc, manifest or MANIFEST_ROW)
    await index_document(db_session, pd_id)
    return pd_id


async def _com_vetores(db_session, vetores: list[list[float] | None], **kw) -> list[ClauseChunk]:
    """Documento indexado (4 chunks) com os vetores plantados na ordem dada.

    `None` deixa a linha como a R2a a criou — `embedding IS NULL`, "ainda não indexada".
    """
    pd_id = await _indexar(db_session, **kw)
    chunks = await _chunks(db_session, pd_id)
    assert len(chunks) == len(vetores), "o documento canônico da suíte tem 4 chunks"
    for chunk, vetor in zip(chunks, vetores, strict=True):
        if vetor is not None:
            chunk.embedding = vetor
            chunk.embedding_model = EMBED_MODEL
    await db_session.flush()
    return chunks


async def test_ordena_por_distancia_e_respeita_o_k(db_session):
    """O contrato central: os `k` vizinhos mais próximos, do mais próximo ao mais
    distante. Os vetores são plantados FORA da ordem de id, senão um `ORDER BY` ignorado
    (ou trocado por ordem de inserção) passaria verde."""
    c_longe, c_perto, c_meio, c_oposto = await _com_vetores(
        db_session, [LONGE, PERTO, MEIO, OPOSTO]
    )
    fake = FakeQueryVoyage()

    hits = await search_clauses(db_session, PERGUNTA, k=2, client=fake)

    assert [h.chunk_id for h in hits] == [c_perto.id, c_meio.id]
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert hits[1].distance == pytest.approx(1 - 2 ** -0.5, abs=1e-6)

    # A origem viaja junto: sem ela o texto recuperado é uma frase sem procedência.
    assert all((h.exclusion_id is None) != (h.coverage_id is None) for h in hits)
    assert all(h.text for h in hits)

    # `input_type="query"` é o outro lado do par assimétrico da indexação
    # (`"document"`). Errar não levanta erro nenhum — só piora os vizinhos em silêncio.
    assert fake.calls[0]["input_type"] == "query"
    assert fake.calls[0]["texts"] == [PERGUNTA]

    # k maior que o corte anterior traz o resto, na mesma ordem — o ranking é global,
    # não uma janela do que o k anterior devolveu.
    todos = await search_clauses(db_session, PERGUNTA, k=10, client=FakeQueryVoyage())
    assert [h.chunk_id for h in todos] == [c_perto.id, c_meio.id, c_longe.id, c_oposto.id]


async def test_max_distance_corta_os_distantes(db_session):
    """O corte é de RELEVÂNCIA, e acontece depois do LIMIT: a busca já devolveu os `k`
    vizinhos, e o limiar decide quais deles merecem virar resposta."""
    _, c_perto, c_meio, _ = await _com_vetores(db_session, [LONGE, PERTO, MEIO, OPOSTO])

    hits = await search_clauses(
        db_session, PERGUNTA, k=4, max_distance=0.5, client=FakeQueryVoyage()
    )

    assert [h.chunk_id for h in hits] == [c_perto.id, c_meio.id]
    assert all(h.distance <= 0.5 for h in hits)


async def test_lista_vazia_so_pode_vir_do_limiar(db_session):
    """A busca vetorial **sempre** devolveria `k` resultados: o `ORDER BY ... LIMIT`
    entrega os k vizinhos menos distantes ainda que o melhor deles esteja do outro lado
    do espaço. Uma pergunta fora do escopo do corpus (bicicleta, seguro de viagem) volta
    com 4 cláusulas de seguro residencial, e um sintetizador confiante transforma isso
    numa resposta errada com citação. É o limiar — e só ele — que produz o "não sei".
    """
    await _com_vetores(db_session, [LONGE, OPOSTO, LONGE, OPOSTO])
    fake = FakeQueryVoyage()

    sem_limiar = await search_clauses(db_session, PERGUNTA, k=4, client=fake)
    assert len(sem_limiar) == 4                      # nada "não encontrado" aqui
    assert min(h.distance for h in sem_limiar) >= 1.0

    com_limiar = await search_clauses(
        db_session, PERGUNTA, k=4, max_distance=0.5, client=fake
    )
    assert com_limiar == []


async def test_document_ids_filtra_antes_do_ranking(db_session):
    """Restringir a um documento é *identidade* (seguradora, produto, processo SUSEP),
    que a R2a deixou de fora do texto do chunk justamente porque um `WHERE` responde isso
    de graça e com precisão exata.

    O chunk mais próximo da pergunta mora no documento B, e a busca é restrita ao A: se o
    filtro fosse aplicado depois do ranking (ou não fosse aplicado), ele apareceria.
    """
    outro_manifest = {**MANIFEST_ROW, "processo": "15414.111111/2024-11", "sha256": "def456"}

    # Documento A: nada perto. Documento B: o vizinho mais próximo de todos.
    a_chunks = await _com_vetores(db_session, [LONGE, LONGE, OPOSTO, OPOSTO])
    b_chunks = await _com_vetores(
        db_session, [PERTO, MEIO, LONGE, OPOSTO], manifest=outro_manifest, version="v2"
    )
    doc_a = a_chunks[0].document_id
    doc_b = b_chunks[0].document_id
    assert doc_a != doc_b

    hits = await search_clauses(
        db_session, PERGUNTA, k=5, document_ids=[doc_a], client=FakeQueryVoyage()
    )

    assert hits, "o recorte não está vazio — o documento A tem chunks indexados"
    assert {h.document_id for h in hits} == {doc_a}
    assert b_chunks[0].id not in {h.chunk_id for h in hits}   # o mais próximo, barrado
    assert min(h.distance for h in hits) >= 1.0              # e o que sobrou é distante

    # Sem o recorte, o vizinho do documento B é o primeiro colocado — é o que prova que
    # o filtro, e não a geometria, foi o que o tirou da lista acima.
    livre = await search_clauses(db_session, PERGUNTA, k=1, client=FakeQueryVoyage())
    assert livre[0].chunk_id == b_chunks[0].id


async def test_lista_de_documentos_vazia_nao_chama_a_api(db_session):
    """`document_ids=[]` é um recorte vazio: a resposta é `[]` sem depender do vetor, e
    embeddar a pergunta seria uma chamada paga por nada."""
    await _com_vetores(db_session, [PERTO, MEIO, LONGE, OPOSTO])
    fake = FakeQueryVoyage()

    assert await search_clauses(db_session, PERGUNTA, document_ids=[], client=fake) == []
    assert fake.calls == []


async def test_chunk_sem_embedding_nunca_entra(db_session):
    """`embedding IS NULL` é "materializado pela R2a, ainda não embeddado pela R2b": a
    linha não está no índice HNSW e não tem distância. Sem o filtro explícito ela viria
    com `distance = None` e envenenaria a ordenação — pior, entraria numa resposta como
    se fosse uma cláusula recuperada.
    """
    chunks = await _com_vetores(db_session, [None, None, MEIO, None])
    unico = chunks[2]

    hits = await search_clauses(db_session, PERGUNTA, k=5, client=FakeQueryVoyage())

    assert [h.chunk_id for h in hits] == [unico.id]


async def test_grava_uma_linha_de_custo_por_busca(db_session, cost_rows):
    """Embeddar a pergunta é uma chamada paga como qualquer outra, e o grão do
    `cost_event` continua sendo a CHAMADA.

    A diferença pra passada de indexação (`agent_name="embedder"`, offline) é que a busca
    roda dentro de um request — daí o nome próprio, que é o que permite responder "quanto
    da conta é recuperação e quanto é geração".
    """
    await _com_vetores(db_session, [PERTO, MEIO, LONGE, OPOSTO])
    fake = FakeQueryVoyage()

    await search_clauses(db_session, PERGUNTA, client=fake)

    async with cost_rows() as s:
        eventos = list((await s.execute(select(CostEvent))).scalars())

    assert len(eventos) == 1
    ev = eventos[0]
    assert ev.agent_name == "rag_search"
    assert ev.model == EMBED_MODEL
    # O número gravado é o que a API COBROU, não uma estimativa por caracteres — é por
    # isso que a busca usa `embed_query_with_tokens` e não `embed_query`.
    assert ev.input_tokens == max(1, len(PERGUNTA) // 4)
    assert ev.output_tokens == 0          # embedding não tem token de saída
    assert ev.batch is False               # não é a Batch API da Anthropic

    # `cost_usd` bate com o cálculo de `app.cost`, e o valor é ZERO — não por bug: a
    # coluna é Numeric(12, 6) e uma pergunta de ~6 tokens a US$ 0,02/Mtok dá 1,2e-7, que
    # arredonda pra baixo. Seriam necessários ~50 tokens numa única chamada pra registrar
    # o primeiro microdólar. O que fica gravado e é verdadeiro é a CONTAGEM DE TOKENS
    # (`input_tokens`), que é o que permite somar o custo real da recuperação depois;
    # somar `cost_usd` de milhares de buscas devolveria zero.
    assert ev.cost_usd == cost_usd(EMBED_MODEL, ev.input_tokens, 0)
    assert ev.cost_usd == 0


async def test_ef_search_acompanha_o_k(db_session):
    """`hnsw.ef_search` (default 40) é um TETO SILENCIOSO: com o índice em uso, um
    `LIMIT 100` volta com 40 linhas e nenhum erro — verificado no corpus real forçando
    `enable_seqscan=off`. Isso quebraria a garantia de "`k` resultados" que o docstring
    promete, e pior no caminho filtrado, onde o `WHERE document_id` é aplicado depois da
    varredura aproximada e sobra ~1 candidato por documento.

    Aqui a suíte tem 4 chunks e o planner usa seq scan de qualquer jeito, então o teste
    não consegue ver o truncamento — o que ele prova é que o parâmetro foi **de fato
    ajustado na transação**, que é a parte que dá pra travar. `SET LOCAL` vale até o fim
    da transação, então o `SHOW` depois da busca enxerga o valor.
    """
    await _com_vetores(db_session, [PERTO, MEIO, LONGE, OPOSTO])

    await search_clauses(db_session, PERGUNTA, k=50, client=FakeQueryVoyage())
    assert int(await db_session.scalar(text("SHOW hnsw.ef_search"))) == 200   # 50 * 4

    # Piso: um k pequeno não pode REDUZIR a varredura abaixo do default do pgvector.
    await search_clauses(db_session, PERGUNTA, k=2, client=FakeQueryVoyage())
    assert int(await db_session.scalar(text("SHOW hnsw.ef_search"))) == 40

    # Teto: 1000 é o máximo que o pgvector aceita — pedir mais é erro de configuração.
    await search_clauses(db_session, PERGUNTA, k=500, client=FakeQueryVoyage())
    assert int(await db_session.scalar(text("SHOW hnsw.ef_search"))) == 1000


async def test_vetor_de_outro_modelo_nao_entra_no_ranking(db_session):
    """`embed_pending` commita por lote (de propósito), então um `--remodel` interrompido
    deixa metade do corpus num modelo e metade no outro. Distâncias de cosseno de modelos
    diferentes não são comparáveis: sem este filtro a busca ordenaria os dois juntos e
    devolveria vizinhos errados **sem erro nenhum** — a mesma "índice que mente" que
    `contar_desatualizados` guarda do lado da escrita, entrando pela leitura.

    O chunk do modelo antigo é o MAIS PRÓXIMO da pergunta de propósito: se o filtro não
    existisse, ele seria o primeiro colocado.
    """
    chunks = await _com_vetores(db_session, [PERTO, MEIO, LONGE, OPOSTO])
    antigo = chunks[0]
    antigo.embedding_model = "voyage-modelo-antigo"
    await db_session.flush()

    hits = await search_clauses(db_session, PERGUNTA, k=5, client=FakeQueryVoyage())

    assert antigo.id not in {h.chunk_id for h in hits}
    assert len(hits) == 3

    # E o caso extremo: corpus INTEIRO no modelo antigo devolve vazio, em vez de devolver
    # uma ordenação que mistura dois espaços vetoriais. Vazio é recuperável (é só rodar
    # o --remodel até o fim); ranking errado com cara de certo, não.
    for c in await _chunks(db_session, antigo.document_id):
        c.embedding_model = "voyage-modelo-antigo"
    await db_session.flush()
    assert await search_clauses(db_session, PERGUNTA, k=5, client=FakeQueryVoyage()) == []


async def test_k_invalido_falha_alto(db_session):
    """`k=0` viraria `LIMIT 0` e devolveria `[]` — indistinguível de "nada passou do
    limiar", que é a única lista vazia legítima desta função."""
    await _com_vetores(db_session, [PERTO, MEIO, LONGE, OPOSTO])
    fake = FakeQueryVoyage()

    for ruim in (0, -3):
        with pytest.raises(ValueError, match="k tem que ser >= 1"):
            await search_clauses(db_session, PERGUNTA, k=ruim, client=fake)

    assert fake.calls == []   # a validação vem antes da chamada paga
