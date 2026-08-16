"""A resposta declara sua BASE — e a declaração é montada em CÓDIGO, nunca pelo LLM.

Este é o único módulo que exercita os rodapés CONTADOS de verdade: os da rota SQL e do
NADA_RELEVANTE consultam o banco, então aqui o `SessionLocal` aponta pro container e há
documentos REALMENTE inseridos e commitados, pra que o número da resposta venha do banco e
não de uma constante. Os outros módulos do grafo desligam essas contagens com um no-op
explícito, do mesmo jeito que já desligam `record_call_cost`: contar exige banco, e uma
asserção sobre o texto da resposta não pode depender de quantos documentos outro módulo
deixou commitados.

O corpus de teste é desenhado pra que os três números possíveis sejam DIFERENTES entre si
— com um corpus uniforme, usar a contagem errada passaria verde:

  - 5 linhas em `policy_document`, porque um dos produtos tem duas versões;
  - 4 PRODUTOS (`distinct susep_process`), que é o número honesto do rodapé;
  - 2 produtos ALCANÇÁVEIS pela busca — dos outros dois, um não tem vetor nenhum e o
    outro tem vetor de um modelo antigo.

O que se prova aqui, e por que cada um importa:

  - o número da rota SQL vem do BANCO e conta PRODUTOS, não linhas de tabela;
  - o da rota RAG conta os hits que sobraram DEPOIS do corte por distância — um hit
    descartado não pode inflar nem as cláusulas nem as apólices;
  - `NADA_RELEVANTE` declara a base PESQUISÁVEL, que é menor que o corpus;
  - quem declara é o WORKER: sem declaração reconhecida (ou com erro de query), sem
    rodapé — o default é fail-closed;
  - `FORA_DE_ESCOPO` e `FALHA_INTERNA` **não** declaram (nada foi consultado nos dois
    casos — o rodapé afirmaria uma verificação que não houve);
  - o prompt do synthesizer não menciona base nenhuma, que é a guarda contra alguém
    reintroduzir a instrução no lugar do código.

ZERO chamada real: Anthropic, busca e SQL são fakes. Nenhum teste gasta.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents import graph as graph_mod
from app.models import EMBEDDING_DIM, ClauseChunk, Exclusion, PolicyDocument
from app.rag.embedding import EMBED_MODEL
from app.rag.search import Hit

TOKEN = "token-de-teste"
CLIENTE = "teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

FRASE = "A cláusula exclui danos por infiltração de janela aberta."

# A fixture planta QUATRO produtos e deixa só DOIS ao alcance da busca. Os outros dois são
# as duas maneiras de um documento estar na base e fora do alcance, e as duas são estados
# reais e documentados: um sem vetor nenhum (o intervalo normal entre a extração e
# `scripts/embed_chunks.py`) e um com vetor de OUTRO modelo (um `--remodel` interrompido,
# que o commit-por-lote do `embed_pending` torna possível). São eles que tornam as duas
# bases numericamente diferentes, e é essa diferença que prova que a rota SQL e o "não
# achei" não podem usar a mesma contagem.
PRODUTOS_NA_BASE = 4
PRODUTOS_PESQUISAVEIS = 2
MODELO_ANTIGO = "voyage-modelo-anterior"

# Um dos produtos entra com DUAS versões, então a tabela tem 5 linhas pra 4 produtos. O
# grão de `policy_document` é `(susep_process, version)` e `susep_harvest.py
# --all-versions` existe pra baixar o histórico inteiro — sem esta linha extra, `count(*)`
# e `count(distinct susep_process)` dariam o mesmo número e o rodapé poderia contar versões
# como se fossem seguradoras diferentes sem nada ficar vermelho.
LINHAS_NA_BASE = 5

# As strings LITERAIS, escritas à mão. Montá-las a partir de constantes do módulo faria o
# teste concordar com o código por construção.
SUFIXO_CORPUS = "Base: corpus de 4 apólice(s)."
SUFIXO_PESQUISADO = "Base: 2 apólice(s) pesquisada(s)."

# Marca de todo documento plantado por este módulo: a fixture apaga por ela no teardown.
# Estas linhas são COMMITADAS (o rodapé é lido por uma sessão própria, fora da transação
# do teste), então o rollback do `db_session` não as alcançaria.
MARCA = "base-declarada-teste"


# --- Fakes do client Anthropic (mesma superfície dos outros módulos do grafo) ---
class _FakeToolUse:
    type = "tool_use"

    def __init__(self, payload: dict):
        self.input = payload


class _FakeText:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, blocks: list):
        self.content = blocks


class _FakeMessages:
    def __init__(self, decisions: list[str], synth_explode: bool = False):
        self._decisions = decisions
        self._synth_explode = synth_explode
        self.supervisor_calls = 0
        self.synth_calls = 0
        self.synth_prompts: list[str] = []
        self.synth_systems: list[str] = []

    async def create(self, **kwargs):
        # Sem tools = synthesizer: é o único nó que pede texto livre.
        if not kwargs.get("tools"):
            self.synth_calls += 1
            self.synth_prompts.append(kwargs["messages"][0]["content"])
            self.synth_systems.append(kwargs["system"])
            if self._synth_explode:
                raise RuntimeError("API do synthesizer fora do ar")
            return _FakeResponse([_FakeText(FRASE)])

        if kwargs["tools"][0]["name"] == graph_mod._DECISION_TOOL:
            i = self.supervisor_calls
            self.supervisor_calls += 1
            # `min(i, len-1)` daria `[-1]` com a lista vazia, ou seja, um IndexError vindo
            # de dentro do fake em vez de uma mensagem legível. Os testes que chamam o
            # synthesizer direto constroem `_FakeClient([])` e nunca chegam aqui — este
            # assert é o que faz o dia em que um deles passar a rotear por `/ask` sair como
            # diagnóstico, e não como erro de indexação.
            assert self._decisions, "supervisor chamado sem roteiro de decisões"
            nxt = self._decisions[min(i, len(self._decisions) - 1)]
            return _FakeResponse([_FakeToolUse({"next": nxt, "reasoning": f"fake #{i}"})])
        return _FakeResponse([_FakeToolUse({"sql": "SELECT count(*) FROM peril"})])


class _FakeClient:
    def __init__(self, decisions: list[str], synth_explode: bool = False):
        self.messages = _FakeMessages(decisions, synth_explode)


@pytest_asyncio.fixture
async def corpus(engine, monkeypatch):
    """Planta o corpus de teste COMMITADO e aponta `SessionLocal` pro container.

    `PRODUTOS_NA_BASE` produtos em `LINHAS_NA_BASE` linhas (um produto entra com duas
    versões), dos quais `PRODUTOS_PESQUISAVEIS` recebem uma exclusão com chunk embeddado no
    modelo atual. Os outros dois ficam fora do alcance da busca de maneiras diferentes — um
    sem vetor, outro com vetor de modelo antigo. As três contagens têm que sair diferentes,
    senão usar a errada passaria verde.

    As contagens são lidas por uma sessão que os nós abrem sozinhos (mesmo padrão de
    `record_call_cost`), fora da transação do teste — então documento inserido via
    `db_session` seria invisível pra elas. Daí commitar, e daí a limpeza no teardown: do
    contrário estas linhas vazariam pros módulos seguintes.
    """
    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.SessionLocal", Session)

    async def _limpa():
        async with Session() as s:
            docs = select(PolicyDocument.id).where(PolicyDocument.pdf_hash.like(f"{MARCA}%"))
            await s.execute(delete(ClauseChunk).where(ClauseChunk.document_id.in_(docs)))
            await s.execute(delete(Exclusion).where(Exclusion.document_id.in_(docs)))
            await s.execute(delete(PolicyDocument).where(PolicyDocument.pdf_hash.like(f"{MARCA}%")))
            await s.commit()

    await _limpa()
    async with Session() as s:
        # PRÉ-CONDIÇÃO explícita: os testes abaixo asserem números ABSOLUTOS de tabela, o
        # que só é verdade num banco que ninguém mais povoou. Hoje é o caso (os outros
        # módulos escrevem `policy_document` por `db_session`, que dá rollback), e o dia em
        # que deixar de ser tem que sair como esta mensagem — e não como um rodapé com o
        # número errado, que é o sintoma que este módulo inteiro existe pra pegar.
        sobrando = await s.scalar(select(func.count()).select_from(PolicyDocument))
        assert sobrando == 0, (
            f"{sobrando} policy_document commitado(s) por outro teste: as contagens "
            "absolutas deste módulo não valem mais"
        )

        docs = [
            PolicyDocument(
                insurer=f"Seguradora {i}",
                product="Residencial",
                susep_process=f"15414.{i:06d}/2024-11",
                version="1",
                pdf_url=f"https://exemplo/{i}.pdf",
                pdf_hash=f"{MARCA}-{i}",
            )
            for i in range(PRODUTOS_NA_BASE)
        ]
        # A segunda versão do PRIMEIRO produto: linha nova, produto que já existe. É ela
        # que separa `count(*)` de `count(distinct susep_process)`.
        docs.append(
            PolicyDocument(
                insurer="Seguradora 0",
                product="Residencial",
                susep_process=docs[0].susep_process,
                version="2",
                pdf_url="https://exemplo/0-v2.pdf",
                pdf_hash=f"{MARCA}-0-v2",
            )
        )
        assert len(docs) == LINHAS_NA_BASE
        s.add_all(docs)
        await s.flush()

        # O vetor é plantado à mão (mesma técnica de tests/test_search.py): o que se conta
        # aqui é o recorte `embedding IS NOT NULL AND embedding_model = EMBED_MODEL`, não
        # semântica. Os `DOCS_PESQUISAVEIS` primeiros ficam dentro dele; o penúltimo fica
        # sem chunk nenhum; o ÚLTIMO tem chunk com vetor de outro modelo — dentro do
        # `IS NOT NULL` e fora do alcance da busca, que é o caso que um filtro de contagem
        # escrito à mão (em vez de compartilhado com a busca) erraria.
        # A quinta linha (segunda versão do produto 0) TAMBÉM é embeddada, e isso é
        # deliberado: assim três *linhas* são alcançáveis mas só dois *produtos*, e uma
        # contagem por `distinct document_id` diria 3 onde a honesta diz 2. Sem essa
        # sobreposição, contar linha e contar produto dariam o mesmo número aqui.
        modelos = [EMBED_MODEL] * PRODUTOS_PESQUISAVEIS + [None, MODELO_ANTIGO, EMBED_MODEL]
        for doc, modelo in zip(docs, modelos):
            if modelo is None:
                continue
            exc = Exclusion(document_id=doc.id, scope="general", clause_text="Danos por guerra.")
            s.add(exc)
            await s.flush()
            s.add(ClauseChunk(
                document_id=doc.id,
                exclusion_id=exc.id,
                chunk_index=0,
                text="Exclusão geral da apólice: danos por guerra.",
                embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
                embedding_model=modelo,
            ))
        await s.commit()

    yield Session
    await _limpa()


@pytest.fixture
def fake_graph(monkeypatch):
    """LLM, schema/SQL e busca falsos. O rodapé NÃO é mockado — é o assunto do módulo."""

    def _install(
        decisions: list[str],
        hits: list[Hit] | None = None,
        busca_explode: str | None = None,
        rows: str = "Columns: count\nRows:\n7\n",
        synth_explode: bool = False,
    ):
        client = _FakeClient(decisions, synth_explode)
        monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)
        monkeypatch.setattr(graph_mod, "get_schema", lambda: "TABLE peril(id int, nome text)")
        monkeypatch.setattr(graph_mod, "run_query", lambda sql: rows)

        async def _busca(session, question, *, k=5, max_distance=None, **kw):
            if busca_explode is not None:
                raise RuntimeError(busca_explode)
            return list(hits or [])

        monkeypatch.setattr(graph_mod, "search_clauses", _busca)

        async def _sem_custo(**kwargs):
            return None

        monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)
        return SimpleNamespace(client=client)

    return _install


@pytest_asyncio.fixture
async def ask_client(corpus, monkeypatch):
    monkeypatch.setenv("API_TOKENS", f"{CLIENTE}:{TOKEN}")
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _perguntar(ask_client, pergunta: str) -> str:
    r = await ask_client.post("/ask", json={"question": pergunta}, headers=AUTH)
    assert r.status_code == 200
    return r.json()["answer"]


# --- Rota SQL: a base é o CORPUS, e o número vem do banco ---
async def test_rota_sql_declara_a_base_do_corpus(ask_client, fake_graph):
    """Quatro produtos no banco ⇒ "4 apólice(s)". O 4 é CONTADO, não escrito.

    A asserção de igualdade cobre as três metades de uma vez: a frase do modelo, o
    separador (linha em branco — o rodapé é um bloco à parte, não uma oração da resposta)
    e o rodapé literal. Um sufixo com outra contagem, ou colado com um espaço, falha aqui.
    """
    f = fake_graph(["sql_worker"])

    answer = await _perguntar(ask_client, "Quantos perigos existem na base?")

    assert answer.endswith(SUFIXO_CORPUS)
    assert answer == f"{FRASE}\n\n{SUFIXO_CORPUS}"

    # E o rodapé NÃO saiu do modelo: o fake do synthesizer devolve só FRASE, e o payload
    # que ele recebeu não continha número nenhum de documento pra ele copiar.
    assert f.client.messages.synth_calls == 1
    assert "Base:" not in f.client.messages.synth_prompts[0]


async def test_o_numero_da_base_sai_do_banco_e_nao_de_uma_constante(
    ask_client, fake_graph, corpus
):
    """Apagar os documentos muda o rodapé. Sem isto, um literal "4" passaria igual.

    É o mesmo `/ask`, com a base esvaziada entre uma pergunta e outra: se o número fosse
    fixo (ou cacheado no processo), a segunda resposta continuaria dizendo "4".
    """
    fake_graph(["sql_worker"])

    primeira = await _perguntar(ask_client, "Quantos perigos existem na base?")
    assert primeira.endswith(SUFIXO_CORPUS)

    async with corpus() as s:
        docs = select(PolicyDocument.id).where(PolicyDocument.pdf_hash.like(f"{MARCA}%"))
        await s.execute(delete(ClauseChunk).where(ClauseChunk.document_id.in_(docs)))
        await s.execute(delete(Exclusion).where(Exclusion.document_id.in_(docs)))
        await s.execute(delete(PolicyDocument).where(PolicyDocument.pdf_hash.like(f"{MARCA}%")))
        await s.commit()

    segunda = await _perguntar(ask_client, "Quantos perigos existem na base?")
    assert segunda.endswith("Base: corpus de 0 apólice(s).")


async def test_erro_de_query_nao_declara_base(ask_client, fake_graph):
    """SQL inválido volta como TEXTO — e texto de erro não é apólice consultada.

    `run_query` devolve as recusas do guard e o erro do Postgres como string, pelo mesmo
    contrato que faz o SQL inválido ser reportável em vez de fatal. Mas aí o worker produz
    uma mensagem normal, o synthesizer a parafraseia ("não consegui obter essa
    informação") e, sem esta guarda, o rodapé afirmaria em cima disso que N apólices foram
    corpus — numa resposta em que o SELECT não chegou a ler nada.
    """
    from mcp_servers.postgres_mcp_server import ERRO_PREFIXO

    erro = f'{ERRO_PREFIXO} executing query: column "nome_errado" does not exist'
    f = fake_graph(["sql_worker"], rows=erro)

    answer = await _perguntar(ask_client, "Quantos perigos existem na base?")

    assert answer == FRASE          # o erro ainda vira frase (isso não mudou)...
    assert "Base:" not in answer    # ...mas sem declaração de base nenhuma
    assert f.client.messages.synth_calls == 1


async def test_nada_relevante_sem_declaracao_nao_ganha_rodape(monkeypatch, corpus):
    """Até o "não achei" tira a base da MENSAGEM — a frase não escolhe uma base sozinha.

    É o que impede o fail-open de voltar por essa porta. Enquanto a busca for global os
    dois caminhos dão o mesmo número, então nenhum teste de ponta a ponta os distingue:
    este pega a diferença estruturalmente, com uma mensagem que não declarou nada. Um
    synthesizer que chamasse `_sufixo_pesquisado()` direto no branch carimbaria a base
    global aqui — e faria o mesmo, no dia do recorte por `document_ids`, numa busca que
    varreu um documento só.
    """
    client = _FakeClient([])
    monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)

    async def _sem_custo(**kwargs):
        return None

    monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)

    state = {
        "iterations": 1,
        "next": "END",
        "messages": [
            HumanMessage(content="Meteorito é coberto?"),
            AIMessage(content=graph_mod.NADA_RELEVANTE, name="rag_worker"),  # sem declaração
        ],
    }

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == graph_mod.NADA_RELEVANTE
    assert "Base:" not in out["messages"][0].content
    assert client.messages.synth_calls == 0


@pytest.mark.parametrize(
    "sql_recusado",
    [
        pytest.param("SELECT 1; DROP TABLE peril", id="statements-empilhados"),
        pytest.param("UPDATE peril SET name = 'x'", id="nao-e-select"),
        pytest.param("SELECT * FROM clause_chunk", id="tabela-fora-da-allowlist"),
    ],
)
def test_toda_recusa_do_run_query_carrega_o_prefixo(sql_recusado):
    """As recusas do guard também têm que ser reconhecíveis como "não consultei".

    `test_erro_de_query_nao_declara_base` monta a string de erro A PARTIR de
    `ERRO_PREFIXO`, então ele prova que o LEITOR reconhece o prefixo — nunca que o
    PRODUTOR o emite. Um quinto `return` adicionado ao `run_query` sem o prefixo (uma
    guarda nova, ou uma mensagem reescrita em pt-BR) faria `consultou` virar `True` em
    silêncio, e toda query recusada ganharia "Base: corpus de N apólice(s)" embaixo da
    paráfrase do erro, com a suíte verde. Este teste fecha o outro lado do par, exercitando
    os três caminhos que rejeitam ANTES de tocar no banco (então roda sem container).
    """
    from mcp_servers.postgres_mcp_server import ERRO_PREFIXO, run_query

    saida = run_query(sql_recusado)

    assert saida.startswith(ERRO_PREFIXO), saida
    # E é isso, literalmente, que o sql_worker vai perguntar.
    assert not saida.startswith("Columns:")


async def test_rodape_sobrevive_a_queda_do_synthesizer(ask_client, fake_graph):
    """A degradação (`frase or resultado`) mantém o rodapé, e é onde ele vale mais.

    O que foi consultado não muda quando a síntese cai. E um despejo cru de resultado é
    justamente a resposta mais difícil de interpretar — tirar dela a procedência seria
    tirar a única coisa que a torna legível.
    """
    fake_graph(["sql_worker"], synth_explode=True)

    answer = await _perguntar(ask_client, "Quantos perigos existem na base?")

    assert "Rows:" in answer        # caiu no texto cru do worker
    assert answer.endswith(SUFIXO_CORPUS)


async def test_declaracao_de_base_malformada_nao_derruba_o_ask(monkeypatch, corpus):
    """Payload de base quebrado sai como resposta sem rodapé, nunca como 500.

    Os subscritos de `_sufixo_da_resposta` leem um dicionário montado por outro nó. Um
    `KeyError` ali escaparia do synthesizer — o ÚLTIMO nó —, abortaria o grafo e devolveria
    500 num request que já pagou supervisor e worker e já tem a resposta na mão, que é
    exatamente a invariante que o resto do módulo sustenta.
    """
    client = _FakeClient([])
    monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)

    async def _sem_custo(**kwargs):
        return None

    monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)

    state = {
        "iterations": 1,
        "next": "END",
        "messages": [
            HumanMessage(content="Infiltração é coberta?"),
            AIMessage(
                content="Cláusulas recuperadas:\n[chunk 1 | doc 1 | dist 0.100] texto",
                name="rag_worker",
                # `tipo` certo, chaves de contagem ausentes: o formato que um refactor
                # parcial (ou um checkpointer que serializou outra versão) produziria.
                additional_kwargs={graph_mod.BASE: {"tipo": graph_mod.BASE_RECUPERADO}},
            ),
        ],
    }

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == FRASE
    assert "Base:" not in out["messages"][0].content


@pytest.mark.parametrize(
    "declaracao",
    [
        pytest.param({}, id="nao-declarou-nada"),
        pytest.param({"tipo": "recuperado_v2"}, id="declarou-tipo-desconhecido"),
    ],
)
async def test_worker_que_nao_declara_base_nao_ganha_rodape(monkeypatch, corpus, declaracao):
    """Sem declaração RECONHECIDA, sem rodapé — o default é FECHADO, e isso é a decisão.

    Os dois casos são branches diferentes e precisam dos dois: a chave ausente é o worker
    novo que esqueceu, e o `tipo` desconhecido é o mesmo worker depois de alguém renomear
    a constante de um lado só. Fail-open em qualquer um deles carimba "o corpus inteiro
    foi lido" numa resposta que leu outra coisa.

    O desenho anterior era o inverso: quem não dissesse nada herdava o rodapé do corpus.
    Aí a saída INSEGURA era a padrão — bastava um worker novo (uma segunda rota de
    recuperação, o worker de extração) entrar em `WORKERS` sem preencher a chave pra que
    toda resposta dele afirmasse que o corpus inteiro foi lido, sem erro nenhum e sem
    nada ficar vermelho. Rodapé a menos é uma resposta sem procedência; rodapé a mais é
    uma afirmação falsa de cobertura, que é o único erro que este rodapé não pode cometer.

    O `corpus` está na assinatura de propósito: com os documentos plantados, uma versão
    fail-open teria de fato o que contar e devolveria SUFIXO_CORPUS. Sem ele o teste
    passaria pelo motivo errado (a contagem falharia e o rodapé sumiria de qualquer jeito).
    """
    client = _FakeClient([])
    monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)

    async def _sem_custo(**kwargs):
        return None

    monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)

    state = {
        "iterations": 1,
        "next": "END",
        "messages": [
            HumanMessage(content="Infiltração é coberta?"),
            # Um worker de `WORKERS` cuja declaração o synthesizer não reconhece.
            AIMessage(
                content="Achado do worker novo.",
                name="rag_worker",
                additional_kwargs=({graph_mod.BASE: declaracao} if declaracao else {}),
            ),
        ],
    }

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == FRASE
    assert "Base:" not in out["messages"][0].content
    assert SUFIXO_CORPUS not in out["messages"][0].content


async def test_falha_ao_contar_a_base_nao_derruba_a_resposta(ask_client, fake_graph, monkeypatch):
    """Best effort, como `_record_cost`: sem contagem, sai a resposta SEM rodapé.

    Quando chegamos ao rodapé a pergunta já foi paga e já tem resposta. Trocar uma
    resposta boa por um 500 — ou por FALHA_INTERNA — só porque o denominador não pôde ser
    lido seria destruir o que já se comprou; e inventar um número seria pior ainda, porque
    o rodapé existe justamente pra ser confiável.
    """
    f = fake_graph(["sql_worker"])

    async def _conta_quebrada(session):
        raise RuntimeError("contagem fora do ar")

    monkeypatch.setattr(graph_mod, "_base_do_corpus", _conta_quebrada)

    answer = await _perguntar(ask_client, "Quantos perigos existem na base?")

    assert answer == FRASE          # a resposta chega inteira...
    assert "Base:" not in answer    # ...só sem a declaração que não pôde ser verificada
    assert f.client.messages.synth_calls == 1


# --- Rota RAG: a base é o que SOBROU do corte ---
async def test_rota_rag_conta_so_os_hits_que_passaram_do_limiar(ask_client, fake_graph):
    """Um hit descartado pelo limiar não conta — nem como cláusula, nem como apólice.

    Os dois números são load-bearing e o teste separa os dois: o descartado está num
    TERCEIRO documento, então uma contagem feita sobre `hits` (antes do corte) diria
    "3 cláusula(s) de 3 apólice(s)" e as duas metades estariam erradas de uma vez. É a
    diferença entre declarar o que o synthesizer leu e declarar o que a busca trouxe —
    e a segunda faz a resposta parecer mais apoiada do que é.
    """
    perto_a = Hit(chunk_id=11, document_id=3, exclusion_id=42, coverage_id=None,
                  text="Exclusão geral da apólice: danos por infiltração.", distance=0.21)
    perto_b = Hit(chunk_id=12, document_id=5, exclusion_id=None, coverage_id=7,
                  text="Regra de franquia da cobertura Vendaval: 10%.", distance=0.44)
    longe = Hit(chunk_id=99, document_id=9, exclusion_id=1, coverage_id=None,
                text="Exclusão geral da apólice: danos por guerra.", distance=0.90)
    assert longe.distance > graph_mod.MAX_DISTANCE_PADRAO   # o corte é o que exclui

    f = fake_graph(["rag_worker"], hits=[perto_a, perto_b, longe])

    answer = await _perguntar(ask_client, "Infiltração por janela aberta é coberta?")

    assert answer == f"{FRASE}\n\nBase: 2 cláusula(s) de 2 apólice(s)."
    # E o descartado também não chegou ao synthesizer — as duas metades do mesmo corte.
    assert "chunk 99" not in f.client.messages.synth_prompts[0]


async def test_rota_rag_nao_declara_a_base_do_corpus(ask_client, fake_graph):
    """A rota RAG declara CLÁUSULAS, não o corpus inteiro.

    O rodapé do corpus embaixo de uma resposta escrita em cima de duas cláusulas diria
    que as 2 apólices foram lidas quando 2 cláusulas foram — a resposta pareceria muito
    mais apoiada do que é, que é o oposto do que o rodapé existe pra fazer.
    """
    hits = [
        Hit(chunk_id=11, document_id=3, exclusion_id=42, coverage_id=None,
            text="Exclusão geral da apólice: danos por infiltração.", distance=0.21),
        Hit(chunk_id=12, document_id=3, exclusion_id=None, coverage_id=7,
            text="Regra de franquia da cobertura Vendaval: 10%.", distance=0.44),
    ]
    fake_graph(["rag_worker"], hits=hits)

    answer = await _perguntar(ask_client, "Infiltração por janela aberta é coberta?")

    assert answer == f"{FRASE}\n\nBase: 2 cláusula(s) de 1 apólice(s)."
    assert SUFIXO_CORPUS not in answer
    assert "corpus" not in answer


# --- As quatro frases estáticas: só uma declara base ---
async def test_nada_relevante_declara_a_base_PESQUISADA_e_nao_a_do_corpus(
    ask_client, fake_graph
):
    """"Não achei" declara o que a BUSCA alcança, que é menos do que o corpus.

    Duas propriedades num teste só, e as duas são load-bearing.

    A primeira: o rodapé existe. Sem ele, "não encontrei nenhuma cláusula sobre isso" lê
    como "isso não existe"; com ele, lê como "não achei nas que eu tenho".

    A segunda, e é o bug que este teste guarda: o número **não** pode ser o do corpus. A
    busca só alcança chunk com vetor do modelo atual, então o documento que a fixture
    deixou sem embedding — o estado normal entre a extração e `embed_chunks.py` — está na
    base e fora do alcance. Contar `policy_document` diria "procurei em 4" quando 2 foram
    procuradas, inflando justamente a frase cujo trabalho é dizer "não achei NAS QUE EU
    TENHO". A desigualdade explícita no fim é o que impede a regressão: com uma contagem
    só, os dois rodapés voltam a ser o mesmo e nada mais reclama.
    """
    f = fake_graph(["rag_worker"], hits=[])

    answer = await _perguntar(ask_client, "Meteorito é coberto?")

    assert answer == f"{graph_mod.NADA_RELEVANTE}\n\n{SUFIXO_PESQUISADO}"
    assert f.client.messages.synth_calls == 0   # continua sem pagar pra dizer "não achei"

    assert SUFIXO_PESQUISADO != SUFIXO_CORPUS
    assert SUFIXO_CORPUS not in answer


async def test_fora_de_escopo_nao_declara_base(ask_client, fake_graph):
    """Nada foi consultado: o supervisor classificou antes de gastar qualquer coisa.

    Um "Base: corpus de 4 apólice(s)" aqui afirmaria uma verificação que não houve — e
    a frase nem é sobre o corpus, é sobre o escopo do sistema.
    """
    fake_graph(["unsupported"])

    answer = await _perguntar(ask_client, "Seguro de vida cobre suicídio?")

    assert answer == graph_mod.FORA_DE_ESCOPO
    assert "Base:" not in answer


async def test_falha_interna_nao_declara_base(ask_client, fake_graph):
    """A consulta não chegou a rodar — declarar base seria afirmar o contrário.

    Pior: o rodapé daria ar de completude à única frase que significa "não sei o que
    aconteceu do meu lado".
    """
    fake_graph(["rag_worker"], busca_explode="pgvector fora do ar")

    answer = await _perguntar(ask_client, "Vendaval tem franquia?")

    assert answer == graph_mod.FALHA_INTERNA
    assert "Base:" not in answer


async def test_no_answer_nao_declara_base(ask_client, fake_graph):
    """Ninguém trabalhou (falha de roteamento): mesma razão das outras duas."""
    fake_graph(["END"])

    answer = await _perguntar(ask_client, "Qual a capital da França?")

    assert answer == graph_mod.NO_ANSWER
    assert "Base:" not in answer


# --- A guarda contra reintroduzir a instrução no prompt ---
def test_o_prompt_do_synthesizer_nao_menciona_base():
    """A declaração é responsabilidade do CÓDIGO, e o prompt não pode disputá-la.

    Pedir pro modelo declarar a base é pedir pra ele afirmar quantos documentos foram
    consultados — exatamente o tipo de número que ele inventa com confiança quando não
    está no payload, e reescreve quando está. Uma declaração de cobertura errada é pior
    do que nenhuma: ela dá ao usuário uma razão falsa pra confiar na resposta.
    """
    assert "base" not in graph_mod.SYNTHESIZER_SYSTEM.lower()


async def test_o_rodape_nao_passa_pelo_modelo(ask_client, fake_graph):
    """O outro lado da mesma guarda: o rodapé é colado DEPOIS da chamada.

    O `system` que o synthesizer manda não fala de base, e o rodapé que aparece na
    resposta não estava em nada que o modelo viu — o fake devolve só FRASE.
    """
    f = fake_graph(["sql_worker"])

    answer = await _perguntar(ask_client, "Quantos perigos existem na base?")

    assert SUFIXO_CORPUS in answer
    assert "base" not in f.client.messages.synth_systems[0].lower()
    assert SUFIXO_CORPUS not in f.client.messages.synth_prompts[0]
