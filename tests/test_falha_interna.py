"""Falha de worker: o USUÁRIO recebe uma frase genérica, o OPERADOR recebe o traceback.

Este módulo cobre a fatia que fechou o vazamento: antes, os dois workers devolviam
`f"SQL error: {exc}"` / `f"RAG error: {exc}"` como se fosse resultado, e esse texto seguia
pro prompt do synthesizer — que podia parafraseá-lo — e, no caminho de degradação dele
(`frase or resultado`), virava LITERALMENTE a resposta do `/ask`. O que vazava não era
ruído inofensivo: um `OperationalError` do Postgres carrega host, porta, usuário e banco.

Por isso a asserção load-bearing de cada teste não é `answer == FALHA_INTERNA` — é a
AUSÊNCIA do texto da exceção no corpo da resposta, conferida contra uma string que só
existe dentro da exceção plantada. Sem ela, uma implementação que devolvesse
`f"{FALHA_INTERNA} ({exc})"` passaria pela metade dos asserts.

Roda sem container: nenhum nó chega ao banco (o `get_schema` é falso no caminho SQL, e no
caminho RAG a busca falsa estoura antes de a session ser usada). Nenhum teste gasta.
"""
import logging
import re
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import graph as graph_mod

TOKEN = "token-de-teste"
CLIENTE = "teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

FRASE = "Existem 7 perigos cadastrados na base."

# As exceções plantadas carregam o tipo de detalhe que de fato aparece nesses erros: host
# interno, porta, usuário de banco, chave de API. É esta string que NÃO pode aparecer na
# resposta — e que TEM que aparecer no log.
SEGREDO_SQL = (
    'connection to server at "db.interna.railway" (10.0.0.7), port 5432 failed: '
    'password authentication failed for user "insurance_ro"'
)
SEGREDO_RAG = "Voyage recusou a chave sk-voyage-abc123 do host worker-3"


# --- Fakes do client Anthropic (mesma superfície dos outros módulos) ---
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
    def __init__(self, decisions: list[str], sql_explode: str | None):
        self._decisions = decisions
        self._sql_explode = sql_explode
        self.supervisor_calls = 0
        self.synth_calls = 0
        self.calls = 0                       # TODA chamada de LLM, de qualquer nó
        self.synth_prompts: list[str] = []   # o que o synthesizer recebeu pra sintetizar

    async def create(self, **kwargs):
        self.calls += 1
        # Sem tools = synthesizer: é o único nó que pede texto livre.
        if not kwargs.get("tools"):
            self.synth_calls += 1
            self.synth_prompts.append(kwargs["messages"][0]["content"])
            return _FakeResponse([_FakeText(FRASE)])

        if kwargs["tools"][0]["name"] == graph_mod._DECISION_TOOL:
            i = self.supervisor_calls
            self.supervisor_calls += 1
            nxt = self._decisions[min(i, len(self._decisions) - 1)]
            return _FakeResponse([_FakeToolUse({"next": nxt, "reasoning": f"fake #{i}"})])

        if self._sql_explode is not None:
            raise RuntimeError(self._sql_explode)
        return _FakeResponse([_FakeToolUse({"sql": "SELECT count(*) FROM peril"})])


class _FakeClient:
    def __init__(self, decisions: list[str], sql_explode: str | None = None):
        self.messages = _FakeMessages(decisions, sql_explode)


class _FakeSession:
    """`async with SessionLocal() as s` sem banco nenhum atrás."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _sem_base(monkeypatch):
    """Desliga os rodapés CONTADOS ("corpus de N apólice(s)" / "N apólice(s) pesquisada(s)").

    Mesma forma (e mesmo motivo) do no-op de `record_call_cost`: este módulo roda sem
    container, e o assunto dele é o que a resposta NÃO pode conter. `_contando` é o ponto
    único por onde as duas contagens passam. Sem o no-op o rodapé sumiria por acidente — a
    contagem falharia no `_FakeSession` — e voltaria a aparecer no dia em que a suíte
    rodasse depois de um módulo que sobe o banco. Quem testa a declaração de base é
    tests/test_base_declarada.py.
    """

    async def _nada(rotulo, conta):
        return None

    monkeypatch.setattr(graph_mod, "_contando", _nada)


@pytest.fixture
def fake_graph(monkeypatch):
    """Instala os fakes e escolhe ONDE plantar a exceção."""

    def _install(
        decisions: list[str],
        *,
        schema_explode: str | None = None,
        sql_explode: str | None = None,
        busca_explode: str | None = None,
    ):
        client = _FakeClient(decisions, sql_explode)
        monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)
        monkeypatch.setattr("app.db.SessionLocal", lambda **kw: _FakeSession())

        def _schema():
            if schema_explode is not None:
                raise RuntimeError(schema_explode)
            return "TABLE peril(id int, nome text)"

        monkeypatch.setattr(graph_mod, "get_schema", _schema)
        monkeypatch.setattr(graph_mod, "run_query", lambda sql: "Columns: count\nRows:\n7\n")

        buscas: list[str] = []

        async def _busca(session, question, *, k=5, max_distance=None, **kw):
            buscas.append(question)
            if busca_explode is not None:
                raise RuntimeError(busca_explode)
            return []

        monkeypatch.setattr(graph_mod, "search_clauses", _busca)

        async def _sem_custo(**kwargs):
            return None

        monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)
        _sem_base(monkeypatch)
        return SimpleNamespace(client=client, buscas=buscas)

    return _install


@pytest.fixture
async def ask_client(monkeypatch):
    monkeypatch.setenv("API_TOKENS", f"{CLIENTE}:{TOKEN}")
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _perguntar(ask_client, pergunta: str):
    return await ask_client.post("/ask", json={"question": pergunta}, headers=AUTH)


def _confere_falha(resposta, segredo: str, client) -> None:
    """As quatro coisas que valem em toda falha de worker, num lugar só."""
    # 1. 200, não 500: a superfície é conversa, e um 5xx deixa o usuário sem resposta e
    #    sem saber se vale tentar de novo (ver _final_answer em app/main.py).
    assert resposta.status_code == 200

    corpo = resposta.text          # o corpo INTEIRO, não só `answer`
    assert resposta.json()["answer"] == graph_mod.FALHA_INTERNA

    # 2. Nada da exceção sai daqui. Confere o texto todo e, separadamente, pedaços que
    #    sozinhos já seriam vazamento — uma implementação que truncasse a mensagem da
    #    exceção continuaria vazando host e usuário.
    assert segredo not in corpo
    for pedaco in segredo.split():
        if len(pedaco) > 8:        # ignora preposições e ruído curto
            assert pedaco not in corpo, pedaco

    # 3. A frase é DISTINGUÍVEL das outras três: "quebrou do nosso lado" não é "não achei
    #    no corpus", nem "não trato disso", nem "falhei em rotear".
    for outra in (graph_mod.NO_ANSWER, graph_mod.NADA_RELEVANTE, graph_mod.FORA_DE_ESCOPO):
        assert resposta.json()["answer"] != outra

    # 4. Não se paga pra reescrever uma frase que já está pronta — e mandar o texto de uma
    #    exceção pro synthesizer seria justamente o caminho pelo qual ele voltaria
    #    parafraseado pro usuário.
    assert client.messages.synth_calls == 0


async def test_falha_no_sql_worker_nao_vaza_a_excecao(ask_client, fake_graph):
    """Banco fora do ar no `get_schema`: FALHA_INTERNA, e o erro de conexão fica no log."""
    f = fake_graph(["sql_worker"], schema_explode=SEGREDO_SQL)

    r = await _perguntar(ask_client, "Quantos perigos existem na base?")

    _confere_falha(r, SEGREDO_SQL, f.client)
    assert f.client.messages.calls == 1   # só o supervisor; nem o SQL nem o synthesizer


async def test_falha_no_llm_do_sql_worker_tambem_e_contida(ask_client, fake_graph):
    """O nó INTEIRO está protegido, não só o `get_schema`.

    Antes desta fatia só a chamada de schema tinha `try`: uma falha da Anthropic (ou uma
    resposta sem bloco `tool_use`, ou um payload que não valida) subia do nó, abortava o
    grafo e devolvia 500 — num request que já tinha pago o supervisor. Com a exceção
    plantada na chamada do worker, este teste fica vermelho se o `try` voltar a cobrir só
    o primeiro degrau.
    """
    f = fake_graph(["sql_worker"], sql_explode=SEGREDO_SQL)

    r = await _perguntar(ask_client, "Quantos perigos existem na base?")

    _confere_falha(r, SEGREDO_SQL, f.client)
    assert f.client.messages.calls == 2   # supervisor + a chamada do worker que estourou


async def test_falha_na_busca_do_rag_worker_nao_vaza_a_excecao(ask_client, fake_graph):
    """Mesmo contrato no outro worker — e aqui o que vazava era pior.

    Uma `AuthenticationError` da Voyage fala de chave de API, e um erro de conexão do
    asyncpg fala do host do Postgres. Nenhum dos dois é assunto do usuário.
    """
    f = fake_graph(["rag_worker"], busca_explode=SEGREDO_RAG)

    r = await _perguntar(ask_client, "Infiltração por janela aberta é coberta?")

    _confere_falha(r, SEGREDO_RAG, f.client)
    assert len(f.buscas) == 1             # a busca foi tentada (é o que distingue do resto)
    assert f.client.messages.calls == 1   # só o supervisor


@pytest.mark.parametrize(
    "onde,segredo,worker",
    [
        ({"schema_explode": SEGREDO_SQL}, SEGREDO_SQL, "sql_worker"),
        ({"busca_explode": SEGREDO_RAG}, SEGREDO_RAG, "rag_worker"),
    ],
)
async def test_o_log_leva_traceback_request_id_e_client(
    ask_client, fake_graph, caplog, onde, segredo, worker
):
    """O log é o ÚNICO lugar onde o incidente é descrito — então ele carrega tudo.

    `request_id` e `client` são as mesmas chaves de `cost_event`: é o que liga o traceback
    às linhas de custo, à pergunta e a quem perguntou daquele `/ask`. Sem elas, com dois
    requests concorrentes, sobra um traceback solto — e como o usuário recebeu uma frase
    genérica, ninguém consegue reconstruir qual resposta foi essa.
    """
    decisao = ["sql_worker" if worker == "sql_worker" else "rag_worker"]
    fake_graph(decisao, **onde)

    with caplog.at_level(logging.ERROR, logger="app.agents.graph"):
        r = await _perguntar(ask_client, "Vendaval tem franquia?")

    assert r.json()["answer"] == graph_mod.FALHA_INTERNA

    erros = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert len(erros) == 1, [rec.message for rec in caplog.records]
    registro = erros[0]

    assert worker in registro.message
    assert re.search(r"request_id=[0-9a-f]{8}-[0-9a-f]{4}-", registro.message), registro.message
    assert f"client={CLIENTE}" in registro.message

    # `logger.exception`, não `logger.error`: sem o exc_info o operador sabe QUE caiu e não
    # POR QUE, que é o inverso do que esta fatia trocou de lugar.
    assert registro.exc_info is not None
    # E o detalhe que saiu da resposta está aqui, formatado junto do traceback.
    assert segredo in caplog.text


# --- As outras três frases: nada mudou pra elas ---
async def test_busca_vazia_continua_dizendo_nada_relevante(ask_client, fake_graph):
    """Busca que RODA e não acha nada não é falha: continua sendo fato sobre o corpus."""
    f = fake_graph(["rag_worker"])   # a busca falsa devolve [] sem estourar

    r = await _perguntar(ask_client, "Meteorito é coberto?")

    assert r.status_code == 200
    assert r.json()["answer"] == graph_mod.NADA_RELEVANTE
    assert r.json()["answer"] != graph_mod.FALHA_INTERNA
    assert f.client.messages.synth_calls == 0


async def test_fora_de_escopo_continua_fora_de_escopo(ask_client, fake_graph):
    f = fake_graph(["unsupported"])

    r = await _perguntar(ask_client, "Seguro de vida cobre suicídio?")

    assert r.status_code == 200
    assert r.json()["answer"] == graph_mod.FORA_DE_ESCOPO
    assert r.json()["answer"] != graph_mod.FALHA_INTERNA
    assert f.client.messages.calls == 1


async def test_supervisor_sem_worker_continua_no_answer(ask_client, fake_graph):
    """`END` (fail-safe do enum): ninguém trabalhou e nada quebrou — falha de roteamento.

    É a fronteira que justifica a quarta frase existir: aqui NÃO há traceback nenhum no
    log, então dizer "tive uma falha interna" mandaria o operador procurar um incidente
    que não aconteceu.
    """
    f = fake_graph(["END"])

    r = await _perguntar(ask_client, "Qual a capital da França?")

    assert r.status_code == 200
    assert r.json()["answer"] == graph_mod.NO_ANSWER
    assert r.json()["answer"] != graph_mod.FALHA_INTERNA
    assert f.client.messages.synth_calls == 0


async def test_caminho_feliz_nao_virou_falha(ask_client, fake_graph):
    """Cinto do cinto: com nada estourando, a resposta continua sendo a frase sintetizada.

    Sem este teste, um `except` largo demais (ou um `falhou` sempre verdadeiro) devolveria
    FALHA_INTERNA pra tudo e os testes de falha ficariam todos verdes.
    """
    f = fake_graph(["sql_worker"])

    r = await _perguntar(ask_client, "Quantos perigos existem na base?")

    assert r.status_code == 200
    assert r.json()["answer"] == FRASE
    assert f.client.messages.synth_calls == 1


# --- A ordem das frases quando elas competem (estado que o multi-hop reintroduz) ---
def _synthesizer_state(next_: str, mensagens: list) -> dict:
    return {"iterations": 2, "next": next_, "messages": mensagens}


@pytest.fixture
def synth_sem_custo(monkeypatch):
    """Synthesizer chamado direto, com LLM falso e custo no-op."""
    client = _FakeClient([])
    monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)

    async def _sem_custo(**kwargs):
        return None

    monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)
    _sem_base(monkeypatch)
    return client


async def test_falha_vence_fora_de_escopo(synth_sem_custo):
    """Worker caiu E supervisor disse `unsupported` -> FALHA_INTERNA.

    Chamado direto porque via `/ask` o caso é inalcançável hoje (single-hop: o supervisor
    classifica antes de qualquer worker), e é exatamente o estado que o multi-hop
    reintroduz. FORA_DE_ESCOPO aqui seria a pior saída possível: manda o usuário embora,
    afirmando algo sobre o ESCOPO da pergunta, por causa de um banco fora do ar.
    """
    state = _synthesizer_state(
        "unsupported",
        [
            HumanMessage(content="Vendaval tem franquia?"),
            AIMessage(content="rag_worker: falha interna.", name=graph_mod.WORKER_ERROR),
        ],
    )

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == graph_mod.FALHA_INTERNA
    assert synth_sem_custo.messages.synth_calls == 0


async def test_falha_vence_nada_relevante(synth_sem_custo):
    """Um worker caiu e a busca do outro voltou vazia -> FALHA_INTERNA.

    "Não encontrei no corpus" é uma afirmação sobre o CORPUS, e ela não foi verificada:
    metade da consulta nem rodou.
    """
    state = _synthesizer_state(
        "END",
        [
            HumanMessage(content="Vendaval tem franquia?"),
            AIMessage(content="sql_worker: falha interna.", name=graph_mod.WORKER_ERROR),
            AIMessage(content=graph_mod.NADA_RELEVANTE, name="rag_worker"),
        ],
    )

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == graph_mod.FALHA_INTERNA
    assert synth_sem_custo.messages.synth_calls == 0


async def test_falha_de_um_worker_nao_apaga_o_resultado_do_outro(synth_sem_custo):
    """Mesma regra das outras frases estáticas: nenhuma vale se há dado real a apresentar.

    Um `rag_worker` que caiu não pode fazer o `/ask` esquecer a contagem que o
    `sql_worker` já produziu (e pela qual já se pagou).
    """
    state = _synthesizer_state(
        "END",
        [
            HumanMessage(content="Quantas seguradoras cobrem vendaval?"),
            AIMessage(content="SQL: SELECT count(*)\nResult: [(7,)]", name="sql_worker"),
            AIMessage(content="rag_worker: falha interna.", name=graph_mod.WORKER_ERROR),
        ],
    )

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == FRASE
    assert out["messages"][0].content != graph_mod.FALHA_INTERNA
    assert synth_sem_custo.messages.synth_calls == 1


async def test_rag_worker_contem_falha_depois_da_busca(monkeypatch):
    """O `try` do `rag_worker` cobre o CORPO INTEIRO, não só a chamada de busca.

    A busca é o degrau mais provável, não o único: o que vem depois dela também mexe em
    dado que veio do banco. Um `Hit` com `distance=None` (nada no dataclass `frozen`
    proíbe) estoura no filtro do limiar e no `f"{...:.3f}"` do `_formatar_hits` — depois do
    `except`, se ele cobrisse só a busca. Aí a exceção escapa do nó, aborta o grafo e
    devolve o 500 que esta fatia existe pra eliminar. Era a mesma assimetria que o
    `sql_worker` corrigiu ao envolver o corpo todo, e o code review pegou que o irmão
    tinha ficado pela metade.
    """
    from app.rag.search import Hit

    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: _FakeSession())

    async def _busca(session, question, **kw):
        return [
            Hit(
                chunk_id=1, document_id=1, exclusion_id=1, coverage_id=None,
                text="cláusula", distance=None,   # a busca devolve, o resto quebra
            )
        ]

    monkeypatch.setattr(graph_mod, "search_clauses", _busca)

    out = await graph_mod.rag_worker(
        {"iterations": 1, "next": "rag_worker", "messages": [HumanMessage(content="oi?")]}
    )

    assert out["messages"][0].name == graph_mod.WORKER_ERROR


# --- A última porta: o texto de erro do `run_query` ---
async def test_falha_de_conexao_no_run_query_nao_vaza(ask_client, fake_graph, monkeypatch):
    """`run_query` real, `_connect` quebrado: era o caminho que sobrava pro vazamento.

    O `get_schema` propaga a `OperationalError` (nunca teve `except`), mas o `run_query`
    a devolvia como TEXTO, indistinguível de um erro de SQL — e aí ela seguia como
    *resultado* do worker: entrava no prompt do synthesizer e, no fallback dele (`frase or
    resultado`), virava a resposta do `/ask` ao pé da letra. É o mesmo `OperationalError`
    nos dois casos, e agora tem o mesmo destino.

    O cenário é concreto e nada exótico: o banco reinicia (ou a senha da role RO é
    rotacionada) DEPOIS do `get_schema` e ANTES do `run_query`.
    """
    f = fake_graph(["sql_worker"])
    # Devolve o run_query de verdade pro grafo (o fixture o troca por um dublê) e quebra a
    # conexão por baixo dele.
    from mcp_servers import postgres_mcp_server as mcp_mod

    monkeypatch.setattr(graph_mod, "run_query", mcp_mod.run_query)

    def _connect_quebrado():
        raise mcp_mod.psycopg.OperationalError(SEGREDO_SQL)

    monkeypatch.setattr(mcp_mod, "_connect", _connect_quebrado)

    r = await _perguntar(ask_client, "Quantos perigos existem na base?")

    _confere_falha(r, SEGREDO_SQL, f.client)
    # O worker gastou a chamada que gera o SQL (supervisor + worker), e o synthesizer não:
    # um erro de conexão não se parafraseia.
    assert f.client.messages.calls == 2


@pytest.mark.parametrize(
    "erro_nome,mensagem",
    [
        ("QueryCanceled", "canceling statement due to statement timeout"),
        ("StatementTooComplex", "stack depth limit exceeded"),
    ],
)
def test_operational_error_da_query_nao_e_incidente(monkeypatch, erro_nome, mensagem):
    """`OperationalError` NÃO é sinônimo de infraestrutura, e esta é a exceção que importa.

    `QueryCanceled` (57014, `statement_timeout`) e a classe 54 (`program_limit_exceeded`:
    `StatementTooComplex`, `TooManyColumns`, `TooManyArguments`) são `OperationalError`
    **causados pela query** — um cartesian join escrito pelo modelo num Postgres gerenciado,
    que costuma vir com `statement_timeout` configurado, chega exatamente aqui. Se subissem
    junto com as falhas de conexão, o operador seria paginado com traceback pra investigar
    infra por causa de SQL ruim, e o worker perderia a chance de reportar o que houve.
    Continuam voltando como texto — é o que `_ERROS_DA_QUERY` subtrai do `raise`.
    """
    from mcp_servers import postgres_mcp_server as mcp_mod

    erro = getattr(mcp_mod.psycopg.errors, erro_nome)(mensagem)

    class _Cursor:
        def execute(self, sql):
            raise erro

    class _Conn:
        closed = False

        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    monkeypatch.setattr(mcp_mod, "_connect", lambda: _Conn())

    saida = mcp_mod.run_query("SELECT * FROM coverage, peril, exclusion")

    assert saida.startswith("Error executing query:")
    assert mensagem in saida
    # E o texto é seguro de repassar: nenhuma dessas mensagens carrega conninfo.
    assert "password" not in saida


def test_erro_de_query_continua_voltando_como_texto(monkeypatch):
    """A outra metade da divisão, e ela é o que impede a correção de virar exagero.

    SQL inválido é RESULTADO: o LLM errou a query, e o worker tem que poder dizer isso ao
    synthesizer em vez de estourar — se tudo subisse, uma coluna com nome errado viraria
    "tive uma falha interna", mandando o operador procurar um incidente que não houve.
    A divisão é a do próprio psycopg: `ProgrammingError` (sintaxe, coluna/tabela
    inexistente, permissão) é do LLM; `OperationalError` é da infraestrutura.
    """
    from mcp_servers import postgres_mcp_server as mcp_mod

    class _Cursor:
        def execute(self, sql):
            raise mcp_mod.psycopg.errors.UndefinedColumn(
                'column "nome_errado" does not exist'
            )

    class _Conn:
        closed = False

        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    monkeypatch.setattr(mcp_mod, "_connect", lambda: _Conn())

    saida = mcp_mod.run_query("SELECT nome_errado FROM peril")

    assert saida.startswith("Error executing query:")
    assert "nome_errado" in saida


async def test_falha_de_um_turno_anterior_nao_contamina_este(synth_sem_custo):
    """A leitura do synthesizer é do TURNO (o que veio depois da última pergunta).

    Hoje o `/ask` monta um State novo por request, então o caso é inalcançável — e é
    justamente por isso que precisa de teste: quando o histórico atravessar turnos (a
    superfície de WhatsApp), uma falha do turno 1 faria o turno 2 responder FALHA_INTERNA
    sobre uma busca que rodou e simplesmente não achou nada, mandando o operador procurar
    um traceback de OUTRO request. Ninguém ligaria o sintoma a esta função.

    O estado abaixo é o discriminante: a falha antiga só muda a resposta quando ESTE turno
    não tem resultado substantivo, porque nos outros casos o `if not substantivos` já
    protege. (Foi verificado por mutação — o primeiro estado que escrevi passava sem a
    fatia do turno.)
    """
    state = {
        "iterations": 1,
        "next": "END",
        "messages": [
            HumanMessage(content="Infiltração é coberta?"),          # turno 1: quebrou
            AIMessage(content="rag_worker: falha interna.", name=graph_mod.WORKER_ERROR),
            HumanMessage(content="Meteorito é coberto?"),            # turno 2: rodou e não achou
            AIMessage(content=graph_mod.NADA_RELEVANTE, name="rag_worker"),
        ],
    }

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == graph_mod.NADA_RELEVANTE
    assert out["messages"][0].content != graph_mod.FALHA_INTERNA
    assert synth_sem_custo.messages.synth_calls == 0


async def test_resultado_de_um_turno_anterior_nao_e_reaproveitado(synth_sem_custo):
    """O outro lado da mesma fatia: `resultados` também é do turno.

    Um turno sem worker nenhum (supervisor classificou como fora de escopo) não pode
    sintetizar a resposta do turno anterior — seria devolver, com a confiança de quem
    acabou de consultar, um dado que ninguém pediu nesta pergunta.
    """
    state = {
        "iterations": 1,
        "next": "unsupported",
        "messages": [
            HumanMessage(content="Quantas seguradoras cobrem vendaval?"),   # turno 1
            AIMessage(content="SQL: SELECT count(*)\nResult: [(7,)]", name="sql_worker"),
            HumanMessage(content="Qual a capital da França?"),              # turno 2
        ],
    }

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == graph_mod.FORA_DE_ESCOPO
    assert synth_sem_custo.messages.synth_calls == 0


def test_a_marca_da_falha_e_o_name_e_nao_o_conteudo(caplog):
    """A falha é reconhecida por campo estruturado, e o conteúdo dela é ESTÉRIL.

    Duas propriedades, e a segunda precisa ser testada AQUI e não pelo `/ask`: hoje o
    synthesizer curto-circuita pela marca do `name` e nem lê o conteúdo, então uma versão
    que voltasse a interpolar a exceção na mensagem devolveria FALHA_INTERNA do mesmo
    jeito e passaria por todos os testes de ponta a ponta deste módulo. O conteúdo
    contaminado só cobraria a conta depois — no dia em que alguém logasse o histórico do
    grafo, o mandasse pra um trace, ou reintroduzisse o multi-hop (aí a mensagem volta a
    entrar num prompt). Esta é a única asserção que fixa a propriedade na fonte.

    `WORKER_ERROR` fora de `WORKERS` é a outra metade: se estivesse dentro, o texto da
    falha seria elegível como payload de síntese, e o synthesizer o mandaria pro modelo.
    """
    assert graph_mod.WORKER_ERROR not in graph_mod.WORKERS

    # De dentro de um `except` de verdade: é o único estado em que a função consegue ver a
    # exceção, e portanto o único em que ela conseguiria vazá-la.
    try:
        raise RuntimeError(SEGREDO_SQL)
    except RuntimeError:
        saida = graph_mod._falha_de_worker("sql_worker")

    mensagem = saida["messages"][0]
    assert mensagem.name == graph_mod.WORKER_ERROR
    assert "sql_worker" in mensagem.content          # diz QUAL nó caiu
    assert SEGREDO_SQL not in mensagem.content
    for pedaco in SEGREDO_SQL.split():
        if len(pedaco) > 8:
            assert pedaco not in mensagem.content, pedaco
    assert mensagem.content == mensagem.content.strip()
