"""
Testes do POST /ask — ZERO chamada real à API Anthropic.

Todo o grafo é exercitado de verdade (supervisor -> route -> sql_worker -> synthesizer),
só que com o client Anthropic e o acesso ao Postgres trocados por fakes: o que importa
aqui é o roteamento e a TERMINAÇÃO, não o modelo nem o banco. O grafo é single-hop — o
supervisor fala uma vez —, então o caso load-bearing é a contagem de chamadas; o circuit
breaker, hoje inalcançável por aqui, tem teste próprio que forja o State.

/ask é autenticado (ver tests/test_auth.py); aqui a auth é só um pré-requisito, então
a fixture cadastra um token e todas as requisições mandam o header.
"""
import logging
import re

import pytest
from httpx import ASGITransport, AsyncClient

from app.agents import graph as graph_mod

TOKEN = "token-de-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# Frase que o synthesizer (mockado) devolve — a resposta do /ask passa por ele agora.
FRASE_FINAL = "Existem 7 perigos cadastrados na base."


# --- Fakes do client Anthropic (mesma superfície: .messages.create -> .content[]) ---
class _FakeToolUse:
    """Bloco tool_use canned: o grafo só lê .type e .input."""

    type = "tool_use"

    def __init__(self, payload: dict):
        self.input = payload


class _FakeText:
    """Bloco de texto: é o que o synthesizer lê (ele não força tool)."""

    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, blocks: list):
        self.content = blocks


class _FakeMessages:
    def __init__(self, decisions: list[str], sql: str):
        self._decisions = decisions   # uma decisão de rota por chamada do supervisor
        self._sql = sql
        self.supervisor_calls = 0

    async def create(self, **kwargs):
        # Sem tools = synthesizer, o único nó que pede texto livre.
        if not kwargs.get("tools"):
            return _FakeResponse([_FakeText(FRASE_FINAL)])

        # Despacha pelo nome da tool forçada: o mesmo client atende supervisor e worker.
        tool_name = kwargs["tools"][0]["name"]
        if tool_name == graph_mod._DECISION_TOOL:
            i = self.supervisor_calls
            self.supervisor_calls += 1
            # Esgotou o roteiro? repete a última decisão. Com o grafo single-hop o
            # supervisor é chamado uma vez só, então a lista tem UM elemento na prática
            # — a repetição sobrou como rede pro caso de o ciclo voltar.
            nxt = self._decisions[min(i, len(self._decisions) - 1)]
            return _FakeResponse([_FakeToolUse({"next": nxt, "reasoning": f"fake decision #{i}"})])
        return _FakeResponse([_FakeToolUse({"sql": self._sql})])


class _FakeClient:
    def __init__(self, decisions: list[str], sql: str):
        self.messages = _FakeMessages(decisions, sql)


@pytest.fixture
def fake_graph(monkeypatch):
    """Instala os fakes em app.agents.graph e devolve um instalador parametrizável."""

    def _install(decisions: list[str], sql: str = "SELECT count(*) FROM peril"):
        client = _FakeClient(decisions, sql)
        # get_async_client é chamado a cada nó; devolver sempre a MESMA instância
        # preserva o contador de chamadas do supervisor entre as iterações.
        monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)
        monkeypatch.setattr(graph_mod, "get_schema", lambda: "TABLE peril(id int, nome text)")
        monkeypatch.setattr(graph_mod, "run_query", lambda sql: "[(7,)]")

        # Gravação de custo vira no-op: quem testa isso é tests/test_cost_graph.py.
        # Explícito de propósito — sem isto, o custo só não é gravado por ACIDENTE
        # (os fakes não têm .model/.usage, e o AttributeError cai no best-effort do
        # grafo). Completar os fakes um dia faria estes testes, que não têm fixture
        # de banco, tentarem escrever no Postgres de verdade.
        async def _sem_custo(**kwargs):
            return None

        monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)

        # O rodapé "Base: ..." também vira no-op explícito, e pelo mesmo motivo do custo:
        # contá-lo exige banco, e este módulo não tem nenhum. O patch é em `_contando`, o
        # ponto único por onde passam TODAS as contagens de base (corpus e pesquisável),
        # pra que uma terceira não reintroduza a dependência sem ninguém notar. Sem isso o
        # rodapé sumiria sozinho (a contagem falha, o sufixo some), mas só POR ACIDENTE —
        # e se a suíte rodar depois de um teste que subiu o container a contagem funciona,
        # e a resposta passa a depender de quantos documentos outro módulo deixou
        # commitados. Quem testa a base, com banco de verdade, é tests/test_base_declarada.py.
        async def _sem_base(rotulo, conta):
            return None

        monkeypatch.setattr(graph_mod, "_contando", _sem_base)
        return client

    return _install


@pytest.fixture
async def ask_client(monkeypatch):
    """Client HTTP puro: /ask não toca no Postgres, então não usa a fixture de DB."""
    monkeypatch.setenv("API_TOKENS", f"teste:{TOKEN}")
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_ask_happy_path(ask_client, fake_graph):
    """supervisor -> sql_worker -> synthesizer: a resposta vem em prosa, em UMA passada."""
    client = fake_graph(["sql_worker"])

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == FRASE_FINAL    # frase do synthesizer, não o resultado cru
    assert body["iterations"] == 1          # UMA passada pelo supervisor
    assert client.messages.supervisor_calls == 1


async def test_ask_end_e_fail_safe_e_nao_inventa_resposta(ask_client, fake_graph, caplog):
    """`END` saiu do prompt do supervisor, mas continua no enum — e tem que ser inócuo.

    O supervisor não encerra mais nada (os workers vão direto pro synthesizer) e o
    prompt manda explicitamente NUNCA escolher `END`, então ele só apareceria por um
    modelo desalinhado. Fica no `Literal` justamente pra esse caso: fora do enum a
    resposta quebraria a validação e derrubaria o request; dentro, cai no mesmo caminho
    do valor inválido — nenhum worker rodou, então NO_ANSWER, sem inventar resposta.

    E não pode acontecer em silêncio: de fora, esse NO_ANSWER é indistinguível de uma
    falha de infra, então o supervisor loga WARNING quando vê `END`.
    """
    fake_graph(["END"])

    with caplog.at_level(logging.WARNING, logger="app.agents.graph"):
        r = await ask_client.post(
            "/ask", json={"question": "Qual a capital da França?"}, headers=AUTH
        )

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Não consegui responder essa pergunta com os dados disponíveis."
    assert body["iterations"] == 1

    avisos = [rec.message for rec in caplog.records if "END" in rec.message]
    assert avisos
    # Com o request_id junto — é a mesma chave de `cost_event.request_id`, e sem ela o
    # aviso não se liga a request nenhum: com dois /ask concorrentes o operador vê um
    # WARNING solto e continua sem saber qual resposta foi essa.
    assert re.search(r"request_id=[0-9a-f]{8}-[0-9a-f]{4}-", avisos[0]), avisos[0]
    assert "client=teste" in avisos[0]


async def test_ask_nao_reroteia_nem_com_supervisor_teimoso(ask_client, fake_graph):
    """A terminação é ESTRUTURAL: um supervisor que só sabe dizer "sql_worker" para mesmo assim.

    Antes as arestas voltavam do worker pro supervisor, e este teste media o circuit
    breaker: MAX_ITERATIONS passadas, MAX_ITERATIONS chamadas pagas. Medido num /ask
    real, era o caso NORMAL e não o patológico — o supervisor reclassificava a mesma
    pergunta e reroteava pro mesmo worker, 10 chamadas onde bastavam 3. Com o worker
    indo direto pro synthesizer não há laço pra frear: o fake aqui é o mesmo supervisor
    teimoso de antes, e ele só consegue falar uma vez.
    """
    client = fake_graph(["sql_worker"])  # nunca escolheria END

    r = await ask_client.post(
        "/ask", json={"question": "Pergunta que antes gerava loop"}, headers=AUTH
    )

    assert r.status_code == 200
    assert r.json()["iterations"] == 1
    assert client.messages.supervisor_calls == 1


def test_circuit_breaker_continua_armado():
    """O guard de MAX_ITERATIONS fica, mesmo inalcançável pelo caminho normal.

    Hoje `route` roda uma vez por request, então `iterations` nunca chega perto do
    limite — por isso o estado é forjado à mão. Ele não é código morto: é o que segura
    um ciclo reintroduzido por engano (basta uma aresta de worker voltando pro
    supervisor), e guarda que nunca dispara continua sendo guarda.
    """
    from langgraph.graph import END

    estado = {"iterations": graph_mod.MAX_ITERATIONS, "next": "sql_worker", "messages": []}
    assert graph_mod.route(estado) == END        # nem com `next` válido ele deixa seguir
    estado["iterations"] = graph_mod.MAX_ITERATIONS - 1
    assert graph_mod.route(estado) == "sql_worker"


async def test_ask_rejects_empty_question(ask_client):
    """Validação do Pydantic barra antes de qualquer chamada de LLM."""
    r = await ask_client.post("/ask", json={"question": ""}, headers=AUTH)
    assert r.status_code == 422
