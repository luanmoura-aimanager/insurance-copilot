"""
Testes do POST /ask — ZERO chamada real à API Anthropic.

Todo o grafo é exercitado de verdade (supervisor -> route -> sql_worker -> supervisor),
só que com o client Anthropic e o acesso ao Postgres trocados por fakes: o que importa
aqui é o roteamento e o circuit breaker, não o modelo nem o banco.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from app.agents import graph as graph_mod


# --- Fakes do client Anthropic (mesma superfície: .messages.create -> .content[]) ---
class _FakeToolUse:
    """Bloco tool_use canned: o grafo só lê .type e .input."""

    type = "tool_use"

    def __init__(self, payload: dict):
        self.input = payload


class _FakeResponse:
    def __init__(self, payload: dict):
        self.content = [_FakeToolUse(payload)]


class _FakeMessages:
    def __init__(self, decisions: list[str], sql: str):
        self._decisions = decisions   # uma decisão de rota por chamada do supervisor
        self._sql = sql
        self.supervisor_calls = 0

    async def create(self, **kwargs):
        # Despacha pelo nome da tool forçada: o mesmo client atende supervisor e worker.
        tool_name = kwargs["tools"][0]["name"]
        if tool_name == graph_mod._DECISION_TOOL:
            i = self.supervisor_calls
            self.supervisor_calls += 1
            # Esgotou o roteiro? repete a última decisão (é o que segura o teste do
            # circuit breaker: supervisor que NUNCA escolhe END).
            nxt = self._decisions[min(i, len(self._decisions) - 1)]
            return _FakeResponse({"next": nxt, "reasoning": f"fake decision #{i}"})
        return _FakeResponse({"sql": self._sql})


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
        return client

    return _install


@pytest.fixture
async def ask_client():
    """Client HTTP puro: /ask não toca no Postgres, então não usa a fixture de DB."""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_ask_happy_path(ask_client, fake_graph):
    """supervisor -> sql_worker -> supervisor -> END: a resposta vem do worker."""
    fake_graph(["sql_worker", "END"])

    r = await ask_client.post("/ask", json={"question": "Quantos perigos existem?"})

    assert r.status_code == 200
    body = r.json()
    assert "Result:" in body["answer"]      # veio do sql_worker, não do reasoning do supervisor
    assert body["iterations"] == 2          # duas passadas pelo supervisor


async def test_ask_end_immediately_returns_fallback(ask_client, fake_graph):
    """Supervisor encerra de cara: nenhum worker rodou, então não inventamos resposta."""
    fake_graph(["END"])

    r = await ask_client.post("/ask", json={"question": "Qual a capital da França?"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Não consegui responder essa pergunta com os dados disponíveis."
    assert body["iterations"] == 1


async def test_ask_circuit_breaker_stops_the_loop(ask_client, fake_graph):
    """O teste que mais importa: supervisor que SEMPRE roteia pro worker não roda para sempre.

    O guard de MAX_ITERATIONS em route() é mecânico (não pergunta pro LLM), então um
    modelo teimoso — ou em loop — para de queimar chamadas pagas no limite.
    """
    client = fake_graph(["sql_worker"])  # nunca escolhe END

    r = await ask_client.post("/ask", json={"question": "Pergunta que gera loop"})

    assert r.status_code == 200
    assert r.json()["iterations"] == graph_mod.MAX_ITERATIONS
    assert client.messages.supervisor_calls == graph_mod.MAX_ITERATIONS


async def test_ask_rejects_empty_question(ask_client):
    """Validação do Pydantic barra antes de qualquer chamada de LLM."""
    r = await ask_client.post("/ask", json={"question": ""})
    assert r.status_code == 422
