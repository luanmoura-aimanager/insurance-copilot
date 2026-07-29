"""
Custo por chamada DENTRO do grafo — ZERO chamada real à API Anthropic.

O grafo roda inteiro (supervisor -> sql_worker -> supervisor -> END), com o client
Anthropic e o acesso ao Postgres trocados por fakes; o que se verifica aqui é a
CONTABILIDADE: uma linha de cost_event por chamada de LLM, cada uma carimbada com o
request_id do /ask e com o cliente autenticado.

Diferente dos outros testes de banco: `record_call_cost` abre a PRÓPRIA transação e
commita (o custo não pode ficar pendurado na transação do request), então o padrão de
rollback do `db_session` não alcança essas linhas — a fixture `cost_rows` limpa a
tabela na mão, antes e depois.
"""
from decimal import Decimal

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents import graph as graph_mod
from app.cost import load_pricing
from app.models import CostEvent

TOKEN = "token-de-teste"
CLIENTE = "teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# O id que a API DEVOLVE (resp.model) — versionado. Mandamos o alias "claude-haiku-4-5";
# quem é cobrado, e quem tem preço no pricing.json, é a versão resolvida.
MODEL_COBRADO = "claude-haiku-4-5-20251001"


# --- Fakes do client Anthropic: além do tool_use, agora .model e .usage ---
class _FakeToolUse:
    type = "tool_use"

    def __init__(self, payload: dict):
        self.input = payload


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, payload: dict):
        self.content = [_FakeToolUse(payload)]
        self.model = MODEL_COBRADO      # id versionado, como vem da API
        self.usage = _FakeUsage(1_000, 200)


class _FakeMessages:
    def __init__(self, decisions: list[str], sql: str):
        self._decisions = decisions
        self._sql = sql
        self.supervisor_calls = 0

    async def create(self, **kwargs):
        tool_name = kwargs["tools"][0]["name"]
        if tool_name == graph_mod._DECISION_TOOL:
            i = self.supervisor_calls
            self.supervisor_calls += 1
            nxt = self._decisions[min(i, len(self._decisions) - 1)]
            return _FakeResponse({"next": nxt, "reasoning": f"fake decision #{i}"})
        return _FakeResponse({"sql": self._sql})


class _FakeClient:
    def __init__(self, decisions: list[str], sql: str):
        self.messages = _FakeMessages(decisions, sql)


@pytest.fixture
def fake_graph(monkeypatch):
    def _install(decisions: list[str], sql: str = "SELECT count(*) FROM peril"):
        client = _FakeClient(decisions, sql)
        monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)
        monkeypatch.setattr(graph_mod, "get_schema", lambda: "TABLE peril(id int, nome text)")
        monkeypatch.setattr(graph_mod, "run_query", lambda sql: "[(7,)]")
        return client

    return _install


@pytest_asyncio.fixture
async def cost_rows(engine, monkeypatch):
    """Aponta a session de `record_call_cost` pro container e limpa cost_event.

    `record_call_cost` resolve `SessionLocal` de `app.db` na hora da chamada (import
    lá dentro), então trocar o atributo do módulo basta. A limpeza é obrigatória
    porque essas linhas são COMMITADAS: sem ela vazariam para o `test_cost.py`, que
    conta a tabela inteira.
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


@pytest_asyncio.fixture
async def ask_client(monkeypatch):
    monkeypatch.setenv("API_TOKENS", f"{CLIENTE}:{TOKEN}")
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _eventos(Session) -> list[CostEvent]:
    async with Session() as s:
        return list((await s.execute(select(CostEvent).order_by(CostEvent.id))).scalars())


def test_modelo_cobrado_tem_preco_cadastrado():
    """Guarda-corpo: o id VERSIONADO (o que `resp.model` devolve) precisa ter preço.

    O grafo manda o alias, mas grava `resp.model` — se só o alias estivesse no
    pricing.json, todo custo do grafo cairia no fail-loud do cost_usd e, por ser best
    effort, sumiria num log. Aqui isso falha alto, onde dá pra ver.
    """
    assert MODEL_COBRADO in load_pricing()


async def test_ask_grava_uma_linha_de_custo_por_chamada(ask_client, fake_graph, cost_rows):
    """supervisor -> sql_worker -> supervisor -> END = 3 chamadas pagas = 3 linhas.

    São 3, não 2: o supervisor roda de novo DEPOIS do worker (é ele quem decide o END),
    e o grão do cost_event é a CHAMADA, não o agente. Cada linha sai carimbada com o
    mesmo request_id — é o que permite somar "quanto custou este /ask".
    """
    fake_graph(["sql_worker", "END"])

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )
    assert r.status_code == 200

    eventos = await _eventos(cost_rows)
    assert [e.agent_name for e in eventos] == ["supervisor", "sql_worker", "supervisor"]

    request_ids = {e.request_id for e in eventos}
    assert len(request_ids) == 1              # um id por /ask, compartilhado pelas chamadas
    assert request_ids.pop()                  # preenchido, não None
    assert {e.client for e in eventos} == {CLIENTE}
    assert all(e.model == MODEL_COBRADO for e in eventos)   # o id versionado, não o alias
    assert all(e.cost_usd > Decimal("0") for e in eventos)
    assert all(e.batch is False for e in eventos)           # /ask é online, preço cheio


async def test_cada_ask_tem_seu_proprio_request_id(ask_client, fake_graph, cost_rows):
    """O ContextVar é resetado no fim do request: dois /ask não compartilham id.

    Se o reset falhasse (ou o var fosse global), o segundo request herdaria o id do
    primeiro e o custo de dois pedidos viraria um só na hora de somar.
    """
    fake_graph(["END"])  # encerra de cara: 1 chamada por request, basta pro id

    for _ in range(2):
        r = await ask_client.post("/ask", json={"question": "Oi?"}, headers=AUTH)
        assert r.status_code == 200

    eventos = await _eventos(cost_rows)
    assert len(eventos) == 2
    assert eventos[0].request_id != eventos[1].request_id


async def test_falha_ao_gravar_custo_nao_derruba_o_ask(
    ask_client, fake_graph, cost_rows, monkeypatch
):
    """Best effort: a chamada ao LLM JÁ FOI PAGA e já tem resposta.

    Perder a linha de custo é muito mais barato do que devolver 500 num /ask que
    custou dinheiro — então uma falha na gravação vira log e o request segue.
    """
    fake_graph(["sql_worker", "END"])

    async def _explode(**kwargs):
        raise RuntimeError("banco de custo fora do ar")

    monkeypatch.setattr(graph_mod, "record_call_cost", _explode)

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )

    assert r.status_code == 200
    assert "Result:" in r.json()["answer"]     # a resposta do worker chegou inteira
    assert await _eventos(cost_rows) == []     # e nenhuma linha foi gravada
