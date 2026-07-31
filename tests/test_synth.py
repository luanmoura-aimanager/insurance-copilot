"""
Nó synthesizer — ZERO chamada real à API Anthropic.

O que se verifica aqui é a ÚLTIMA milha: o /ask devolve uma frase em pt-BR, não o
resultado cru da query. O grafo roda inteiro com o client Anthropic e o acesso ao
Postgres trocados por fakes.

O fake do client despacha por `tools`: chamada COM tool forçada é supervisor/sql_worker,
chamada SEM tools é o synthesizer (texto livre). Só o teste de custo usa banco.
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents import graph as graph_mod
from app.models import CostEvent

TOKEN = "token-de-teste"
CLIENTE = "teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

FRASE = "Existem 7 perigos cadastrados na base."
MODEL_COBRADO = "claude-haiku-4-5-20251001"  # id versionado, como a API devolve


# --- Fakes do client Anthropic (superfície: .messages.create -> .content[]/.model/.usage) ---
class _FakeToolUse:
    type = "tool_use"

    def __init__(self, payload: dict):
        self.input = payload


class _FakeText:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, blocks: list):
        self.content = blocks
        self.model = MODEL_COBRADO
        self.usage = _FakeUsage(1_000, 200)


class _FakeMessages:
    def __init__(self, decisions: list[str], sql: str, synth_explode: bool):
        self._decisions = decisions
        self._sql = sql
        self._synth_explode = synth_explode
        self.supervisor_calls = 0
        self.synth_calls = 0

    async def create(self, **kwargs):
        # Sem tools = synthesizer: ele é o único nó que pede texto livre.
        if not kwargs.get("tools"):
            self.synth_calls += 1
            if self._synth_explode:
                raise RuntimeError("API do synthesizer fora do ar")
            return _FakeResponse([_FakeText(FRASE)])

        tool_name = kwargs["tools"][0]["name"]
        if tool_name == graph_mod._DECISION_TOOL:
            i = self.supervisor_calls
            self.supervisor_calls += 1
            nxt = self._decisions[min(i, len(self._decisions) - 1)]
            return _FakeResponse([_FakeToolUse({"next": nxt, "reasoning": f"fake #{i}"})])
        return _FakeResponse([_FakeToolUse({"sql": self._sql})])


class _FakeClient:
    def __init__(self, decisions: list[str], sql: str, synth_explode: bool):
        self.messages = _FakeMessages(decisions, sql, synth_explode)


@pytest.fixture
def fake_graph(monkeypatch):
    def _install(
        decisions: list[str],
        sql: str = "SELECT count(*) FROM peril",
        synth_explode: bool = False,
        com_custo: bool = False,
    ):
        client = _FakeClient(decisions, sql, synth_explode)
        monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)
        monkeypatch.setattr(graph_mod, "get_schema", lambda: "TABLE peril(id int, nome text)")
        monkeypatch.setattr(graph_mod, "run_query", lambda sql: "[(7,)]")

        if not com_custo:
            # Sem banco neste teste: a gravação de custo vira no-op explícito (quem
            # testa custo é o teste que pede a fixture cost_rows).
            async def _sem_custo(**kwargs):
                return None

            monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)
        return client

    return _install


@pytest_asyncio.fixture
async def cost_rows(engine, monkeypatch):
    """Aponta a session de `record_call_cost` pro container e limpa cost_event.

    Mesmo motivo de tests/test_cost_graph.py: essas linhas são COMMITADAS, então o
    rollback do `db_session` não as alcança.
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
async def ask_client(db_url, monkeypatch):
    """`db_url` vem primeiro DE PROPÓSITO, mesmo nos testes que não tocam no banco.

    `app.db` lê DATABASE_URL no nível do módulo, então importar `app.main` antes do
    container subir só funciona se houver um .env por perto. Com a fixture de banco
    na frente, rodar só este arquivo (`pytest tests/test_synth.py`) funciona num
    checkout limpo — e o container é session-scoped, então não custa nada a mais.
    """
    monkeypatch.setenv("API_TOKENS", f"{CLIENTE}:{TOKEN}")
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_ask_devolve_frase_e_nao_resultado_cru(ask_client, fake_graph):
    """O ponto da fatia: o usuário recebe português, não `SQL: ... / Result: [(7,)]`."""
    fake_graph(["sql_worker", "END"])

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )

    assert r.status_code == 200
    answer = r.json()["answer"]
    assert answer == FRASE
    assert "Result:" not in answer   # o texto cru do worker não vaza pra resposta
    assert "SELECT" not in answer


async def test_caminho_sem_worker_usa_fallback_estatico_sem_llm(ask_client, fake_graph):
    """Nenhum worker rodou: frase fixa e NENHUMA chamada paga pra dizer "não sei"."""
    client = fake_graph(["END"])

    r = await ask_client.post(
        "/ask", json={"question": "Qual a capital da França?"}, headers=AUTH
    )

    assert r.status_code == 200
    assert r.json()["answer"] == graph_mod.NO_ANSWER
    assert client.messages.synth_calls == 0   # não há o que sintetizar, não se paga


async def test_synthesizer_grava_seu_proprio_cost_event(cost_rows, ask_client, fake_graph):
    """O synthesizer é uma chamada de LLM como qualquer outra: tem que aparecer no custo.

    São 4 linhas agora, não 3: supervisor -> sql_worker -> supervisor -> synthesizer.
    """
    fake_graph(["sql_worker", "END"], com_custo=True)

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )
    assert r.status_code == 200

    async with cost_rows() as s:
        eventos = list((await s.execute(select(CostEvent).order_by(CostEvent.id))).scalars())

    assert [e.agent_name for e in eventos] == [
        "supervisor", "sql_worker", "supervisor", "synthesizer",
    ]
    # A linha do synthesizer carrega o mesmo request_id das outras (é o mesmo /ask).
    assert len({e.request_id for e in eventos}) == 1


async def test_falha_do_llm_do_synthesizer_nao_derruba_o_ask(ask_client, fake_graph):
    """Best effort no último nó: a resposta já foi paga, então degrada pro texto cru.

    Feio de ler, mas é um dado REAL — muito melhor do que 500 num /ask que já custou
    dinheiro e já tinha o resultado na mão.
    """
    fake_graph(["sql_worker", "END"], synth_explode=True)

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )

    assert r.status_code == 200
    assert "Result:" in r.json()["answer"]   # caiu no texto cru do worker
