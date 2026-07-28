"""
Testes da proteção do POST /ask: Bearer com identidade + rate limit.

O grafo inteiro é substituído por um fake (`_FakeGraph`) — aqui não interessa o que
o agente responde, só quem consegue chegar até ele e quantas vezes. Zero chamada paga.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

TOKEN = "token-valido-do-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
QUESTION = {"question": "Quantos perigos existem?"}


class _FakeGraph:
    """Mesma superfície usada pelo /ask: ainvoke -> state com messages/iterations."""

    async def ainvoke(self, state):
        return {
            "iterations": 1,
            "messages": [AIMessage(content="Result: [(7,)]", name="sql_worker")],
        }


@pytest.fixture
def protected_client(monkeypatch):
    """Client HTTP com um token cadastrado, grafo fake e limites folgados.

    Cada teste sobrescreve o que precisa apertar (o env é lido a cada request, tanto
    em _load_tokens quanto nos callables de limite).
    """
    monkeypatch.setenv("API_TOKENS", f"cliente-teste:{TOKEN}")
    monkeypatch.setenv("ASK_RATE_LIMIT_CLIENT", "100/minute")
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "100/minute")

    from app import main as main_mod

    monkeypatch.setattr(main_mod, "graph", _FakeGraph())

    async def _make():
        transport = ASGITransport(app=main_mod.app)
        return AsyncClient(transport=transport, base_url="http://test")

    return _make


@pytest.fixture
async def client(protected_client):
    async with await protected_client() as c:
        yield c


async def test_sem_header_401(client):
    """Sem Authorization: 401 + WWW-Authenticate (é isso que diz ao cliente COMO autenticar)."""
    r = await client.post("/ask", json=QUESTION)

    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


async def test_token_invalido_401(client):
    """Token que não está no API_TOKENS não entra — e a resposta não repete o token."""
    r = await client.post("/ask", json=QUESTION, headers={"Authorization": "Bearer nao-e-esse"})

    assert r.status_code == 401
    assert "nao-e-esse" not in r.text


async def test_token_valido_200(client):
    r = await client.post("/ask", json=QUESTION, headers=AUTH)

    assert r.status_code == 200
    assert r.json() == {"answer": "Result: [(7,)]", "iterations": 1}


async def test_health_continua_aberto(client):
    """O healthcheck do Railway não manda Authorization: protegê-lo derrubaria o deploy."""
    assert (await client.get("/health")).status_code == 200


async def test_estourar_o_limite_429(monkeypatch, protected_client):
    """3ª requisição com o limite por cliente em 2/minute -> 429."""
    monkeypatch.setenv("ASK_RATE_LIMIT_CLIENT", "2/minute")

    async with await protected_client() as c:
        assert (await c.post("/ask", json=QUESTION, headers=AUTH)).status_code == 200
        assert (await c.post("/ask", json=QUESTION, headers=AUTH)).status_code == 200

        r = await c.post("/ask", json=QUESTION, headers=AUTH)

    assert r.status_code == 429


async def test_limite_por_ip_429(monkeypatch, protected_client):
    """O limite por IP vale em cima do de cliente: nem token válido escapa do burst."""
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "1/minute")

    async with await protected_client() as c:
        assert (await c.post("/ask", json=QUESTION, headers=AUTH)).status_code == 200

        r = await c.post("/ask", json=QUESTION, headers=AUTH)

    assert r.status_code == 429


async def test_api_tokens_vazio_500(monkeypatch, protected_client):
    """Fail closed na configuração: sem tokens ninguém entra, e o erro é 500 (não 401),
    porque 401 faria um deploy quebrado parecer credencial errada do cliente."""
    monkeypatch.setenv("API_TOKENS", "")

    async with await protected_client() as c:
        r = await c.post("/ask", json=QUESTION, headers=AUTH)

    assert r.status_code == 500
