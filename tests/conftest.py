"""
Test fixtures.

Two scopes on purpose:
- the Postgres *container* is session-scoped (slow to boot, so boot it once);
- each test's *session* is function-scoped and rolls back at the end,
  so tests stay isolated even though they share one container.
"""
import os
import subprocess

# Ryuk (testcontainers' cleanup sidecar) doesn't play well with Colima on macOS.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

# testcontainers' docker SDK reads DOCKER_HOST, which Colima doesn't set globally.
# Resolve the socket the working docker CLI uses and point the SDK at it.
if not os.environ.get("DOCKER_HOST"):
    try:
        host = subprocess.check_output(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            text=True,
        ).strip()
        if host:
            os.environ["DOCKER_HOST"] = host
    except Exception:
        pass

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer


@pytest.fixture(autouse=True)
def reset_limiter():
    """O limiter do slowapi é global (storage in-memory no processo do teste).

    Sem isso, o consumo de um teste vazaria pro próximo e a ordem de execução da
    suíte viraria parte do resultado.
    """
    from app.limits import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(autouse=True)
def sem_varredura_automatica(monkeypatch):
    """Nenhum teste chama o grafo por acidente: a varredura do webhook vira NO-OP.

    Sem isto, qualquer teste que POSTe em /webhook/whatsapp passa a rodar a varredura da W2b
    DENTRO do `await client.post(...)`. O mecanismo é exato: `Response.__call__` faz
    `await self.background()` depois de mandar `http.response.start`/`body`, mas ainda dentro
    do `await self.app(...)` do `RequestIdMiddleware` — e o `ASGITransport` do httpx só
    devolve a resposta quando a chamada ASGI termina. Com a sessão real de
    `tests/test_whatsapp_inbox.py` a linha existe, então seria grafo real -> Anthropic real
    -> DINHEIRO, em dev e no CI, num módulo que não é sobre isso.

    Fail-CLOSED de propósito, e autouse pelo mesmo motivo do `reset_limiter`: a invariante
    "nenhum teste gasta dinheiro" passa a ser estrutural em vez de depender de cada fixture
    lembrar. Quem QUER exercitar a varredura chama `processar_pendentes` direto (com grafo e
    sender falsos) ou troca este seam por um espião — sempre explicitamente.
    """
    async def _nao_roda(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.inbox.varrer_em_background", _nao_roda)


@pytest.fixture(scope="session")
def db_url():
    """Boot one throwaway Postgres for the whole suite and apply migrations to it."""
    # Mesma imagem do docker-compose: a migration do clause_chunk faz
    # CREATE EXTENSION vector, que o postgres:16 puro não tem.
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        # testcontainers gives a sync (psycopg2) URL; the app engine wants asyncpg.
        async_url = pg.get_connection_url().replace("+psycopg2", "+asyncpg")
        os.environ["DATABASE_URL"] = async_url  # env.py reads this (and swaps to sync for Alembic)

        # apply the real migrations against the clean container — this also exercises them.
        #
        # `configure_logger=False` é lido pelo `alembic/env.py` e é o que impede a migration
        # de RECONFIGURAR O LOGGING DO PYTEST: rodando in-process, o `fileConfig` do env.py
        # desligava todos os loggers já criados (`disable_existing_loggers`) e substituía os
        # handlers da raiz — onde vive o `LogCaptureHandler` que alimenta o `caplog`. Nos
        # dois casos o sintoma é o mesmo e é o pior possível: teste de log que passa sem ver
        # log nenhum. Ver o comentário longo em `alembic/env.py`.
        cfg = Config("alembic.ini")
        cfg.attributes["configure_logger"] = False
        command.upgrade(cfg, "head")

        yield async_url
    # container is torn down here, at the end of the whole session


@pytest_asyncio.fixture(scope="session")
async def engine(db_url):
    eng = create_async_engine(db_url)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine):
    """Function-scoped session wrapped in a transaction that is always rolled back."""
    conn = await engine.connect()
    txn = await conn.begin()
    Session = async_sessionmaker(bind=conn, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        await session.close()
        await txn.rollback()   # undo whatever the test wrote — next test starts clean
        await conn.close()


@pytest_asyncio.fixture
async def client(db_session):
    """FastAPI client whose DB dependency uses the rolled-back test session."""
    from app.db import get_session
    from app.main import app

    async def _override():
        yield db_session

    app.dependency_overrides[get_session] = _override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
