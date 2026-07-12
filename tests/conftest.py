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


@pytest.fixture(scope="session")
def db_url():
    """Boot one throwaway Postgres for the whole suite and apply migrations to it."""
    with PostgresContainer("postgres:16") as pg:
        # testcontainers gives a sync (psycopg2) URL; the app engine wants asyncpg.
        async_url = pg.get_connection_url().replace("+psycopg2", "+asyncpg")
        os.environ["DATABASE_URL"] = async_url  # env.py reads this (and swaps to sync for Alembic)

        # apply the real migrations against the clean container — this also exercises them
        command.upgrade(Config("alembic.ini"), "head")

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
