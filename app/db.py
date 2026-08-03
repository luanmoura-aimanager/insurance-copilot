import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db_url import normalize_url

# Continua no topo: em dev local o .env é a fonte do DATABASE_URL. Em CI/produção
# não existe .env e a chamada é no-op — ela não lê nem exige a variável.
load_dotenv()


@lru_cache(maxsize=1)
def _sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Constrói engine + fábrica de sessions na PRIMEIRA chamada, não no import.

    É aqui que DATABASE_URL é lida — em runtime. Importar `app.db` (e portanto
    `app.main`, que importa a cadeia toda) não exige mais banco nenhum, que é o que
    permite `import app.main` num ambiente limpo: CI, `--help` de script, coleta do
    pytest antes do testcontainers subir o Postgres.

    O cache é o que garante UM pool de conexões por processo: sem ele cada chamada
    de `SessionLocal()` criaria uma engine nova e vazaria conexões.
    """
    # Railway pode entregar `postgres://` ou `postgresql://`; a engine da app precisa
    # de `postgresql+asyncpg://` (ver app/db_url.py).
    url = normalize_url(os.environ["DATABASE_URL"], "asyncpg")
    engine = create_async_engine(url)                        # pool de conexões async
    return async_sessionmaker(engine, expire_on_commit=False)


def SessionLocal(**kwargs) -> AsyncSession:
    """Fábrica de sessions — mesma API de antes (`async with SessionLocal() as s`).

    Continua sendo um atributo chamável do módulo de propósito: 6 scripts fazem
    `from app.db import SessionLocal` e dois testes trocam o atributo via
    `monkeypatch.setattr("app.db.SessionLocal", ...)`.

    Só a CHAMADA é a API pública: como isto virou função, os métodos do
    `async_sessionmaker` (`.begin()`, `.configure()`) não existem mais aqui. Ninguém
    os usa hoje; se precisar, pegue a fábrica com `_sessionmaker()`.
    """
    return _sessionmaker()(**kwargs)


async def get_session() -> AsyncSession:            # dependência pro Depends
    async with SessionLocal() as session:
        yield session
