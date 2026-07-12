import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db_url import normalize_url

load_dotenv()

# Railway pode entregar `postgres://` ou `postgresql://`; a engine da app precisa
# de `postgresql+asyncpg://` (ver app/db_url.py).
DATABASE_URL = normalize_url(os.environ["DATABASE_URL"], "asyncpg")

engine = create_async_engine(DATABASE_URL)          # pool de conexões async
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)  # fábrica de sessions

async def get_session() -> AsyncSession:            # dependência pro Depends
    async with SessionLocal() as session:
        yield session
