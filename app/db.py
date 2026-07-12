import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

load_dotenv()


def _async_url(url: str) -> str:
    """Normalize any Postgres URL to the async (asyncpg) driver.

    Railway may hand us `postgres://...` or `postgresql://...`; the app engine
    needs `postgresql+asyncpg://...`.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    return url


DATABASE_URL = _async_url(os.environ["DATABASE_URL"])

engine = create_async_engine(DATABASE_URL)          # pool de conexões async
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)  # fábrica de sessions

async def get_session() -> AsyncSession:            # dependência pro Depends
    async with SessionLocal() as session:
        yield session
