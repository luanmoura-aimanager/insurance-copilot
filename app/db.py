import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

DATABASE_URL = os.environ["DATABASE_URL"]          # ex.: postgresql+asyncpg://insurance:insurance@localhost:5432/insurance

engine = create_async_engine(DATABASE_URL)          # pool de conexões async
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)  # fábrica de sessions

async def get_session() -> AsyncSession:            # dependência pro Depends
    async with SessionLocal() as session:
        yield session
