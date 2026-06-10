from sqlalchemy import text
from fastapi import Depends
from app.db import get_session

@app.get("/health/db")
async def health_db(session=Depends(get_session)):
    await session.execute(text("SELECT 1"))
    return {"db": "ok"}
