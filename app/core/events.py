from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.postgres.database import engine
from sqlalchemy import text
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL connected")
    except Exception as e:
        print("❌ PostgreSQL connection failed:", e)

    yield

    await engine.dispose()
    print("🔌 PostgreSQL engine disposed")
