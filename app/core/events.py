from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.postgres.database import engine
from app.db.redis.redis import redis_client
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

    try:
        pong = await redis_client.ping()
        print("✅ Redis connected:", pong)
    except Exception as e:
        print("❌ Redis connection failed:", e)

    yield

    await engine.dispose()
    print("🔌 PostgreSQL engine disposed")

    await redis_client.aclose()
    print("🔌 Redis connection closed")
