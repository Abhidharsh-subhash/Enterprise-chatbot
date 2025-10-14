from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.postgres.database import async_session
from app.db.redis.redis import redis_client


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_redis():
    try:
        yield redis_client
    finally:
        pass
