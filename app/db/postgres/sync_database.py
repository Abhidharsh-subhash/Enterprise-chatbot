from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Synchronous PostgreSQL connection URL
SYNC_DATABASE_URL = (
    f"postgresql://{settings.database_username}:{settings.database_password}"
    f"@{settings.database_host}:{settings.database_port}/{settings.database_name}"
)

# Create Sync Engine
sync_engine = create_engine(SYNC_DATABASE_URL, pool_pre_ping=True)

# Create Sync Session Factory
SyncSessionLocal = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)
