from datetime import datetime, timedelta, timezone
from app.core.config import settings
from jose import jwt


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expiry_time
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, settings.algorithm)


def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.refresh_token_expiry_time
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, settings.algorithm)
