from openai import OpenAI
from app.core.config import settings

if not settings.openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not configured")

client = OpenAI(api_key=settings.openai_api_key)

# Centralized config
CHAT_MODEL = settings.openai_model or "gpt-4o-mini"
DEFAULT_TEMPERATURE = float(getattr(settings, "model_temperature", 0) or 0)
EMBEDDING_MODEL = "text-embedding-3-small"
