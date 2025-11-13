from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.1
    max_context_chars: int = 12000  # safeguard to avoid over-long prompts
    session_id: Optional[str] = None  # NEW: scope memory per chat thread
