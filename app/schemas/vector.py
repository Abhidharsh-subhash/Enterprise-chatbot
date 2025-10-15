from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.1
    max_context_chars: int = 12000  # safeguard to avoid over-long prompts
