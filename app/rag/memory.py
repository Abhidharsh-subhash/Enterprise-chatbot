import os
from typing import Optional, List
import redis
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.core.config import settings

_r = redis.Redis.from_url(settings.memory_url, decode_responses=True)

os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
# Fast/cheap LLM for summary only
_SUMMARIZER = ChatOpenAI(
    model=settings.openai_model, temperature=settings.model_temperature
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    "You maintain a brief, factual running summary of this conversation (<=120 tokens). "
    "Capture goals, decisions, constraints, key entities, and stable preferences. "
    "Do not include chain-of-thought or step-by-step reasoning.\n\n"
    "Prior summary:\n{summary}\n\nNew turns:\n{turns}\n\nUpdated summary:"
)
_SUMMARY_CHAIN = SUMMARY_PROMPT | _SUMMARIZER | StrOutputParser()


def _summary_key(user_id: str, session_id: Optional[str]) -> str:
    return f"convsum:{user_id}:{session_id or 'default'}"


def get_summary(user_id: str, session_id: Optional[str]) -> str:
    return _r.get(_summary_key(user_id, session_id)) or ""


def set_summary(
    user_id: str, session_id: Optional[str], summary: str, ttl_sec: int = 24 * 3600
):
    key = _summary_key(user_id, session_id)
    _r.set(key, summary)
    _r.expire(key, ttl_sec)


def get_history(user_id: str, session_id: Optional[str]) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        url=settings.memory_url,
        session_id=f"{user_id}:{session_id or 'default'}",
        ttl=7 * 24 * 3600,
    )


def messages_to_text(messages, max_pairs: int = 3, max_chars: int = 1200) -> str:
    # keep a small recent buffer (like *BufferMemory*)
    # messages are BaseMessage objects with .type in {"human","ai"} and .content
    lines: List[str] = []
    for m in messages[-(max_pairs * 2) :]:
        role = "User" if getattr(m, "type", "") in ("human", "user") else "Assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content}")
    txt = "\n".join(lines)
    return txt[-max_chars:]


def update_memory(user_id: str, session_id: Optional[str], question: str, answer: str):
    # 1) append turn to chat history
    history = get_history(user_id, session_id)
    history.add_user_message(question)
    history.add_ai_message(answer)
    # 2) update rolling summary
    prior = get_summary(user_id, session_id)
    turns = f"User: {question}\nAssistant: {answer}"
    try:
        updated = _SUMMARY_CHAIN.invoke({"summary": prior, "turns": turns}).strip()
        set_summary(user_id, session_id, updated)
    except Exception:
        # if summarizer fails, keep prior summary
        pass
