# app/rag/utils.py
import re
from typing import Optional, Dict, Any, List
from app.core.openai_client import client, CHAT_MODEL


def is_idk(s: str) -> bool:
    if not s:
        return True
    s = s.strip().strip('"').strip().lower()
    s = s.rstrip(".")
    return s == "i don't know"


def evidence_is_strong(item: Dict[str, Any]) -> bool:
    rank_score = item.get("rank_score") or 0.0
    sim = item.get("similarity") or 0.0
    return rank_score >= 0.75 or sim >= 0.55


def extract_short_definition(term: str, text: str) -> Optional[str]:
    if not text:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cands = [s.strip() for s in sentences if term.lower() in s.lower()] or [
        s.strip() for s in sentences if s.strip()
    ]
    if not cands:
        return None
    cands.sort(key=len)
    return cands[0][:300]


def extractive_answer_from_snippets(
    question: str, snippets: List[str]
) -> Optional[str]:
    joined = "\n\n---\n\n".join(s[:1600] for s in snippets if s)
    if not joined:
        return None
    system = (
        "You extract answers ONLY from the provided text snippets. "
        "Do not use outside knowledge. "
        'Return JSON {"answer": string}. If no direct answer exists, return {"answer": "#UNANSWERABLE"}.'
    )
    user_payload = {"question": question, "snippets": joined}
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user_payload)},
        ],
    )
    try:
        import json

        data = json.loads(resp.choices[0].message.content)
    except Exception:
        return None
    ans = (data.get("answer") or "").strip()
    return None if not ans or ans == "#UNANSWERABLE" else ans[:800]


def strip_domain_preface(answer: str, original_question: str) -> str:
    """
    Remove leading domain qualifiers like 'In computer science,' if the original_question
    does not include that domain. Heuristic and conservative.
    """
    if not answer:
        return answer
    ql = original_question.lower()
    if any(
        tok in ql for tok in ["computer science", "programming", "python", "software"]
    ):
        return answer  # user explicitly hinted the domain

    # Remove leading 'In <domain>,' or 'In <domain> terms,' etc.
    cleaned = re.sub(r"^\s*in [^,]{2,60},\s*", "", answer, flags=re.IGNORECASE)
    return cleaned.strip()


def build_conv_hint(
    summary: Optional[str], recent_text: Optional[str], max_chars: int = 800
) -> str:
    """
    Build a compact, model-friendly hint from summary + recent messages.
    The hint is used only by refine_query to resolve references and keep intent.
    """
    parts = []
    if summary and summary.strip():
        parts.append(f"Summary: {summary.strip()}")
    if recent_text and recent_text.strip():
        parts.append(f"Recent: {recent_text.strip()}")
    hint = "\n".join(parts)
    # Cap length to keep refiner prompt tight
    return hint[:max_chars]
