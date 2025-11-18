import re
from typing import Optional, Dict, Any, List
from app.core.openai_client import client, CHAT_MODEL


def is_idk(s: str) -> bool:
    if not s:
        return True
    s = s.strip().strip('"').strip().lower()
    s = s.rstrip(".")
    return s == "i don't know"


def extract_short_definition(term: str, text: str) -> Optional[str]:
    if not text:
        return None
    term_l = term.lower()
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Prefer sentences that mention the term
    cands = [s.strip() for s in sentences if term_l in s.lower()]
    if not cands:
        cands = [s.strip() for s in sentences if s.strip()]
    if not cands:
        return None
    # Prefer shorter sentences that look definitional
    cands.sort(key=len)
    best = cands[0]
    return best[:300]


def evidence_is_strong(item: Dict[str, Any]) -> bool:
    rank_score = item.get("rank_score") or 0.0
    sim = item.get("similarity") or 0.0
    return rank_score >= 0.75 or sim >= 0.55


def extractive_answer_from_snippets(
    question: str, snippets: List[str]
) -> Optional[str]:
    """
    LLM-powered, strictly-extractive fallback from the provided snippets.
    Returns a short answer or None if it truly can’t extract.
    """
    # Build a compact context
    joined = "\n\n---\n\n".join(s[:1600] for s in snippets if s)
    if not joined:
        return None

    system = (
        "You extract answers ONLY from the provided text snippets. "
        "Do not use outside knowledge. "
        'Return JSON { answer: string }. If no direct answer exists, return { answer: "#UNANSWERABLE" }.'
    )
    user_payload = {
        "question": question,
        "snippets": joined,
    }
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user_payload)},
        ],
    )
    content = resp.choices[0].message.content
    try:
        import json

        data = json.loads(content)
    except Exception:
        return None
    ans = (data.get("answer") or "").strip()
    if not ans or ans == "#UNANSWERABLE":
        return None
    # Keep it short
    return ans[:800]
