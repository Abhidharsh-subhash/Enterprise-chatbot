import re
from typing import Optional, Dict, Any


def is_idk(s: str) -> bool:
    if not s:
        return True
    s = s.strip().strip('"').strip().lower()
    s = s.rstrip(".")
    return s == "i don't know"


def extract_short_definition(term: str, text: str) -> Optional[str]:
    """
    Simple extractive fallback: return a short sentence from `text` that likely
    defines or describes `term`.
    """
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
    """
    Decide if a ranked candidate is strong enough to 'must answer'.
    Combines LLM reranker score and vector similarity if present.
    """
    rank_score = item.get("rank_score") or 0.0  # 0..1 from rank_evidence
    sim = item.get("similarity") or 0.0  # cosine similarity proxy (1 - distance)
    return rank_score >= 0.75 or sim >= 0.55
