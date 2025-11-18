import json
from typing import List, Dict, Tuple, Optional
from app.core.openai_client import client, CHAT_MODEL


def refine_query(question: str) -> Dict:
    """
    Improve retrieval with a careful rewrite and (optional) sub-questions.
    - Do NOT broaden the intent of the original question.
    - If the original is a single word or very short, prefer a definitional rewrite:
      e.g., "What is a <term> (in the likely domain of the provided context)?"
    """
    system = (
        "You improve search queries for retrieval without changing the user's intent. "
        "If the question is a single word or very short and ambiguous, rewrite it into a definitional question, "
        "keeping scope narrow and aligned with likely domain (e.g., programming if the context suggests it). "
        "Return JSON: { rewrite (string), sub_questions (<=3), keywords (array) }."
    )
    user = f"Original question: {question}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "rewrite": data.get("rewrite") or question,
        "sub_questions": data.get("sub_questions", [])[:3],
        "keywords": data.get("keywords", []),
    }


def rank_evidence(question: str, candidates: List[Dict]) -> List[Dict]:
    """
    Step 2: Ask the model to pick the most relevant chunks.
    Input candidates: [{id, text, file_name, chunk_index, similarity}, ...]
    Returns a ranked subset with reason (short) to aid debugging (not chain-of-thought).
    """
    system = (
        "You are ranking text snippets by relevance to the question. "
        "Return JSON: { selected: [{id, reason, score}] }, where score is 0..1. "
        "Keep reason brief (<= 1 sentence)."
    )
    doc_summaries = [
        {
            "id": c["id"],
            "text": c["text"][:1200],  # small cap for safety
            "similarity": c.get("similarity"),
            "file_name": c.get("file_name"),
            "chunk_index": c.get("chunk_index"),
        }
        for c in candidates
    ]
    user = json.dumps({"question": question, "candidates": doc_summaries})
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    selection = data.get("selected", [])
    selected_ids = {s["id"] for s in selection}

    # Reorder candidates based on selected list order
    ordered = []
    id_to_item = {c["id"]: c for c in candidates}
    for s in selection:
        c = id_to_item.get(s["id"])
        if c:
            c["rank_reason"] = s.get("reason")
            c["rank_score"] = s.get("score")
            ordered.append(c)
    # Fallback: if model returns nothing, return top by similarity
    if not ordered:
        ordered = sorted(
            candidates, key=lambda x: (x.get("similarity") or 0), reverse=True
        )
    return ordered


# def compose_answer(
#     question: str, context_blocks: List[str], temperature: float = 0.1
# ) -> Tuple[str, str]:
#     """
#     Step 3: Compose final answer using only selected context.
#     Returns (answer, brief_explanation). We do not expose full chain-of-thought.
#     """
#     system = (
#         "Answer the user's question using ONLY the provided context. "
#         "If unknown, reply exactly: I don't know. "
#         "Do not include citations or source markers. No preamble. "
#         "Respond as JSON with fields: answer (string), brief_explanation (<= 1 sentence)."
#     )
#     context = "\n\n---\n\n".join(context_blocks)
#     user = json.dumps({"context": context, "question": question})
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         temperature=temperature,
#         response_format={"type": "json_object"},
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ],
#     )
#     data = json.loads(resp.choices[0].message.content)
#     answer = (data.get("answer") or "").strip()
#     brief_explanation = (data.get("brief_explanation") or "").strip()
#     return answer, brief_explanation


def compose_answer(
    original_question: str,
    context_blocks: List[str],
    rewrite: Optional[str] = None,
    sub_questions: Optional[List[str]] = None,
    temperature: float = 0.1,
    must_answer: bool = False,
) -> Tuple[str, str]:
    """
    Use ONLY the provided context to answer.
    Prefer the original question; use rewrite/sub_questions to disambiguate.
    If must_answer=True, DO NOT return "I don't know." — extract and summarize from the context.
    """
    # Put clear constraints in the system message
    system = (
        "Answer the user's original question using ONLY the provided context. "
        "Prefer original_question; use rewrite/sub_questions only to disambiguate. "
        "The context may include 'Source [n]' blocks (these are the only factual sources) "
        "and optional conversation notes (not factual sources). "
        "If must_answer is true, DO NOT respond with 'I don't know.' "
        "Instead, infer the likely interpretation supported by the sources and answer using quotes or paraphrases from the sources. "
        "Only if the context truly lacks relevant information may you answer exactly 'I don't know.' "
        "No citations or source markers in the output. "
        "Respond as JSON: { answer: string, brief_explanation: string }."
    )
    context = "\n\n---\n\n".join(context_blocks)
    payload = {
        "context": context,
        "original_question": original_question,
        "rewrite": rewrite,
        "sub_questions": sub_questions or [],
        "must_answer": bool(must_answer),
    }
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    answer = (data.get("answer") or "").strip()
    brief_explanation = (data.get("brief_explanation") or "").strip()
    return answer, brief_explanation
