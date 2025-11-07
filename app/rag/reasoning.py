import json
from typing import List, Dict, Tuple
from app.core.openai_client import client, CHAT_MODEL


def refine_query(question: str) -> Dict:
    """
    Step 1: Improve retrieval with a lightweight rewrite and sub-questions.
    Returns JSON: { rewrite, sub_questions[], keywords[] }
    """
    system = (
        "You improve search queries for retrieval. "
        "Return a JSON object with fields: rewrite (string), sub_questions (array of up to 3 strings), keywords (array). "
        "Keep it concise."
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


def compose_answer(
    question: str, context_blocks: List[str], temperature: float = 0.1
) -> Tuple[str, str]:
    """
    Step 3: Compose final answer using only selected context.
    Returns (answer, brief_explanation). We do not expose full chain-of-thought.
    """
    system = (
        "Answer the user's question using ONLY the provided context. "
        "If unknown, reply exactly: I don't know. "
        "Do not include citations or source markers. No preamble. "
        "Respond as JSON with fields: answer (string), brief_explanation (<= 1 sentence)."
    )
    context = "\n\n---\n\n".join(context_blocks)
    user = json.dumps({"context": context, "question": question})
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    answer = (data.get("answer") or "").strip()
    brief_explanation = (data.get("brief_explanation") or "").strip()
    return answer, brief_explanation
