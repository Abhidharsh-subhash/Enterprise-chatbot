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


def compose_answer(
    original_question: str,
    context_blocks: List[str],
    temperature: float = 0.0,
    must_answer: bool = False,
    include_example: bool = False,
    max_sentences: int = 2,
) -> Tuple[str, str]:
    """
    Answer ONLY the original_question using ONLY the provided context.
    - Do NOT broaden or add domain qualifiers not present in the question.
    - Keep it concise (<= max_sentences).
    - If include_example=True, include exactly one short example drawn from the context.
    - If must_answer is True and evidence is present, do not return 'I don't know.'
    - Math/LaTeX: see system prompt below.
    """
    system = (
        "Answer the user's original_question using ONLY the provided context. "
        "Do not broaden or add domain qualifiers (e.g., 'in computer science') that are not explicitly present in the question. "
        "Do not restate the question. Keep the answer concise and direct. "
        "If include_example is true, include exactly one short example grounded in the context. "
        "Use at most max_sentences sentences. "
        "If must_answer is true and relevant evidence exists in the context, do NOT reply 'I don't know.' "
        "Only if the context truly lacks relevant information may you answer exactly 'I don't know.' "
        "Do not include citations or source markers. "
        "\n\n"
        "Math and LaTeX formatting rules:\n"
        "- When mathematical notation is needed, typeset it in LaTeX.\n"
        "- Use $...$ for inline math and $$...$$ for display equations on their own lines.\n"
        "- Do NOT wrap LaTeX in code fences or HTML; return plain Markdown + LaTeX only.\n"
        "- Keep to standard LaTeX macros only (e.g., \\frac, \\sum, \\alpha); do not define new macros.\n"
        "- Ensure the JSON string escapes backslashes (e.g., use \\\\frac not \\frac in the JSON value). "
        "The consumer will render the string as Markdown with LaTeX.\n"
        "- Keep display equations self-contained (avoid \\begin{equation} ... \\end{equation}); $$...$$ is sufficient.\n"
        "- Keep punctuation outside math when possible (e.g., '$a^2+b^2=c^2$.' not '$a^2+b^2=c^2.$').\n"
        "\n"
        "Respond as JSON: { answer: string, brief_explanation: string }."
    )

    context = "\n\n---\n\n".join(context_blocks)
    payload = {
        "context": context,
        "original_question": original_question,
        "must_answer": bool(must_answer),
        "include_example": bool(include_example),
        "max_sentences": int(max_sentences),
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


def refine_query_with_history(
    question: str, conversation_hint: Optional[str] = None
) -> Dict:
    """
    Improve retrieval queries using the user's question plus conversation context.
    - Use conversation_hint (summary + recent turns) ONLY to resolve references and narrow scope.
    - Do not broaden intent beyond what the user asked.
    Returns JSON: { rewrite: str, sub_questions: [str], keywords: [str] }
    """
    system = (
        "You refine user questions for retrieval without changing their intent. "
        "Use the conversation_hint (summary + recent chat) only to resolve references, scope, and ellipses. "
        "Do not broaden the topic beyond what the user asked. Keep it concise. "
        "Return JSON: { rewrite: string, sub_questions: array<=3, keywords: array }."
    )
    print("after system")
    user_payload = {
        "question": question,
        "conversation_hint": conversation_hint or "",
    }
    print("after user_payload")
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )
    print("after resp")
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}
    return {
        "rewrite": (data.get("rewrite") or question),
        "sub_questions": (data.get("sub_questions") or [])[:3],
        "keywords": data.get("keywords") or [],
    }
