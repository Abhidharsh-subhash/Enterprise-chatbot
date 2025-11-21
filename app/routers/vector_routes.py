import os
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    status,
    Depends,
    BackgroundTasks,
    Request,
)
from app.tasks.vector_tasks import process_file_task
from app.models.users import Users
from app.dependencies import get_current_user
from app.schemas.vector import AskRequest
from app.vector_store.chrome_store import query_user_vectors, get_collection
from app.utils.embeddings import get_embedding
from app.rag.reasoning import compose_answer, refine_query_with_history, rank_evidence
from app.rag.memory import get_summary, get_history, messages_to_text, update_memory
from app.rag.utils import (
    evidence_is_strong,
    is_idk,
    extract_short_definition,
    extractive_answer_from_snippets,
    strip_domain_preface,
    build_conv_hint,
)
from app.core.logger import logger
import uuid
from typing import Dict, Any

router = APIRouter(prefix="/vector", tags=["convertion"])


@router.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...), current_user: Users = Depends(get_current_user)
):
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Send Celery task to background
        process_file_task.delay(file_path, current_user.id)

        return {
            "status": status.HTTP_201_CREATED,
            "message": "File uploaded successfully. Processing in background.",
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/ask")
async def ask_question(
    payload: AskRequest,
    background_tasks: BackgroundTasks,
    current_user: Users = Depends(get_current_user),
):
    try:
        logger.info(f"asked the question")
        q = (payload.question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Question is required.")
        user_id = str(current_user.id)
        logger.info(f"user_id fetched is {user_id}")

        # 0) Session management: create a session_id if none provided
        session_id = (payload.session_id or "").strip() or str(uuid.uuid4())
        session_created = not bool(payload.session_id)
        logger.info("handled the session_id")

        # A) Memory (scoped to session_id)
        conv_summary = get_summary(user_id, session_id)
        try:
            recent_text = messages_to_text(get_history(user_id, session_id).messages)
        except Exception:
            recent_text = ""
        logger.info("conv summary if fetched")

        # Build a compact hint for the refiner from summary + recent turns
        conv_hint = build_conv_hint(conv_summary, recent_text)
        logger.info("con_hint is created")

        # B) Query rewrite WITH conversation hint (rewrite used only for retrieval)
        try:
            r = refine_query_with_history(q, conversation_hint=conv_hint)
        except Exception as e:
            logger.error(f"refine_query_with_history failed: {e}")
            r = {"rewrite": q, "sub_questions": [], "keywords": []}
        print(f"refined query : {r}")

        search_queries = [q, r.get("rewrite")] + r.get("sub_questions", [])
        search_queries = [s for s in search_queries if s]
        seen = set()
        search_queries = [s for s in search_queries if not (s in seen or seen.add(s))][
            :3
        ]

        col = get_collection()
        candidates_map = {}

        def add_results(res):
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            ids = res.get("ids", [[]])[0]
            for doc, meta, dist, _id in zip(docs, metas, dists, ids):
                sim = (1 - (dist or 0)) if dist is not None else None
                existing = candidates_map.get(_id)
                if not existing:
                    candidates_map[_id] = {
                        "id": _id,
                        "text": doc,
                        "file_name": meta.get("file_name"),
                        "chunk_index": meta.get("chunk_index"),
                        "doc_id": meta.get("doc_id"),
                        "upload_time": meta.get("upload_time"),
                        "similarity": sim,
                    }
                else:
                    if sim is not None:
                        existing["similarity"] = max(
                            existing.get("similarity") or 0, sim
                        )

        TOP_K = 6
        for sq in search_queries:
            emb = get_embedding(sq)
            res = query_user_vectors(emb, user_id, top_k=TOP_K)
            add_results(res)
        print(f"emb is {emb}")
        print(f"res is {res}")

        if not candidates_map:
            return {
                "status_code": status.HTTP_200_OK,
                "answer": "No matching context found. Upload a file first.",
            }

        # C) Neighbor expansion
        def expand_neighbor_ids(_id: str, window: int = 1):
            try:
                base, idx = _id.rsplit(":", 1)
                idx = int(idx)
            except Exception:
                return []
            return [
                f"{base}:{j}" for j in range(max(0, idx - window), idx + window + 1)
            ]

        neighbor_ids = set()
        for _id in list(candidates_map.keys()):
            neighbor_ids.update(expand_neighbor_ids(_id, window=1))

        missing_neighbors = list(neighbor_ids.difference(candidates_map.keys()))
        if missing_neighbors:
            nb = col.get(ids=missing_neighbors, include=["documents", "metadatas"])
            for _id, doc, meta in zip(
                nb.get("ids", []), nb.get("documents", []), nb.get("metadatas", [])
            ):
                candidates_map[_id] = {
                    "id": _id,
                    "text": doc,
                    "file_name": meta.get("file_name"),
                    "chunk_index": meta.get("chunk_index"),
                    "doc_id": meta.get("doc_id"),
                    "upload_time": meta.get("upload_time"),
                    "similarity": None,
                }

        candidates = list(candidates_map.values())

        # D) Rerank
        try:
            ranked = rank_evidence(q, candidates)[:3]
        except Exception as e:
            logger.error(f"rank_evidence failed: {e}")
            ranked = sorted(
                candidates, key=lambda c: c.get("similarity") or 0, reverse=True
            )[:3]

        # E) Guard
        max_sim = max((c.get("similarity") or 0) for c in ranked) if ranked else 0.0
        if max_sim < 0.40 and not any(evidence_is_strong(c) for c in ranked):
            return {
                "status_code": status.HTTP_200_OK,
                "answer": "I don't know.",
                "session_id": session_id,
                "session_created": session_created,
            }

        # F) Build context: Sources FIRST, then optional summary/recent (labelled)
        context_parts = []
        sources = []
        for i, c in enumerate(ranked, start=1):
            header = f"Source [{i}] | file: {c.get('file_name')} | chunk: {c.get('chunk_index')}"
            context_parts.append(f"{header}\n{(c.get('text') or '')}")
            sources.append(
                {
                    "rank": i,
                    "id": c["id"],
                    "file_name": c.get("file_name"),
                    "chunk_index": c.get("chunk_index"),
                    "doc_id": c.get("doc_id"),
                    "upload_time": c.get("upload_time"),
                    "similarity": c.get("similarity"),
                    "rank_score": c.get("rank_score"),
                    "rank_reason": c.get("rank_reason"),
                }
            )

        if conv_summary.strip():
            context_parts.append(
                f"Conversation summary (not a factual source):\n{conv_summary}"
            )
        if recent_text.strip():
            context_parts.append(
                f"Recent chat excerpt (not a factual source):\n{recent_text}"
            )

        # Optional cap
        max_chars = getattr(payload, "max_context_chars", None)
        if max_chars:
            used, trimmed = 0, []
            for block in context_parts:
                if used + len(block) > max_chars:
                    break
                trimmed.append(block)
                used += len(block)
            context_parts = trimmed

        # G) Decide must_answer
        must_answer = any(evidence_is_strong(c) for c in ranked)

        # H) Composer: strictly answer the original question; include example if asked
        q_lower = q.lower()
        include_example = (
            ("example" in q_lower) or ("e.g." in q_lower) or (" eg " in f" {q_lower} ")
        )

        try:
            answer, brief_explanation = compose_answer(
                original_question=q,
                context_blocks=context_parts,
                temperature=payload.temperature,
                must_answer=must_answer,
                include_example=include_example,
                max_sentences=2,
            )
        except Exception as e:
            logger.error(f"compose_answer failed: {e}")
            answer = "I'm sorry, I encountered an issue generating the answer."
            brief_explanation = "Fallback response"

        # I) Strip domain prefaces if the user didn't include them
        answer = strip_domain_preface(answer, q)

        # J) Fallbacks if still IDK but evidence is strong
        fallback_used = False
        if is_idk(answer) and must_answer and ranked:
            top_snippets = [
                (ranked[i].get("text") or "") for i in range(min(3, len(ranked)))
            ]
            extracted = extractive_answer_from_snippets(q, top_snippets)
            if extracted:
                answer = strip_domain_preface(extracted, q)
                brief_explanation = "Extracted from the most relevant snippets."
                fallback_used = True
            else:
                term = q.strip().strip("?.!").lower()
                if term and len(term.split()) <= 3:
                    defn = extract_short_definition(
                        term, top_snippets[0] if top_snippets else ""
                    )
                    if defn:
                        answer = strip_domain_preface(defn, q)
                        brief_explanation = "Extracted from the most relevant snippet."
                        fallback_used = True

        # K) Update memory (scoped to this session_id)
        background_tasks.add_task(update_memory, user_id, session_id, q, answer)

        return {
            "status_code": status.HTTP_200_OK,
            "answer": answer,
            "brief_explanation": brief_explanation,
            "sources": sources,
            "query_used": q,
            "rewritten_query": r.get("rewrite"),
            "sub_questions": r.get("sub_questions", []),
            "must_answer": must_answer,
            "fallback_used": fallback_used,
            "session_id": session_id,
            "session_created": session_created,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/dummy_webhook", status_code=status.HTTP_200_OK)
async def capture_details(
    request: Request,  # Access to the raw request object
) -> Dict[str, Any]:
    """
    Captures and prints the incoming request and dependency values,
    then always returns a 200 OK success message.
    """

    ("\n--- 📥 CAPTURE ENDPOINT ACTIVATED ---")

    # 2. Capture and Print Request Details (Headers, Query, Body)
    print("\n▶️ Captured Request Details:")

    # Headers
    print(f"  Request Headers: {dict(request.headers)}")

    # Query Parameters
    query_params = dict(request.query_params)
    print(f"  Query Parameters: {query_params}")

    # Body (Asynchronously)
    try:
        body = await request.json()
        logger.info(f"chatbot >> webhook >> Request Body >> {body}")
    except Exception:
        # Fallback for non-JSON content (e.g., plain text or empty)
        body_bytes = await request.body()
        if body_bytes:
            logger.info(
                f"chatbot >> webhook >> Request Body(Raw) >> {body_bytes.decode("utf-8", errors="ignore")}"
            )
        else:
            logger.info(f"chatbot >> webhook >> Request Body >> Request Body: (Empty)")

    print("--- ✅ CAPTURE COMPLETE ---")

    # 3. Always return success and 200 OK (status_code is set in the decorator)
    return {
        "status_code": status.HTTP_200_OK,
        "message": "Request successfull.",
    }
