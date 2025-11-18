import os
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    status,
    Depends,
    BackgroundTasks,
)
from app.tasks.vector_tasks import process_file_task
from app.models.users import Users
from app.dependencies import get_current_user
from app.schemas.vector import AskRequest
from app.vector_store.chrome_store import query_user_vectors, get_collection
from app.utils.embeddings import get_embedding
from app.rag.reasoning import compose_answer, refine_query, rank_evidence
from app.rag.memory import get_summary, get_history, messages_to_text, update_memory
from app.rag.utils import evidence_is_strong, is_idk, extract_short_definition

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


# @router.post("/ask")
# async def ask_question(
#     payload: AskRequest, current_user: Users = Depends(get_current_user)
# ):
#     try:
#         # 0) Refine question for better retrieval
#         rq = refine_query(payload.question)
#         print(f"rq: {rq}")
#         # rq = payload.question
#         variants = [payload.question, rq["rewrite"]] + rq["sub_questions"]
#         # Deduplicate and cap the number of variants t  o control cost
#         seen = set()
#         queries = []
#         for q in variants:
#             q = (q or "").strip()
#             if q and q not in seen:
#                 seen.add(q)
#                 queries.append(q)
#             if len(queries) >= 3:  # at most 3 variants
#                 break

#         # 1) Retrieve for each query variant
#         all_hits = []
#         for q in queries:
#             q_emb = get_embedding(q)
#             results = query_user_vectors(
#                 q_emb, str(current_user.id), top_k=max(2, payload.top_k)
#             )
#             docs = results.get("documents", [[]])[0]
#             metas = results.get("metadatas", [[]])[0]
#             distances = results.get("distances", [[]])[0]
#             ids = results.get("ids", [[]])[0]

#             for doc, meta, dist, _id in zip(docs, metas, distances, ids):
#                 sim = (1 - dist) if dist is not None else None
#                 all_hits.append(
#                     {
#                         "id": _id,
#                         "text": doc,
#                         "file_name": meta.get("file_name"),
#                         "chunk_index": meta.get("chunk_index"),
#                         "doc_id": meta.get("doc_id"),
#                         "upload_time": meta.get("upload_time"),
#                         "similarity": sim,
#                     }
#                 )

#         if not all_hits:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No matching context found. Upload a file first.",
#             )

#         # 2) Deduplicate by ID and keep top by similarity
#         dedup = {}
#         for h in all_hits:
#             existing = dedup.get(h["id"])
#             if not existing or (h["similarity"] or 0) > (existing["similarity"] or 0):
#                 dedup[h["id"]] = h
#         candidates = list(dedup.values())
#         candidates.sort(key=lambda x: (x.get("similarity") or 0), reverse=True)
#         candidates = candidates[: max(10, payload.top_k)]  # trim candidate pool

#         # 3) Rank evidence with a brief reason (not full chain-of-thought)
#         ranked = rank_evidence(payload.question, candidates)
#         top = ranked[: payload.top_k]

#         # 4) Build context blocks with a character ceiling
#         context_parts = []
#         used = 0
#         for i, c in enumerate(top):
#             header = f"Source [{i+1}] | file: {c.get('file_name')} | chunk: {c.get('chunk_index')}"
#             block = f"{header}\n{c['text']}"
#             if used + len(block) > payload.max_context_chars:
#                 break
#             context_parts.append(block)
#             used += len(block)

#         # 5) Compose the final answer (no citations, no CoT leakage)
#         answer, brief_explanation = compose_answer(
#             payload.question, context_parts, temperature=payload.temperature
#         )

#         # 6) Build sources payload
#         sources = []
#         for i, c in enumerate(top):
#             sources.append(
#                 {
#                     "rank": i + 1,
#                     "id": c["id"],
#                     "file_name": c.get("file_name"),
#                     "chunk_index": c.get("chunk_index"),
#                     "doc_id": c.get("doc_id"),
#                     "upload_time": c.get("upload_time"),
#                     "similarity": c.get("similarity"),
#                     "rank_reason": c.get(
#                         "rank_reason"
#                     ),  # short reason from ranking step
#                     "rank_score": c.get("rank_score"),
#                 }
#             )

#         return {
#             "status_code": status.HTTP_200_OK,
#             "answer": answer,
#             "brief_explanation": brief_explanation,  # short rationale, optional
#             "sources": sources,
#             "query_variants_used": queries,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# @router.post("/ask")
# async def ask_question(
#     payload: AskRequest, current_user: Users = Depends(get_current_user)
# ):
#     try:
#         q = (payload.question or "").strip()
#         if not q:
#             raise HTTPException(status_code=400, detail="Question is required.")

#         # 1) Embed raw question and fetch top-2
#         q_emb = get_embedding(q)
#         results = query_user_vectors(q_emb, str(current_user.id), top_k=2)

#         docs = results.get("documents", [[]])[0]
#         metas = results.get("metadatas", [[]])[0]
#         distances = results.get("distances", [[]])[0]
#         ids = results.get("ids", [[]])[0]

#         if not docs:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No matching context found. Upload a file first.",
#             )

#         # 2) Build context blocks and sources
#         context_parts = []
#         sources = []
#         for i, (doc, meta, dist, _id) in enumerate(
#             zip(docs, metas, distances, ids), start=1
#         ):
#             sim = (1 - (dist or 0)) if dist is not None else None
#             header = f"Source [{i}] | file: {meta.get('file_name')} | chunk: {meta.get('chunk_index')}"
#             context_parts.append(f"{header}\n{doc}")
#             sources.append(
#                 {
#                     "rank": i,
#                     "id": _id,
#                     "file_name": meta.get("file_name"),
#                     "chunk_index": meta.get("chunk_index"),
#                     "doc_id": meta.get("doc_id"),
#                     "upload_time": meta.get("upload_time"),
#                     "similarity": sim,
#                 }
#             )

#         # Optional: cap context length if you kept max_context_chars in your schema
#         max_chars = getattr(payload, "max_context_chars", None)
#         if max_chars:
#             used = 0
#             trimmed = []
#             for block in context_parts:
#                 if used + len(block) > max_chars:
#                     break
#                 trimmed.append(block)
#                 used += len(block)
#             context_parts = trimmed

#         answer, brief_explanation = compose_answer(
#             payload.question, context_parts, temperature=payload.temperature
#         )
#         return {
#             "status_code": status.HTTP_200_OK,
#             "answer": answer,
#             "brief_explanation": brief_explanation,
#             "sources": sources,
#             "query_used": q,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# @router.post("/ask")
# async def ask_question(
#     payload: AskRequest,
#     background_tasks: BackgroundTasks,
#     current_user: Users = Depends(get_current_user),
# ):
#     try:
#         q = (payload.question or "").strip()
#         if not q:
#             raise HTTPException(status_code=400, detail="Question is required.")

#         user_id = str(current_user.id)

#         # A) Load memory (summary + recent buffer)
#         conv_summary = get_summary(user_id, payload.session_id)
#         recent_text = ""
#         try:
#             recent_text = messages_to_text(get_history(user_id, payload.session_id).messages)
#         except Exception:
#             pass

#         # B) Embed + retrieve user-scoped docs (unchanged)
#         q_emb = get_embedding(q)
#         results = query_user_vectors(q_emb, user_id, top_k=2)

#         docs = results.get("documents", [[]])[0]
#         metas = results.get("metadatas", [[]])[0]
#         distances = results.get("distances", [[]])[0]
#         ids = results.get("ids", [[]])[0]

#         if not docs:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No matching context found. Upload a file first.",
#             )

#         # C) Build context blocks (prepend summary and a tiny recent buffer)
#         context_parts = []
#         if conv_summary.strip():
#             context_parts.append(f"Conversation summary:\n{conv_summary}")
#         if recent_text.strip():
#             context_parts.append(f"Recent chat excerpt:\n{recent_text}")

#         sources = []
#         for i, (doc, meta, dist, _id) in enumerate(zip(docs, metas, distances, ids), start=1):
#             sim = (1 - (dist or 0)) if dist is not None else None
#             header = f"Source [{i}] | file: {meta.get('file_name')} | chunk: {meta.get('chunk_index')}"
#             context_parts.append(f"{header}\n{doc}")
#             sources.append(
#                 {
#                     "rank": i,
#                     "id": _id,
#                     "file_name": meta.get("file_name"),
#                     "chunk_index": meta.get("chunk_index"),
#                     "doc_id": meta.get("doc_id"),
#                     "upload_time": meta.get("upload_time"),
#                     "similarity": sim,
#                 }
#             )

#         # D) Optional: cap prompt size
#         max_chars = getattr(payload, "max_context_chars", None)
#         if max_chars:
#             used, trimmed = 0, []
#             for block in context_parts:
#                 if used + len(block) > max_chars:
#                     break
#                 trimmed.append(block)
#                 used += len(block)
#             context_parts = trimmed

#         # E) Ask the model (your existing RAG composer)
#         answer, brief_explanation = compose_answer(
#             payload.question, context_parts, temperature=payload.temperature
#         )

#         # F) Update memory asynchronously (summary + history)
#         background_tasks.add_task(update_memory, user_id, payload.session_id, q, answer)

#         return {
#             "status_code": status.HTTP_200_OK,
#             "answer": answer,
#             "brief_explanation": brief_explanation,
#             "sources": sources,
#             "query_used": q,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

# vector_routes.py (diff-style)


@router.post("/ask")
async def ask_question(
    payload: AskRequest,
    background_tasks: BackgroundTasks,
    current_user: Users = Depends(get_current_user),
):
    try:
        q = (payload.question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Question is required.")
        user_id = str(current_user.id)

        # A) Load memory (summary + recent buffer)
        conv_summary = get_summary(user_id, payload.session_id)
        recent_text = ""
        try:
            recent_text = messages_to_text(
                get_history(user_id, payload.session_id).messages
            )
        except Exception:
            pass

        # B) Query rewrite + multi-query retrieval
        r = refine_query(q)  # { rewrite, sub_questions[], keywords[] }
        search_queries = [q, r.get("rewrite")] + r.get("sub_questions", [])
        search_queries = [s for s in search_queries if s]
        # Dedup while preserving order; cap to 3 distinct queries
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

        TOP_K = 6  # give the reranker more candidates
        for sq in search_queries:
            emb = get_embedding(sq)
            res = query_user_vectors(emb, user_id, top_k=TOP_K)
            add_results(res)

        if not candidates_map:
            raise HTTPException(
                status_code=404,
                detail="No matching context found. Upload a file first.",
            )

        # C) Expand neighbors around top hits (prev/next chunks)
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
                    "similarity": None,  # neighbor added without a sim score
                }

        candidates = list(candidates_map.values())

        # D) LLM rerank to find the best small set
        ranked = rank_evidence(q, candidates)[:4]

        # E) Guard: if everything is weak, ask for clarification or return 404
        max_sim = max((c.get("similarity") or 0) for c in ranked) if ranked else 0.0
        if max_sim < 0.40 and not any(evidence_is_strong(c) for c in ranked):
            return {
                "status_code": status.HTTP_200_OK,
                "answer": "I don't know",
            }

        # F) Build context (prepend summary + recent chat)
        context_parts = []
        if conv_summary.strip():
            context_parts.append(f"Conversation summary:\n{conv_summary}")
        if recent_text.strip():
            context_parts.append(f"Recent chat excerpt:\n{recent_text}")

        sources = []
        for i, c in enumerate(ranked, start=1):
            header = f"Source [{i}] | file: {c.get('file_name')} | chunk: {c.get('chunk_index')}"
            context_parts.append(f"{header}\n{c.get('text','')}")
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

        # G) Decide if we should "must answer"
        must_answer = any(evidence_is_strong(c) for c in ranked)

        # H) Compose the answer using BOTH original and rewrite as hints
        answer, brief_explanation = compose_answer(
            original_question=q,
            context_blocks=context_parts,
            rewrite=r.get("rewrite"),
            sub_questions=r.get("sub_questions", []),
            temperature=payload.temperature,
            must_answer=must_answer,
        )

        # I) Fallback: extract a short definition if model still says "I don't know" but evidence is strong
        fallback_used = False
        if is_idk(answer) and must_answer and ranked:
            # If the original query is very short, try extractive fallback from top ranked
            term = q.strip().strip("?.!").lower()
            if term and len(term.split()) <= 3:
                top_text = ranked[0].get("text", "") or ""
                extracted = extract_short_definition(term, top_text)
                if extracted:
                    answer = extracted
                    brief_explanation = "Extracted from the most relevant snippet."
                    fallback_used = True

        # J) Update memory async
        background_tasks.add_task(update_memory, user_id, payload.session_id, q, answer)

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
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
