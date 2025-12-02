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
from app.rag.pandas_executor import execute_pandas_retrieval
from app.core.openai_client import client, CHAT_MODEL

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
#     payload: AskRequest,
#     background_tasks: BackgroundTasks,
#     current_user: Users = Depends(get_current_user),
# ):
#     try:
#         logger.info(f"asked the question")
#         q = (payload.question or "").strip()
#         if not q:
#             raise HTTPException(status_code=400, detail="Question is required.")
#         user_id = str(current_user.id)
#         logger.info(f"user_id fetched is {user_id}")

#         # 0) Session management: create a session_id if none provided
#         session_id = (payload.session_id or "").strip() or str(uuid.uuid4())
#         session_created = not bool(payload.session_id)
#         logger.info("handled the session_id")

#         # A) Memory (scoped to session_id)
#         conv_summary = get_summary(user_id, session_id)
#         try:
#             recent_text = messages_to_text(get_history(user_id, session_id).messages)
#         except Exception:
#             recent_text = ""
#         logger.info("conv summary if fetched")

#         # Build a compact hint for the refiner from summary + recent turns
#         conv_hint = build_conv_hint(conv_summary, recent_text)
#         logger.info("conv_hint is created")

#         logger.info("Before refine_query_with_history")
#         # B) Query rewrite WITH conversation hint (rewrite used only for retrieval)
#         try:
#             r = refine_query_with_history(q, conversation_hint=conv_hint)
#         except Exception as e:
#             logger.error(f"refine_query_with_history failed: {e}")
#             r = {"rewrite": q, "sub_questions": [], "keywords": []}
#         logger.info(f"refined query with history : {r}")

#         search_queries = [q, r.get("rewrite")] + r.get("sub_questions", [])
#         search_queries = [s for s in search_queries if s]
#         seen = set()
#         search_queries = [s for s in search_queries if not (s in seen or seen.add(s))][
#             :3
#         ]

#         col = get_collection()
#         candidates_map = {}

#         def add_results(res):
#             docs = res.get("documents", [[]])[0]
#             metas = res.get("metadatas", [[]])[0]
#             dists = res.get("distances", [[]])[0]
#             ids = res.get("ids", [[]])[0]
#             for doc, meta, dist, _id in zip(docs, metas, dists, ids):
#                 sim = (1 - (dist or 0)) if dist is not None else None
#                 existing = candidates_map.get(_id)
#                 if not existing:
#                     candidates_map[_id] = {
#                         "id": _id,
#                         "text": doc,
#                         "file_name": meta.get("file_name"),
#                         "chunk_index": meta.get("chunk_index"),
#                         "doc_id": meta.get("doc_id"),
#                         "upload_time": meta.get("upload_time"),
#                         "similarity": sim,
#                     }
#                 else:
#                     if sim is not None:
#                         existing["similarity"] = max(
#                             existing.get("similarity") or 0, sim
#                         )

#         logger.info(f"Search queries: {search_queries}")
#         for sq in search_queries:
#             try:
#                 emb = get_embedding(sq)
#                 logger.info(f"get embedding has no error")
#             except Exception as e:
#                 logger.error(f"get embedding error: {e}")
#                 raise
#             try:
#                 res = query_user_vectors(emb, user_id, top_k=4)
#                 logger.info(f"query user vecotrs has no error")
#             except Exception as e:
#                 logger.error(f"vector DB query error: {e}")
#                 raise
#             add_results(res)
#         # print(f"emb is {emb}")
#         # print(f"res is {res}")

#         if not candidates_map:
#             return {
#                 "status_code": status.HTTP_200_OK,
#                 "answer": "No matching context found. Upload a file first.",
#             }

#         # C) Neighbor expansion
#         def expand_neighbor_ids(_id: str, window: int = 1):
#             try:
#                 base, idx = _id.rsplit(":", 1)
#                 idx = int(idx)
#             except Exception:
#                 return []
#             return [
#                 f"{base}:{j}" for j in range(max(0, idx - window), idx + window + 1)
#             ]

#         neighbor_ids = set()
#         for _id in list(candidates_map.keys()):
#             neighbor_ids.update(expand_neighbor_ids(_id, window=1))

#         missing_neighbors = list(neighbor_ids.difference(candidates_map.keys()))
#         if missing_neighbors:
#             nb = col.get(ids=missing_neighbors, include=["documents", "metadatas"])
#             for _id, doc, meta in zip(
#                 nb.get("ids", []), nb.get("documents", []), nb.get("metadatas", [])
#             ):
#                 candidates_map[_id] = {
#                     "id": _id,
#                     "text": doc,
#                     "file_name": meta.get("file_name"),
#                     "chunk_index": meta.get("chunk_index"),
#                     "doc_id": meta.get("doc_id"),
#                     "upload_time": meta.get("upload_time"),
#                     "similarity": None,
#                 }

#         candidates = list(candidates_map.values())

#         # D) Rerank
#         try:
#             ranked = rank_evidence(q, candidates)[:3]
#         except Exception as e:
#             logger.error(f"rank_evidence failed: {e}")
#             ranked = sorted(
#                 candidates, key=lambda c: c.get("similarity") or 0, reverse=True
#             )[:3]

#         # =========================================================================
#         # NEW LOGIC: EXCEL INTERCEPTOR
#         # =========================================================================

#         # 1. Check the single best file
#         top_candidate = ranked[0] if ranked else None
#         file_name = top_candidate.get("file_name", "") if top_candidate else ""
#         is_excel = file_name.lower().endswith((".xlsx", ".xls"))

#         if is_excel:
#             logger.info(f"Excel Detected ({file_name}). Switching to Pandas Executor.")
#             try:
#                 # 2. Execute Pandas to get the RAW data (e.g., "9840204060" or "419")
#                 raw_data = execute_pandas_retrieval(file_name, q)

#                 # 2. Generate a Precise Response
#                 # We change the prompt to allow Tables and Lists
#                 nl_system_prompt = (
#                     "You are a precise data assistant. "
#                     "I will provide raw data retrieved from an Excel file based on a user's query. "
#                     "\n\nRULES:"
#                     "\n1. If the data contains multiple records/rows, format them as a clean **Markdown Table** or a structured list. Do NOT summarize vague details (e.g. do not say 'There are 3 records', show the records)."
#                     "\n2. If the data is a single value (like a count or one specific phone number), state it in a natural sentence."
#                     "\n3. Present exactly what is returned in the raw data without omitting columns."
#                 )

#                 nl_user_prompt = f"User Query: {q}\n\nRaw Database Result:\n{raw_data}"

#                 nl_response = client.chat.completions.create(
#                     model=CHAT_MODEL,
#                     messages=[
#                         {"role": "system", "content": nl_system_prompt},
#                         {"role": "user", "content": nl_user_prompt},
#                     ],
#                     temperature=0,  # Slightly higher for natural flow
#                 )

#                 answer = nl_response.choices[0].message.content.strip()

#                 # 4. Return Early
#                 background_tasks.add_task(update_memory, user_id, session_id, q, answer)

#                 return {
#                     "status_code": status.HTTP_200_OK,
#                     "answer": answer,
#                     "brief_explanation": "Derived directly from Excel data.",
#                     "sources": [
#                         {"file_name": file_name, "rank": 1, "type": "pandas_exact"}
#                     ],
#                     "query_used": q,
#                     "must_answer": True,
#                     "session_id": session_id,
#                     "session_created": session_created,
#                 }

#             except Exception as pd_error:
#                 logger.error(
#                     f"Pandas execution failed: {pd_error}. Falling back to standard RAG."
#                 )
#                 # If Pandas fails, we simply do nothing here and let the code
#                 # flow down to 'E) Guard' and 'F) Build context'
#         # =========================================================================
#         # END NEW LOGIC - Standard RAG continues below
#         # =========================================================================

#         # E) Guard
#         max_sim = max((c.get("similarity") or 0) for c in ranked) if ranked else 0.0
#         if max_sim < 0.40 and not any(evidence_is_strong(c) for c in ranked):
#             return {
#                 "status_code": status.HTTP_200_OK,
#                 "answer": "I don't know.",
#                 "session_id": session_id,
#                 "session_created": session_created,
#             }

#         # F) Build context: Sources FIRST, then optional summary/recent (labelled)
#         context_parts = []
#         sources = []
#         for i, c in enumerate(ranked, start=1):
#             header = f"Source [{i}] | file: {c.get('file_name')} | chunk: {c.get('chunk_index')}"
#             context_parts.append(f"{header}\n{(c.get('text') or '')}")
#             sources.append(
#                 {
#                     "rank": i,
#                     "id": c["id"],
#                     "file_name": c.get("file_name"),
#                     "chunk_index": c.get("chunk_index"),
#                     "doc_id": c.get("doc_id"),
#                     "upload_time": c.get("upload_time"),
#                     "similarity": c.get("similarity"),
#                     "rank_score": c.get("rank_score"),
#                     "rank_reason": c.get("rank_reason"),
#                 }
#             )

#         if conv_summary.strip():
#             context_parts.append(
#                 f"Conversation summary (not a factual source):\n{conv_summary}"
#             )
#         if recent_text.strip():
#             context_parts.append(
#                 f"Recent chat excerpt (not a factual source):\n{recent_text}"
#             )

#         # Optional cap
#         max_chars = getattr(payload, "max_context_chars", None)
#         if max_chars:
#             used, trimmed = 0, []
#             for block in context_parts:
#                 if used + len(block) > max_chars:
#                     break
#                 trimmed.append(block)
#                 used += len(block)
#             context_parts = trimmed

#         # G) Decide must_answer
#         must_answer = any(evidence_is_strong(c) for c in ranked)

#         # H) Composer: strictly answer the original question; include example if asked
#         q_lower = q.lower()
#         include_example = (
#             ("example" in q_lower) or ("e.g." in q_lower) or (" eg " in f" {q_lower} ")
#         )

#         try:
#             answer, brief_explanation = compose_answer(
#                 original_question=q,
#                 context_blocks=context_parts,
#                 temperature=payload.temperature,
#                 must_answer=must_answer,
#                 include_example=include_example,
#                 max_sentences=2,
#             )
#         except Exception as e:
#             logger.error(f"compose_answer failed: {e}")
#             answer = "I'm sorry, I encountered an issue generating the answer."
#             brief_explanation = "Fallback response"

#         # I) Strip domain prefaces if the user didn't include them
#         answer = strip_domain_preface(answer, q)

#         # J) Fallbacks if still IDK but evidence is strong
#         fallback_used = False
#         if is_idk(answer) and must_answer and ranked:
#             top_snippets = [
#                 (ranked[i].get("text") or "") for i in range(min(3, len(ranked)))
#             ]
#             extracted = extractive_answer_from_snippets(q, top_snippets)
#             if extracted:
#                 answer = strip_domain_preface(extracted, q)
#                 brief_explanation = "Extracted from the most relevant snippets."
#                 fallback_used = True
#             else:
#                 term = q.strip().strip("?.!").lower()
#                 if term and len(term.split()) <= 3:
#                     defn = extract_short_definition(
#                         term, top_snippets[0] if top_snippets else ""
#                     )
#                     if defn:
#                         answer = strip_domain_preface(defn, q)
#                         brief_explanation = "Extracted from the most relevant snippet."
#                         fallback_used = True

#         # K) Update memory (scoped to this session_id)
#         background_tasks.add_task(update_memory, user_id, session_id, q, answer)

#         return {
#             "status_code": status.HTTP_200_OK,
#             "answer": answer,
#             "brief_explanation": brief_explanation,
#             "sources": sources,
#             "query_used": q,
#             "rewritten_query": r.get("rewrite"),
#             "sub_questions": r.get("sub_questions", []),
#             "must_answer": must_answer,
#             "fallback_used": fallback_used,
#             "session_id": session_id,
#             "session_created": session_created,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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

        # --- REMOVED HISTORY/CHAIN OF THOUGHT RETRIEVAL HERE ---
        # We no longer fetch get_summary, get_history, or build_conv_hint.

        # B) Search Query Setup (No history refinement)
        # Since we aren't looking at history, we don't refine the query based on previous turns.
        # We treat the current question 'q' as the standalone query.
        r = {"rewrite": q, "sub_questions": [], "keywords": []}
        logger.info(f"Query used (no history refinement): {q}")

        search_queries = [q]
        # If you have a standalone synonym expander, you could add it here,
        # but we removed the history-based refiner.

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

        logger.info(f"Search queries: {search_queries}")
        for sq in search_queries:
            try:
                emb = get_embedding(sq)
                logger.info(f"get embedding has no error")
            except Exception as e:
                logger.error(f"get embedding error: {e}")
                raise
            try:
                res = query_user_vectors(emb, user_id, top_k=4)
                logger.info(f"query user vecotrs has no error")
            except Exception as e:
                logger.error(f"vector DB query error: {e}")
                raise
            add_results(res)

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

        # In vector_routes.py - Replace the EXCEL INTERCEPTOR section with this:

        # =========================================================================
        # EXCEL INTERCEPTOR
        # =========================================================================

        # 1. Check the single best file
        top_candidate = ranked[0] if ranked else None
        file_name = top_candidate.get("file_name", "") if top_candidate else ""
        is_excel = file_name.lower().endswith((".xlsx", ".xls"))

        if is_excel:
            logger.info(f"Excel Detected ({file_name}). Switching to Pandas Executor.")
            try:
                # 2. Execute Pandas to get structured result
                pandas_result = execute_pandas_retrieval(file_name, q)

                if pandas_result.get("success"):
                    response_type = pandas_result.get("response_type")

                    # Handle TABLE responses - direct data, minimal LLM processing
                    if response_type == "table":
                        intro_message = pandas_result.get(
                            "intro_message", "Here are the results:"
                        )
                        table_markdown = pandas_result.get("table_markdown", "")
                        total_rows = pandas_result.get("total_rows", 0)

                        # Combine intro with table - NO LLM processing of the data
                        answer = f"{intro_message}\n\n{table_markdown}"

                        # Update memory
                        background_tasks.add_task(
                            update_memory, user_id, session_id, q, answer
                        )

                        return {
                            "status_code": status.HTTP_200_OK,
                            "answer": answer,
                            "response_type": "table",
                            "table_data": pandas_result.get("table_data"),
                            "table_html": pandas_result.get("table_html"),
                            "columns": pandas_result.get("columns"),
                            "total_rows": total_rows,
                            "brief_explanation": f"Retrieved {total_rows} records directly from Excel.",
                            "sources": [
                                {
                                    "file_name": file_name,
                                    "rank": 1,
                                    "type": "pandas_table",
                                }
                            ],
                            "query_used": q,
                            "must_answer": True,
                            "session_id": session_id,
                            "session_created": session_created,
                        }

                    # Handle COUNT/VALUE responses - simple answers
                    elif response_type in ["count", "value"]:
                        answer = pandas_result.get("answer", "")
                        intro = pandas_result.get("intro_message", "")

                        if intro:
                            answer = intro

                        background_tasks.add_task(
                            update_memory, user_id, session_id, q, answer
                        )

                        return {
                            "status_code": status.HTTP_200_OK,
                            "answer": answer,
                            "response_type": response_type,
                            "brief_explanation": "Calculated directly from the Excel file.",
                            "sources": [
                                {
                                    "file_name": file_name,
                                    "rank": 1,
                                    "type": "pandas_exact",
                                }
                            ],
                            "query_used": q,
                            "must_answer": True,
                            "session_id": session_id,
                            "session_created": session_created,
                        }

                # If pandas execution wasn't successful, try the old LLM approach
                else:
                    # Fall through to let the original LLM-based approach handle it
                    logger.info(
                        "Pandas returned unsuccessful result, trying LLM approach..."
                    )

                    raw_data = pandas_result.get("answer", "NO_MATCH")

                    if raw_data and raw_data != "NO_MATCH":
                        # Use the original LLM approach for formatting
                        nl_system_prompt = (
                            "You are a data analyst assistant. "
                            "I will provide a User Query and the 'Raw Execution Result' derived from running Python code on the Excel file. "
                            "\n\nRULES:"
                            "\n1. **Calculated Values**: If the result is a single number (count, sum, average), answer the user's question directly with that number in a sentence."
                            "\n2. **Data Lists**: If the result is a table/list, format it as a clean Markdown Table."
                            "\n3. **No Match**: If the raw result is exactly 'NO_MATCH' or an empty table/dataframe, clearly say there are no records matching the user query."
                            "\n4. **Errors**: If the result mentions an error, politely explain that the data couldn't be calculated."
                            "\n5. Do not hallucinate data not present in the 'Raw Execution Result'."
                        )

                        nl_user_prompt = (
                            f"User Query: {q}\n\nRaw Execution Result:\n{raw_data}"
                        )

                        nl_response = client.chat.completions.create(
                            model=CHAT_MODEL,
                            messages=[
                                {"role": "system", "content": nl_system_prompt},
                                {"role": "user", "content": nl_user_prompt},
                            ],
                            temperature=0,
                        )

                        answer = nl_response.choices[0].message.content.strip()

                        background_tasks.add_task(
                            update_memory, user_id, session_id, q, answer
                        )

                        return {
                            "status_code": status.HTTP_200_OK,
                            "answer": answer,
                            "brief_explanation": "Processed from the Excel file.",
                            "sources": [
                                {
                                    "file_name": file_name,
                                    "rank": 1,
                                    "type": "pandas_llm",
                                }
                            ],
                            "query_used": q,
                            "must_answer": True,
                            "session_id": session_id,
                            "session_created": session_created,
                        }

            except Exception as pd_error:
                logger.error(
                    f"Pandas execution failed: {pd_error}. Falling back to standard RAG."
                )
                # If pandas fails, the code will automatically continue down
                # to the 'Standard RAG' section below.

        # =========================================================================
        # Standard RAG (Contextual Fallback)
        # =========================================================================

        # E) Guard
        max_sim = max((c.get("similarity") or 0) for c in ranked) if ranked else 0.0
        if max_sim < 0.40 and not any(evidence_is_strong(c) for c in ranked):
            return {
                "status_code": status.HTTP_200_OK,
                "answer": "I don't know.",
                "session_id": session_id,
                "session_created": session_created,
            }

        # F) Build context: Sources ONLY.
        # Removed: "Conversation summary" and "Recent chat excerpt".
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
        # We still SAVE the interaction, but we didn't READ it above.
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
