import os
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from app.tasks.vector_tasks import process_file_task
from app.models.users import Users
from app.dependencies import get_current_user
from app.schemas.vector import AskRequest
from app.vector_store.chrome_store import query_user_vectors
from app.utils.embeddings import get_embedding
import openai

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
    payload: AskRequest, current_user: Users = Depends(get_current_user)
):
    """
    RAG endpoint:
    - Embed the question
    - Query Chroma for this user's docs
    - Build a context from top_k chunks
    - Ask OpenAI to answer using only the provided context
    """
    try:
        # 1) Embed the query
        q_emb = get_embedding(payload.question)

        # 2) Retrieve user-specific context from Chroma
        results = query_user_vectors(q_emb, str(current_user.id), top_k=payload.top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        if not docs:
            raise HTTPException(
                status_code=404,
                detail="No matching context found for this user. Upload a file first.",
            )

        # 3) Build a concise context block with a character ceiling
        context_parts = []
        used = 0
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            header = f"Source [{i+1}] | file: {meta.get('file_name')} | chunk: {meta.get('chunk_index')}"
            block = f"{header}\n{doc}"
            if used + len(block) > payload.max_context_chars:
                break
            context_parts.append(block)
            used += len(block)

        context = "\n\n---\n\n".join(context_parts)

        # 4) Ask OpenAI using only provided context
        system_prompt = (
            "You are a helpful assistant that answers the user's question using ONLY the provided context. "
            "If the answer is not contained in the context, say you don't know. "
            "Cite sources inline as [<number>] where <number> matches the 'Source [n]' blocks."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=payload.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content
        # print(f"answer from llm is \n {answer}")

        # 5) Return answer + useful source metadata
        sources = []
        for i, (meta, dist, id_) in enumerate(zip(metas, distances, ids)):
            sources.append(
                {
                    "rank": i + 1,
                    "id": id_,
                    "file_name": meta.get("file_name"),
                    "chunk_index": meta.get("chunk_index"),
                    "doc_id": meta.get("doc_id"),
                    "upload_time": meta.get("upload_time"),
                    # Convert cosine distance to similarity (1 - distance)
                    "similarity": (1 - dist) if dist is not None else None,
                }
            )

        return {"status_code": status.HTTP_200_OK, "answer": answer, "sources": sources}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
