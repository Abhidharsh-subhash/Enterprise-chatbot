import os
import chromadb
from app.core.logger import logger

CHROMA_PATH = os.path.abspath("./vector_db")


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=chromadb.Settings(anonymized_telemetry=False)
    )
    # Set space if you want cosine distance
    return client.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )


def add_to_vector_db(chunks, embeddings, metadatas, ids):
    col = get_collection()
    col.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)


def query_user_vectors(query_embedding, user_id: str, top_k: int = 5):
    col = get_collection()

    # 1. Sanitize Input (Force to list)
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    elif hasattr(query_embedding, "shape"):
        query_embedding = query_embedding.tolist()

    # 2. GUARDRAIL: Check Dimensions
    # We peek at 1 item to see what the DB expects
    try:
        # Only check if collection is not empty
        if col.count() > 0:
            # Peek returns a dictionary, get the first embedding
            existing_data = col.peek(limit=1)
            if existing_data and existing_data["embeddings"]:
                db_dim = len(existing_data["embeddings"][0])
                query_dim = len(query_embedding)

                if db_dim != query_dim:
                    error_msg = f"CRITICAL: Dimension Mismatch. DB expects {db_dim}, got {query_dim}."
                    logger.error(error_msg)
                    # Return an empty result or raise a handled error, BUT DO NOT QUERY
                    return {"documents": [], "distances": [], "metadatas": []}
    except Exception as e:
        logger.warning(f"Skipping dimension check due to error: {e}")

    # 3. Safe to Query
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": str(user_id)},
        include=["documents", "metadatas", "distances"],
    )
    return results
