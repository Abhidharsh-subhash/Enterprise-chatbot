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
    """
    Return top_k results for this user_id using the query embedding.
    """
    logger.info("before get_collection")
    col = get_collection()
    logger.info(f"the type of embedding is {type(query_embedding)}")

    try:
        peek = col.peek(limit=1)
        if peek["embeddings"]:
            expected_dim = len(peek["embeddings"][0])
            actual_dim = len(query_embedding)
            logger.info(
                f"DEBUG: Collection Dim: {expected_dim}, Query Dim: {actual_dim}"
            )

            if expected_dim != actual_dim:
                logger.error("CRITICAL: Dimension mismatch! This will crash the app.")
                return {"error": "Dimension mismatch"}  # Return early to save the app
    except Exception as e:
        logger.warning(f"Could not verify dimensions: {e}")

    try:
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": str(user_id)},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"The issue in results value is : {e}")
        raise
    logger.info("after results")
    return results
