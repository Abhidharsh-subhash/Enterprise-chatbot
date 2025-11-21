import os
import chromadb
from app.core.logger import logger
import numpy as np

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
    logger.info(f"before converting the type of embedding is {type(query_embedding)}")
    # --- CRITICAL FIX: SANITIZE INPUT ---
    # The C++ layer crashes if it gets a Numpy Array or Tensor directly.
    # We must force it into a standard Python List[float].

    # 1. Convert from Tensor/Numpy to List
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    elif isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    # 2. Ensure it's a flat list of floats (not a list of lists for a single query)
    # If embedding came in as [[0.1, 0.2]], flatten it to [0.1, 0.2]
    if len(query_embedding) > 0 and isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]

    logger.info(f"Sanitized query_embedding type: {type(query_embedding)}")
    # -------
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
