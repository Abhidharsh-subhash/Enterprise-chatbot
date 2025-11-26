import os
import chromadb
from app.core.logger import logger
from app.core.config import settings

CHROMA_PATH = os.path.abspath(f"./{settings.vector_database}")


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

    # 3. Safe to Query
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": str(user_id)},
        include=["documents", "metadatas", "distances"],
    )
    return results
