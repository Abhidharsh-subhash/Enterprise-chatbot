import os
import chromadb

CHROMA_PATH = os.path.abspath("./vector_db")


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Set space if you want cosine distance
    return client.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )


def add_to_vector_db(chunks, embeddings, metadatas, ids):
    col = get_collection()
    col.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
