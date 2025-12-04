import os
import chromadb
from typing import List, Dict, Any
from app.core.config import settings

CHROMA_PATH = os.path.abspath(f"./{settings.vector_database}")


def get_client():
    """Get ChromaDB client"""
    return chromadb.PersistentClient(
        path=CHROMA_PATH, settings=chromadb.Settings(anonymized_telemetry=False)
    )


def get_collection():
    """Get documents collection (for PDFs, DOCs, etc.)"""
    client = get_client()
    return client.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )


def get_excel_schema_collection():
    """Get Excel schema collection (for Excel files only)"""
    client = get_client()
    return client.get_or_create_collection(
        name="excel_schemas", metadata={"hnsw:space": "cosine"}
    )


# ============================================
# EXISTING FUNCTIONS (for PDFs, DOCs, etc.)
# ============================================


def add_to_vector_db(chunks, embeddings, metadatas, ids):
    """Add document chunks to vector DB (for non-Excel files)"""
    col = get_collection()
    col.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)


def query_user_vectors(query_embedding, user_id: str, top_k: int = 5):
    """Query document vectors for a user"""
    col = get_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": str(user_id)},
        include=["documents", "metadatas", "distances"],
    )
    return results


# ============================================
# NEW FUNCTIONS (for Excel schemas)
# ============================================


def add_excel_schema(
    schema_info: Dict[str, Any], embedding: List[float], schema_text: str
):
    """
    Add Excel table schema to vector DB.
    This stores ONLY the schema, not the actual data.
    """
    col = get_excel_schema_collection()

    # Create prompt-ready schema text
    prompt_text = _create_schema_prompt(schema_info)

    col.upsert(
        ids=[schema_info["table_name"]],
        documents=[schema_text],  # Text used for embedding
        embeddings=[embedding],
        metadatas=[
            {
                "table_name": schema_info["table_name"],
                "original_filename": schema_info["original_filename"],
                "sheet_name": schema_info["sheet_name"],
                "row_count": schema_info["row_count"],
                "user_id": str(schema_info["user_id"]),
                "doc_id": schema_info["doc_id"],
                "columns": ",".join([c["name"] for c in schema_info["columns"]]),
                "prompt_text": prompt_text,  # Ready-to-use in LLM prompt
            }
        ],
    )


def query_excel_schemas(
    query_embedding: List[float], user_id: str, top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Find relevant Excel tables based on query.
    Returns schemas that match the user's question.
    """
    col = get_excel_schema_collection()

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": str(user_id)},
        include=["metadatas", "distances"],
    )

    tables = []
    if results["ids"] and results["ids"][0]:
        for i, table_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance

            tables.append(
                {
                    "table_name": metadata["table_name"],
                    "original_filename": metadata["original_filename"],
                    "sheet_name": metadata["sheet_name"],
                    "row_count": metadata["row_count"],
                    "prompt_text": metadata["prompt_text"],
                    "columns": metadata["columns"].split(","),
                    "similarity": round(similarity, 3),
                }
            )

    return tables


def delete_excel_schemas(user_id: str, doc_id: str = None):
    """Delete Excel schemas for a user"""
    col = get_excel_schema_collection()

    try:
        if doc_id:
            # Delete specific document's schemas
            col.delete(where={"$and": [{"user_id": str(user_id)}, {"doc_id": doc_id}]})
        else:
            # Delete all user's schemas
            col.delete(where={"user_id": str(user_id)})
    except Exception as e:
        print(f"Warning: Could not delete schemas: {e}")


def _create_schema_prompt(schema_info: Dict[str, Any]) -> str:
    """Create detailed schema text for LLM prompt"""
    lines = [
        f"Table: {schema_info['table_name']}",
        f"Source: {schema_info['original_filename']} (Sheet: {schema_info['sheet_name']})",
        f"Total Rows: {schema_info['row_count']}",
        "",
        "Columns:",
    ]

    for col in schema_info["columns"]:
        line = f"  - {col['name']} ({col['type']})"
        if col.get("samples"):
            samples_str = ", ".join(str(s) for s in col["samples"][:3])
            line += f" | Examples: {samples_str}"
        if col.get("stats") and col["stats"].get("min") is not None:
            line += f" | Range: {col['stats']['min']} to {col['stats']['max']}"
        lines.append(line)

    return "\n".join(lines)


def _create_embedding_text(schema_info: Dict[str, Any]) -> str:
    """Create text optimized for embedding/semantic search"""
    parts = [
        f"Excel file {schema_info['original_filename']}",
        f"sheet {schema_info['sheet_name']}",
        f"with {schema_info['row_count']} rows.",
        "Contains columns:",
    ]

    for col in schema_info["columns"]:
        col_text = f"{col['name']} which stores {col['type']} data"
        if col.get("samples"):
            col_text += (
                f" with values like {', '.join(str(s) for s in col['samples'][:3])}"
            )
        parts.append(col_text)

    return " ".join(parts)
