from app.core.celery_app import celery_app
from app.utils.text_utils import (
    extract_text_from_file,
    split_text_into_chunks,
    create_schema_embedding_text,
    is_excel_file,
)
from app.utils.embeddings import get_embedding
from app.vector_store.chrome_store import add_to_vector_db, add_excel_schema
from app.vector_store.sqlite_store import sqlite_store
from app.db.postgres.sync_database import SyncSessionLocal
from app.models.files import UploadedFiles
from datetime import datetime, timezone
import os
import uuid


@celery_app.task
def process_file_task(file_path: str, user_id: str):
    """Background task for processing uploaded files."""

    db = SyncSessionLocal()

    try:
        file_name = os.path.basename(file_path)
        doc_id = f"{file_name}:{uuid.uuid4().hex}"

        print(f"🚀 Background task started for: {file_name} (user: {user_id})")

        # ════════════════════════════════════════════════════════
        # CHECK FILE TYPE AND PROCESS ACCORDINGLY
        # ════════════════════════════════════════════════════════

        if is_excel_file(file_path):
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # EXCEL FILE: Use SQLite + Schema Embedding approach
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            _process_excel_file(file_path, user_id, doc_id, file_name)
        else:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # OTHER FILES: Use original chunking + embedding approach
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            _process_document_file(file_path, user_id, doc_id, file_name)

        # ════════════════════════════════════════════════════════
        # RECORD IN DATABASE
        # ════════════════════════════════════════════════════════
        uploaded_record = UploadedFiles(
            user_id=user_id,
            original_filename=file_name,
            unique_filename=doc_id,
        )
        db.add(uploaded_record)
        db.commit()
        db.refresh(uploaded_record)

        print(
            f"✅ Successfully processed: {file_name} (record id: {uploaded_record.id})"
        )

    except Exception as e:
        db.rollback()
        print(f"❌ Error processing {file_path}: {str(e)}")
        raise

    finally:
        db.close()
        print("🔒 DB connection closed")


def _process_excel_file(file_path: str, user_id: str, doc_id: str, file_name: str):
    """
    Process Excel file using SQLite approach.

    1. Store actual DATA in SQLite (for SQL queries)
    2. Store SCHEMA in ChromaDB (for finding relevant tables)
    """
    print(f"📊 Processing Excel file: {file_name}")

    # ────────────────────────────────────────────────────────
    # STEP 1: Import data into SQLite
    # ────────────────────────────────────────────────────────
    print("  → Importing data to SQLite...")
    created_tables = sqlite_store.import_excel(
        file_path=file_path, user_id=user_id, doc_id=doc_id
    )
    print(f"  ✓ Created {len(created_tables)} table(s) in SQLite")

    # ────────────────────────────────────────────────────────
    # STEP 2: Create embeddings for SCHEMAS only
    # ────────────────────────────────────────────────────────
    print("  → Creating schema embeddings...")

    for schema_info in created_tables:
        # Create text for embedding (schema description, not data!)
        schema_text = create_schema_embedding_text(schema_info)

        # Generate embedding for schema
        schema_embedding = get_embedding(schema_text)

        # Store in ChromaDB
        add_excel_schema(
            schema_info=schema_info, embedding=schema_embedding, schema_text=schema_text
        )

        print(
            f"    ✓ Schema embedded: {schema_info['table_name']} "
            f"({schema_info['row_count']} rows, "
            f"{len(schema_info['columns'])} columns)"
        )

    print(f"  ✓ Excel processing complete")


def _process_document_file(file_path: str, user_id: str, doc_id: str, file_name: str):
    """
    Process non-Excel files using original chunking approach.
    (PDFs, Word docs, text files, etc.)
    """
    print(f"📄 Processing document file: {file_name}")

    # ────────────────────────────────────────────────────────
    # STEP 1: Extract text
    # ────────────────────────────────────────────────────────
    print("  → Extracting text...")
    text = extract_text_from_file(file_path)
    print(f"  ✓ Extracted {len(text)} characters")

    # ────────────────────────────────────────────────────────
    # STEP 2: Split into chunks
    # ────────────────────────────────────────────────────────
    print("  → Splitting into chunks...")
    chunks = split_text_into_chunks(text)
    print(f"  ✓ Created {len(chunks)} chunks")

    # ────────────────────────────────────────────────────────
    # STEP 3: Generate embeddings
    # ────────────────────────────────────────────────────────
    print("  → Generating embeddings...")
    embeddings = [get_embedding(chunk) for chunk in chunks]
    print(f"  ✓ Generated {len(embeddings)} embeddings")

    # ────────────────────────────────────────────────────────
    # STEP 4: Store in ChromaDB
    # ────────────────────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()

    metadatas = [
        {
            "file_name": file_name,
            "chunk_index": i,
            "upload_time": now_iso,
            "source_path": file_path,
            "doc_id": doc_id,
            "user_id": str(user_id),
            "file_type": "document",
        }
        for i in range(len(chunks))
    ]

    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]

    print("  → Adding to vector DB...")
    add_to_vector_db(chunks, embeddings, metadatas, ids)
    print(f"  ✓ Document processing complete")
