from app.core.celery_app import celery_app
from app.utils.text_utils import extract_text_from_file, split_text_into_chunks
from app.utils.embeddings import get_embedding
from app.vector_store.chrome_store import add_to_vector_db
from app.db.postgres.sync_database import SyncSessionLocal
from app.models.files import UploadedFiles
from datetime import datetime, timezone
import os
import uuid


@celery_app.task
def process_file_task(file_path: str, user_id: str):
    """Background task for extracting, chunking, embedding, and storing file."""
    db = SyncSessionLocal()  # ✅ real sync session
    try:
        file_name = os.path.basename(file_path)
        print(f"background task started for the file {file_name} of user {user_id}")

        # --- text and embedding generation ---
        text = extract_text_from_file(file_path)
        print(f"text extraction completed")

        chunks = split_text_into_chunks(text)
        print(f"chunk splitting completed: {len(chunks)} chunks")

        embeddings = [get_embedding(chunk) for chunk in chunks]
        print("embeddings generated successfully")

        # --- prepare metadata for vector DB ---
        now_iso = datetime.now(timezone.utc).isoformat()
        doc_id = f"{file_name}:{uuid.uuid4().hex}"

        metadatas = [
            {
                "file_name": file_name,
                "chunk_index": i,
                "upload_time": now_iso,
                "source_path": file_path,
                "doc_id": doc_id,
                "user_id": str(user_id),
            }
            for i in range(len(chunks))
        ]
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]

        print("adding to vector DB…")
        add_to_vector_db(chunks, embeddings, metadatas, ids)
        print("✅ added to vector DB successfully")

        # --- record upload in UploadedFiles ---
        uploaded_record = UploadedFiles(
            user_id=user_id,
            original_filename=file_name,
            unique_filename=doc_id,
        )
        db.add(uploaded_record)
        db.commit()
        db.refresh(uploaded_record)
        print(
            f"📁 Added UploadedFiles record with id: {uploaded_record.id} of user {user_id}"
        )

    except Exception as e:
        db.rollback()
        print(f"❌ Error processing {file_name} with user {user_id}: {str(e)}")

    finally:
        db.close()
        print("🔒 DB connection closed")
