from app.core.celery_app import celery_app
from app.utils.text_utils import extract_text_from_file, split_text_into_chunks
from app.utils.embeddings import get_embedding
from app.vector_store.chrome_store import add_to_vector_db
import os
from datetime import datetime, timezone
import uuid


@celery_app.task
def process_file_task(file_path: str):
    """Background task for extracting, chunking, embedding and storing file."""
    try:
        file_name = os.path.basename(file_path)
        text = extract_text_from_file(file_path)
        print(f"text extraction is completed\n {text}")
        chunks = split_text_into_chunks(text)
        print(f"chunk splitting is completed\n {chunks}")
        embeddings = [get_embedding(chunk) for chunk in chunks]
        print(f"get embedding completed\n {embeddings}")

        now_iso = datetime.now(timezone.utc).isoformat()
        doc_id = f"{file_name}:{uuid.uuid4().hex}"

        metadatas = [
            {
                "file_name": file_name,
                "chunk_index": i,
                "upload_time": now_iso,
                "source_path": file_path,
                "doc_id": doc_id,
            }
            for i in range(len(chunks))
        ]
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]  # unique IDs
        print("adding to vectordb is calling")
        add_to_vector_db(chunks, embeddings, metadatas, ids)
        print("adding to vector db is completed")
        print(
            f"✅ File processed successfully: {file_path} ({len(chunks)} chunks added)"
        )
    except Exception as e:
        print(f"❌ Error processing file {file_path}: {str(e)}")
