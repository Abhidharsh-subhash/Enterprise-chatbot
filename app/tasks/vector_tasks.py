from app.core.celery_app import celery_app
from app.utils.text_utils import extract_text_from_file, split_text_into_chunks
from app.utils.embeddings import get_embedding
from app.vector_store.chrome_store import add_to_vector_db
import os
from datetime import datetime, timezone


@celery_app.task
def process_file_task(file_path: str):
    """Background task for extracting, chunking, embedding and storing file."""
    try:
        file_name = os.path.basename(file_path)
        text = extract_text_from_file(file_path)
        chunks = split_text_into_chunks(text)
        embeddings = [get_embedding(chunk) for chunk in chunks]
        metadatas = [
            {
                "file_name": file_name,
                "chunk_index": i,
                "upload_time": datetime.now(timezone.utc).isoformat(),
                "source_path": file_path,
            }
            for i in range(len(chunks))
        ]
        add_to_vector_db(chunks, embeddings, metadatas)
        print(
            f"✅ File processed successfully: {file_path} ({len(chunks)} chunks added)"
        )
    except Exception as e:
        print(f"❌ Error processing file {file_path}: {str(e)}")
