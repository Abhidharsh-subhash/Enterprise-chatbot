import os
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    status,
    Depends,
)
from app.tasks.vector_tasks import process_file_task
from app.models.users import Users
from app.dependencies import get_current_user
from app.core.logger import logger
import uuid
from typing import Dict, Any, Optional
from app.schemas.vector import AskRequest, AskResponse, DataResult
from app.services.query_service import query_service

router = APIRouter(prefix="/vector", tags=["convertion"])


@router.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...), current_user: Users = Depends(get_current_user)
):
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Send Celery task to background
        process_file_task.delay(file_path, current_user.id)

        return {
            "status": status.HTTP_201_CREATED,
            "message": "File uploaded successfully. Processing in background.",
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    payload: AskRequest,
    current_user: Users = Depends(get_current_user),
):
    """
    Ask a question about your uploaded data.

    - For Excel files: Uses SQL queries for accurate aggregations and filtering
    - For documents: Uses RAG (Retrieval Augmented Generation)

    Supports follow-up questions within the same session.
    """
    try:
        result = await query_service.process_question(
            question=payload.question,
            user_id=str(current_user.id),
            session_id=payload.session_id,
            top_k=payload.top_k,
            temperature=payload.temperature,
            show_sql=payload.show_sql,
        )

        # Build response
        data_result = None
        if result.get("data"):
            data_result = DataResult(
                columns=result["data"]["columns"],
                rows=result["data"]["rows"],
                row_count=result["data"]["row_count"],
            )

        return AskResponse(
            status_code=status.HTTP_200_OK,
            success=result["success"],
            answer=result["answer"],
            query_type=result.get("query_type", "unknown"),
            data=data_result,
            sql=result.get("sql"),
            sources=result.get("sources"),
            session_id=result["session_id"],
            error=result.get("error"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}",
        )


@router.get("/tables")
async def list_user_tables(
    current_user: Users = Depends(get_current_user),
):
    """List all Excel tables available for the current user"""
    from app.vector_store.sqlite_store import sqlite_store

    tables = sqlite_store.get_user_tables(str(current_user.id))

    return {
        "tables": [
            {
                "table_name": t["table_name"],
                "original_filename": t["original_filename"],
                "sheet_name": t["sheet_name"],
                "row_count": t["row_count"],
                "columns": [c["name"] for c in t["columns"]],
                "created_at": t["created_at"],
            }
            for t in tables
        ]
    }


@router.get("/tables/{table_name}/preview")
async def preview_table(
    table_name: str,
    limit: int = 10,
    current_user: Users = Depends(get_current_user),
):
    """Preview data from a specific table"""
    from app.vector_store.sqlite_store import sqlite_store

    # Verify table belongs to user
    user_tables = sqlite_store.get_user_tables(str(current_user.id))
    table_names = [t["table_name"] for t in user_tables]

    if table_name not in table_names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Table not found or access denied",
        )

    # Get sample data
    sample_df = sqlite_store.get_sample_data(table_name, limit)

    if sample_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not retrieve table data",
        )

    return {
        "table_name": table_name,
        "columns": list(sample_df.columns),
        "rows": sample_df.to_dict("records"),
        "row_count": len(sample_df),
    }


@router.post("/clear-session")
async def clear_session(
    session_id: Optional[str] = None,
    current_user: Users = Depends(get_current_user),
):
    """Clear conversation history for a session"""
    from app.services.session_manager import session_manager

    if session_id:
        # Clear specific session
        session_manager.sessions.pop(session_id, None)
        return {"message": f"Session {session_id} cleared"}

    return {"message": "No session specified"}


# Add to vector_routes.py


@router.get("/tables/{table_name}/columns")
async def get_table_column_info(
    table_name: str,
    current_user: Users = Depends(get_current_user),
):
    """Get detailed column information including distinct values for categorical columns"""
    from app.vector_store.sqlite_store import sqlite_store
    from app.utils.column_analyzer import column_analyzer

    # Verify table belongs to user
    user_tables = sqlite_store.get_user_tables(str(current_user.id))
    table_info = None
    for t in user_tables:
        if t["table_name"] == table_name:
            table_info = t
            break

    if not table_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Table not found or access denied",
        )

    # Get column context
    context = column_analyzer.get_column_context(
        table_name=table_name,
        columns=table_info.get("columns", []),
        user_id=str(current_user.id),
    )

    return {"table_name": table_name, "column_context": context}
