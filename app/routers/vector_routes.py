import os
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    status,
    Depends,
    BackgroundTasks,
    Request,
)
from app.tasks.vector_tasks import process_file_task
from app.models.users import Users
from app.dependencies import get_current_user
from app.core.logger import logger
import uuid
from typing import Dict, Any
from app.schemas.vector import AskRequest

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


@router.post("/ask")
async def ask_question(
    payload: AskRequest,
    background_tasks: BackgroundTasks,
    current_user: Users = Depends(get_current_user),
):
    pass
