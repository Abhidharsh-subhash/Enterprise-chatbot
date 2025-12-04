from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.1
    max_context_chars: int = 12000
    session_id: Optional[str] = None
    show_sql: bool = False  # Option to show generated SQL


class DataResult(BaseModel):
    """Represents query result data"""

    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int


class AskResponse(BaseModel):
    status_code: int
    success: bool
    answer: str
    query_type: str  # "excel_sql" or "document_rag"
    data: Optional[DataResult] = None
    sql: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    session_id: str
    error: Optional[str] = None


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
