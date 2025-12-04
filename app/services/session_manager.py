from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import threading


class SessionManager:
    """
    Manages conversation sessions for contextual follow-up questions.
    Uses in-memory storage (consider Redis for production).
    """

    def __init__(self, max_history: int = 10, session_ttl_hours: int = 24):
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._lock = threading.Lock()

    def get_or_create_session(self, session_id: Optional[str], user_id: str) -> str:
        """Get existing session or create new one"""
        with self._lock:
            # Clean expired sessions periodically
            self._cleanup_expired()

            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                # Verify session belongs to user
                if session["user_id"] == str(user_id):
                    session["last_accessed"] = datetime.utcnow()
                    return session_id

            # Create new session
            new_session_id = str(uuid.uuid4())
            self.sessions[new_session_id] = {
                "user_id": str(user_id),
                "messages": [],
                "context": {},  # Store relevant table info, etc.
                "created_at": datetime.utcnow(),
                "last_accessed": datetime.utcnow(),
            }
            return new_session_id

    def add_message(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None
    ):
        """Add message to session history"""
        with self._lock:
            if session_id not in self.sessions:
                return

            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }

            self.sessions[session_id]["messages"].append(message)

            # Trim to max history
            if len(self.sessions[session_id]["messages"]) > self.max_history * 2:
                # Keep last N exchanges (user + assistant pairs)
                self.sessions[session_id]["messages"] = self.sessions[session_id][
                    "messages"
                ][-self.max_history * 2 :]

    def get_history(self, session_id: str, max_messages: int = None) -> List[Dict]:
        """Get conversation history"""
        with self._lock:
            if session_id not in self.sessions:
                return []

            messages = self.sessions[session_id]["messages"]
            if max_messages:
                return messages[-max_messages:]
            return messages

    def set_context(self, session_id: str, key: str, value: any):
        """Store context information for session"""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]["context"][key] = value

    def get_context(self, session_id: str, key: str) -> Optional[any]:
        """Get context information from session"""
        with self._lock:
            if session_id in self.sessions:
                return self.sessions[session_id]["context"].get(key)
            return None

    def _cleanup_expired(self):
        """Remove expired sessions"""
        now = datetime.utcnow()
        expired = [
            sid
            for sid, session in self.sessions.items()
            if now - session["last_accessed"] > self.session_ttl
        ]
        for sid in expired:
            del self.sessions[sid]


# Singleton instance
session_manager = SessionManager()
