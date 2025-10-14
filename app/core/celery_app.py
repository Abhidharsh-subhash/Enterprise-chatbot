from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "chatbot",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.email", "app.tasks.vector_tasks"],
)

celery_app.conf.update(
    task_routes={
        "app.tasks.agent_tasks.*": {"queue": "celery"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
)
