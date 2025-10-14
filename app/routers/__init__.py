from fastapi import APIRouter
from app.routers import users, vector_routes

api_router = APIRouter()

api_router.include_router(users.router)
api_router.include_router(vector_routes.router)

__all__ = ["api_router"]
