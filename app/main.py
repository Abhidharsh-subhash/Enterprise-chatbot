from fastapi import FastAPI, Request
from app.core.events import lifespan
from app.routers import api_router
from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI(title="self-hosted chatbot", lifespan=lifespan)
app.include_router(api_router)


@app.get("/test")
async def root():
    return {"message": "Hello World"}

