from fastapi import FastAPI
from app.core.events import lifespan
from app.routers import api_router

app = FastAPI(title="self-hosted chatbot", lifespan=lifespan)
app.include_router(api_router)


@app.get("/test")
async def root():
    return {"message": "Hello World"}
