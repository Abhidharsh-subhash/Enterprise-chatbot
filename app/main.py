from fastapi import FastAPI
from app.core.events import lifespan

app = FastAPI(title="self-hosted chatbot", lifespan=lifespan)


@app.get("/test")
async def root():
    return {"message": "Hello World"}
