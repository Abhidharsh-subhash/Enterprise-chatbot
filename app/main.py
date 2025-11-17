from fastapi import FastAPI, Request
from app.core.events import lifespan
from app.routers import api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="self-hosted chatbot", lifespan=lifespan)
app.include_router(api_router)

# Allow all origins (any domain or IP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- this allows any domain
    allow_credentials=True,  # allow cookies/auth headers
    allow_methods=["*"],  # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # allow all headers
)

@app.get("/test")
async def root():
    return {"message": "Hello World"}

