from fastapi import FastAPI, HTTPException, Request
from app.core.events import lifespan
from app.routers import api_router
from fastapi.responses import JSONResponse

app = FastAPI(title="self-hosted chatbot", lifespan=lifespan)
app.include_router(api_router)


@app.get("/test")
async def root():
    return {"message": "Hello World"}


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "message": exc.detail,  # rename 'detail' → 'message'
        },
    )
