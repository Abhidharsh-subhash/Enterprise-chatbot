from fastapi import FastAPI, Request
from app.core.events import lifespan
from app.routers import api_router
from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI(title="self-hosted chatbot", lifespan=lifespan)
app.include_router(api_router)


@app.get("/test")
async def root():
    return {"message": "Hello World"}


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request):
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="API",
    )
