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
async def get_swagger(request: Request):
    return get_swagger_ui_html(
        openapi_url=request.scope.get("root_path") + "/openapi.json", title="Swagger"
    )
