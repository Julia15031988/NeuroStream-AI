from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.admin.router import router as admin_router
from app.websocket.router import router as websocket_router

app = FastAPI(
    title="NeuroStream AI",
    description="Real-time EEG brainwave processing service",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

app.include_router(admin_router)
app.include_router(websocket_router)


@app.get("/dashboard", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
    )


@app.get("/")
async def root():
    return {"message": "NeuroStream AI is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}
