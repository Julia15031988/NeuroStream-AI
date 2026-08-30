from fastapi import FastAPI

from app.admin.router import router as admin_router
from app.websocket.router import router as websocket_router

app = FastAPI(
    title="NeuroStream AI",
    description="Real-time EEG brainwave processing service",
    version="0.1.0",
)

app.include_router(admin_router)
app.include_router(websocket_router)


@app.get("/")
async def root():
    return {"message": "NeuroStream AI is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}