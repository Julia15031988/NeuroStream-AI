from fastapi import APIRouter

from app.config import settings


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"]
)

@router.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "NeuroStream AI",
        "admin": settings.admin_username,
        "version": "0.1.0",
    }



