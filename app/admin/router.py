from fastapi import APIRouter


router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"]
)

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "NeuroStream AI",
        "version": "0.1.0",
    }


