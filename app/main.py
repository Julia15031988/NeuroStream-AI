from fastapi import FastAPI


from app.admin.router import router as admin_router

app = FastAPI(
    title="NeuroStream AI",
    version="0.1.0",
)

app.include_router(admin_router)


@app.get("/")
async def root():
    return {
        "message": "NeuroStream AI is running"
    }
