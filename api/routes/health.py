from fastapi import APIRouter
from core.camera.manager import camera_manager

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "cameras": camera_manager.count,
    }