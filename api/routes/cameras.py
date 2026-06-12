import time
import uuid
from fastapi import APIRouter, HTTPException, status

from api.schemas.models import CameraCreate, CameraUpdate, CameraResponse, StatusResponse
from core.camera.manager import camera_manager

router = APIRouter()

# In-memory store for camera metadata
_cameras: dict = {}


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(body: CameraCreate):
    camera_id = str(uuid.uuid4())[:8]
    adapter = camera_manager.add(
        camera_id=camera_id,
        source=body.source,
        target_fps=body.target_fps,
    )
    record = {
        "id": camera_id,
        "name": body.name,
        "source": body.source,
        "source_type": adapter.source_type.value,
        "target_fps": body.target_fps,
        "enabled": body.enabled,
        "streaming": False,
        "created_at": time.time(),
    }
    _cameras[camera_id] = record
    return _build_response(camera_id)


@router.get("", response_model=list[CameraResponse])
async def list_cameras():
    return [_build_response(cid) for cid in _cameras]


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str):
    _require(camera_id)
    return _build_response(camera_id)


@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(camera_id: str, body: CameraUpdate):
    _require(camera_id)
    rec = _cameras[camera_id]
    if body.name is not None:
        rec["name"] = body.name
    if body.target_fps is not None:
        rec["target_fps"] = body.target_fps
    if body.enabled is not None:
        rec["enabled"] = body.enabled
    return _build_response(camera_id)


@router.delete("/{camera_id}", response_model=StatusResponse)
async def delete_camera(camera_id: str):
    _require(camera_id)
    camera_manager.remove(camera_id)
    del _cameras[camera_id]
    return {"status": "ok", "message": f"Camera '{camera_id}' removed."}


@router.post("/{camera_id}/start", response_model=StatusResponse)
async def start_camera(camera_id: str):
    _require(camera_id)
    _cameras[camera_id]["streaming"] = True
    return {"status": "ok", "message": f"Camera '{camera_id}' stream started."}


@router.post("/{camera_id}/stop", response_model=StatusResponse)
async def stop_camera(camera_id: str):
    _require(camera_id)
    adapter = camera_manager.get(camera_id)
    if adapter:
        adapter.stop()
    _cameras[camera_id]["streaming"] = False
    return {"status": "ok", "message": f"Camera '{camera_id}' stream stopped."}


def _require(camera_id: str):
    if camera_id not in _cameras:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")


def _build_response(camera_id: str) -> CameraResponse:
    rec = _cameras[camera_id]
    adapter = camera_manager.get(camera_id)
    info = adapter.info if adapter else None
    return CameraResponse(
        **{k: v for k, v in rec.items() if k not in ("width", "height", "fps")},
        width=info.width if info else None,
        height=info.height if info else None,
        fps=info.fps if info else None,
    )