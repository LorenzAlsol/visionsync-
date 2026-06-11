from typing import Dict, Optional
from loguru import logger

from core.camera.adapter import CameraAdapter, CameraInfo


class CameraManager:
    def __init__(self):
        self._cameras: Dict[str, CameraAdapter] = {}

    def add(
        self,
        camera_id: str,
        source: str,
        target_fps: int = 10,
        reconnect_delay: float = 5.0,
    ) -> CameraAdapter:
        if camera_id in self._cameras:
            raise ValueError(f"Camera '{camera_id}' already exists.")
        adapter = CameraAdapter(
            camera_id=camera_id,
            source=source,
            target_fps=target_fps,
            reconnect_delay=reconnect_delay,
        )
        self._cameras[camera_id] = adapter
        logger.info(f"Camera '{camera_id}' registered (source={source})")
        return adapter

    def get(self, camera_id: str) -> Optional[CameraAdapter]:
        return self._cameras.get(camera_id)

    def remove(self, camera_id: str):
        adapter = self._cameras.pop(camera_id, None)
        if adapter:
            adapter.stop()

    def list(self) -> Dict[str, CameraInfo]:
        return {
            cid: adapter.info
            for cid, adapter in self._cameras.items()
            if adapter.info
        }

    @property
    def count(self) -> int:
        return len(self._cameras)


# Singleton
camera_manager = CameraManager()