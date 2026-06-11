import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
from loguru import logger


class SourceType(str, Enum):
    RTSP = "rtsp"
    USB = "usb"
    HTTP = "http"
    FILE = "file"
    ONVIF = "onvif"
    UNKNOWN = "unknown"

def detect_source_type(source: str) -> SourceType:
    s = str(source).strip()
    if s.isdigit():
        return SourceType.USB
    if s.startswith("rtsp://"):
        return SourceType.RTSP
    if s.startswith("onvif://"):
        return SourceType.ONVIF
    if s.startswith("http://") or s.startswith("https://"):
        return SourceType.HTTP
    if s.endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
        return SourceType.FILE
    return SourceType.UNKNOWN
@dataclass
class CameraInfo:
    source: str
    source_type: SourceType
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""


@dataclass
class Frame:
    data: np.ndarray
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0    

class CameraAdapter:
    def __init__(
        self,
        camera_id: str,
        source: str,
        target_fps: int = 10,
        reconnect_delay: float = 5.0,
        max_reconnects: int = -1,
    ):
        self.camera_id = camera_id
        self.source = source
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects

        self.source_type = detect_source_type(source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_count = 0
        self._info: Optional[CameraInfo] = None

    @property
    def info(self) -> Optional[CameraInfo]:
        return self._info

    @property
    def is_live(self) -> bool:
        return self.source_type in (
            SourceType.RTSP,
            SourceType.HTTP,
            SourceType.USB,
            SourceType.ONVIF,
        )

    def stop(self):
        self._running = False
    def _open(self) -> bool:
        raw_source = int(self.source) if self.source_type == SourceType.USB else self.source

        if self.source_type == SourceType.ONVIF:
            raw_source = self.source.replace("onvif://", "http://", 1)

        cap = cv2.VideoCapture(raw_source)
        if not cap.isOpened():
            logger.warning(f"[{self.camera_id}] Could not open source: {self.source}")
            return False

        self._cap = cap
        self._info = CameraInfo(
            source=self.source,
            source_type=self.source_type,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS) or self.target_fps,
            codec=str(int(cap.get(cv2.CAP_PROP_FOURCC))),
        )
        logger.info(
            f"[{self.camera_id}] Opened {self.source_type.value} "
            f"{self._info.width}x{self._info.height} @ {self._info.fps:.1f}fps"
        )
        return True

    def _close(self):
        if self._cap:
            self._cap.release()
            self._cap = None    

    async def stream(self) -> AsyncGenerator[Frame, None]:
        self._running = True
        reconnects = 0
        interval = 1.0 / self.target_fps

        while self._running:
            if not self._open():
                if not self.is_live:
                    logger.error(f"[{self.camera_id}] Cannot open source. Stopping.")
                    break
                reconnects += 1
                if self.max_reconnects != -1 and reconnects > self.max_reconnects:
                    logger.error(f"[{self.camera_id}] Max reconnects reached. Stopping.")
                    break
                logger.info(f"[{self.camera_id}] Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                continue

            reconnects = 0
            loop_start = time.monotonic()

            while self._running:
                ret, raw = self._cap.read()
                if not ret:
                    logger.warning(f"[{self.camera_id}] Frame read failed — reconnecting.")
                    self._close()
                    break

                self._frame_count += 1
                yield Frame(
                    data=raw,
                    camera_id=self.camera_id,
                    frame_number=self._frame_count,
                )

                elapsed = time.monotonic() - loop_start
                sleep_for = interval - (elapsed % interval)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                loop_start = time.monotonic()

            if not self.is_live:
                break

        self._close()
        self._running = False
        logger.info(f"[{self.camera_id}] Stream ended. Frames: {self._frame_count}")        