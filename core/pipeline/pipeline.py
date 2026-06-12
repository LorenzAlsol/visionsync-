import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from loguru import logger

from core.camera.adapter import CameraAdapter, Frame


@dataclass
class PipelineConfig:
    target_fps: int = 10
    max_queue_size: int = 30
    skip_on_full: bool = True


class FramePipeline:
    def __init__(self, adapter: CameraAdapter, config: PipelineConfig = None):
        self.adapter = adapter
        self.config = config or PipelineConfig()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._running = False
        self._frames_received = 0
        self._frames_dropped = 0

    @property
    def camera_id(self) -> str:
        return self.adapter.camera_id

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    def stop(self):
        self._running = False
        self.adapter.stop()

    async def run(self):
        self._running = True
        logger.info(f"[{self.camera_id}] Pipeline started.")

        async for frame in self.adapter.stream():
            if not self._running:
                break

            self._frames_received += 1

            if self._queue.full():
                if self.config.skip_on_full:
                    self._frames_dropped += 1
                    logger.debug(f"[{self.camera_id}] Queue full, dropping frame {frame.frame_number}.")
                    continue
                else:
                    await self._queue.put(frame)
            else:
                await self._queue.put(frame)

        self._running = False
        logger.info(
            f"[{self.camera_id}] Pipeline stopped. "
            f"Received: {self._frames_received}, Dropped: {self._frames_dropped}"
        )

    @property
    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "frames_received": self._frames_received,
            "frames_dropped": self._frames_dropped,
        }