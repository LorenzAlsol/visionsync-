import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger


@dataclass
class BoundingBox:
    x: float       # top-left x (0-1 normalised)
    y: float       # top-left y (0-1 normalised)
    width: float   # 0-1 normalised
    height: float  # 0-1 normalised


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None


@dataclass
class DetectionEvent:
    camera_id: str
    frame_number: int
    timestamp: float = field(default_factory=time.time)
    detections: List[Detection] = field(default_factory=list)
    model_id: str = ""
    inference_ms: float = 0.0

def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "inference_ms": round(self.inference_ms, 2),
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "track_id": d.track_id,
                    "bbox": {
                        "x": round(d.bbox.x, 4),
                        "y": round(d.bbox.y, 4),
                        "width": round(d.bbox.width, 4),
                        "height": round(d.bbox.height, 4),
                    },
                }
                for d in self.detections
            ],
        }    

class EventBus:
    def __init__(self, max_queue_size: int = 100):
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._max_queue_size = max_queue_size
        self._running = False

    async def start(self):
        self._running = True
        logger.info("Event bus started.")

    async def stop(self):
        self._running = False
        for q in self._subscribers.values():
            q.put_nowait(None)  # poison pill to unblock listeners
        logger.info("Event bus stopped.")

    def subscribe(self, subscriber_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[subscriber_id] = q
        logger.debug(f"Subscriber '{subscriber_id}' added.")
        return q

    def unsubscribe(self, subscriber_id: str):
        self._subscribers.pop(subscriber_id, None)
        logger.debug(f"Subscriber '{subscriber_id}' removed.")

    async def publish(self, event: DetectionEvent):
        for sid, q in list(self._subscribers.items()):
            if q.full():
                logger.warning(f"Queue full for '{sid}', dropping oldest frame.")
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await q.put(event)


# Singleton
event_bus = EventBus()