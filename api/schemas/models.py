from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import time


# ── Cameras ───────────────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    source: str
    target_fps: int = Field(default=10, ge=1, le=60)
    enabled: bool = True


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    target_fps: Optional[int] = Field(default=None, ge=1, le=60)
    enabled: Optional[bool] = None


class CameraResponse(BaseModel):
    id: str
    name: str
    source: str
    source_type: str
    target_fps: int
    enabled: bool
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    streaming: bool = False
    created_at: float


# ── Models ────────────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    FACE = "face"
    CUSTOM = "custom"


class ModelResponse(BaseModel):
    id: str
    name: str
    type: ModelType
    description: str
    labels: List[str]
    input_size: List[int]
    loaded: bool
    default: bool


# ── Webhooks ──────────────────────────────────────────────────────────────────

class WebhookTrigger(str, Enum):
    ANY_DETECTION = "any_detection"
    LABEL_MATCH = "label_match"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CAMERA_OFFLINE = "camera_offline"


class WebhookCreate(BaseModel):
    name: str
    url: str
    trigger: WebhookTrigger = WebhookTrigger.ANY_DETECTION
    filter_labels: Optional[List[str]] = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    camera_ids: Optional[List[str]] = None
    enabled: bool = True


class WebhookResponse(BaseModel):
    id: str
    name: str
    url: str
    trigger: WebhookTrigger
    filter_labels: Optional[List[str]]
    min_confidence: float
    camera_ids: Optional[List[str]]
    enabled: bool
    created_at: float
    last_fired_at: Optional[float] = None
    fire_count: int = 0


# ── Generic ───────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str
    message: str = ""