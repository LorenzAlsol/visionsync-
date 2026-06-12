import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import onnxruntime as ort
from loguru import logger

from core.events.bus import DetectionEvent, Detection, BoundingBox


@dataclass
class ModelConfig:
    model_id: str
    model_path: str
    input_size: tuple = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    device: str = "cpu"        # cpu | cuda

class YOLOEngine:
    def __init__(self, config: ModelConfig, labels: List[str]):
        self.config = config
        self.labels = labels
        self._session: Optional[ort.InferenceSession] = None

    def load(self):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.config.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(
            self.config.model_path, providers=providers
        )
        logger.info(f"[{self.config.model_id}] Model loaded from {self.config.model_path}")

    def is_loaded(self) -> bool:
        return self._session is not None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_h, target_w = self.config.input_size

        # Resize
        import cv2
        resized = cv2.resize(frame, (target_w, target_h))

        # BGR to RGB, normalize, add batch dim
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        transposed = rgb.transpose(2, 0, 1)
        return np.expand_dims(transposed, axis=0)

    def _postprocess(
        self, output: np.ndarray, orig_h: int, orig_w: int
    ) -> List[Detection]:
        detections = []
        predictions = output[0].transpose(1, 0)  # (8400, 84)

        for pred in predictions:
            scores = pred[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < self.config.confidence_threshold:
                continue

            cx, cy, bw, bh = pred[:4]

            # Normalise to 0-1
            x = float((cx - bw / 2) / self.config.input_size[1])
            y = float((cy - bh / 2) / self.config.input_size[0])
            width = float(bw / self.config.input_size[1])
            height = float(bh / self.config.input_size[0])

            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                )
            )

        return detections

    def infer(self, frame: np.ndarray, camera_id: str, frame_number: int) -> DetectionEvent:
        if not self._session:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        input_tensor = self._preprocess(frame)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})

        detections = self._postprocess(outputs[0], orig_h, orig_w)
        inference_ms = (time.perf_counter() - start) * 1000

        return DetectionEvent(
            camera_id=camera_id,
            frame_number=frame_number,
            detections=detections,
            model_id=self.config.model_id,
            inference_ms=inference_ms,
        )