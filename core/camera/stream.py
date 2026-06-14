import threading
import cv2
from loguru import logger


class LatencyFreeStream:
    """
    Runs capture in a background thread, always serves the latest frame.
    Eliminates buffer buildup that causes delay.
    """

    def __init__(self, source: str, name: str = "stream"):
        self.source = source
        self.name = name
        self._cap = None
        self._frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._running = True
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()
        logger.info(f"[{self.name}] Latency-free stream started.")
        return self

    def _grab(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"[{self.name}] Frame grab failed.")
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame is not None, self._frame

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        logger.info(f"[{self.name}] Stream stopped.")