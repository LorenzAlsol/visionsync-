import cv2
import time
from core.camera.stream import LatencyFreeStream
from core.engine.inference import YOLOEngine, ModelConfig
from core.engine.labels import COCO_LABELS


def main():
    # --- Engine ---
    config = ModelConfig(
        model_id='yolov8m',
        model_path='models/onnx/yolov8m.onnx',
        confidence_threshold=0.45,
    )
    engine = YOLOEngine(config, COCO_LABELS)
    engine.load()

    # --- Stream ---
    stream = LatencyFreeStream(
        'rtsp://admin:p%40%24%24w0rd@192.168.1.236:554/h264Preview_01_sub',
        name='reolink'
    ).start()

    time.sleep(2)

    frame_count = 0
    fps_timer = time.time()
    display_fps = 0
    last_event = None

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        frame_count += 1

        # --- Inference every 2nd frame ---
        if frame_count % 2 == 0:
            last_event = engine.infer(frame, 'reolink', frame_count)

        # --- FPS counter ---
        if frame_count % 10 == 0:
            display_fps = 10 / (time.time() - fps_timer)
            fps_timer = time.time()

        # --- Draw detections ---
        if last_event:
            for d in last_event.detections:
                h, w = frame.shape[:2]
                x1 = int(d.bbox.x * w)
                y1 = int(d.bbox.y * h)
                x2 = int((d.bbox.x + d.bbox.width) * w)
                y2 = int((d.bbox.y + d.bbox.height) * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{d.label} {d.confidence:.0%}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # --- HUD ---
        detections = len(last_event.detections) if last_event else 0
        inf_ms = last_event.inference_ms if last_event else 0
        cv2.rectangle(frame, (0, 0), (400, 40), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"VisionSync  |  FPS: {display_fps:.1f}  |  Objects: {detections}  |  Inference: {inf_ms:.0f}ms",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        cv2.imshow("VisionSync", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()


main()