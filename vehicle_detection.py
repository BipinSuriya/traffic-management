"""OpenCV video input and optional Ultralytics YOLOv8 vehicle detection."""

from typing import Dict, List, Optional, Tuple

from config import CLASS_COLORS, DEFAULT_MODEL, VEHICLE_CLASSES
from vehicle_tracking import CentroidTracker


class YOLOVehicleDetector:
    """Run YOLOv8 on frames and return detections for four vehicle classes."""

    def __init__(self, model_path: str = DEFAULT_MODEL, confidence: float = 0.35) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install requirements.txt to use YOLO mode."
            ) from exc
        self.confidence = confidence
        self.model = YOLO(model_path)

    def detect(self, frame) -> List[Dict[str, object]]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=list(VEHICLE_CLASSES),
            verbose=False,
        )
        detections = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id not in VEHICLE_CLASSES:
                continue
            coordinates = box.xyxy[0].tolist()
            detections.append({
                "bbox": tuple(float(value) for value in coordinates),
                "class_name": VEHICLE_CLASSES[class_id],
                "confidence": float(box.conf[0].item()),
            })
        return detections


class VideoVehicleMonitor:
    """Read a source, detect and track vehicles, annotate, and release resources."""

    def __init__(
        self,
        source: str,
        detector: YOLOVehicleDetector,
        tracker: Optional[CentroidTracker] = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed. Install requirements.txt.") from exc
        self._cv2 = cv2
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Could not open video or CCTV stream: {}".format(source))
        self.detector = detector
        self.tracker = tracker or CentroidTracker()
        self._fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= self._fps <= 120.0:
            self._fps = 25.0
        self._last_frame = None

    def read(self, elapsed_seconds: float = 1.0) -> Optional[Dict[str, object]]:
        # Skip the frames that elapsed since the last read so video time keeps
        # pace with the signal clock (and live streams drop stale buffered frames).
        for _ in range(min(300, max(0, int(round(self._fps * elapsed_seconds)) - 1))):
            if not self.capture.grab():
                break
        success, frame = self.capture.read()
        if not success:
            self.capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.capture.read()
            if not success:
                return None
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, elapsed_seconds)
        annotated = frame.copy()
        counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        waits = []
        for track in tracks:
            class_name = str(track["class_name"])
            counts[class_name] += 1
            waits.append(float(track["waiting_seconds"]))
            x1, y1, x2, y2 = (int(value) for value in track["bbox"])
            color = CLASS_COLORS.get(class_name, (40, 220, 40))
            self._cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = "{} #{}".format(class_name, track["id"])
            self._cv2.putText(
                annotated, label, (x1, max(18, y1 - 6)),
                self._cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
        self._last_frame = annotated
        rgb_frame = self._cv2.cvtColor(annotated, self._cv2.COLOR_BGR2RGB)
        return {
            "frame": rgb_frame,
            "counts": counts,
            "total": sum(counts.values()),
            "unique_total": self.tracker.total_unique,
            "waiting": max(waits, default=0.0),
        }

    def close(self) -> None:
        self.capture.release()
