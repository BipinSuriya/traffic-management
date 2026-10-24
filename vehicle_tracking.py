"""Lightweight centroid-based vehicle tracking for YOLO detections."""

from math import hypot
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Detection = Mapping[str, object]


class CentroidTracker:
    """Associate nearby detections over time and count each track only once."""

    def __init__(self, max_disappeared: int = 12, max_distance: float = 75.0) -> None:
        if max_disappeared < 0 or max_distance <= 0:
            raise ValueError("Tracker limits must be non-negative and positive.")
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._tracks: Dict[int, Dict[str, object]] = {}
        self._next_id = 1
        self.total_unique = 0

    def update(
        self, detections: Sequence[Detection], elapsed_seconds: float = 1.0
    ) -> List[Dict[str, object]]:
        """Update tracks and return tracks detected in this frame."""
        elapsed = max(0.0, float(elapsed_seconds))
        candidates = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                center = self._center(detection)
                distance = hypot(
                    center[0] - track["center"][0], center[1] - track["center"][1]
                )
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, index))
        candidates.sort()

        matched_tracks = set()
        matched_detections = set()
        for _, track_id, index in candidates:
            if track_id in matched_tracks or index in matched_detections:
                continue
            track = self._tracks[track_id]
            detection = detections[index]
            center = self._center(detection)
            old_center = track["center"]
            movement = hypot(center[0] - old_center[0], center[1] - old_center[1])
            if movement <= 3.0:
                track["waiting_seconds"] += elapsed
            else:
                track["waiting_seconds"] = 0.0
            track.update(
                center=center,
                bbox=tuple(detection["bbox"]),
                class_name=str(detection["class_name"]),
                confidence=float(detection.get("confidence", 0.0)),
                disappeared=0,
            )
            matched_tracks.add(track_id)
            matched_detections.add(index)

        for track_id in list(self._tracks):
            if track_id not in matched_tracks:
                self._tracks[track_id]["disappeared"] += 1
                if self._tracks[track_id]["disappeared"] > self.max_disappeared:
                    del self._tracks[track_id]

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.total_unique += 1
            self._tracks[track_id] = {
                "id": track_id,
                "center": self._center(detection),
                "bbox": tuple(detection["bbox"]),
                "class_name": str(detection["class_name"]),
                "confidence": float(detection.get("confidence", 0.0)),
                "disappeared": 0,
                "waiting_seconds": 0.0,
            }

        return [
            dict(track)
            for track in self._tracks.values()
            if track["disappeared"] == 0
        ]

    @staticmethod
    def _center(detection: Detection) -> Tuple[float, float]:
        bbox = detection["bbox"]
        return ((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2)

    @property
    def active_track_count(self) -> int:
        return sum(1 for track in self._tracks.values() if track["disappeared"] == 0)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.total_unique = 0
