"""Synthetic four-approach traffic for camera-free demonstrations."""

import random
from typing import Dict, List, Tuple

import numpy as np

from config import (
    ARRIVAL_RATES, DIRECTIONS, FRAME_HEIGHT, FRAME_WIDTH, SATURATION_HEADWAY_SECONDS,
)


class IntersectionSimulation:
    """Maintain simple synthetic queues and render one feed per approach."""

    def __init__(self, seed: int = 7) -> None:
        self._random = random.Random(seed)
        self._queues: Dict[str, List[Dict[str, object]]] = {
            direction: [] for direction in DIRECTIONS
        }
        self._spawn_accumulator = {direction: 0.0 for direction in DIRECTIONS}
        self._discharge_accumulator = {direction: 0.0 for direction in DIRECTIONS}
        self._next_id = 1
        self.reset()

    def reset(self) -> None:
        self._queues = {direction: [] for direction in DIRECTIONS}
        self._spawn_accumulator = {direction: 0.0 for direction in DIRECTIONS}
        self._discharge_accumulator = {direction: 0.0 for direction in DIRECTIONS}
        self._next_id = 1

    def step(
        self, delta_seconds: float, lights: Dict[str, str]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, int],
        Dict[str, float],
        Dict[str, Dict[str, int]],
    ]:
        """Advance queues and return feeds, totals, waits, and class counts."""
        dt = max(0.0, float(delta_seconds))
        frames = {}
        counts = {}
        waits = {}
        class_counts = {}
        for direction in DIRECTIONS:
            queue = self._queues[direction]
            self._spawn_accumulator[direction] += dt * ARRIVAL_RATES[direction]
            while self._spawn_accumulator[direction] >= 1.0 and len(queue) < 22:
                self._spawn_accumulator[direction] -= 1.0
                queue.append({
                    "id": self._next_id,
                    "class_name": self._random.choices(
                        ("car", "motorcycle", "bus", "truck"), (0.66, 0.18, 0.08, 0.08)
                    )[0],
                    "wait": 0.0,
                })
                self._next_id += 1

            moving = lights.get(direction) == "green"
            if moving:
                # First-in-first-out discharge at one vehicle per headway.
                self._discharge_accumulator[direction] += dt / SATURATION_HEADWAY_SECONDS
                while self._discharge_accumulator[direction] >= 1.0 and queue:
                    queue.pop(0)
                    self._discharge_accumulator[direction] -= 1.0
                if not queue:
                    self._discharge_accumulator[direction] = 0.0
            else:
                self._discharge_accumulator[direction] = 0.0
            for vehicle in queue:
                vehicle["wait"] = 0.0 if moving else float(vehicle["wait"]) + dt

            counts[direction] = len(queue)
            waits[direction] = max((float(v["wait"]) for v in queue), default=0.0)
            class_counts[direction] = {
                class_name: sum(
                    1 for vehicle in queue if vehicle["class_name"] == class_name
                )
                for class_name in ("car", "motorcycle", "bus", "truck")
            }
            frames[direction] = self._render(direction, queue, lights.get(direction, "red"))
        return frames, counts, waits, class_counts

    @staticmethod
    def _render(
        direction: str, vehicles: List[Dict[str, object]], light: str
    ) -> np.ndarray:
        image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        image[:] = (45, 48, 52)
        image[65:295, :] = (66, 68, 72)
        image[175:185, :] = (210, 205, 120)
        image[70:290:36, :] = (83, 85, 88)
        image[90:270, 0:FRAME_WIDTH:48] = (110, 112, 115)
        vehicle_colors = {
            "car": (42, 180, 75),
            "motorcycle": (255, 165, 45),
            "bus": (50, 130, 235),
            "truck": (205, 60, 180),
        }
        for index, vehicle in enumerate(vehicles[-12:]):
            x = 28 + (index % 6) * 100
            y = 105 + (index // 6) * 105
            color = vehicle_colors[str(vehicle["class_name"])]
            image[y:y + 40, x:x + 70] = color
            image[y + 7:y + 15, x + 8:x + 27] = (195, 225, 240)
            image[y + 7:y + 15, x + 42:x + 61] = (195, 225, 240)
        lamp = {"green": (40, 220, 70), "yellow": (240, 205, 35)}.get(light, (220, 45, 45))
        image[15:53, 14:52] = (24, 24, 24)
        image[23:45, 21:45] = lamp
        return image
