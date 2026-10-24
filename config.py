"""Shared settings for the academic traffic-intersection prototype."""

from typing import Dict, Tuple

DIRECTIONS: Tuple[str, ...] = ("North", "South", "East", "West")
PHASES: Dict[str, Tuple[str, ...]] = {
    "NS": ("North", "South"),
    "EW": ("East", "West"),
}
DIRECTION_TO_PHASE = {
    direction: phase for phase, directions in PHASES.items() for direction in directions
}

MIN_GREEN_SECONDS = 10.0
MAX_GREEN_SECONDS = 45.0
YELLOW_SECONDS = 3.0
ALL_RED_SECONDS = 1.5
STARVATION_SECONDS = 60.0
GREEN_VOLUME_WEIGHT = 1.4
GREEN_WAIT_WEIGHT = 0.12
# Green-time allocation: opposing lanes discharge in parallel, so green length
# follows the longest queue at one vehicle per saturation headway.
SATURATION_HEADWAY_SECONDS = 2.0
GREEN_WAIT_BONUS = 0.08
# Dashboard clock: one tick per second; a single tick may cover at most this
# much real time so a stalled browser tab cannot jump the signal clock.
TICK_SECONDS = 1.0
MAX_TICK_SECONDS = 5.0
# Synthetic arrivals (vehicles/second) -- below discharge capacity so queues
# can actually drain and green times can adapt.
ARRIVAL_RATES = {"North": 0.20, "South": 0.15, "East": 0.20, "West": 0.15}
DENSITY_CAPACITY = 20
DENSITY_LEVELS = ((5, "Low"), (12, "Medium"), (20, "High"))

DEFAULT_MODEL = "yolov8n.pt"
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CLASS_COLORS = {
    "car": (50, 220, 50),
    "motorcycle": (50, 180, 255),
    "bus": (255, 190, 40),
    "truck": (230, 70, 220),
}

SIMULATION_FPS = 12
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
