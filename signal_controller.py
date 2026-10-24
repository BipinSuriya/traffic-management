"""Adaptive, conflict-free signal scheduler for a four-way intersection.

The controller is simulation-only: it computes signal states and never drives
physical traffic-light hardware.
"""

from typing import Dict, Mapping, Optional, Tuple

from config import (
    ALL_RED_SECONDS,
    DIRECTIONS,
    GREEN_VOLUME_WEIGHT,
    GREEN_WAIT_BONUS,
    GREEN_WAIT_WEIGHT,
    MAX_GREEN_SECONDS,
    MIN_GREEN_SECONDS,
    PHASES,
    SATURATION_HEADWAY_SECONDS,
    STARVATION_SECONDS,
    YELLOW_SECONDS,
)


class SignalController:
    """Cycle opposing approaches together and separate conflicting movements."""

    def __init__(
        self,
        min_green: float = MIN_GREEN_SECONDS,
        max_green: float = MAX_GREEN_SECONDS,
        yellow: float = YELLOW_SECONDS,
        all_red: float = ALL_RED_SECONDS,
        starvation_seconds: float = STARVATION_SECONDS,
    ) -> None:
        if min_green <= 0 or max_green < min_green:
            raise ValueError("Green limits must satisfy 0 < minimum <= maximum.")
        if yellow < 0 or all_red < 0 or starvation_seconds <= 0:
            raise ValueError("Transition times must be non-negative; starvation must be positive.")
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.yellow_seconds = float(yellow)
        self.all_red_seconds = float(all_red)
        self.starvation_seconds = float(starvation_seconds)
        self.stage = "green"
        self.phase = "NS"
        self.elapsed = 0.0
        self.green_duration = self.min_green
        self._initialized_green = False
        self._waiting = {direction: 0.0 for direction in DIRECTIONS}
        self._manual_phase: Optional[str] = None
        self._last_demand: Dict[str, Dict[str, float]] = {}

    def update(
        self, delta_seconds: float, counts: Optional[Mapping[str, int]] = None,
        waiting_seconds: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, object]:
        """Advance the state machine by elapsed time and return a signal snapshot."""
        if delta_seconds < 0:
            raise ValueError("delta_seconds cannot be negative.")
        counts = counts or {}
        supplied_waiting = waiting_seconds or {}
        if not self._initialized_green:
            self.green_duration = self._allocate_green(self.phase, counts, supplied_waiting)
            self._initialized_green = True
        for direction in DIRECTIONS:
            # The phase that just had green is still clearing during yellow,
            # so it must not start banking waiting time yet.
            is_being_served = (
                self.stage in ("green", "yellow") and direction in PHASES[self.phase]
            )
            self._waiting[direction] = 0.0 if is_being_served else max(
                self._waiting[direction] + delta_seconds,
                max(0.0, float(supplied_waiting.get(direction, 0.0))),
            )

        remaining_delta = float(delta_seconds)
        while remaining_delta > 0:
            duration = self._stage_duration()
            remaining = max(0.0, duration - self.elapsed)
            consumed = min(remaining_delta, remaining)
            self.elapsed += consumed
            remaining_delta -= consumed
            if self.elapsed + 1e-9 < duration:
                break
            self.elapsed = 0.0
            self._advance(counts, supplied_waiting)
        self._last_demand = self._demand_summary(counts, supplied_waiting)
        return self.snapshot()

    def request_manual_phase(self, phase: str) -> None:
        """Request NS or EW for the next safe phase boundary."""
        normalized = str(phase).upper()
        if normalized not in PHASES:
            raise ValueError("Manual phase must be 'NS' or 'EW'.")
        self._manual_phase = normalized

    def clear_manual_override(self) -> None:
        self._manual_phase = None

    def reset(self) -> None:
        self.stage = "green"
        self.phase = "NS"
        self.elapsed = 0.0
        self.green_duration = self.min_green
        self._initialized_green = False
        self._waiting = {direction: 0.0 for direction in DIRECTIONS}
        self._manual_phase = None
        self._last_demand = {}

    def snapshot(self) -> Dict[str, object]:
        """Expose phase, countdown, per-approach lights, and override status."""
        lights = {direction: "red" for direction in DIRECTIONS}
        if self.stage == "green":
            for direction in PHASES[self.phase]:
                lights[direction] = "green"
        elif self.stage == "yellow":
            for direction in PHASES[self.phase]:
                lights[direction] = "yellow"
        remaining = max(0.0, self._stage_duration() - self.elapsed)
        return {
            "stage": self.stage,
            "phase": self.phase,
            "remaining": remaining,
            "green_duration": self.green_duration,
            "lights": lights,
            "waiting": dict(self._waiting),
            "manual_override": self._manual_phase,
            "demand": dict(self._last_demand),
        }

    def _stage_duration(self) -> float:
        if self.stage == "green":
            return self.green_duration
        if self.stage == "yellow":
            return self.yellow_seconds
        return self.all_red_seconds

    def _advance(self, counts: Mapping[str, int], waiting: Mapping[str, float]) -> None:
        if self.stage == "green":
            if self.yellow_seconds > 0:
                self.stage = "yellow"
            else:
                self.stage = "all_red"
        elif self.stage == "yellow":
            if self.all_red_seconds > 0:
                self.stage = "all_red"
            else:
                self._begin_next_green(counts, waiting)
        else:
            self._begin_next_green(counts, waiting)

    def _begin_next_green(self, counts: Mapping[str, int], waiting: Mapping[str, float]) -> None:
        self.phase = self._choose_phase(counts, waiting)
        self.stage = "green"
        self.green_duration = self._allocate_green(self.phase, counts, waiting)
        for direction in PHASES[self.phase]:
            self._waiting[direction] = 0.0

    def _choose_phase(self, counts: Mapping[str, int], waiting: Mapping[str, float]) -> str:
        if self._manual_phase is not None:
            selected = self._manual_phase
            self._manual_phase = None
            return selected

        starved = []
        for phase, directions in PHASES.items():
            longest_wait = max(
                (self._waiting[d] for d in directions), default=0.0
            )
            longest_wait = max(
                longest_wait,
                max((float(waiting.get(d, 0.0)) for d in directions), default=0.0),
            )
            if longest_wait >= self.starvation_seconds:
                starved.append((longest_wait, phase))
        if starved:
            starved.sort(reverse=True)
            return starved[0][1]

        scores = {
            phase: sum(max(0, int(counts.get(d, 0))) * GREEN_VOLUME_WEIGHT
                       + max(self._waiting[d], float(waiting.get(d, 0.0))) * GREEN_WAIT_WEIGHT
                       for d in directions)
            for phase, directions in PHASES.items()
        }
        if scores["NS"] == scores["EW"]:
            return "EW" if self.phase == "NS" else "NS"
        return max(scores, key=scores.get)

    def _allocate_green(
        self, phase: str, counts: Mapping[str, int], waiting: Mapping[str, float]
    ) -> float:
        directions = PHASES[phase]
        queue = max((max(0, int(counts.get(d, 0))) for d in directions), default=0)
        wait = max(
            (max(self._waiting[d], float(waiting.get(d, 0.0))) for d in directions),
            default=0.0,
        )
        duration = (
            self.min_green
            + queue * SATURATION_HEADWAY_SECONDS
            + wait * GREEN_WAIT_BONUS
        )
        return min(self.max_green, max(self.min_green, duration))

    def _demand_summary(
        self, counts: Mapping[str, int], waiting: Mapping[str, float]
    ) -> Dict[str, Dict[str, float]]:
        return {
            phase: {
                "vehicles": float(sum(max(0, int(counts.get(d, 0))) for d in directions)),
                "waiting_seconds": max(
                    (max(self._waiting[d], float(waiting.get(d, 0.0))) for d in directions),
                    default=0.0,
                ),
            }
            for phase, directions in PHASES.items()
        }
