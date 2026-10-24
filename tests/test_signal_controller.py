import unittest

from config import DIRECTIONS
from intersection_simulation import IntersectionSimulation
from signal_controller import SignalController
from traffic_density import classify_density, summarize_traffic
from vehicle_tracking import CentroidTracker


class SignalControllerTests(unittest.TestCase):
    def test_signals_are_conflict_free_through_a_cycle(self):
        controller = SignalController(min_green=4, max_green=8, yellow=2, all_red=1)
        observed = set()
        for _ in range(30):
            snapshot = controller.update(1, {"North": 3, "South": 2, "East": 1, "West": 1})
            green = [direction for direction, color in snapshot["lights"].items() if color == "green"]
            yellow = [direction for direction, color in snapshot["lights"].items() if color == "yellow"]
            self.assertFalse(green and yellow)
            self.assertTrue(
                not green
                or set(green) == {"North", "South"}
                or set(green) == {"East", "West"}
            )
            if snapshot["stage"] == "all_red":
                self.assertTrue(all(color == "red" for color in snapshot["lights"].values()))
            observed.add(snapshot["phase"])
        self.assertEqual(observed, {"NS", "EW"})

    def test_green_allocation_is_bounded_and_uses_vehicle_volume(self):
        controller = SignalController(min_green=5, max_green=12, yellow=2, all_red=1)
        counts = {"North": 100, "South": 100, "East": 0, "West": 0}
        controller.update(0, counts)
        self.assertEqual(controller.snapshot()["green_duration"], 12)
        controller.update(12, counts)
        self.assertEqual(controller.snapshot()["stage"], "yellow")
        controller.update(3, counts)
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["phase"], "NS")
        self.assertEqual(snapshot["green_duration"], 12)
        self.assertGreaterEqual(snapshot["green_duration"], 5)
        self.assertLessEqual(snapshot["green_duration"], 12)

    def test_green_allocation_increases_with_volume_and_waiting(self):
        low_demand = SignalController(min_green=5, max_green=30)
        high_demand = SignalController(min_green=5, max_green=30)
        low_demand.update(0, {"North": 1, "South": 0})
        high_demand.update(
            0,
            {"North": 5, "South": 0},
            {"North": 20, "South": 0},
        )
        self.assertGreater(
            high_demand.snapshot()["green_duration"],
            low_demand.snapshot()["green_duration"],
        )

    def test_manual_override_waits_for_safe_phase_boundary(self):
        controller = SignalController(min_green=4, max_green=4, yellow=2, all_red=1)
        controller.request_manual_phase("EW")
        self.assertEqual(controller.snapshot()["phase"], "NS")
        controller.update(7, {"East": 1, "West": 1})
        self.assertEqual(controller.snapshot()["stage"], "green")
        self.assertEqual(controller.snapshot()["phase"], "EW")
        self.assertIsNone(controller.snapshot()["manual_override"])

    def test_starvation_prevention_selects_longest_waiting_phase(self):
        controller = SignalController(
            min_green=2, max_green=2, yellow=1, all_red=1, starvation_seconds=3
        )
        controller.update(0, {"North": 1, "South": 1})
        controller.update(2, {"North": 1, "South": 1})
        controller.update(1, {"North": 1, "South": 1})
        controller.update(1, {"North": 1, "South": 1})
        self.assertEqual(controller.snapshot()["phase"], "EW")
        self.assertEqual(controller.snapshot()["stage"], "green")

    def test_invalid_phase_and_negative_time_are_rejected(self):
        controller = SignalController()
        with self.assertRaises(ValueError):
            controller.request_manual_phase("NE")
        with self.assertRaises(ValueError):
            controller.update(-1)


class TrackingAndDensityTests(unittest.TestCase):
    def test_nearby_detections_keep_track_and_unique_count(self):
        tracker = CentroidTracker(max_distance=30)
        first = tracker.update([{"bbox": (0, 0, 20, 20), "class_name": "car"}])
        second = tracker.update([{"bbox": (3, 2, 23, 22), "class_name": "car"}])
        self.assertEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(tracker.total_unique, 1)

    def test_density_summary_includes_all_approaches(self):
        self.assertEqual(classify_density(2)[0], "Low")
        summary = summarize_traffic({"North": 15})
        self.assertEqual(set(summary), set(DIRECTIONS))
        self.assertEqual(summary["East"]["count"], 0)
        self.assertEqual(summary["North"]["density"], "High")

    def test_simulation_produces_four_feeds_and_class_totals(self):
        simulation = IntersectionSimulation(seed=3)
        frames, totals, _, class_counts = simulation.step(
            8, {direction: "red" for direction in DIRECTIONS}
        )
        self.assertEqual(set(frames), set(DIRECTIONS))
        for direction in DIRECTIONS:
            self.assertEqual(frames[direction].shape, (360, 640, 3))
            self.assertEqual(sum(class_counts[direction].values()), totals[direction])


if __name__ == "__main__":
    unittest.main()
