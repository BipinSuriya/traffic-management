"""Traffic volume and density calculations."""

from typing import Dict, Iterable, Mapping, Tuple

from config import DENSITY_CAPACITY, DENSITY_LEVELS, DIRECTIONS


def classify_density(count: int, capacity: int = DENSITY_CAPACITY) -> Tuple[str, float]:
    """Return a Low/Medium/High label and a normalized 0..1 density value."""
    safe_count = max(0, int(count))
    safe_capacity = max(1, int(capacity))
    ratio = min(1.0, safe_count / float(safe_capacity))
    for threshold, label in DENSITY_LEVELS:
        if safe_count <= threshold:
            return label, ratio
    return "High", ratio


def summarize_traffic(
    counts: Mapping[str, int], capacity: int = DENSITY_CAPACITY
) -> Dict[str, Dict[str, object]]:
    """Build a complete direction-indexed traffic summary."""
    result = {}
    for direction in DIRECTIONS:
        count = max(0, int(counts.get(direction, 0)))
        label, ratio = classify_density(count, capacity)
        result[direction] = {"count": count, "density": label, "ratio": ratio}
    return result


def count_by_class(class_names: Iterable[str]) -> Dict[str, int]:
    """Count observed vehicle classes, including zero-valued known categories."""
    result = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    for name in class_names:
        if name in result:
            result[name] += 1
    return result
