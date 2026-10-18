"""Box → distance on the floor, for SF-06 (`person_distance` read by forklift_safety's supervisor).

Flat-floor pinhole model: the bottom-centre of a person box is where the feet touch the floor; intersecting
that pixel's ray with the floor plane gives the distance. Camera = fork-carriage camera, `height_m` above the
floor (changes with the lift), pitched down by `pitch_rad`. No ROS: the node passes CameraInfo K and the lift.

When a box touches the bottom image edge the feet are cut off and are closer than the bottom row's floor
point; the bottom row's distance is returned then. On the 1:14 forklift that row is < 0.1 m ahead (camera
0.054 m + lift, 20° down: forklift_description dimensions.yaml), below the 0.5 m stop distance, so a cut-off
person always stops the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .detector import Detection


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    height_m: float          # optical centre above the floor
    pitch_rad: float = 0.0   # positive = looking down

    @classmethod
    def from_k(cls, k: Iterable[float], width: int, height: int, height_m: float, pitch_rad: float = 0.0):
        """From a row-major 3×3 K (sensor_msgs/CameraInfo.k)."""
        k = list(k)
        return cls(k[0], k[4], k[2], k[5], width, height, height_m, pitch_rad)


def floor_point(cam: Camera, u: float, v: float) -> tuple[float, float] | None:
    """Pixel → (forward, right) metres on the floor from the point below the camera; None above the horizon."""
    a, b = (u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy
    c, s = math.cos(cam.pitch_rad), math.sin(cam.pitch_rad)
    forward, down = c - s * b, s + c * b
    if down <= 1e-9:
        return None
    t = cam.height_m / down
    return t * forward, t * a


def detection_distance(cam: Camera, det: Detection) -> float:
    """Floor distance to the bottom-centre of the box (feet); inf when that point is above the horizon."""
    v = min(det.y2, cam.height - 1)
    p = floor_point(cam, (det.x1 + det.x2) / 2, v)
    return math.inf if p is None else math.hypot(*p)


def nearest_distance(cam: Camera, dets: Iterable[Detection], class_name: str = "person",
                     min_score: float = 0.0) -> float:
    """Closest detection of `class_name` (inf when none) — publish as `person_distance` (Float32)."""
    return min((detection_distance(cam, d) for d in dets if d.class_name == class_name and d.score >= min_score),
               default=math.inf)
