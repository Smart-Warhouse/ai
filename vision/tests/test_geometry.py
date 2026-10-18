import math

import pytest

from forklift_vision.detector import Detection
from forklift_vision.geometry import Camera, detection_distance, floor_point, nearest_distance

CAM = Camera(fx=500, fy=500, cx=320, cy=240, width=640, height=480, height_m=0.10, pitch_rad=0.0)


def person(x1, y1, x2, y2, score=0.9, name="person"):
    return Detection(0, name, score, x1, y1, x2, y2)


def test_floor_point_level_camera():
    # level camera: a pixel fy*h/d below the centre row sees the floor at distance d
    fwd, right = floor_point(CAM, 320, 240 + 500 * 0.10 / 1.0)
    assert fwd == pytest.approx(1.0) and right == pytest.approx(0.0)
    fwd, right = floor_point(CAM, 320 + 100, 290)
    assert right == pytest.approx(0.2) and fwd == pytest.approx(1.0)
    assert floor_point(CAM, 320, 240) is None and floor_point(CAM, 320, 100) is None  # horizon / above


def test_floor_point_pitched_camera():
    pitch = math.radians(20)
    cam = Camera(500, 500, 320, 240, 640, 480, height_m=0.10, pitch_rad=pitch)
    fwd, _ = floor_point(cam, 320, 240)  # optical axis hits the floor at h / tan(pitch)
    assert fwd == pytest.approx(0.10 / math.tan(pitch))


def test_from_k():
    cam = Camera.from_k([600, 0, 330, 0, 610, 250, 0, 0, 1], 640, 480, 0.12, 0.1)
    assert (cam.fx, cam.fy, cam.cx, cam.cy, cam.height_m) == (600, 610, 330, 250, 0.12)


def test_nearest_person_distance():
    far = person(300, 100, 340, 265)    # feet at v=265 → 2 m
    near = person(100, 50, 200, 290)    # feet at v=290 → 1 m
    pallet = person(300, 300, 340, 400, name="pallet")
    low = person(300, 300, 340, 440, score=0.1)
    assert detection_distance(CAM, far) == pytest.approx(2.0)
    assert nearest_distance(CAM, [far, near, pallet, low], min_score=0.25) == pytest.approx(math.hypot(1.0, 0.34))
    assert nearest_distance(CAM, [pallet]) == math.inf
    assert nearest_distance(CAM, [person(0, 0, 10, 200)]) == math.inf  # feet above the horizon


def test_box_cut_by_bottom_edge_uses_bottom_row():
    d = detection_distance(CAM, person(300, 200, 340, 480))
    assert d == pytest.approx(0.10 * 500 / (479 - 240))
