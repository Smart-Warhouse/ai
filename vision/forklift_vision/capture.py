"""Grab frames for labelling from a webcam / USB camera / video file / RTSP stream or a ROS 2 image topic.

One run = one session folder `<out>/<YYYYmmdd-HHMMSS>_<label>/images/<label>_000001.jpg` + `session.json`
(source, interval, lighting, robot, notes). Frames are saved every `--interval` seconds; blurred frames and
frames almost identical to the last saved one (forklift standing still) are skipped.

    python -m forklift_vision.capture --source 0 --label fl1_lux300 --lux 300 --interval 1.0 --max 300
    python -m forklift_vision.capture --ros-topic /fl1/camera/image_raw --label fl1_lux150 --lux 150
    python -m forklift_vision.capture --ros-topic /fl1/camera/image_raw/compressed --label fl1_dim

The ROS mode needs a sourced ROS 2 environment (rclpy + sensor_msgs); cv_bridge is not required.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — low = motion blur / out of focus."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def change(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference (0..255) of two frames at 1/8 resolution."""
    sa = cv2.resize(a, (a.shape[1] // 8, a.shape[0] // 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    sb = cv2.resize(b, (sa.shape[1], sa.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(sa - sb).mean())


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "session"


class FrameSaver:
    """Applies the interval / blur / duplicate filters and writes the session folder. Source-agnostic."""

    def __init__(self, out: Path, label: str, interval: float, max_frames: int, min_sharpness: float,
                 min_change: float, meta: dict, jpeg_quality: int = 95, clock=time.monotonic):
        self.label = safe_label(label)
        self.dir = Path(out) / f"{datetime.now():%Y%m%d-%H%M%S}_{self.label}"
        (self.dir / "images").mkdir(parents=True, exist_ok=False)
        (self.dir / "labels").mkdir()
        self.interval, self.max_frames = interval, max_frames
        self.min_sharpness, self.min_change = min_sharpness, min_change
        self.jpeg_quality, self.clock = jpeg_quality, clock
        self.saved = 0
        self.skipped = {"blur": 0, "duplicate": 0}
        self._last_t = -1e18
        self._last_img: np.ndarray | None = None
        self.meta = {"label": self.label, "started": datetime.now().isoformat(timespec="seconds"),
                     "interval_s": interval, "min_sharpness": min_sharpness, "min_change": min_change, **meta}
        self._write_meta()

    @property
    def done(self) -> bool:
        return 0 < self.max_frames <= self.saved

    def offer(self, img: np.ndarray) -> bool:
        """Consider one incoming frame; returns True when it was written."""
        now = self.clock()
        if now - self._last_t < self.interval or self.done:
            return False
        if self.min_sharpness and sharpness(img) < self.min_sharpness:
            self.skipped["blur"] += 1
            return False
        if self.min_change and self._last_img is not None and change(img, self._last_img) < self.min_change:
            self.skipped["duplicate"] += 1
            return False
        self.saved += 1
        path = self.dir / "images" / f"{self.label}_{self.saved:06d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        self._last_t, self._last_img = now, img
        if self.saved % 25 == 0:
            self._write_meta()
        return True

    def close(self) -> None:
        self.meta.update(ended=datetime.now().isoformat(timespec="seconds"), frames=self.saved,
                         skipped=self.skipped)
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "session.json").write_text(json.dumps(self.meta, indent=2))


def ros_image_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image or CompressedImage -> BGR uint8, without cv_bridge."""
    if hasattr(msg, "format"):  # CompressedImage
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        return img if enc == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.step // 4, 4)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if enc == "bgra8" else cv2.COLOR_RGBA2BGR)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, :msg.width], cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422", "yuyv", "yuv422_yuy2"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.step // 2, 2)[:, :msg.width], cv2.COLOR_YUV2BGR_YUYV)
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def run_opencv(args, saver: FrameSaver) -> None:
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera/video {args.source!r}")
    try:
        while not saver.done:
            ok, frame = cap.read()
            if not ok:
                break
            if saver.offer(frame) and args.verbose:
                print(f"saved {saver.saved}")
            if args.preview:
                cv2.imshow("capture (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()


def run_ros(args, saver: FrameSaver) -> None:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image

    rclpy.init()
    node = rclpy.create_node("yolo_dataset_capture")
    msg_type = CompressedImage if args.ros_topic.endswith("/compressed") else Image

    def cb(msg):
        if saver.offer(ros_image_to_bgr(msg)) and args.verbose:
            node.get_logger().info(f"saved {saver.saved}")

    node.create_subscription(msg_type, args.ros_topic, cb, qos_profile_sensor_data)
    try:
        while rclpy.ok() and not saver.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="camera index (0), device (/dev/video0), video file or rtsp:// URL")
    src.add_argument("--ros-topic", help="sensor_msgs/Image topic, or .../compressed for CompressedImage")
    ap.add_argument("--out", default="datasets/capture", help="parent folder of the session folders")
    ap.add_argument("--label", required=True, help="session label, e.g. fl1_lux300_aisleA")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between saved frames")
    ap.add_argument("--max", type=int, default=0, help="stop after N frames (0 = until Ctrl-C / end)")
    ap.add_argument("--min-sharpness", type=float, default=30.0, help="Laplacian variance; 0 disables")
    ap.add_argument("--min-change", type=float, default=2.0, help="mean abs diff vs last saved; 0 disables")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--lux", type=float, help="measured illuminance at the scene (PC-11 levels)")
    ap.add_argument("--robot", default="", help="fl1 / fl2 / fl3")
    ap.add_argument("--notes", default="")
    ap.add_argument("--preview", action="store_true", help="show the live image (OpenCV sources only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    meta = {"source": args.source or args.ros_topic, "lux": args.lux, "robot": args.robot, "notes": args.notes,
            "resolution": [args.width, args.height]}
    saver = FrameSaver(Path(args.out), args.label, args.interval, args.max, args.min_sharpness,
                       args.min_change, meta)
    print(f"capturing into {saver.dir}")
    try:
        (run_ros if args.ros_topic else run_opencv)(args, saver)
    except KeyboardInterrupt:
        pass
    finally:
        saver.close()
    print(f"{saver.saved} frames saved, skipped {saver.skipped} -> {saver.dir}")


if __name__ == "__main__":
    main()
