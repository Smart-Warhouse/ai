"""YOLO perception layer for the 1:14 forklift camera (requirements §6, PC-06..PC-08).

Pure Python on purpose: `detector` needs only numpy + opencv + onnxruntime, so a ROS 2 node can import it
without pulling in Ultralytics or PyTorch.
"""

CLASSES: tuple[str, ...] = ("person", "pallet", "forklift", "obstacle")  # PC-06 — order = YOLO class id
TARGET_IMAGES_PER_CLASS = 400  # plan.md B6
