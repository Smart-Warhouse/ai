"""Synthetic `pallet` / `forklift` / `obstacle` images rendered with MuJoCo, boxes taken from the segmentation
render (pixel-exact, occlusion-aware), optional COCO person cut-outs pasted in (see fetch_coco_persons.py).

The scene re-uses the 1:14 dimensions of ai/training/forklift_rl/config.yaml (read only): forklifts, pallets
with a load and the DICT_4X4_50 marker, racks (unlabelled distractors), floor tape, walls, and obstacles
(boxes, cans, balls, traffic cones). Every object is a mocap body, so one compiled scene renders many frames:
per frame the camera pose (fork-carriage heights), field of view, object poses/colours and the light level
(dim / normal / bright, PC-11) are re-drawn.

    python scripts/make_synthetic.py --n 1800 --out datasets/raw/synth_mujoco \
        --person-cutouts datasets/raw/coco_person_cutouts --person-prob 0.35
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forklift_vision import CLASSES  # noqa: E402

TRAINING_CFG = ROOT.parent / "training" / "forklift_rl" / "config.yaml"
CLS = {c: i for i, c in enumerate(CLASSES)}
NOPHYS = 'contype="0" conaffinity="0"'
H, W = 480, 640
HIDDEN = (0.0, 200.0, 0.0)
LIGHT_LEVELS = {"dim": (0.25, 0.45), "normal": (0.6, 0.9), "bright": (1.0, 1.35)}  # PC-11, light multiplier

# ------------------------------------------------------------------------------------------------ MJCF
def aruco_png(path: Path, marker_id: int = 7, px: int = 240) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        inner = cv2.aruco.generateImageMarker(d, marker_id, px * 3 // 4)
        img = np.full((px, px), 255, np.uint8)
        o = (px - inner.shape[0]) // 2
        img[o:o + inner.shape[0], o:o + inner.shape[1]] = inner
        cv2.imwrite(str(path), img)
    return path


def g(kind, pos, size, rgba="0.5 0.5 0.5 1", extra=""):
    return (f'<geom type="{kind}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{" ".join(f"{s:.4f}" for s in size)}" rgba="{rgba}" {NOPHYS} {extra}/>')


def forklift_xml(name: str, f: dict, lift: float) -> str:
    hx, hw = f["body_length"] / 2, f["body_width"] / 2
    fl, fw, ft, fs = f["fork_length"], f["fork_width"], f["fork_thickness"], f["fork_spacing"] / 2
    cx, mh, wb = f["carriage_x"], f["mast_height"], f["wheelbase"]
    z0 = f["fork_clearance"] + lift
    geoms = [
        g("box", (-hx, 0, 0.085), (hx, hw, 0.055), extra='name="%s_chassis"' % name),
        g("box", (-2 * hx + 0.04, 0, 0.17), (0.04, hw - 0.01, 0.035), "0.12 0.14 0.16 1"),
        g("box", (-hx, 0, 0.155), (0.045, 0.05, 0.015), "0.12 0.14 0.16 1"),
        f'<geom type="capsule" fromto="0 {hw - 0.015:.4f} 0.14 -0.15 {hw - 0.015:.4f} 0.27" size="0.006" rgba="0.12 0.14 0.16 1" {NOPHYS}/>',
        f'<geom type="capsule" fromto="0 {-hw + 0.015:.4f} 0.14 -0.15 {-hw + 0.015:.4f} 0.27" size="0.006" rgba="0.12 0.14 0.16 1" {NOPHYS}/>',
        g("box", (-0.08, 0, 0.275), (0.09, hw - 0.01, 0.006), "0.12 0.14 0.16 1"),
        g("cylinder", (-0.08, 0, 0.295), (0.035, 0.015), "0.12 0.14 0.16 1"),
        g("cylinder", (-0.08, 0, 0.313), (0.02, 0.004), "0.85 0.2 0.2 1"),
        g("box", (0.012, fs + 0.012, mh / 2), (0.008, 0.008, mh / 2), "0.55 0.58 0.62 1"),
        g("box", (0.012, -fs - 0.012, mh / 2), (0.008, 0.008, mh / 2), "0.55 0.58 0.62 1"),
        g("box", (cx + 0.004, 0, z0 + 0.045), (0.004, fs + 0.02, 0.045), "0.55 0.58 0.62 1"),
        g("box", (cx + fl / 2, fs, z0 + ft / 2), (fl / 2, fw / 2, ft / 2), "0.55 0.58 0.62 1"),
        g("box", (cx + fl / 2, -fs, z0 + ft / 2), (fl / 2, fw / 2, ft / 2), "0.55 0.58 0.62 1"),
    ]
    for x, y, r, w in ((0, hw - 0.012, 0.035, 0.012), (0, -hw + 0.012, 0.035, 0.012),
                       (-wb, hw - 0.015, 0.028, 0.011), (-wb, -hw + 0.015, 0.028, 0.011)):
        geoms.append(g("cylinder", (x, y, r), (r, w), "0.1 0.1 0.1 1", 'euler="1.5708 0 0"'))
    return f'<body name="{name}" mocap="true" pos="0 200 0">{"".join(geoms)}</body>'


def pallet_xml(name: str, p: dict, scale: float, rng: np.random.Generator) -> str:
    pd, pw, ph = p["depth"] * scale / 2, p["width"] * scale / 2, p["height"] * scale
    dt, sw = p["deck_thickness"] * scale, p["stringer_width"] * scale / 2
    leg = (ph - dt) / 2
    ms = p["marker_size"] / 2
    load_h = rng.uniform(0.03, 0.12)
    geoms = [g("box", (pd, 0, ph - dt / 2), (pd, pw, dt / 2), extra='name="%s_deck"' % name)]
    for y in (pw - sw, 0, -pw + sw):
        geoms.append(g("box", (pd, y, leg), (pd, sw, leg), extra='name="%s_str%d"' % (name, len(geoms))))
    geoms.append(g("box", (pd, 0, ph + load_h / 2), (pd - 0.01, pw - 0.01, load_h / 2), "0.80 0.66 0.45 1",
                   'name="%s_load"' % name))
    geoms.append(f'<geom type="box" pos="0.0105 0 {ph + min(load_h / 2, 0.035):.4f}" size="0.0004 {ms:.4f} '
                 f'{ms:.4f}" material="aruco" {NOPHYS}/>')
    return f'<body name="{name}" mocap="true" pos="0 200 0">{"".join(geoms)}</body>'


def obstacle_xml(name: str, rng: np.random.Generator) -> str:
    kind = rng.choice(["box", "box", "can", "ball", "cone", "pile"])
    if kind == "box":
        s = rng.uniform(0.015, 0.09, 3)
        geoms = [g("box", (0, 0, s[2]), s)]
    elif kind == "can":
        r, h = rng.uniform(0.012, 0.05), rng.uniform(0.02, 0.10)
        geoms = [g("cylinder", (0, 0, h), (r, h))]
    elif kind == "ball":
        r = rng.uniform(0.015, 0.05)
        geoms = [g("sphere", (0, 0, r), (r,))]
    elif kind == "cone":  # traffic cone: base plate + 3 shrinking cylinders
        r, h = rng.uniform(0.02, 0.04), rng.uniform(0.05, 0.11)
        geoms = [g("box", (0, 0, 0.003), (r * 1.3, r * 1.3, 0.003))]
        for k in range(3):
            geoms.append(g("cylinder", (0, 0, 0.006 + h / 6 * (2 * k + 1)), (r * (1 - 0.28 * k), h / 6)))
    else:  # pile of small boxes
        geoms = []
        for _ in range(rng.integers(2, 5)):
            s = rng.uniform(0.012, 0.035, 3)
            o = rng.uniform(-0.04, 0.04, 2)
            geoms.append(g("box", (o[0], o[1], s[2] + rng.uniform(0, 0.02)), s))
    return f'<body name="{name}" mocap="true" pos="0 200 0">{"".join(geoms)}</body>'


def rack_xml(name: str, rng: np.random.Generator) -> tuple[str, float]:
    d, w, h = rng.uniform(0.22, 0.3), rng.uniform(0.25, 0.6), rng.uniform(0.25, 0.45)
    shelf_z = 0.06
    geoms = [g("box", (sx * d / 2, sy * w / 2, h / 2), (0.008, 0.008, h / 2))
             for sx in (-1, 1) for sy in (-1, 1)]
    for z in (shelf_z, h * 0.6, h):
        geoms.append(g("box", (0, 0, z - 0.006), (d / 2 + 0.01, w / 2 + 0.01, 0.006)))
    return f'<body name="{name}" mocap="true" pos="0 200 0">{"".join(geoms)}</body>', shelf_z


def build_scene(cfg: dict, rng: np.random.Generator, asset_dir: Path, n=dict(pal=6, fl=3, obs=8, rack=3, tape=8)):
    f, p = cfg["forklift"], cfg["pallet"]
    tex = aruco_png(asset_dir / "aruco_4x4_50_id7.png").as_posix()
    c1, c2 = rng.uniform(0.25, 0.85, 3), rng.uniform(0.25, 0.85, 3)
    builtin = rng.choice(["checker", "flat", "gradient"])
    floor_tex = (f'<texture name="floor" type="2d" builtin="{builtin}" rgb1="{c1[0]:.2f} {c1[1]:.2f} {c1[2]:.2f}" '
                 f'rgb2="{c2[0]:.2f} {c2[1]:.2f} {c2[2]:.2f}" width="512" height="512"/>')
    s1, s2 = rng.uniform(0.3, 0.9, 3), rng.uniform(0.1, 0.5, 3)
    sky = (f'<texture type="skybox" builtin="gradient" rgb1="{s1[0]:.2f} {s1[1]:.2f} {s1[2]:.2f}" '
           f'rgb2="{s2[0]:.2f} {s2[1]:.2f} {s2[2]:.2f}" width="256" height="256"/>')
    bodies = []
    pallet_scale = rng.uniform(0.85, 1.25)
    for k in range(n["pal"]):
        bodies.append(pallet_xml(f"pal{k}", p, pallet_scale, rng))
    for k in range(n["fl"]):
        bodies.append(forklift_xml(f"fl{k}", f, rng.uniform(0, f["lift_max"])))
    for k in range(n["obs"]):
        bodies.append(obstacle_xml(f"obs{k}", rng))
    shelf_z = 0.06
    for k in range(n["rack"]):
        xml, shelf_z = rack_xml(f"rack{k}", rng)
        bodies.append(xml)
    for k in range(n["tape"]):
        bodies.append(f'<body name="tape{k}" mocap="true" pos="0 200 0">'
                      f'{g("box", (0, 0, 0.0006), (rng.uniform(0.3, 1.5), rng.uniform(0.006, 0.015), 0.0005), "0.95 0.8 0.1 1")}</body>')
    walls = "".join(f'<geom name="wall{k}" type="box" pos="{x} {y} 1.5" size="{sx} {sy} 1.5" rgba="0.7 0.7 0.7 1" {NOPHYS}/>'
                    for k, (x, y, sx, sy) in enumerate(((4, 0, 0.05, 4), (-4, 0, 0.05, 4), (0, 4, 4, 0.05), (0, -4, 4, 0.05))))
    xml = f"""
<mujoco model="vision_synth">
  <statistic extent="2" center="0 0 0.3"/>  <!-- fixes znear/zfar scale (hidden bodies sit far away) -->
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality shadowsize="2048" offsamples="4"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <map znear="0.005" zfar="30"/>
  </visual>
  <asset>
    {floor_tex}
    {sky}
    <material name="floor" texture="floor" texrepeat="{rng.uniform(4, 16):.1f} {rng.uniform(4, 16):.1f}" reflectance="{rng.uniform(0, 0.15):.2f}"/>
    <texture name="aruco" type="2d" file="{tex}"/>
    <material name="aruco" texture="aruco"/>
  </asset>
  <worldbody>
    <light name="l0" pos="0 0 2.5" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light name="l1" pos="1.5 1.5 2.0" dir="-0.5 -0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <light name="l2" pos="-1.5 -1 2.0" dir="0.5 0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.05" material="floor" {NOPHYS}/>
    {walls}
    {"".join(bodies)}
  </worldbody>
</mujoco>"""
    return xml, shelf_z


# ---------------------------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, cfg, rng, asset_dir):
        xml, self.shelf_z = build_scene(cfg, rng, asset_dir)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.rng = rng
        self.r = mujoco.Renderer(self.m, H, W)
        # segmentation needs its own context without multisampling: MSAA blends the id colours at edges and
        # decodes them as random other objects, which inflates the boxes
        self.m.vis.quality.offsamples = 0
        self.rs = mujoco.Renderer(self.m, H, W)
        self.rs.enable_segmentation_rendering()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        m = self.m
        self.bodies = {}
        for b in range(m.nbody):
            name = m.body(b).name
            for kind in ("pal", "fl", "obs", "rack", "tape"):
                if name.startswith(kind) and name[len(kind):].isdigit():
                    self.bodies.setdefault(kind, []).append(b)
        self.cls_of_body = {}
        for kind, cls in (("pal", "pallet"), ("fl", "forklift"), ("obs", "obstacle")):
            for b in self.bodies[kind]:
                self.cls_of_body[b] = CLS[cls]
        self.geom_body = m.geom_bodyid.copy()
        self.base_rgba = m.geom_rgba.copy()
        self.base_light = m.light_diffuse.copy()
        self.walls = [m.geom(f"wall{k}").id for k in range(4)]

    def close(self):
        self.r.close()
        self.rs.close()

    def place(self, b, x, y, z, yaw):
        mid = self.m.body_mocapid[b]
        self.d.mocap_pos[mid] = (x, y, z)
        self.d.mocap_quat[mid] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))

    def hide_all(self):
        self.d.mocap_pos[:] = HIDDEN  # beyond zfar → culled (below the floor would show in reflections)

    def recolor(self):
        m, rng = self.m, self.rng
        m.geom_rgba[:] = self.base_rgba
        for b in self.bodies["fl"]:
            body = rng.choice([[0.96, 0.71, 0.0], [0.95, 0.5, 0.05], [0.9, 0.15, 0.1], [0.15, 0.35, 0.8],
                               [0.2, 0.6, 0.25]], p=[0.5, 0.2, 0.1, 0.1, 0.1]) * rng.uniform(0.8, 1.1)
            chassis = m.geom(f"{m.body(b).name}_chassis").id
            m.geom_rgba[chassis, :3] = np.clip(body, 0, 1)
        for b in self.bodies["pal"]:
            wood = rng.choice([[0.36, 0.55, 0.78], [0.62, 0.45, 0.25], [0.75, 0.6, 0.4], [0.3, 0.3, 0.32],
                               [0.15, 0.5, 0.2]])
            for gi in np.where(self.geom_body == b)[0]:
                gname = m.geom(gi).name
                if gname.endswith("_load"):
                    m.geom_rgba[gi, :3] = rng.choice([[0.8, 0.66, 0.45], [0.7, 0.55, 0.35], [0.9, 0.9, 0.88],
                                                      rng.uniform(0.1, 0.9, 3)])
                elif gname:
                    m.geom_rgba[gi, :3] = np.clip(np.array(wood) * rng.uniform(0.85, 1.15), 0, 1)
        for b in self.bodies["obs"]:
            col = rng.choice([[0.95, 0.4, 0.05], [0.7, 0.55, 0.35], [0.1, 0.3, 0.8], [0.8, 0.1, 0.1],
                              rng.uniform(0, 1, 3)])
            for gi in np.where(self.geom_body == b)[0]:
                m.geom_rgba[gi, :3] = np.clip(np.array(col) * rng.uniform(0.85, 1.1), 0, 1)
        for b in self.bodies["rack"]:
            col = rng.choice([[0.62, 0.42, 0.25], [0.2, 0.3, 0.6], [0.5, 0.5, 0.52], [0.85, 0.45, 0.1]])
            for gi in np.where(self.geom_body == b)[0]:
                m.geom_rgba[gi, :3] = col
        wall = rng.uniform(0.35, 0.95, 3)
        for gi in self.walls:
            m.geom_rgba[gi, :3] = wall * rng.uniform(0.9, 1.0)

    def light(self, level: str) -> float:
        k = self.rng.uniform(*LIGHT_LEVELS[level])
        tint = self.rng.uniform(0.9, 1.1, 3)
        self.m.light_diffuse[:] = np.clip(self.base_light * k * tint, 0, 1.5)
        self.m.vis.headlight.ambient[:] = 0.3 * k
        self.m.vis.headlight.diffuse[:] = 0.35 * k
        return k

    def frame(self, person_cutouts, person_prob):
        rng = self.rng
        self.hide_all()
        self.recolor()
        level = rng.choice(list(LIGHT_LEVELS), p=[0.3, 0.4, 0.3])
        k = self.light(level)
        # camera: fork carriage height (+ lift) or a bit higher, looking ahead and slightly down
        cx, cy = rng.uniform(-1.5, 1.5, 2)
        cz = rng.choice([rng.uniform(0.03, 0.22), rng.uniform(0.2, 0.6)], p=[0.7, 0.3])
        yaw = rng.uniform(-math.pi, math.pi)
        pitch = -rng.uniform(2, 30) if cz < 0.25 else -rng.uniform(10, 45)
        fovy = rng.uniform(38, 56)
        self.m.vis.global_.fovy = fovy
        hfov = 2 * math.degrees(math.atan(math.tan(math.radians(fovy) / 2) * W / H))
        placed = []  # (x, y, radius)

        def spot(radius, near=0.12, far=2.8):
            for _ in range(40):
                dist = math.exp(rng.uniform(math.log(near), math.log(far)))
                bearing = math.radians(rng.uniform(-hfov / 2 - 15, hfov / 2 + 15))
                x, y = cx + dist * math.cos(yaw + bearing), cy + dist * math.sin(yaw + bearing)
                if math.hypot(x - cx, y - cy) < radius + 0.05:
                    continue
                if all(math.hypot(x - px, y - py) > radius + pr for px, py, pr in placed):
                    placed.append((x, y, radius))
                    return x, y
            return None

        racks = list(rng.permutation(self.bodies["rack"]))[:rng.integers(0, 3)]
        pallets = list(rng.permutation(self.bodies["pal"]))[:rng.integers(0, 5)]
        for b in racks:
            s = spot(0.35, 0.3, 3.0)
            if s is None:
                continue
            ryaw = rng.uniform(-math.pi, math.pi)
            self.place(b, s[0], s[1], 0, ryaw)
            if pallets and rng.random() < 0.6:  # pallet on the lower shelf (rack level, task elevation)
                pb = pallets.pop()
                self.place(pb, s[0] - 0.1 * math.cos(ryaw), s[1] - 0.1 * math.sin(ryaw), self.shelf_z, ryaw)
        for b in pallets:
            s = spot(0.17, 0.18, 2.8)
            if s:
                self.place(b, s[0], s[1], 0, rng.uniform(-math.pi, math.pi))
        for b in list(rng.permutation(self.bodies["fl"]))[:rng.choice([0, 1, 1, 2, 3])]:
            s = spot(0.32, 0.35, 3.0)
            if s:
                self.place(b, s[0], s[1], 0, rng.uniform(-math.pi, math.pi))
        for b in list(rng.permutation(self.bodies["obs"]))[:rng.integers(0, 6)]:
            s = spot(0.08, 0.18, 2.2)
            if s:
                self.place(b, s[0], s[1], 0, rng.uniform(-math.pi, math.pi))
        for b in self.bodies["tape"]:
            if rng.random() < 0.5:
                self.place(b, rng.uniform(-3, 3), rng.uniform(-3, 3), 0, rng.choice([0, math.pi / 2]))
        mujoco.mj_forward(self.m, self.d)

        fwd = np.array([math.cos(yaw) * math.cos(math.radians(pitch)), math.sin(yaw) * math.cos(math.radians(pitch)),
                        math.sin(math.radians(pitch))])
        self.cam.lookat[:] = np.array([cx, cy, cz]) + fwd
        self.cam.distance = 1.0
        self.cam.azimuth = math.degrees(yaw)
        self.cam.elevation = pitch
        self.r.update_scene(self.d, camera=self.cam)
        rgb = self.r.render().copy()
        refl = self.m.mat_reflectance.copy()
        self.m.mat_reflectance[:] = 0  # mirrored objects in a shiny floor must not enlarge the boxes
        self.rs.update_scene(self.d, camera=self.cam)
        seg = self.rs.render()
        self.m.mat_reflectance[:] = refl
        inst = np.full((H, W), -1, np.int32)
        geom_px = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        inst[geom_px] = self.geom_body[seg[..., 0][geom_px]]
        cls_of = dict(self.cls_of_body)

        img = rgb[:, :, ::-1].copy()  # BGR
        n_persons = 0
        if person_cutouts and rng.random() < person_prob:
            for _ in range(rng.choice([1, 1, 1, 2, 3])):
                pid = 100000 + n_persons
                if paste_person(img, inst, pid, person_cutouts, rng, k):
                    cls_of[pid] = CLS["person"]
                    n_persons += 1
        labels = boxes_from_instances(inst, cls_of)
        img = degrade(img, rng)
        return img, labels, {"light": level, "light_k": round(float(k), 3), "cam_z": round(float(cz), 3),
                             "fovy": round(float(fovy), 1), "persons_pasted": n_persons}


def boxes_from_instances(inst: np.ndarray, cls_of: dict, min_area: int = 30, min_side: int = 4) -> list:
    out = []
    h, w = inst.shape
    for iid, cls in cls_of.items():
        ys, xs = np.nonzero(inst == iid)
        if len(xs) < min_area:
            continue
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if x2 - x1 < min_side or y2 - y1 < min_side:
            continue
        out.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return out


def paste_person(img, inst, pid, cutouts, rng, light_k) -> bool:
    path = cutouts[rng.integers(len(cutouts))]
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        return False
    th = int(rng.uniform(70, 560))  # person height in pixels (near people tower over a 1:14 camera)
    s = th / rgba.shape[0]
    tw = max(4, int(rgba.shape[1] * s))
    rgba = cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    foot_y = int(rng.uniform(0.45, 1.05) * H)
    x0, y0 = int(rng.uniform(-0.2 * tw, W - 0.8 * tw)), foot_y - th
    xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(W, x0 + tw), min(H, y0 + th)
    if xs1 - xs0 < 4 or ys1 - ys0 < 4:
        return False
    crop = rgba[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0].astype(np.float32)
    alpha = cv2.GaussianBlur(crop[..., 3] / 255.0, (3, 3), 0)[..., None]
    fg = np.clip(crop[..., :3] * np.clip(light_k * rng.uniform(0.8, 1.1), 0.25, 1.3), 0, 255)
    roi = img[ys0:ys1, xs0:xs1].astype(np.float32)
    img[ys0:ys1, xs0:xs1] = (alpha * fg + (1 - alpha) * roi).astype(np.uint8)
    inst[ys0:ys1, xs0:xs1][alpha[..., 0] > 0.5] = pid
    return True


def degrade(img, rng):
    if rng.random() < 0.25:  # motion blur (moving forklift)
        k = int(rng.integers(3, 9))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        if rng.random() < 0.5:
            kern = kern.T
        img = cv2.filter2D(img, -1, kern)
    elif rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    sigma = rng.uniform(0, 7)
    if sigma > 0.5:
        img = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(60, 96))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/raw/synth_mujoco")
    ap.add_argument("--frames-per-scene", type=int, default=60)
    ap.add_argument("--person-cutouts", type=Path)
    ap.add_argument("--person-prob", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=14)
    args = ap.parse_args()

    cfg = yaml.safe_load(TRAINING_CFG.read_text())
    rng = np.random.default_rng(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    cutouts = sorted(args.person_cutouts.glob("*.png")) if args.person_cutouts else []
    print(f"{len(cutouts)} person cut-outs")
    meta = open(args.out / "frames.jsonl", "w")
    counts = [0] * len(CLASSES)
    i = 0
    scene = None
    while i < args.n:
        if i % args.frames_per_scene == 0:
            if scene:
                scene.close()
            scene = Scene(cfg, rng, args.out.parent / "synth_assets")
        img, labels, info = scene.frame(cutouts, args.person_prob)
        i += 1
        name = f"synth_{i:06d}"
        cv2.imwrite(str(args.out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (args.out / "labels" / f"{name}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
        for c in {lab[0] for lab in labels}:
            counts[c] += 1
        meta.write(json.dumps({"image": name, **info, "boxes": len(labels)}) + "\n")
        if i % 200 == 0:
            print(f"{i}/{args.n}  images per class so far: {dict(zip(CLASSES, counts))}")
    meta.close()
    print(f"done: {args.n} images -> {args.out}; images per class {dict(zip(CLASSES, counts))}")


if __name__ == "__main__":
    main()
