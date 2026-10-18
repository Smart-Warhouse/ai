"""MJCF scene for the pallet-docking task: 1:14 forklift (rear-wheel steering), pallet with ArUco, optional rack shelf.

The chassis is driven kinematically through three world-frame joints (x, y, yaw) with stiff velocity actuators,
so the bicycle model decides the motion while MuJoCo still resolves contacts: forks or body touching the pallet
push it, which the environment detects as a collision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"


def aruco_texture(marker_id: int, px: int = 240) -> Path:
    """Real DICT_4X4_50 marker with a white quiet zone, cached as PNG (PC-03)."""
    ASSET_DIR.mkdir(exist_ok=True)
    path = ASSET_DIR / f"aruco_4x4_50_id{marker_id}.png"
    if not path.exists():
        import cv2

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        inner = cv2.aruco.generateImageMarker(dictionary, marker_id, px * 3 // 4)
        img = np.full((px, px), 255, np.uint8)
        o = (px - inner.shape[0]) // 2
        img[o:o + inner.shape[0], o:o + inner.shape[1]] = inner
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return path


def build_xml(cfg: dict) -> str:
    f, p, c = cfg["forklift"], cfg["pallet"], cfg["control"]
    tex = aruco_texture(int(p["marker_id"])).as_posix()
    hx = f["body_length"] / 2
    hw = f["body_width"] / 2
    fl, fw, ft = f["fork_length"], f["fork_width"], f["fork_thickness"]
    fs = f["fork_spacing"] / 2
    cx = f["carriage_x"]
    pd, pw, ph = p["depth"] / 2, p["width"] / 2, p["height"]
    dt_ = p["deck_thickness"]
    sw = p["stringer_width"] / 2
    leg_h = (ph - dt_) / 2
    ms = p["marker_size"] / 2
    wb = f["wheelbase"]
    # robot geoms: group 1, contype 2 (never touch the floor; do touch pallet/shelf)
    rg = 'group="1" contype="2" conaffinity="2"'
    vis = 'group="3" contype="0" conaffinity="0"'
    return f"""
<mujoco model="forklift_docking">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{c['physics_dt']}" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.78 0.80 0.82" rgb2="0.72 0.74 0.77" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="12 12" reflectance="0"/>
    <texture name="aruco" type="2d" file="{tex}"/>
    <material name="aruco" texture="aruco" texuniform="false"/>
    <material name="lift" rgba="0.96 0.71 0.0 1"/>
    <material name="dark" rgba="0.12 0.14 0.16 1"/>
    <material name="steel" rgba="0.55 0.58 0.62 1" reflectance="0.2"/>
    <material name="pallet" rgba="0.36 0.55 0.78 1"/>
    <material name="shelf" rgba="0.62 0.42 0.25 1"/>
  </asset>

  <worldbody>
    <light pos="0.3 -0.6 1.6" dir="-0.2 0.4 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor" contype="1" conaffinity="1" friction="0.9 0.005 0.0001"/>

    <!-- forklift: origin = front (drive) axle centre on the floor, +x forward -->
    <body name="base" pos="0 0 0">
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <inertial pos="{-hx:.4f} 0 0.1" mass="1.6" diaginertia="0.02 0.03 0.03"/>
      <geom name="chassis" type="box" pos="{-hx:.4f} 0 0.085" size="{hx:.4f} {hw:.4f} 0.055" material="lift" {rg}/>
      <geom name="counterweight" type="box" pos="{-2 * hx + 0.04:.4f} 0 0.17" size="0.04 {hw - 0.01:.4f} 0.035" material="dark" {rg}/>
      <geom name="seat" type="box" pos="{-hx:.4f} 0 0.155" size="0.045 0.05 0.015" material="dark" {vis}/>
      <geom name="guard_l" type="capsule" fromto="0.0 {hw - 0.015:.4f} 0.14 -0.15 {hw - 0.015:.4f} 0.27" size="0.006" material="dark" {vis}/>
      <geom name="guard_r" type="capsule" fromto="0.0 {-hw + 0.015:.4f} 0.14 -0.15 {-hw + 0.015:.4f} 0.27" size="0.006" material="dark" {vis}/>
      <geom name="roof" type="box" pos="-0.08 0 0.275" size="0.09 {hw - 0.01:.4f} 0.006" material="dark" {vis}/>
      <geom name="lidar" type="cylinder" pos="-0.08 0 0.295" size="0.035 0.015" material="dark" {vis}/>
      <geom name="lidar_top" type="cylinder" pos="-0.08 0 0.313" size="0.02 0.004" rgba="0.85 0.2 0.2 1" {vis}/>
      <geom name="mast_l" type="box" pos="0.012 {fs + 0.012:.4f} {f['mast_height'] / 2:.4f}" size="0.008 0.008 {f['mast_height'] / 2:.4f}" material="steel" {rg}/>
      <geom name="mast_r" type="box" pos="0.012 {-fs - 0.012:.4f} {f['mast_height'] / 2:.4f}" size="0.008 0.008 {f['mast_height'] / 2:.4f}" material="steel" {rg}/>
      <geom name="wheel_fl" type="cylinder" pos="0 {hw - 0.012:.4f} 0.035" euler="1.5708 0 0" size="0.035 0.012" material="dark" {vis}/>
      <geom name="wheel_fr" type="cylinder" pos="0 {-hw + 0.012:.4f} 0.035" euler="1.5708 0 0" size="0.035 0.012" material="dark" {vis}/>
      <body name="rear_axle" pos="{-wb} 0 0.03">
        <joint name="steer" type="hinge" axis="0 0 1" range="-0.6 0.6" damping="0.01" armature="0.001"/>
        <geom name="wheel_rl" type="cylinder" pos="0 {hw - 0.015:.4f} 0" euler="1.5708 0 0" size="0.028 0.011" material="dark" {vis}/>
        <geom name="wheel_rr" type="cylinder" pos="0 {-hw + 0.015:.4f} 0" euler="1.5708 0 0" size="0.028 0.011" material="dark" {vis}/>
      </body>
      <body name="carriage" pos="{cx} 0 {f['fork_clearance']}" gravcomp="1">
        <joint name="lift" type="slide" axis="0 0 1" range="0 {f['lift_max']}" damping="2" armature="0.05"/>
        <inertial pos="0.05 0 0.02" mass="0.15" diaginertia="0.001 0.001 0.001"/>
        <geom name="backrest" type="box" pos="0.004 0 0.045" size="0.004 {fs + 0.02:.4f} 0.045" material="steel" {rg}/>
        <geom name="fork_l" type="box" pos="{fl / 2:.4f} {fs:.4f} {ft / 2:.4f}" size="{fl / 2:.4f} {fw / 2:.4f} {ft / 2:.4f}" material="steel" {rg}/>
        <geom name="fork_r" type="box" pos="{fl / 2:.4f} {-fs:.4f} {ft / 2:.4f}" size="{fl / 2:.4f} {fw / 2:.4f} {ft / 2:.4f}" material="steel" {rg}/>
        <site name="tip_l" pos="{fl:.4f} {fs:.4f} {ft:.4f}" size="0.004" rgba="1 0 0 1"/>
        <site name="tip_r" pos="{fl:.4f} {-fs:.4f} {ft:.4f}" size="0.004" rgba="1 0 0 1"/>
        <camera name="fork_cam" pos="0.0 0 0.09" xyaxes="0 -1 0 0.34 0 0.94" fovy="55"/>
      </body>
      <site name="tof_l" pos="0.02 {hw - 0.01:.4f} 0.02" size="0.004"/>
      <site name="tof_c" pos="0.02 0 0.02" size="0.004"/>
      <site name="tof_r" pos="0.02 {-hw + 0.01:.4f} 0.02" size="0.004"/>
    </body>

    <!-- pallet: origin = centre of the front face on its support, +x = insertion direction -->
    <body name="pallet" pos="0.6 0 0">
      <freejoint name="pallet"/>
      <inertial pos="{pd:.4f} 0 {ph / 2:.4f}" mass="{p['mass']}" diaginertia="0.002 0.002 0.003"/>
      <geom name="deck" type="box" pos="{pd:.4f} 0 {ph - dt_ / 2:.4f}" size="{pd:.4f} {pw:.4f} {dt_ / 2:.4f}" material="pallet" contype="3" conaffinity="3" friction="0.8 0.005 0.0001"/>
      <geom name="str_l" type="box" pos="{pd:.4f} {pw - sw:.4f} {leg_h:.4f}" size="{pd:.4f} {sw:.4f} {leg_h:.4f}" material="pallet" contype="3" conaffinity="3"/>
      <geom name="str_c" type="box" pos="{pd:.4f} 0 {leg_h:.4f}" size="{pd:.4f} {sw:.4f} {leg_h:.4f}" material="pallet" contype="3" conaffinity="3"/>
      <geom name="str_r" type="box" pos="{pd:.4f} {-pw + sw:.4f} {leg_h:.4f}" size="{pd:.4f} {sw:.4f} {leg_h:.4f}" material="pallet" contype="3" conaffinity="3"/>
      <geom name="load" type="box" pos="{pd:.4f} 0 {ph + 0.04:.4f}" size="{pd - 0.015:.4f} {pw - 0.015:.4f} 0.04" rgba="0.80 0.66 0.45 1" contype="3" conaffinity="3" mass="0.05"/>
      <geom name="marker" type="box" pos="0.0145 0 {ph + 0.04:.4f}" size="0.0004 {ms:.4f} {ms:.4f}" material="aruco" contype="0" conaffinity="0"/>
    </body>

    <!-- rack shelf (mocap): moved under the pallet for elevated episodes, buried otherwise -->
    <body name="shelf" mocap="true" pos="0.6 0 -1">
      <geom name="shelf" type="box" pos="{pd + 0.02:.4f} 0 0" size="{pd + 0.04:.4f} {pw + 0.06:.4f} 0.03" material="shelf" contype="3" conaffinity="3"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="vx" joint="x" kv="400" ctrlrange="-1 1"/>
    <velocity name="vy" joint="y" kv="400" ctrlrange="-1 1"/>
    <velocity name="wz" joint="yaw" kv="60" ctrlrange="-5 5"/>
    <position name="steer" joint="steer" kp="2" ctrlrange="-0.6 0.6"/>
    <position name="lift" joint="lift" kp="400" ctrlrange="0 {f['lift_max']}"/>
  </actuator>
</mujoco>
"""
