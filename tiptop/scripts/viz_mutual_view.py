"""Mutual-visibility check of tiptop's camera calibration on the bimanual YAM.

Two commands, deliberately separate:

``pixi run viz-mutual-pose``
    Hand-guide both arms into ONE pose in which each wrist camera sees the other arm's gripper and
    the top camera sees both. While you pose, every camera's image is shown in Rerun with both
    arms' silhouettes drawn on it, and the terminal reports how much of each image each arm covers.
    Press ``r`` to record; then ``s`` save and quit, ``t`` retry, ``q`` quit without saving.

``pixi run viz-mutual-result``
    Move both arms to the saved pose, capture one frame per camera and show, in Rerun, the captured
    images overlaid with the embodiment mask and the live point clouds with the robot.

Everything is drawn in tiptop's frame, from tiptop's own calibration entries and the URDF cuTAMP
plans with -- not raiden's -- so a mask that misses the real arm means tiptop's belief about the
robot and cameras is wrong, wherever the error comes from:

    top camera:    world_from_cam                                (``cameras.hand.mount: world``)
    wrist camera:  FK_urdf(q)[<arm>_grasp_frame] @ ee_from_cam   (``cameras.external`` = left,
                                                                  ``cameras.external_2`` = right)

Calibration entries are looked up by camera serial exactly as a tiptop run does
(``calibration_info_<DC_WORKSPACE>.json``), or from ``--calibration-file``. Produce them with
``scripts/raiden_to_tiptop_calib.py``. Run with ``TIPTOP_CONFIG=tiptop_yam_real.yml`` and the arm
server up (see YAM.md).

Occlusion between the arms is not modelled (each arm's silhouette is drawn independently), and
lens distortion is ignored (the RealSense colour stream tiptop uses reports none that it applies).
"""

import json
import logging
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh
from cutamp.robots import bimanual_yam
from cutamp.robots.bimanual_yam import YAM_FINGER_OPEN
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from tiptop.config import load_calibration, tiptop_cfg
from tiptop.perception.cameras import _build_camera, camera_mount
from tiptop.utils import get_robot_client, get_tiptop_cache_dir, setup_logging

_log = logging.getLogger(__name__)

SIDES = ("left", "right")
_NEAR = 0.03  # triangles with a vertex closer to the camera than this (m) are dropped
_ALPHA = 0.45
MIN_VISIBLE_FRAC = 0.01  # an arm is "in view" when its silhouette covers this fraction of the image
_WARMUP_FRAMES = 8
REOPEN_INTERVAL_S = 2.0  # how often a dropped camera is tried again; each try stalls the loop while it opens
_SIDE_RGB = {"left": (60, 120, 220), "right": (220, 80, 60)}

# slot in tiptop's `cameras:` config -> (name used here, arm the camera is mounted on, or None)
_SLOTS = {"hand": ("top_cam", None), "external": ("left_wrist_cam", "left"), "external_2": ("right_wrist_cam", "right")}


# ---------------------------------------------------------------------------
# Geometry: URDF meshes posed by forward kinematics
# ---------------------------------------------------------------------------


def default_urdf_path() -> Path:
    return Path(bimanual_yam.__file__).parent / "assets" / "yam_description" / "bimanual_yam.urdf"


class YamMeshes:
    """Both arms' visual meshes from the URDF, posed from measured joints, in the URDF world frame."""

    def __init__(self, urdf_path: Path):
        self.urdf = URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=True)
        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._parts: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {s: [] for s in SIDES}
        for name, link in self.urdf.link_map.items():
            side = next((s for s in SIDES if name.startswith(s)), None)
            if side is None:
                continue
            for visual in link.visuals:
                mesh = visual.geometry.mesh
                if mesh is None:
                    continue
                if mesh.filename not in cache:
                    m = trimesh.load(str(urdf_path.parent / mesh.filename), force="mesh", process=False)
                    cache[mesh.filename] = (np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
                verts, faces = cache[mesh.filename]
                if mesh.scale is not None:
                    verts = verts * np.asarray(mesh.scale)
                if visual.origin is not None:
                    verts = verts @ visual.origin[:3, :3].T + visual.origin[:3, 3]
                self._parts[side].append((name, verts, faces))
        if not all(self._parts.values()):
            raise RuntimeError(f"no visual meshes found for both arms in {urdf_path}")

    def set_state(self, arms: dict, signs: tuple[float, ...] = (1.0,) * 6) -> None:
        """``arms[side] = {"q": 6 joints, "gripper": 0 (closed)..1 (open)}``. ``signs`` flips joints for diagnosis."""
        cfg = {}
        for side in SIDES:
            q = np.asarray(arms[side]["q"], dtype=np.float64)[:6]
            for i in range(6):
                cfg[f"{side}_joint{i + 1}"] = float(signs[i] * q[i])
            finger = YAM_FINGER_OPEN * float(arms[side].get("gripper", 1.0))
            cfg[f"{side}_left_finger"] = finger
            cfg[f"{side}_right_finger"] = finger
        self.urdf.update_cfg(cfg)

    def grasp_frame(self, side: str) -> np.ndarray:
        """``<side>_grasp_frame`` in world: tiptop's ee_link."""
        return np.asarray(self.urdf.get_transform(f"{side}_grasp_frame", "world"), dtype=np.float64)

    def mesh(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """The arm as one (vertices in world, triangles) pair."""
        verts, faces, offset = [], [], 0
        for link, v, f in self._parts[side]:
            T = np.asarray(self.urdf.get_transform(link, "world"), dtype=np.float64)
            verts.append(v @ T[:3, :3].T + T[:3, 3])
            faces.append(f + offset)
            offset += len(v)
        return np.concatenate(verts), np.concatenate(faces)

    def meshes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {s: self.mesh(s) for s in SIDES}

    def world_parts(self, side: str):
        """(link, vertices in the link frame, triangles, world_from_link) for each visual mesh of an arm."""
        for link, verts, faces in self._parts[side]:
            yield link, verts, faces, np.asarray(self.urdf.get_transform(link, "world"), dtype=np.float64)


def render_masks(
    world_from_cam: np.ndarray, K: np.ndarray, size: tuple[int, int], meshes: dict
) -> dict[str, np.ndarray]:
    """Per-arm silhouette (uint8 0/1, image-sized) of the meshes seen from the camera.

    Triangles are filled one at a time: a single cv2.fillPoly over all of them uses the even-odd
    rule, so overlapping front and back faces would cut holes in the silhouette.
    """
    W, H = size
    cam_from_world = np.linalg.inv(world_from_cam)
    K = np.asarray(K, dtype=np.float64)
    masks = {}
    for side, (verts, faces) in meshes.items():
        pc = verts @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
        in_front = pc[:, 2] > _NEAR
        keep = in_front[faces].all(axis=1)
        mask = np.zeros((H, W), np.uint8)
        if keep.any():
            z = np.where(in_front, pc[:, 2], 1.0)
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
            uv = np.clip(np.stack([u, v], axis=1), -20000, 20000).astype(np.int32)
            tris = uv[faces[keep]]
            lo, hi = tris.min(axis=1), tris.max(axis=1)
            on_screen = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < W) & (lo[:, 1] < H)
            for tri in tris[on_screen]:
                cv2.fillConvexPoly(mask, tri, 1)
        masks[side] = mask
    return masks


def coverage(masks: dict[str, np.ndarray]) -> dict[str, float]:
    return {side: float(m.mean()) for side, m in masks.items()}


def overlay(rgb: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """``rgb`` with each arm's silhouette tinted and outlined (left blue, right red)."""
    out = rgb.copy()
    for side, m in masks.items():
        if not m.any():
            continue
        color = np.array(_SIDE_RGB[side], dtype=np.float64)
        sel = m.astype(bool)
        out[sel] = (out[sel] * (1.0 - _ALPHA) + color * _ALPHA).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 2)
    return out


def in_view(side: Optional[str], cov: dict[str, float]) -> bool:
    """A wrist camera must see the OTHER gripper; the top camera must see both."""
    needed = list(SIDES) if side is None else [s for s in SIDES if s != side]
    return all(cov.get(s, 0.0) >= MIN_VISIBLE_FRAC for s in needed)


# ---------------------------------------------------------------------------
# Cameras and calibration
# ---------------------------------------------------------------------------


@dataclass
class Cam:
    name: str
    slot: str
    side: Optional[str]  # arm it is mounted on; None for the fixed top camera
    camera: Optional[object]  # None while a dropped camera waits to be reopened
    serial: str
    pose: np.ndarray  # world_from_cam (top) or ee_from_cam (wrist)
    cfg: object  # its cameras.<slot> config and depth setting, kept so it can be reopened
    depth: bool
    retry_at: float = 0.0


def pose6_to_matrix(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def load_pose(serial: str, calibration_file: Optional[str]) -> np.ndarray:
    """The calibration entry for a camera serial, as tiptop itself reads it."""
    if calibration_file is None:
        return load_calibration(serial)
    with open(calibration_file) as f:
        entries = json.load(f)
    if serial not in entries:
        raise ValueError(f"{serial} not in {calibration_file} (has {sorted(entries)})")
    return pose6_to_matrix(entries[serial]["pose"])


def open_cams(calibration_file: Optional[str], depth: bool, wrist_depth: bool = False) -> list[Cam]:
    """Open every configured camera that has a calibration entry; warn about and skip the rest."""
    cfg = tiptop_cfg()
    if camera_mount("hand") != "world":
        raise RuntimeError("cameras.hand.mount must be 'world' for the YAM (use TIPTOP_CONFIG=tiptop_yam_real.yml)")
    cams = []
    for slot, (name, side) in _SLOTS.items():
        cam_cfg = cfg.cameras.get(slot)
        if cam_cfg is None:
            _log.warning(f"{name}: cameras.{slot} is not configured; skipping")
            continue
        serial = str(cam_cfg.serial)
        try:
            pose = load_pose(serial, calibration_file)
        except ValueError as e:
            _log.warning(f"{name}: no calibration for serial {serial}; skipping ({e})")
            continue
        cam_depth = depth if side is None else (depth and wrist_depth)
        try:
            camera = _build_camera(cam_cfg, depth=cam_depth)
        except Exception as e:  # a camera that will not open is a hardware problem, not a reason to stop
            _log.warning(f"{name}: could not open serial {serial}: {e}")
            continue
        cams.append(Cam(name, slot, side, camera, serial, pose, cam_cfg, cam_depth))
        _log.info(f"{name}: serial {serial}, {'world_from_cam' if side is None else f'ee_from_cam on {side} arm'}")
    if not cams:
        raise RuntimeError("no camera could be opened with a calibration entry")
    return cams


def camera_pose(cam: Cam, meshes: YamMeshes) -> np.ndarray:
    if cam.side is None:
        return cam.pose
    return meshes.grasp_frame(cam.side) @ cam.pose


def read_arms(client) -> dict:
    return {
        side: {
            "q": np.asarray(state["q"], dtype=np.float64)[:6],
            "gripper": float(state.get("gripper", 1.0)),
        }
        for side in SIDES
        for state in [client.get_arm_state(side)]
    }


def render_all(cams: list[Cam], frames: dict, meshes: YamMeshes, arms: dict, signs) -> dict:
    """{camera name: (rgb, overlay rgb, coverage)} for every camera that delivered a frame."""
    meshes.set_state(arms, signs)
    posed = meshes.meshes()
    out = {}
    for cam in cams:
        frame = frames.get(cam.name)
        if frame is None:
            continue
        rgb = np.asarray(frame.rgb)
        H, W = rgb.shape[:2]
        masks = render_masks(camera_pose(cam, meshes), frame.intrinsics, (W, H), posed)
        out[cam.name] = (rgb, overlay(rgb, masks), coverage(masks))
    return out


def display(cam: Cam, rgb: np.ndarray, flip_right_wrist: bool) -> np.ndarray:
    """The right wrist camera is mounted upside down; flip only what is shown, not what is projected."""
    if flip_right_wrist and cam.side == "right":
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    return np.ascontiguousarray(rgb)


def coverage_text(cams: list[Cam], rendered: dict) -> str:
    parts = []
    for cam in cams:
        if cam.name not in rendered:
            parts.append(f"{cam.name}: --")
            continue
        cov = rendered[cam.name][2]
        mark = "OK" if in_view(cam.side, cov) else "  "
        parts.append(f"{cam.name} L{cov['left'] * 100:4.1f}% R{cov['right'] * 100:4.1f}% {mark}")
    return " | ".join(parts)


def _reopen(cam: Cam) -> None:
    """Open a fresh pipeline for a dropped camera, at most once per ``REOPEN_INTERVAL_S``.

    A wrist D405 on a moving arm can fall off USB and come back under a new /dev/video node; the old
    pipeline is still bound to the vanished one and never reads again, so only a new one recovers.
    """
    if time.monotonic() < cam.retry_at:
        return
    try:
        cam.camera = _build_camera(cam.cfg, depth=cam.depth)
        _log.info(f"{cam.name}: reopened serial {cam.serial}")
    except Exception as e:
        cam.retry_at = time.monotonic() + REOPEN_INTERVAL_S
        _log.warning(f"{cam.name}: still unavailable, retrying every {REOPEN_INTERVAL_S:.0f} s ({e})")


def read_frames(cams: list[Cam]) -> dict:
    frames = {}
    for cam in cams:
        if cam.camera is None:
            _reopen(cam)
        frames[cam.name] = None
        if cam.camera is None:
            continue
        try:
            frames[cam.name] = cam.camera.read_camera()
        except Exception as e:
            _log.warning(f"{cam.name}: read failed, reopening: {e}")
            try:
                cam.camera.close()
            except Exception:
                pass  # the device is already gone; there is nothing left to stop
            cam.camera = None
    return frames


def close_cams(cams: list[Cam]) -> None:
    for cam in cams:
        if cam.camera is None:
            continue
        try:
            cam.camera.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Terminal + robot helpers
# ---------------------------------------------------------------------------


def read_key() -> Optional[str]:
    """One pending key press, or None. Reads the fd: select() cannot see what a buffered read took."""
    fd = sys.stdin.fileno()
    if select.select([fd], [], [], 0)[0]:
        return os.read(fd, 1).decode(errors="ignore").lower()
    return None


class PoseSession:
    """The record -> (save | retry | quit) state machine of ``viz-mutual-pose``, free of I/O."""

    def __init__(self):
        self.recorded: Optional[dict] = None

    def step(self, key: Optional[str], snapshot: dict) -> str:
        """Feed a key press and the current snapshot; returns the action the caller must take:
        ``none`` | ``recorded`` | ``save`` | ``retry`` | ``quit``."""
        if self.recorded is None:
            if key == "q":
                return "quit"
            if key == "r":
                self.recorded = snapshot
                return "recorded"
            return "none"
        if key == "s":
            return "save"
        if key == "t":
            self.recorded = None
            return "retry"
        if key == "q":
            return "quit"
        return "none"


def init_rerun(app_id: str, web: bool, web_port: int, blueprint: Optional[rrb.Blueprint] = None) -> None:
    """Spawn the native viewer, or serve the web viewer when asked or when there is no display (SSH)."""
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not web and not has_display:
        print("\nNo display found: serving the web viewer instead (pass --web to choose this explicitly).")
        web = True
    if web:
        rr.init(app_id)
        grpc_port = web_port + 1
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False)
        print(f"\nOpen in browser: http://localhost:{web_port}?url={quote(server_uri, safe='')}")
        print(f"SSH tunnel:      ssh -L {web_port}:localhost:{web_port} -L {grpc_port}:localhost:{grpc_port} <host>")
    else:
        rr.init(app_id, spawn=True)
    if blueprint is not None:
        rr.send_blueprint(blueprint)


def default_pose_file() -> str:
    return str(get_tiptop_cache_dir() / "mutual_view_pose.json")


def _parse_signs(joint_signs: str) -> tuple[float, ...]:
    signs = tuple(float(x) for x in joint_signs.replace(",", " ").split())
    if len(signs) != 6 or any(abs(s) != 1.0 for s in signs):
        raise ValueError(f"--joint-signs needs six values of +1/-1, got {joint_signs!r}")
    return signs


def _release(client) -> None:
    for side in SIDES:
        res = client.free_drive(arm=side)
        if not res.get("success"):
            raise RuntimeError(f"free_drive({side}) failed: {res.get('error')}")


def _hold(client) -> None:
    for side in SIDES:
        try:
            client.hold(arm=side)
        except Exception as e:
            _log.error(f"hold({side}) failed: {e} -- the arm may still be limp")


# ---------------------------------------------------------------------------
# Command 1: pose the arms
# ---------------------------------------------------------------------------


def viz_mutual_pose(
    calibration_file: Optional[str] = None,
    output_file: Optional[str] = None,
    joint_signs: str = "1 1 1 1 1 1",
    urdf_path: Optional[str] = None,
    flip_right_wrist: bool = True,
    rate: float = 4.0,
    web: bool = False,
    web_port: int = 9090,
):
    """Hand-guide both arms into a mutual-view pose and record exactly one.

    Args:
        calibration_file: tiptop calibration_info json to use instead of the workspace's.
        output_file: where to save the pose (default: tiptop cache dir, mutual_view_pose.json).
        joint_signs: six +1/-1 multipliers applied to the measured joints before the URDF FK. For
            diagnosis only, e.g. "1 1 1 1 1 -1" to test a mirrored wrist-roll joint.
        urdf_path: bimanual_yam.urdf (default: cuTAMP's).
        flip_right_wrist: display the right wrist image rotated 180 deg (mounted upside down).
        rate: viewer updates per second.
        web: serve the viewer over HTTP (for SSH) instead of spawning it.
    """
    setup_logging()
    if not sys.stdin.isatty():
        raise RuntimeError("viz-mutual-pose needs an interactive terminal (single-key input)")
    signs = _parse_signs(joint_signs)
    output_file = output_file or default_pose_file()
    meshes = YamMeshes(Path(urdf_path) if urdf_path else default_urdf_path())

    cams: list[Cam] = []
    client = None
    released = False
    tty_state = termios.tcgetattr(sys.stdin)
    try:
        cams = open_cams(calibration_file, depth=False)
        client = get_robot_client()

        print("\n" + "!" * 60)
        print("  Support BOTH arms by hand before continuing.")
        print("  Releasing makes them limp: an unsupported arm can sag.")
        print("!" * 60)
        input("\n  Press Enter to release the arms...")
        _release(client)
        released = True

        init_rerun("tiptop_mutual_pose", web, web_port)
        print("\nPose the arms so that each wrist camera sees the OTHER gripper and the top camera sees both.")
        print("Rerun shows each arm's projected silhouette (left blue, right red); 'OK' marks a camera that")
        print(f"sees what it needs (>= {MIN_VISIBLE_FRAC:.0%} of the image).")
        print("\n  r  record this pose        q  quit without saving\n")

        tty.setcbreak(sys.stdin.fileno())
        session = PoseSession()
        period = 1.0 / rate
        tick = 0
        while True:
            t0 = time.monotonic()
            rr.set_time("tick", sequence=tick)
            tick += 1

            frames = read_frames(cams)
            arms = read_arms(client)  # right after the frames: same moment
            rendered = render_all(cams, frames, meshes, arms, signs)
            for cam in cams:
                if cam.name in rendered:
                    rr.log(f"posing/{cam.name}", rr.Image(display(cam, rendered[cam.name][1], flip_right_wrist)))

            snapshot = {
                "arms": arms,
                "coverage": {n: v[2] for n, v in rendered.items()},
                "in_view": {c.name: c.name in rendered and in_view(c.side, rendered[c.name][2]) for c in cams},
            }
            action = session.step(read_key(), snapshot)
            if session.recorded is None and action in ("none", "retry"):
                sys.stdout.write("\r\033[K  " + coverage_text(cams, rendered))
                sys.stdout.flush()
            if action == "retry":
                print("\nRetrying -- move the arms and press r again.\n")
            elif action == "recorded":
                print("\n\nRecorded this pose:")
                for cam in cams:
                    cov = snapshot["coverage"].get(cam.name)
                    shown = "no frame" if cov is None else f"L {cov['left']:.1%}  R {cov['right']:.1%}"
                    print(f"  {cam.name:<16} {shown:<22} {'OK' if snapshot['in_view'][cam.name] else 'NOT in view'}")
                print("\n  s  save and quit      t  retry      q  quit without saving")
            elif action == "save":
                data = {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "calibration_file": calibration_file,
                    "joint_signs": list(signs),
                    **{side: {"q": s["q"].tolist(), "gripper": s["gripper"]} for side, s in snapshot["arms"].items()},
                    "coverage": snapshot["coverage"],
                    "in_view": snapshot["in_view"],
                }
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"\n✓ Saved pose to {output_file}\nRun `pixi run viz-mutual-result` to check the calibration.")
                break
            elif action == "quit":
                print("\nQuit without saving.")
                break

            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\nInterrupted (nothing saved).")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, tty_state)
        close_cams(cams)
        if client is not None and released:
            print("\n  !! Support the arms before they re-engage !!")
            try:
                input("  Press Enter to hold the arms where they are...")
            except KeyboardInterrupt:
                print()
            _hold(client)
            print("  (the arms hold their pose; `go-home` moves them back)")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Command 2: show the result
# ---------------------------------------------------------------------------


def _stride_cloud(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, world_from_cam: np.ndarray, stride: int, max_depth: float):
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vv, uu]
    ok = np.isfinite(d) & (d > 0.05) & (d < max_depth)
    u, v, d = uu[ok], vv[ok], d[ok]
    pts = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], axis=1)
    pts = pts @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    return pts.astype(np.float32), rgb[v, u]


def viz_mutual_result(
    calibration_file: Optional[str] = None,
    pose_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    joint_signs: str = "1 1 1 1 1 1",
    urdf_path: Optional[str] = None,
    flip_right_wrist: bool = True,
    move_seconds: float = 6.0,
    wrist_depth: bool = True,
    rate: float = 5.0,
    stride: int = 4,
    max_depth: float = 1.5,
    web: bool = False,
    web_port: int = 9090,
):
    """Move to the saved pose, capture, and show the masks plus the live point clouds.

    Args:
        calibration_file: tiptop calibration_info json to use instead of the workspace's.
        pose_file: pose saved by viz-mutual-pose (default: tiptop cache dir).
        output_dir: where to save captures (default: <cache>/mutual_view/<timestamp>).
        joint_signs: see viz-mutual-pose; pass the same value to compare like with like.
        move_seconds: time per arm to reach the pose; the left arm moves first, then the right.
        wrist_depth: also open the wrist cameras' depth streams for their point clouds.
        rate: viewer updates per second.
        stride: use every N-th pixel for the point clouds.
        max_depth: drop points further than this from the camera, in metres.
    """
    setup_logging()
    signs = _parse_signs(joint_signs)
    pose_file = pose_file or default_pose_file()
    with open(pose_file) as f:
        saved = json.load(f)
    if saved.get("joint_signs") and tuple(saved["joint_signs"]) != signs:
        _log.warning(f"the pose was recorded with joint_signs {saved['joint_signs']}, checking with {list(signs)}")
    targets = {side: np.asarray(saved[side]["q"], dtype=np.float64) for side in SIDES}
    grips = {side: float(saved[side].get("gripper", 1.0)) for side in SIDES}
    out_dir = Path(output_dir or get_tiptop_cache_dir() / "mutual_view" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    meshes = YamMeshes(Path(urdf_path) if urdf_path else default_urdf_path())

    cams: list[Cam] = []
    client = None
    try:
        cams = open_cams(calibration_file, depth=True, wrist_depth=wrist_depth)
        client = get_robot_client()

        print("\n" + "!" * 60)
        print("  The left arm will move straight to the saved pose from where it is, then the right arm.")
        print("  Clear the workspace and keep a hand near the e-stop.")
        print("!" * 60)
        input("\n  Press Enter to move...")
        for side in SIDES:
            res = client.move_to([*targets[side], grips[side]], duration=move_seconds, arm=side)
            if not res.get("success"):
                raise RuntimeError(f"moving the {side} arm to the saved pose failed: {res.get('error')}")
        print("  Settling...")
        time.sleep(1.0)

        frames: dict = {c.name: None for c in cams}
        for _ in range(_WARMUP_FRAMES):
            for cam, frame in zip(cams, read_frames(cams).values()):
                if frame is not None:
                    frames[cam.name] = frame
        arms = read_arms(client)
        for side in SIDES:
            err = float(np.abs(arms[side]["q"] - targets[side]).max())
            if err > 0.05:
                print(f"  ⚠ {side} arm is {np.degrees(err):.1f} deg from the saved pose")
        rendered = render_all(cams, frames, meshes, arms, signs)

        out_dir.mkdir(parents=True, exist_ok=True)
        # One 3D view of "/": the captured overlays ride on the camera frustums instead of getting views of their own.
        # The eye is fixed (behind and to the right of the robot, looking forward over the table) so the view
        # cannot re-frame itself as the point clouds' extent changes from frame to frame.
        eye = rrb.archetypes.EyeControls3D(position=(-0.9, -0.9, 0.9), look_target=(0.25, 0.0, 0.35), eye_up=(0.0, 0.0, 1.0))
        init_rerun("tiptop_mutual_result", web, web_port, rrb.Blueprint(rrb.Spatial3DView(origin="/", name="/", eye_controls=eye)))
        summary = {}
        print("\nCaptured (arm silhouette coverage; OK = camera sees what it needs):")
        for cam in cams:
            if cam.name not in rendered:
                print(f"  {cam.name:<16} no frame")
                continue
            raw, over, cov = rendered[cam.name]
            raw_d, over_d = display(cam, raw, flip_right_wrist), display(cam, over, flip_right_wrist)
            cv2.imwrite(str(out_dir / f"{cam.name}_raw.png"), raw_d[:, :, ::-1])
            cv2.imwrite(str(out_dir / f"{cam.name}_overlay.png"), over_d[:, :, ::-1])
            rr.log(f"world/cameras/{cam.name}", rr.Image(over), static=True)
            summary[cam.name] = {"coverage": cov, "in_view": in_view(cam.side, cov)}
            print(f"  {cam.name:<16} L {cov['left']:.1%}  R {cov['right']:.1%}  {'OK' if summary[cam.name]['in_view'] else ''}")
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"pose_file": pose_file, "calibration_file": calibration_file, "cameras": summary}, f, indent=2)
        print(f"\nSaved captures to {out_dir}")

        # The arms hold still, so the robot and the camera frustums are logged once (static) and only the
        # point clouds are refreshed: nothing about the frame can change between updates.
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        meshes.set_state(arms, signs)
        for side in SIDES:
            for i, (link, verts, faces, world_from_link) in enumerate(meshes.world_parts(side)):
                path = f"world/robot/{side}/{i:02d}_{link}"
                rr.log(path, rr.Transform3D(translation=world_from_link[:3, 3], mat3x3=world_from_link[:3, :3]), static=True)
                rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces, albedo_factor=_SIDE_RGB[side]), static=True)
        poses = {cam.name: camera_pose(cam, meshes) for cam in cams}
        for cam in cams:
            T = poses[cam.name]
            raw = rendered[cam.name][0] if cam.name in rendered else None
            if raw is None:
                continue
            H, W = raw.shape[:2]
            K = frames[cam.name].intrinsics
            rr.log(f"world/cameras/{cam.name}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), static=True)
            rr.log(
                f"world/cameras/{cam.name}",
                rr.Pinhole(image_from_camera=K, resolution=[W, H], image_plane_distance=0.1),
                static=True,
            )
        print("\nStreaming the live point clouds (the arms hold the pose). Compare with the captured overlays:")
        print("the masks should lie on the real arms and the clouds should agree. Ctrl+C to stop.\n")
        period, tick = 1.0 / rate, 0
        while True:
            t0 = time.monotonic()
            rr.set_time("tick", sequence=tick)
            tick += 1
            live = read_frames(cams)
            for cam in cams:
                frame = live.get(cam.name)
                if frame is None or frame.depth is None:
                    continue
                pts, colors = _stride_cloud(frame.rgb, frame.depth, frame.intrinsics, poses[cam.name], stride, max_depth)
                rr.log(f"world/points/{cam.name}", rr.Points3D(pts, colors=colors, radii=0.002))
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        close_cams(cams)
        if client is not None:
            _hold(client)
        sys.stdout.flush()


def viz_mutual_pose_entrypoint():
    import tyro

    tyro.cli(viz_mutual_pose)


def viz_mutual_result_entrypoint():
    import tyro

    tyro.cli(viz_mutual_result)
