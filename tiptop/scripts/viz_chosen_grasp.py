"""Show the grasp cuTAMP chose, on the point cloud the run perceived it from."""

import logging
from pathlib import Path

import dill
import numpy as np
import open3d as o3d
import rerun as rr
import torch
from omegaconf import OmegaConf

from tiptop.chosen_grasp import CHOSEN_GRASPS_FILENAME, extract_chosen_grasps, load_chosen_grasps
from tiptop.perception.m2t2 import m2t2_to_tiptop_transform
from tiptop.planning import load_tiptop_plan
from tiptop.utils import setup_logging
from tiptop.viz_utils import get_gripper_mesh, get_heatmap

_log = logging.getLogger(__name__)

CHOSEN_COLOR = (0, 255, 0)


def _resolve_arm_dirs(path: Path) -> list[Path]:
    """Accept a run dir, an arm dir, a perception dir, or a file inside one; return the arm dirs.

    An arm dir is one holding ``perception/``. A sequential-bimanual run has one per arm
    (``<run>/left``, ``<run>/right``) and nothing at the run level, so pointing at the run dir gives
    both; every other run has exactly one, the run dir itself.
    """
    if path.is_file():
        path = path.parent
    if path.name == "perception":
        path = path.parent
    if (path / "perception").is_dir():
        return [path]
    arm_dirs = sorted(child for child in path.iterdir() if child.is_dir() and (child / "perception").is_dir())
    if not arm_dirs:
        raise FileNotFoundError(f"No perception/ directory found at, above, or one level below {path}")
    return arm_dirs


def _resolve_plan_path(arm_dir: Path) -> Path | None:
    """The plan whose picks were executed with ``arm_dir``'s perception, or None if there is none.

    A bimanual run writes ``tiptop_plan_<arm>.json`` in the RUN dir (one per arm, alongside the arm
    dirs) and only for an arm that had something to do; every other run writes ``tiptop_plan.json``
    next to its perception.
    """
    candidates = [arm_dir / "tiptop_plan.json", arm_dir.parent / f"tiptop_plan_{arm_dir.name}.json"]
    return next((path for path in candidates if path.exists()), None)


def _resolve_robot_type(arm_dir: Path) -> str:
    """Robot type to run FK with, taken from the run's own config.

    A bimanual run copies ONE ``tiptop.yml`` into both arm dirs, so its ``robot.type`` names
    whichever arm the config was written for -- the arm dir's name is the authority instead.
    """
    robot_type = OmegaConf.load(arm_dir / "tiptop.yml")["robot"]["type"]
    if robot_type.startswith("bimanual_yam_") and arm_dir.name in ("left", "right"):
        return f"bimanual_yam_{arm_dir.name}"
    return robot_type


def _log_gripper(entity: str, world_from_grasp: np.ndarray, color) -> None:
    """Log a skeleton gripper mesh at a grasp pose."""
    mesh = get_gripper_mesh()
    vertices = np.asarray(mesh.vertices)
    vertices = (world_from_grasp @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]
    rr.log(
        entity,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=np.asarray(mesh.triangles),
            vertex_colors=np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1)),
        ),
        static=True,
    )


def _chosen_grasps_for(arm_dir: Path, grasps: dict) -> list[dict]:
    """The arm's chosen grasps, recovered from its plan if the run predates chosen_grasps.json."""
    perception_dir = arm_dir / "perception"
    if (perception_dir / CHOSEN_GRASPS_FILENAME).exists():
        return load_chosen_grasps(perception_dir)

    plan_path = _resolve_plan_path(arm_dir)
    if plan_path is None:
        _log.warning(f"{arm_dir} has no plan — this arm never picked anything")
        return []
    _log.info(f"No {CHOSEN_GRASPS_FILENAME} in {perception_dir}; recovering it from {plan_path.name}")
    with open(perception_dir / "cutamp_env.pkl", "rb") as f:
        cutamp_env = dill.load(f)
    return extract_chosen_grasps(load_tiptop_plan(plan_path), cutamp_env, grasps, _resolve_robot_type(arm_dir))


def _viz_arm(arm_dir: Path, prefix: str, show_candidates: bool, num_candidates: int) -> None:
    """Log one arm's point cloud, its M2T2 candidates, and the grasp(s) actually taken."""
    perception_dir = arm_dir / "perception"
    grasps = torch.load(perception_dir / "grasps.pt", weights_only=False, map_location="cpu")

    pcd = o3d.io.read_point_cloud(str(perception_dir / "pointcloud.ply"))
    rr.log(f"{prefix}pcd", rr.Points3D(positions=np.asarray(pcd.points), colors=np.asarray(pcd.colors)), static=True)

    if show_candidates:
        for obj_name, grasp_dict in grasps.items():
            world_from_grasp = np.asarray(grasp_dict["poses"])[:num_candidates] @ m2t2_to_tiptop_transform()
            colors = get_heatmap(np.asarray(grasp_dict["confidences"])[:num_candidates]) * 255
            for idx, (pose, color) in enumerate(zip(world_from_grasp, colors)):
                _log_gripper(f"{prefix}candidates/{obj_name}/{idx:04d}", pose, color)

    for entry in _chosen_grasps_for(arm_dir, grasps):
        match = entry["m2t2"]
        name = f"{prefix}chosen/{entry['object']}/{entry['step_index']:02d}"
        _log_gripper(name, entry["world_from_grasp"], CHOSEN_COLOR)
        rr.log(f"{name}/origin", rr.Points3D(entry["world_from_grasp"][None, :3, 3], radii=0.005), static=True)
        if match["index"] is None:
            _log.info(f"{entry['label']}: no M2T2 candidates saved for this object")
        else:
            _log.info(
                f"{entry['label']}: M2T2 candidate {match['index']} of "
                f"{len(grasps[entry['object']]['confidences'])} (confidence {match['confidence']:.3f}), "
                f"{match['position_error'] * 1000:.1f}mm / {np.rad2deg(match['rotation_error']):.1f}deg from it"
            )


def viz_chosen_grasp(
    path: str,
    show_candidates: bool = True,
    num_candidates: int = 30,
) -> None:
    """Visualize the M2T2 grasp cuTAMP chose on the point cloud it was chosen from, in Rerun.

    The chosen grasp is drawn in green; the candidates it was picked out of are drawn colored by
    M2T2 confidence (turn them off with ``--no-show-candidates``).

    Args:
        path: A run directory, one of its arm directories, or a file inside either (e.g. the
            ``perception/pointcloud.ply`` you want to look at). A bimanual run directory shows
            both arms at once, each under its own entity prefix.
        show_candidates: Also draw the M2T2 candidates that were NOT chosen.
        num_candidates: How many candidates to draw per object.
    """
    setup_logging()
    arm_dirs = _resolve_arm_dirs(Path(path))
    rr.init(application_id="viz_chosen_grasp", spawn=True)
    for arm_dir in arm_dirs:
        prefix = f"{arm_dir.name}/" if len(arm_dirs) > 1 else ""
        _viz_arm(arm_dir, prefix, show_candidates, num_candidates)


def viz_chosen_grasp_entrypoint():
    import tyro

    tyro.cli(viz_chosen_grasp)


if __name__ == "__main__":
    viz_chosen_grasp_entrypoint()
