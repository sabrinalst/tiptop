"""Recover which M2T2 grasp cuTAMP actually chose, and record it next to the perception outputs.

``perception/grasps.pt`` holds every M2T2 candidate; the saved plan only names the grasp
symbolically (``Pick(block, grasp1, q1)``), so on its own neither file says which candidate the
robot went for. cuTAMP does not optimize grasp parameters -- ``ParticleOptimizer`` only takes
``Pose`` and ``Conf`` parameters, so a grasp particle stays exactly the ``grasps_obj`` entry it was
sampled from -- which is what makes the chosen grasp recoverable rather than merely approximable:

  1. the last configuration of the trajectory before each gripper CLOSE is the grasp configuration,
     so ``FK(q) @ inv(tool_from_ee)`` is the world-frame grasp pose cuTAMP planned to;
  2. expressed in the object frame, that pose matches one entry of ``grasps_obj`` to within the IK
     residual, giving the candidate's index and M2T2 confidence.

The recovered match error is reported per grasp so a bad match is visible rather than silent.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from curobo.types.base import TensorDeviceType
from cutamp.robots import load_robot_container
from cutamp.utils.common import pose_list_to_mat4x4

from tiptop.utils import NumpyEncoder

_log = logging.getLogger(__name__)

CHOSEN_GRASPS_FILENAME = "chosen_grasps.json"
CHOSEN_GRASPS_VERSION = "1.0.0"

# Above this the FK-recovered pose did not land on any candidate, so the match is not trustworthy.
MATCH_POSITION_TOL = 0.01  # metres
MATCH_ROTATION_TOL = np.deg2rad(5.0)


def _objects_from_label(label: str, num_arms: int) -> list[str]:
    """The object each closing hand takes, from a cuTAMP operator label, in the label's arm order.

    A single-hand close (``Pick(block, grasp1, q1)``, ``PickGiver``, ``Handover``) always names its
    object first. The one lockstep close is ``PickBoth(obj_a, grasp_a, obj_b, grasp_b, q)``, whose
    two objects pair with ``arms`` positionally -- the same ``zip(arms, (obj_a, obj_b))`` cuTAMP's
    motion solver uses to attach them.
    """
    match = re.match(r"(\w+)\((.*)\)", label)
    if match is None:
        raise ValueError(f"Could not parse a cuTAMP operator out of the label: {label}")
    op_name, args = match.group(1), [arg.strip() for arg in match.group(2).split(",")]

    if num_arms == 1:
        return [args[0]]
    if op_name == "PickBoth":
        return [args[0], args[2]]
    raise ValueError(f"Don't know which object each of the {num_arms} hands takes in: {label}")


def _rotation_angle(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between rotation matrices, broadcasting over leading dims."""
    trace = np.einsum("...ij,...ij->...", r_a, r_b)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def _match_candidate(obj_from_grasp: np.ndarray, grasp_dict: dict) -> dict:
    """Nearest ``grasps_obj`` candidate to a recovered object-frame grasp pose."""
    candidates = grasp_dict.get("grasps_obj")
    if candidates is None or len(candidates) == 0:
        return {"index": None, "confidence": None, "position_error": None, "rotation_error": None}

    candidates = np.asarray(torch.as_tensor(candidates).cpu().numpy(), dtype=np.float64)
    position_error = np.linalg.norm(candidates[:, :3, 3] - obj_from_grasp[:3, 3], axis=-1)
    rotation_error = _rotation_angle(candidates[:, :3, :3], obj_from_grasp[None, :3, :3])
    # Position dominates; the rotation term only breaks ties between the two jaw-flipped twins of a
    # contact point, which sit at the same point but a pi roll apart (see m2t2.jaw_flip_transform).
    index = int(np.argmin(position_error + 0.01 * rotation_error))

    confidences = grasp_dict.get("confidences")
    confidence = float(np.asarray(confidences)[index]) if confidences is not None and len(confidences) else None
    return {
        "index": index,
        "confidence": confidence,
        "position_error": float(position_error[index]),
        "rotation_error": float(rotation_error[index]),
    }


def _arm_spec(container, arm_name: str | None):
    """The named arm's spec, or None on a single-arm container (whose steps name no arm)."""
    return container.arm(arm_name) if arm_name and container.arms else None


def _fk_matrices(kin_model, q: np.ndarray, ee_link: str | None):
    """World-from-EE 4x4s for a batch of configurations, at ``ee_link`` when the robot is multi-arm."""
    q_t = torch.as_tensor(np.atleast_2d(q), dtype=torch.float32, device=kin_model.tensor_args.device)
    state = kin_model.get_state(q_t)
    pose = state.link_pose[ee_link] if ee_link is not None else state.ee_pose
    return pose.get_matrix().detach().cpu().numpy().astype(np.float64)


def extract_chosen_grasps(tiptop_plan: dict, cutamp_env, grasps: dict, robot_type: str) -> list[dict]:
    """The grasp taken at every gripper CLOSE in ``tiptop_plan``, matched to its M2T2 candidate.

    Object poses are tracked through the plan so a re-pick after a place is expressed against the
    object's pose at that moment, not its initial one.
    """
    container = load_robot_container(robot_type, TensorDeviceType())
    kin_model = container.kin_model

    obj_to_pose = {
        obj.name: pose_list_to_mat4x4(obj.pose).numpy().astype(np.float64)
        for obj in cutamp_env.movables + cutamp_env.statics
    }

    chosen: list[dict] = []
    last_q = np.asarray(tiptop_plan["q_init"], dtype=np.float64)
    grasped: dict[str, tuple[str, np.ndarray]] = {}  # arm -> (object, ee_from_obj)

    for step_index, step in enumerate(tiptop_plan["steps"]):
        if step["type"] == "trajectory":
            positions = np.asarray(step["positions"], dtype=np.float64)
            # Carry any grasped object along with the hand holding it, so the next pick of that
            # object sees where it actually ended up.
            for arm_name, (obj_name, ee_from_obj) in grasped.items():
                spec = _arm_spec(container, arm_name)
                world_from_ee = _fk_matrices(kin_model, positions[-1], spec.ee_link if spec else None)[0]
                obj_to_pose[obj_name] = world_from_ee @ ee_from_obj
            last_q = positions[-1]
            continue

        if step["type"] != "gripper":
            raise ValueError(f"Unknown step type in tiptop plan: {step['type']}")

        # A close may actuate several hands at once on cuTAMP's dual-arm path; single-arm steps
        # carry neither key and are handled as one unnamed hand.
        arms = step.get("arms") or ([step["arm"]] if "arm" in step else [None])
        if step["action"] == "open":
            for arm_name in arms:
                grasped.pop(arm_name, None)
            continue
        if step["action"] != "close":
            raise ValueError(f"Unknown gripper action: {step['action']}")

        for arm_name, obj_name in zip(arms, _objects_from_label(step["label"], len(arms))):
            world_from_obj = obj_to_pose[obj_name]
            spec = _arm_spec(container, arm_name)
            tool_from_ee = (spec.tool_from_ee if spec else container.tool_from_ee).cpu().numpy().astype(np.float64)
            world_from_ee = _fk_matrices(kin_model, last_q, spec.ee_link if spec else None)[0]
            world_from_grasp = world_from_ee @ np.linalg.inv(tool_from_ee)
            obj_from_grasp = np.linalg.inv(world_from_obj) @ world_from_grasp

            match = _match_candidate(obj_from_grasp, grasps.get(obj_name, {}))
            if match["index"] is None:
                _log.warning(f"No M2T2 candidates saved for '{obj_name}'; recording the grasp pose only")
            elif match["position_error"] > MATCH_POSITION_TOL or match["rotation_error"] > MATCH_ROTATION_TOL:
                _log.warning(
                    f"Chosen grasp for '{obj_name}' is {match['position_error'] * 100:.1f}cm / "
                    f"{np.rad2deg(match['rotation_error']):.1f}deg from the nearest M2T2 candidate "
                    f"(index {match['index']}) — treat the match as unreliable"
                )

            chosen.append(
                {
                    "object": obj_name,
                    "arm": arm_name,
                    "label": step["label"],
                    "step_index": step_index,
                    "q": last_q,
                    "world_from_obj": world_from_obj,
                    "world_from_grasp": world_from_grasp,
                    "obj_from_grasp": obj_from_grasp,
                    "m2t2": match,
                }
            )
            grasped[arm_name] = (obj_name, np.linalg.inv(world_from_ee) @ world_from_obj)

    return chosen


def save_chosen_grasps(save_dir: Path, tiptop_plan: dict, cutamp_env, grasps: dict, robot_type: str) -> Path | None:
    """Write ``perception/chosen_grasps.json`` for a plan. Returns None if extraction fails.

    Best effort: this is an analysis artifact written after the plan is already on disk, so a
    failure here is logged and must never take down a run that has a perfectly good plan.
    """
    try:
        chosen = extract_chosen_grasps(tiptop_plan, cutamp_env, grasps, robot_type)
    except Exception:
        _log.exception("Failed to extract the chosen grasps; skipping chosen_grasps.json")
        return None

    perception_dir = Path(save_dir) / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    out_path = perception_dir / CHOSEN_GRASPS_FILENAME
    payload = {"version": CHOSEN_GRASPS_VERSION, "robot_type": robot_type, "grasps": chosen}
    with open(out_path, "w") as f:
        json.dump(payload, f, cls=NumpyEncoder, indent=2)
    _log.info(f"Saved {len(chosen)} chosen grasp(s) to {out_path}")
    return out_path


def load_chosen_grasps(perception_dir: Path) -> list[dict]:
    """Load ``chosen_grasps.json``, restoring the transforms as arrays."""
    with open(Path(perception_dir) / CHOSEN_GRASPS_FILENAME) as f:
        payload = json.load(f)
    if payload["version"] != CHOSEN_GRASPS_VERSION:
        raise NotImplementedError(f"chosen_grasps.json version {payload['version']} not supported")
    for entry in payload["grasps"]:
        for key in ("q", "world_from_obj", "world_from_grasp", "obj_from_grasp"):
            entry[key] = np.array(entry[key], dtype=np.float64)
    return payload["grasps"]
