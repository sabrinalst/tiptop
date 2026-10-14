"""Does the recovered grasp really identify the M2T2 candidate cuTAMP picked?

The saved plan names its grasp symbolically (``Pick(block, grasp1, q1)``), so ``chosen_grasp``
recovers the pose by forward kinematics on the configuration before each gripper CLOSE and matches
it against ``grasps_obj``. Two things can silently go wrong: the tool/EE transform can be applied
the wrong way round, and an object re-picked after a place can be matched against its INITIAL pose
instead of where it actually ended up. Both are exercised here against a stub robot whose FK is a
pure translation, so the expected answers are exact and no cuRobo/GPU is needed.

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_chosen_grasp.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tiptop import chosen_grasp as cg

# Fixed non-identity tool offset, so a transform applied the wrong way round cannot pass.
TOOL_FROM_EE = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.05],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
BLOCK_POSE = [0.5, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]  # curobo pose list: xyz + wxyz


def _mat(translation) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = translation
    return m


class _Pose:
    def __init__(self, matrices):
        self._matrices = matrices

    def get_matrix(self):
        return torch.as_tensor(self._matrices, dtype=torch.float32)


class _State:
    """FK stand-in: the EE sits at the configuration's first three joints, with no rotation."""

    def __init__(self, q):
        self.ee_pose = _Pose(np.stack([_mat(row[:3]) for row in q]))
        self.link_pose = {}


class _KinModel:
    tensor_args = type("_TensorArgs", (), {"device": "cpu"})()

    def get_state(self, q):
        return _State(np.atleast_2d(np.asarray(q)))


class _Container:
    name = "stub"
    kin_model = _KinModel()
    tool_from_ee = torch.as_tensor(TOOL_FROM_EE, dtype=torch.float64)
    arms = ()


class _Obj:
    def __init__(self, name, pose):
        self.name = name
        self.pose = pose


class _Env:
    movables = [_Obj("block", BLOCK_POSE), _Obj("bowl", [0.4, -0.2, 0.2, 1.0, 0.0, 0.0, 0.0])]
    statics = [_Obj("table", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]


@pytest.fixture(autouse=True)
def stub_robot(monkeypatch):
    monkeypatch.setattr(cg, "load_robot_container", lambda *a, **k: _Container())


def _grasps_with(expected_obj_from_grasp: np.ndarray, at_index: int, n: int = 5) -> dict:
    """A candidate set whose ``at_index`` entry is the expected grasp, the rest well away from it."""
    candidates = np.stack([_mat([1.0 + i, 1.0, 1.0]) for i in range(n)])
    candidates[at_index] = expected_obj_from_grasp
    return {
        "block": {
            "grasps_obj": torch.as_tensor(candidates, dtype=torch.float64),
            "confidences": np.linspace(0.1, 0.9, n),
        }
    }


def _plan(steps) -> dict:
    return {"q_init": np.zeros(3), "steps": steps}


def _traj(end) -> dict:
    return {"type": "trajectory", "label": "Pick(block, grasp1, q1)", "positions": np.array([[0.0, 0.0, 0.0], end])}


def test_recovers_the_candidate_the_plan_grasped():
    grasp_q = [0.7, 0.3, 0.4]
    world_from_grasp = _mat(grasp_q) @ np.linalg.inv(TOOL_FROM_EE)
    expected = np.linalg.inv(_mat(BLOCK_POSE[:3])) @ world_from_grasp

    grasps = _grasps_with(expected, at_index=3)
    plan = _plan([_traj(grasp_q), {"type": "gripper", "action": "close", "label": "Pick(block, grasp1, q1)"}])

    (chosen,) = cg.extract_chosen_grasps(plan, _Env(), grasps, "stub")
    assert chosen["object"] == "block"
    assert chosen["m2t2"]["index"] == 3
    assert chosen["m2t2"]["confidence"] == pytest.approx(grasps["block"]["confidences"][3])
    assert chosen["m2t2"]["position_error"] == pytest.approx(0.0, abs=1e-6)
    assert chosen["world_from_grasp"] == pytest.approx(world_from_grasp, abs=1e-6)


def test_repick_matches_against_where_the_object_was_left():
    """A second pick is matched at the object's new pose, not the one it started at."""
    grasp_q = [0.7, 0.3, 0.4]
    place_q = [0.2, -0.4, 0.6]
    world_from_grasp = _mat(grasp_q) @ np.linalg.inv(TOOL_FROM_EE)
    expected = np.linalg.inv(_mat(BLOCK_POSE[:3])) @ world_from_grasp

    # The object rides the hand from grasp_q to place_q, so it is displaced by their difference; the
    # same grasp on the moved object is therefore taken from a correspondingly displaced hand pose.
    displacement = np.array(place_q) - np.array(grasp_q)
    label = "Pick(block, grasp2, q3)"
    plan = _plan(
        [
            _traj(grasp_q),
            {"type": "gripper", "action": "close", "label": "Pick(block, grasp1, q1)"},
            _traj(place_q),
            {"type": "gripper", "action": "open", "label": "Place(block, grasp1, pose1, table, q2)"},
            {"type": "trajectory", "label": label, "positions": np.array([place_q, np.array(grasp_q) + displacement])},
            {"type": "gripper", "action": "close", "label": label},
        ]
    )

    first, second = cg.extract_chosen_grasps(plan, _Env(), _grasps_with(expected, at_index=3), "stub")
    assert first["m2t2"]["index"] == 3 and second["m2t2"]["index"] == 3
    assert second["m2t2"]["position_error"] == pytest.approx(0.0, abs=1e-6)
    assert second["world_from_obj"][:3, 3] == pytest.approx(np.array(BLOCK_POSE[:3]) + displacement, abs=1e-6)


def test_pick_both_pairs_each_hand_with_its_own_object():
    """``PickBoth(obj_a, grasp_a, obj_b, grasp_b, q)`` names two objects; each hand takes its own."""
    label = "PickBoth(block, grasp1, bowl, grasp2, q1)"
    plan = _plan(
        [_traj([0.7, 0.3, 0.4]), {"type": "gripper", "action": "close", "label": label, "arms": ["left", "right"]}]
    )
    left, right = cg.extract_chosen_grasps(plan, _Env(), {}, "stub")
    assert (left["arm"], left["object"]) == ("left", "block")
    assert (right["arm"], right["object"]) == ("right", "bowl")
    # Each hand's grasp is expressed against ITS object, so the two differ by the objects' offset.
    offset = np.array(BLOCK_POSE[:3]) - np.array([0.4, -0.2, 0.2])
    assert (right["obj_from_grasp"] - left["obj_from_grasp"])[:3, 3] == pytest.approx(offset, abs=1e-6)

    # A single hand still reads its object off the front, whatever the operator is called.
    assert cg._objects_from_label("Handover(block, gg, gt, hp, q)", num_arms=1) == ["block"]


def test_object_without_candidates_still_records_its_pose():
    plan = _plan([_traj([0.7, 0.3, 0.4]), {"type": "gripper", "action": "close", "label": "Pick(block, g, q)"}])
    (chosen,) = cg.extract_chosen_grasps(plan, _Env(), {}, "stub")
    assert chosen["m2t2"]["index"] is None
    assert chosen["world_from_grasp"].shape == (4, 4)
