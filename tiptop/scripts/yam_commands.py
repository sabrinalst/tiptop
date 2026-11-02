"""Standalone bimanual YAM nudges: home the arms, open the grippers.

``go-home`` and ``gripper-open`` do drive a YAM, but only once ``TIPTOP_CONFIG`` names a YAM config,
and only for the ONE arm ``robot.type`` happens to select. These wrap both: they select a YAM config
themselves and take ``--arm``, so recovering an arm outside a ``tiptop-run`` session (the `home`
robot command at the task prompt) is a single command against the running ``yam_arm_server.py``.

Every tiptop import is deliberately inside a function: ``tiptop.config`` resolves ``TIPTOP_CONFIG``
once, at import time, so the variable has to be set before the first one runs.
"""

import logging
import os
from typing import Literal

import tyro

_log = logging.getLogger(__name__)

# Used unless the caller already named a config -- so an explicit TIPTOP_CONFIG (a sim YAM config,
# say) still wins.
DEFAULT_YAM_CONFIG = "tiptop_yam_real.yml"

ArmChoice = Literal["left", "right", "both"]


def _setup(level: int = logging.INFO) -> None:
    """Point tiptop.config at a YAM config, then set up logging. Must run before any tiptop import."""
    os.environ.setdefault("TIPTOP_CONFIG", DEFAULT_YAM_CONFIG)

    from tiptop.utils import setup_logging

    setup_logging(level)


def _arms_to_drive(arm: ArmChoice) -> list[str]:
    """The arms a command touches, in execution order. Raises if the active config is not a YAM."""
    from tiptop.config import tiptop_cfg, tiptop_config_path
    from tiptop.yam import ARMS, is_yam

    robot_type = tiptop_cfg().robot.type
    if not is_yam(robot_type):
        raise ValueError(
            f"{tiptop_config_path} selects robot type {robot_type!r}, which is not a YAM. "
            f"Unset TIPTOP_CONFIG to use {DEFAULT_YAM_CONFIG}, or point it at a YAM config."
        )
    return list(ARMS) if arm == "both" else [arm]


def _warn_if_other_arm_displaced(arm: str) -> None:
    """Warn when planning for ``arm`` while the other arm is not where cuRobo believes it is.

    The planning arm's cuRobo config locks the other arm at NEUTRAL_Q and collision-checks it there,
    so a displaced arm makes this move a statement about a robot that does not exist. Homing is the
    recovery from exactly that state, so it warns rather than refuses -- same trade
    :func:`tiptop.tiptop_run.home_all_arms` makes, and it carries the same keep-the-workspace-clear
    assumption.
    """
    import numpy as np

    from tiptop.utils import get_robot_client
    from tiptop.yam import ARM_DOF, IDLE_ARM_TOLERANCE_RAD, NEUTRAL_Q, other_arm

    idle = other_arm(arm)
    measured = np.asarray(get_robot_client().get_arm_state(idle)["q"], dtype=np.float64)[:ARM_DOF]
    error = float(np.abs(measured - np.asarray(NEUTRAL_Q)).max())
    if error > IDLE_ARM_TOLERANCE_RAD:
        _log.warning(
            f"Homing the {arm} arm while the {idle} arm is {error:.3f} rad from the neutral posture "
            "-- the collision model does not describe the parked arm for this move. Keep the "
            "workspace clear and watch the arms."
        )


# The zero posture has joints 2 and 3 exactly on their lower limits (0), and cuRobo rejects a start or goal
# state on a joint limit. Moves to or from it therefore go straight through the arm server's ``move_to``,
# which interpolates in joint space with no planning and no collision checking.
ZERO_Q = (0.0,) * 6


def _move_directly(arms: list[str], q_target, time_dilation_factor: float) -> None:
    """Move each arm to ``q_target`` through the arm server, one at a time. Keep the workspace clear."""
    from tiptop.utils import get_robot_client

    client = get_robot_client()
    duration = 2.0 / time_dilation_factor
    for a in arms:
        _log.info(f"Moving the {a} arm directly (no collision checking) over {duration:.1f}s")
        res = client.move_to(q_target, duration=duration, arm=a)
        if not res.get("success"):
            raise RuntimeError(f"moving the {a} arm failed: {res.get('error')}")


def go_home_yam(
    arm: ArmChoice = "both",
    neutral: bool = False,
    zero: bool = False,
    time_dilation_factor: float | None = None,
):
    """Return the YAM arm(s) to the neutral posture (default, ``--neutral``) or the zero posture (``--zero``).

    Neutral is planned through cuRobo, one arm at a time. If cuRobo rejects the arm's current state because
    it rests on a joint limit (the zero posture does), the move is made directly instead.
    """
    from cutamp.motion_solver import MotionPlanningError

    from tiptop.config import tiptop_cfg
    from tiptop.motion_planning import go_to_home
    from tiptop.yam import NEUTRAL_Q, active_arm

    if neutral and zero:
        raise ValueError("--neutral and --zero are mutually exclusive")
    arms = _arms_to_drive(arm)
    if time_dilation_factor is None:
        time_dilation_factor = tiptop_cfg().robot.time_dilation_factor

    if zero:
        _move_directly(arms, ZERO_Q, time_dilation_factor)
        return

    for a in arms:
        with active_arm(a):
            _warn_if_other_arm_displaced(a)
            _log.info(f"Homing the {a} arm")
            # No motion_gen to share: each arm is its own embodiment, so each needs its own solver.
            try:
                go_to_home(time_dilation_factor=time_dilation_factor)
            except MotionPlanningError as e:
                if "JOINT_LIMITS" not in str(e):
                    raise
                _log.warning(f"cuRobo rejected the {a} arm's state on a joint limit ({e}); moving directly instead")
                _move_directly([a], NEUTRAL_Q, time_dilation_factor)


def gripper_open_yam(arm: ArmChoice = "both", speed: float = 1.0, force: float = 0.1):
    """Open the YAM gripper(s)."""
    from tiptop.utils import get_robot_client
    from tiptop.yam import active_arm

    client = get_robot_client()
    for a in _arms_to_drive(arm):
        # The client reads robot.type to decide which arm it is talking to, so this is what picks
        # the gripper -- there is no per-call arm argument.
        with active_arm(a):
            client.open_gripper(speed=speed, force=force)
            _log.info(f"Opened the {a} gripper with speed={speed}, force={force}")


def go_home_yam_entrypoint():
    _setup()
    tyro.cli(go_home_yam)


def gripper_open_yam_entrypoint():
    _setup()
    tyro.cli(gripper_open_yam)
