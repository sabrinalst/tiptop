"""Splitting one instruction's goal between the two YAM arms (tiptop.yam.task_split).

cuTAMP plans one kinematic chain and has no operator for choosing an arm, so this split is what
makes a bimanual episode possible at all. Built against the real cuRobo Mesh because that is what
perception hands over and what carries the centroid the split reads.
"""

import numpy as np
import pytest
from curobo.geom.types import Cuboid, Mesh

from tiptop.yam.task_split import (
    already_satisfied,
    arm_goal_builder,
    atom_crosses_midline,
    handover_goal_builder,
    object_side,
    parallel_goal_builder,
    split_by_arm,
)

UNIT_QUAT = [1.0, 0.0, 0.0, 0.0]
TABLE = Cuboid(name="table", dims=[1.2, 1.6, 0.1], pose=[0.5, 0.0, -0.07, *UNIT_QUAT])


def _box(name: str, center, size) -> Mesh:
    """A Mesh shaped like perception's output: vertices centred on the origin, centre in the pose."""
    cx, cy, cz = np.asarray(size, dtype=float) / 2.0
    corners = np.array(
        [[sx * cx, sy * cy, sz * cz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
    )
    return Mesh(name=name, vertices=corners.tolist(), faces=[[0, 1, 2]], pose=[*map(float, center), *UNIT_QUAT])


class _Scene:
    """The two fields arm_goal_builder reads off a ProcessedScene."""

    def __init__(self, meshes, table=TABLE):
        self.object_meshes = meshes
        self.table_cuboid = table


# A plate straddling the midline, one toy on each side, and one well over on the left.
PLATE = _box("plate", center=(0.5, 0.0, 0.01), size=(0.24, 0.24, 0.02))
LEFT_TOY = _box("blue_toy", center=(0.45, 0.30, 0.02), size=(0.05, 0.05, 0.04))
RIGHT_TOY = _box("pink_toy", center=(0.45, -0.30, 0.02), size=(0.05, 0.05, 0.04))
FAR_LEFT_TOY = _box("brown_toy", center=(0.55, 0.22, 0.02), size=(0.05, 0.05, 0.04))

GOAL = [
    {"predicate": "on", "args": ["blue_toy", "plate"]},
    {"predicate": "on", "args": ["pink_toy", "plate"]},
    {"predicate": "on", "args": ["brown_toy", "plate"]},
]


def test_object_side_splits_at_the_midline():
    assert object_side(0.30) == "left"
    assert object_side(-0.30) == "right"
    # Exactly on the midline goes left, matching _split_by_arm in the sim harness (>= base_y).
    assert object_side(0.0) == "left"


def test_object_side_honours_a_shifted_base():
    # base_y is the robot's midline in the URDF world frame; a rig mounted elsewhere shifts it.
    assert object_side(0.10, base_y=0.25) == "right"
    assert object_side(0.30, base_y=0.25) == "left"


def test_split_by_arm_assigns_every_object_exactly_once():
    meshes = {m.name: m for m in (PLATE, LEFT_TOY, RIGHT_TOY, FAR_LEFT_TOY)}
    split = split_by_arm(meshes, base_y=0.0)
    assert split["left"] == ["blue_toy", "brown_toy", "plate"]
    assert split["right"] == ["pink_toy"]
    assert sorted(split["left"] + split["right"]) == sorted(meshes)


def test_each_arm_gets_only_its_own_side_of_the_goal():
    scene = _Scene({m.name: m for m in (PLATE, LEFT_TOY, RIGHT_TOY, FAR_LEFT_TOY)})

    left_atoms, _ = arm_goal_builder("left")(scene, GOAL)
    right_atoms, _ = arm_goal_builder("right")(scene, GOAL)

    assert [a["args"][0] for a in left_atoms] == ["blue_toy", "brown_toy"]
    assert [a["args"][0] for a in right_atoms] == ["pink_toy"]
    # Between them the two arms cover the whole goal, and neither duplicates the other's work.
    assert len(left_atoms) + len(right_atoms) == len(GOAL)


def test_the_target_surface_is_protected_for_both_arms():
    """Surfaces are otherwise inferred from the kept atoms alone, so an arm with nothing to do would
    leave the plate classified movable and the planner free to pick the plate up."""
    scene = _Scene({m.name: m for m in (PLATE, LEFT_TOY, RIGHT_TOY)})
    for arm in ("left", "right"):
        _, surfaces = arm_goal_builder(arm)(scene, GOAL)
        assert surfaces == {"plate"}


def test_an_object_already_on_the_plate_is_dropped():
    """The second arm re-perceives AFTER the first has placed its objects. A toy that has landed on
    the plate can now sit on the second arm's side of the midline, and picking it back off would
    undo the first arm's work."""
    placed = _box("blue_toy", center=(0.47, -0.03, 0.04), size=(0.05, 0.05, 0.04))  # on the plate, right side
    scene = _Scene({m.name: m for m in (PLATE, placed, RIGHT_TOY, FAR_LEFT_TOY)})

    right_atoms, _ = arm_goal_builder("right")(scene, GOAL)

    assert [a["args"][0] for a in right_atoms] == ["pink_toy"]
    assert "blue_toy" not in [a["args"][0] for a in right_atoms]


def test_an_atom_perception_cannot_see_is_dropped_not_raised():
    """The second arm perceives after the first has moved things, so an object can genuinely be
    missing the second time round. create_tamp_environment raises on a goal naming anything it
    cannot see, which would lose the whole episode over one object."""
    scene = _Scene({m.name: m for m in (PLATE, LEFT_TOY, RIGHT_TOY)})  # brown_toy went undetected

    left_atoms, _ = arm_goal_builder("left")(scene, GOAL)

    assert [a["args"][0] for a in left_atoms] == ["blue_toy"]
    known = set(scene.object_meshes) | {scene.table_cuboid.name}
    assert all(arg in known for atom in left_atoms for arg in atom["args"])


def test_an_atom_whose_surface_is_missing_is_dropped():
    """Both arguments have to be reconstructed, not just the object being moved."""
    scene = _Scene({m.name: m for m in (LEFT_TOY, RIGHT_TOY)})  # the plate itself went undetected
    left_atoms, _ = arm_goal_builder("left")(scene, GOAL)
    assert left_atoms == []


def test_already_satisfied_reads_the_support_relation_not_the_label():
    supported = {"blue_toy": "plate"}
    assert already_satisfied({"predicate": "on", "args": ["blue_toy", "plate"]}, supported)
    # Same object, different surface: not satisfied.
    assert not already_satisfied({"predicate": "on", "args": ["blue_toy", "pad"]}, supported)
    # Nothing holding it up at all.
    assert not already_satisfied({"predicate": "on", "args": ["pink_toy", "plate"]}, {})
    # A non-`on` predicate never counts as satisfied by geometry.
    assert not already_satisfied({"predicate": "holding", "args": ["blue_toy"]}, supported)


def test_an_arm_with_nothing_on_its_side_gets_an_empty_goal():
    """The caller skips planning entirely for that arm rather than handing cuTAMP an empty goal."""
    scene = _Scene({m.name: m for m in (PLATE, LEFT_TOY, FAR_LEFT_TOY)})
    right_atoms, surfaces = arm_goal_builder("right")(scene, GOAL[:1] + GOAL[2:])
    assert right_atoms == []
    assert surfaces == {"plate"}


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="arm must be one of"):
        arm_goal_builder("middle")


def test_a_plate_near_the_midline_does_not_cross():
    # PLATE sits dead on the midline (y=0.0) — within both arms' reach, not "crossing" for either.
    meshes = {m.name: m for m in (PLATE, LEFT_TOY, RIGHT_TOY)}
    assert not atom_crosses_midline({"predicate": "on", "args": ["blue_toy", "plate"]}, meshes)
    assert not atom_crosses_midline({"predicate": "on", "args": ["pink_toy", "plate"]}, meshes)


def test_a_surface_well_past_the_other_arms_reach_crosses():
    # The regression this guards: a movable on one side, its target surface far on the other side —
    # the arm that owns the movable can never reach the surface, so this must be flagged, not planned.
    far_plate = _box("far_plate", center=(0.62, -0.46, 0.01), size=(0.18, 0.18, 0.01))
    meshes = {m.name: m for m in (LEFT_TOY, far_plate)}
    assert atom_crosses_midline({"predicate": "on", "args": ["blue_toy", "far_plate"]}, meshes)


def test_a_crossing_atom_is_dropped_from_the_goal_with_a_clear_reason(caplog):
    far_plate = _box("far_plate", center=(0.62, -0.46, 0.01), size=(0.18, 0.18, 0.01))
    scene = _Scene({m.name: m for m in (LEFT_TOY, far_plate)})
    goal = [{"predicate": "on", "args": ["blue_toy", "far_plate"]}]

    with caplog.at_level("ERROR"):
        atoms, _ = arm_goal_builder("left")(scene, goal)

    assert atoms == []
    assert "needs a handover" in caplog.text or "handover" in caplog.text


# -- handover_goal_builder ------------------------------------------------------------------------

FAR_PLATE = _box("far_plate", center=(0.62, -0.46, 0.01), size=(0.18, 0.18, 0.01))


def test_handover_goal_builder_keeps_exactly_the_crossing_atom():
    scene = _Scene({m.name: m for m in (LEFT_TOY, FAR_PLATE)})
    goal = [{"predicate": "on", "args": ["blue_toy", "far_plate"]}]

    atoms, surfaces = handover_goal_builder()(scene, goal)

    assert [a["args"] for a in atoms] == [["blue_toy", "far_plate"]]
    assert surfaces == {"far_plate"}


def test_handover_goal_builder_drops_atoms_that_dont_need_a_handover():
    """No plain Pick/Place operator exists alongside PickGiver/Handover/PlaceTaker in cuTAMP's
    handover domain, so an atom a single arm could already do has nowhere to be planned this
    episode -- it must be dropped, not silently ignored by cuTAMP later."""
    scene = _Scene({m.name: m for m in (PLATE, LEFT_TOY, FAR_PLATE)})
    goal = [
        {"predicate": "on", "args": ["blue_toy", "plate"]},  # same side, no handover needed
        {"predicate": "on", "args": ["blue_toy", "far_plate"]},  # crosses -> needs a handover
    ]

    atoms, _ = handover_goal_builder()(scene, goal)

    assert [a["args"] for a in atoms] == [["blue_toy", "far_plate"]]


def test_handover_goal_builder_keeps_only_one_crossing_atom_per_episode():
    other_far_plate = _box("other_far_plate", center=(0.6, -0.4, 0.01), size=(0.1, 0.1, 0.01))
    scene = _Scene({m.name: m for m in (LEFT_TOY, FAR_PLATE, other_far_plate)})
    goal = [
        {"predicate": "on", "args": ["blue_toy", "far_plate"]},
        {"predicate": "on", "args": ["blue_toy", "other_far_plate"]},
    ]

    atoms, _ = handover_goal_builder()(scene, goal)

    assert len(atoms) == 1


def test_handover_goal_builder_drops_already_satisfied_atoms():
    # far_plate's top sits at z=0.015 (center 0.01, half-height 0.005); rest this toy's bottom there.
    placed = _box("blue_toy", center=(0.62, -0.46, 0.035), size=(0.05, 0.05, 0.04))  # already on far_plate
    scene = _Scene({m.name: m for m in (FAR_PLATE, placed)})
    goal = [{"predicate": "on", "args": ["blue_toy", "far_plate"]}]

    atoms, _ = handover_goal_builder()(scene, goal)

    assert atoms == []


# -- parallel_goal_builder ------------------------------------------------------------------------
# The lockstep domain (dual_tamp_operators) acts on two objects at once and has no single-arm
# Pick/Place, so its goal is exactly two atoms -- one per hand -- or nothing at all.

LEFT_BOWL = _box("left_bowl", center=(0.5, 0.28, 0.01), size=(0.16, 0.16, 0.02))
RIGHT_BOWL = _box("right_bowl", center=(0.5, -0.28, 0.01), size=(0.16, 0.16, 0.02))
LEFT_CAN = _box("left_can", center=(0.25, 0.22, 0.05), size=(0.06, 0.06, 0.10))
RIGHT_CAN = _box("right_can", center=(0.25, -0.22, 0.05), size=(0.06, 0.06, 0.10))

PARALLEL_SCENE = _Scene({m.name: m for m in (LEFT_BOWL, RIGHT_BOWL, LEFT_CAN, RIGHT_CAN)})
PARALLEL_GOAL = [
    {"predicate": "on", "args": ["left_can", "left_bowl"]},
    {"predicate": "on", "args": ["right_can", "right_bowl"]},
]


def test_parallel_goal_builder_keeps_one_atom_per_arm():
    atoms, surfaces = parallel_goal_builder()(PARALLEL_SCENE, PARALLEL_GOAL)

    assert [a["args"] for a in atoms] == [["left_can", "left_bowl"], ["right_can", "right_bowl"]]
    assert surfaces == {"left_bowl", "right_bowl"}


def test_parallel_goal_builder_drops_a_second_pick_for_the_same_arm():
    """PickBoth grasps one object per hand, so a third atom has no hand to be planned on."""
    second_left_can = _box("other_can", center=(0.25, 0.30, 0.05), size=(0.06, 0.06, 0.10))
    scene = _Scene({**PARALLEL_SCENE.object_meshes, "other_can": second_left_can})
    goal = PARALLEL_GOAL + [{"predicate": "on", "args": ["other_can", "left_bowl"]}]

    atoms, _ = parallel_goal_builder()(scene, goal)

    assert [a["args"] for a in atoms] == [["left_can", "left_bowl"], ["right_can", "right_bowl"]]


def test_parallel_goal_builder_needs_work_on_both_sides(caplog):
    """One atom is as unplannable as none here: every operator in the list names two objects."""
    scene = _Scene({m.name: m for m in (LEFT_BOWL, LEFT_CAN)})
    goal = [{"predicate": "on", "args": ["left_can", "left_bowl"]}]

    with caplog.at_level("WARNING"):
        atoms, _ = parallel_goal_builder()(scene, goal)

    assert atoms == []
    assert "one goal per arm" in caplog.text


def test_parallel_goal_builder_drops_a_crossing_atom():
    """A can whose bowl is on the other side needs a handover, which this operator list lacks."""
    scene = _Scene({m.name: m for m in (LEFT_CAN, FAR_PLATE, RIGHT_CAN, RIGHT_BOWL)})
    goal = [
        {"predicate": "on", "args": ["left_can", "far_plate"]},  # crosses the midline
        {"predicate": "on", "args": ["right_can", "right_bowl"]},
    ]

    atoms, _ = parallel_goal_builder()(scene, goal)

    assert atoms == []  # only the right arm is left with work, and one atom cannot be planned


def test_parallel_goal_builder_drops_already_satisfied_atoms():
    # left_bowl's top sits at z=0.02 (center 0.01, half-height 0.01); rest this can's bottom there.
    placed = _box("left_can", center=(0.5, 0.28, 0.07), size=(0.06, 0.06, 0.10))
    scene = _Scene({m.name: m for m in (LEFT_BOWL, RIGHT_BOWL, placed, RIGHT_CAN)})

    atoms, _ = parallel_goal_builder()(scene, PARALLEL_GOAL)

    assert atoms == []  # the left arm's goal is done; the right arm alone cannot be planned


def test_parallel_goal_builder_honours_a_shifted_midline():
    """Which hand owns which goal is read off base_y, not off the sign of y."""
    # base_y = -0.25 puts BOTH cans on the left arm's side, leaving the right arm nothing to do.
    atoms, _ = parallel_goal_builder(base_y=-0.25)(PARALLEL_SCENE, PARALLEL_GOAL)
    assert atoms == []

    # base_y = 0.25 splits the same pair the other way: left_can (y=0.22) is now the RIGHT arm's,
    # and only a can further out than the midline is the left arm's.
    far_can = _box("far_can", center=(0.25, 0.42, 0.05), size=(0.06, 0.06, 0.10))
    far_bowl = _box("far_bowl", center=(0.5, 0.50, 0.01), size=(0.16, 0.16, 0.02))
    scene = _Scene({**PARALLEL_SCENE.object_meshes, "far_can": far_can, "far_bowl": far_bowl})
    goal = PARALLEL_GOAL + [{"predicate": "on", "args": ["far_can", "far_bowl"]}]

    atoms, _ = parallel_goal_builder(base_y=0.25)(scene, goal)

    assert [a["args"] for a in atoms] == [["far_can", "far_bowl"], ["left_can", "left_bowl"]]
