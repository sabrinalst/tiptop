"""The bimanual YAM URDF places the arms as rig.json says, and the world origin sits on the arms' midline."""

import json
from pathlib import Path

import numpy as np
import pytest
from cutamp.robots import bimanual_yam
from yourdfpy import URDF

ASSETS = Path(bimanual_yam.__file__).parent / "assets" / "yam_description"


@pytest.fixture(scope="module")
def rig() -> dict:
    return json.loads((ASSETS / "rig.json").read_text())


@pytest.fixture(scope="module")
def urdf() -> URDF:
    return URDF.load(str(ASSETS / "bimanual_yam.urdf"), load_meshes=False)


@pytest.mark.parametrize("side", ["left", "right"])
def test_arm_frame_matches_rig(urdf: URDF, rig: dict, side: str):
    T = urdf.get_transform(f"{side}_arm", "world")
    expected = rig["arms"][side]
    np.testing.assert_allclose(T[:2, 3], expected["xyz"][:2], atol=1e-6)
    assert np.arctan2(T[1, 0], T[0, 0]) == pytest.approx(expected["yaw"], abs=1e-6)


def test_world_origin_is_on_the_midline(rig: dict):
    left, right = rig["arms"]["left"]["xyz"], rig["arms"]["right"]["xyz"]
    assert left[0] == pytest.approx(right[0], abs=1e-9)
    assert left[1] == pytest.approx(-right[1], abs=1e-9)
