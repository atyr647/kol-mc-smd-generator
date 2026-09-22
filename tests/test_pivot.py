"""Tests for OPD part pivot handling.

The pivot is an OPD per-part local-space offset that the KO engine applies
BEFORE the shape's world transform to anchor each part's mesh correctly.

Anatomy of the bug (now fixed)
-------------------------------
The opd_parser was silently discarding the pivot: `fp.read(12)  # pivot Vector3`.
This meant every mesh part was placed at the shape's world position (0 pivot),
causing systematic offsets for any part with a non-zero pivot.

Correct formula
---------------
In KO engine:
    part_world = shape_position + R × (S × pivot_local)
where R = quat_to_matrix(shape.rotation), S = diag(shape.scale).

In MC (with X mirror):
    mc_pivot_offset = (-R×S×pivot)[0], (R×S×pivot)[1], (R×S×pivot)[2]
                    = pivot_to_mc_offset(pivot_local, quat, scale)

Separate concerns
-----------------
  - Terrain grounding:  mc_y = terrain_y_at(h_grid, mc_x, mc_z)   ← shape-level
  - Pivot anchoring:    mc_part_pos = mc_pos + pivot_mc_offset      ← part-level
These two steps are ALWAYS separate.  The pivot adjusts local anchoring;
terrain grounding is computed once at the shape level from OPD world position.
"""

import math
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ko2mc.math3d import (
    transform_pivot,
    pivot_to_mc_offset,
    quat_to_matrix,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _identity_quat():
    return (0.0, 0.0, 0.0, 1.0)

def _yaw_quat(yaw_rad: float):
    """Quaternion for a rotation by yaw_rad around the Y axis."""
    return (0.0, math.sin(yaw_rad / 2), 0.0, math.cos(yaw_rad / 2))

def _unit_scale():
    return (1.0, 1.0, 1.0)


# ── transform_pivot: R × (S × pivot) ─────────────────────────────────────────

class TestTransformPivot:
    """Unit tests for math3d.transform_pivot."""

    def test_zero_pivot_returns_zero(self):
        result = transform_pivot((0.0, 0.0, 0.0), _identity_quat(), _unit_scale())
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_identity_rotation_no_change(self):
        """With identity rotation and unit scale, pivot passes through unchanged."""
        pivot = (3.0, 5.0, -2.0)
        result = transform_pivot(pivot, _identity_quat(), _unit_scale())
        np.testing.assert_array_almost_equal(result, pivot, decimal=5)

    def test_scale_applied_before_rotation(self):
        """S × pivot is applied first, then R rotates the result."""
        pivot = (1.0, 0.0, 0.0)
        scale = (2.0, 1.0, 1.0)   # doubles X
        # With identity rotation: R × (S × pivot) = (2, 0, 0)
        result = transform_pivot(pivot, _identity_quat(), scale)
        np.testing.assert_array_almost_equal(result, [2.0, 0.0, 0.0], decimal=5)

    def test_y_rotation_90_deg(self):
        """90° Y rotation: pivot (1,0,0) → (0,0,-1) in KO local space.

        Y-axis 90° CCW rotation:  X → -Z, Z → +X.
        R_y(90°) × (1,0,0) = (0, 0, -1).
        """
        pivot = (1.0, 0.0, 0.0)
        quat  = _yaw_quat(math.pi / 2)
        result = transform_pivot(pivot, quat, _unit_scale())
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, -1.0], decimal=4)

    def test_y_axis_pivot_unaffected_by_y_rotation(self):
        """A pivot purely in Y is unaffected by rotation around Y."""
        pivot = (0.0, 7.0, 0.0)
        quat  = _yaw_quat(math.pi / 3)
        result = transform_pivot(pivot, quat, _unit_scale())
        np.testing.assert_array_almost_equal(result, [0.0, 7.0, 0.0], decimal=4)

    def test_non_uniform_scale_then_rotation(self):
        """Non-uniform scale is applied before rotation."""
        pivot = (1.0, 0.0, 0.0)
        scale = (3.0, 1.0, 1.0)   # triples X component
        quat  = _yaw_quat(math.pi / 2)   # 90° Y rotation
        # S × pivot = (3, 0, 0)
        # R_y(90°) × (3, 0, 0) = (0, 0, -3)
        result = transform_pivot(pivot, quat, scale)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, -3.0], decimal=4)

    def test_returns_float32_array(self):
        result = transform_pivot((1.0, 2.0, 3.0), _identity_quat(), _unit_scale())
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


# ── pivot_to_mc_offset: X mirror applied ─────────────────────────────────────

class TestPivotToMCOffset:
    """Unit tests for math3d.pivot_to_mc_offset."""

    def test_zero_pivot_returns_zero(self):
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (0.0, 0.0, 0.0), _identity_quat(), _unit_scale()
        )
        assert mc_dx == 0
        assert mc_dy == 0
        assert mc_dz == 0

    def test_x_pivot_is_negated_identity_rotation(self):
        """Positive KO-local X pivot maps to NEGATIVE MC X (X mirror)."""
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (5.0, 0.0, 0.0), _identity_quat(), _unit_scale()
        )
        assert mc_dx == -5, f"Expected mc_dx=-5, got {mc_dx}"
        assert mc_dy == 0
        assert mc_dz == 0

    def test_y_pivot_is_positive_identity_rotation(self):
        """Y pivot is additive (Y is not mirrored)."""
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (0.0, 8.0, 0.0), _identity_quat(), _unit_scale()
        )
        assert mc_dx == 0
        assert mc_dy == 8, f"Expected mc_dy=+8, got {mc_dy}"
        assert mc_dz == 0

    def test_z_pivot_is_positive_identity_rotation(self):
        """Z pivot is additive (Z is not mirrored)."""
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (0.0, 0.0, 4.0), _identity_quat(), _unit_scale()
        )
        assert mc_dx == 0
        assert mc_dy == 0
        assert mc_dz == 4, f"Expected mc_dz=+4, got {mc_dz}"

    def test_x_mirror_with_rotation(self):
        """After 90° Y rotation, pivot (1,0,0) becomes world (0,0,-1);
        X mirror: mc_dx = -(0) = 0, mc_dz = -1."""
        pivot = (1.0, 0.0, 0.0)
        quat  = _yaw_quat(math.pi / 2)
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pivot, quat, _unit_scale())
        # Transformed pivot in KO local = (0, 0, -1)
        # X component = 0 → mc_dx = -0 = 0
        # Z component = -1 → mc_dz = -1
        assert mc_dx == 0,  f"mc_dx={mc_dx}"
        assert mc_dy == 0,  f"mc_dy={mc_dy}"
        assert mc_dz == -1, f"mc_dz={mc_dz}"

    def test_returns_integers(self):
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (1.5, 2.7, 3.2), _identity_quat(), _unit_scale()
        )
        assert isinstance(mc_dx, int)
        assert isinstance(mc_dy, int)
        assert isinstance(mc_dz, int)

    def test_rounding(self):
        """Values are rounded to nearest integer."""
        mc_dx, _, _ = pivot_to_mc_offset((2.7, 0.0, 0.0), _identity_quat(), _unit_scale())
        assert mc_dx == -3   # round(2.7) = 3, then negated


# ── Part-level vs shape-level separation ─────────────────────────────────────

class TestPivotSeparationFromGrounding:
    """Verify that pivot adjusts anchoring independently of terrain grounding.

    Terrain grounding is a shape-level operation (computed once from OPD XZ).
    Pivot is a part-level operation that shifts the mesh within that grounding.
    They must not interfere with each other.
    """

    def test_zero_pivot_anchor_unchanged(self):
        """Zero pivot: part anchor == shape anchor exactly."""
        shape_mc_pos = (100, 70, 200)
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (0.0, 0.0, 0.0), _identity_quat(), _unit_scale()
        )
        part_mc_pos = (
            shape_mc_pos[0] + mc_dx,
            shape_mc_pos[1] + mc_dy,
            shape_mc_pos[2] + mc_dz,
        )
        assert part_mc_pos == shape_mc_pos

    def test_nonzero_pivot_shifts_anchor_independently(self):
        """Non-zero pivot shifts the part anchor without changing terrain Y base."""
        shape_mc_pos = (100, 70, 200)
        terrain_y = 70  # grounded at terrain

        # Pivot (0, 5, 0) in local Y → part is 5 blocks above terrain
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (0.0, 5.0, 0.0), _identity_quat(), _unit_scale()
        )
        part_mc_pos = (
            shape_mc_pos[0] + mc_dx,
            shape_mc_pos[1] + mc_dy,
            shape_mc_pos[2] + mc_dz,
        )

        assert part_mc_pos[0] == shape_mc_pos[0], "X should not change"
        assert part_mc_pos[1] == terrain_y + 5,   "Y should be terrain + pivot Y"
        assert part_mc_pos[2] == shape_mc_pos[2], "Z should not change"
        # Verify terrain Y itself is unchanged (grounding not re-triggered)
        assert terrain_y == 70, "Terrain Y must not change due to pivot"

    def test_pivot_x_shift_respects_x_mirror(self):
        """A +X pivot shifts the part in the -MC_X direction (X mirror)."""
        shape_mc_pos = (500, 70, 300)
        # Pivot (10, 0, 0) in local X → MC X should decrease by 10
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(
            (10.0, 0.0, 0.0), _identity_quat(), _unit_scale()
        )
        part_x = shape_mc_pos[0] + mc_dx
        assert part_x == 490, f"Expected 490, got {part_x}"

    def test_multiple_parts_independent_pivots(self):
        """Multiple parts with different pivots each get independent offsets."""
        shape_mc_pos = (200, 70, 400)

        pivots = [
            (0.0,  0.0, 0.0),
            (5.0,  0.0, 0.0),
            (0.0,  3.0, 0.0),
            (-2.0, 0.0, 4.0),
        ]

        part_positions = []
        for pv in pivots:
            mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pv, _identity_quat(), _unit_scale())
            part_positions.append((
                shape_mc_pos[0] + mc_dx,
                shape_mc_pos[1] + mc_dy,
                shape_mc_pos[2] + mc_dz,
            ))

        # Expected per-pivot MC offsets (X negated):
        #   (0,0,0)    → (0,0,0)    part = (200, 70, 400)
        #   (5,0,0)    → (-5,0,0)   part = (195, 70, 400)
        #   (0,3,0)    → (0,+3,0)   part = (200, 73, 400)
        #   (-2,0,4)   → (+2,0,+4)  part = (202, 70, 404)
        assert part_positions[0] == (200, 70, 400)
        assert part_positions[1] == (195, 70, 400)
        assert part_positions[2] == (200, 73, 400)
        assert part_positions[3] == (202, 70, 404)


# ── OPD parser pivot field test ───────────────────────────────────────────────

def test_shapePart_has_pivot_field():
    """ShapePart dataclass must have a pivot field after the parser fix."""
    from ko2mc.opd_parser import ShapePart, Vector3
    part = ShapePart()
    assert hasattr(part, "pivot"), "ShapePart must have a 'pivot' attribute"
    assert isinstance(part.pivot, Vector3), "pivot must be a Vector3"
    assert part.pivot.x == 0.0
    assert part.pivot.y == 0.0
    assert part.pivot.z == 0.0


def test_shapePart_pivot_default_is_zero():
    """Default pivot must be the zero vector (zero offset = no change)."""
    from ko2mc.opd_parser import ShapePart
    part = ShapePart(name="test")
    dx, dy, dz = pivot_to_mc_offset(
        (part.pivot.x, part.pivot.y, part.pivot.z),
        _identity_quat(),
        _unit_scale(),
    )
    assert (dx, dy, dz) == (0, 0, 0)


# ── Known landmark: castle pivot audit ───────────────────────────────────────

def test_castle_zero_pivot_no_offset():
    """Castle shapes typically have near-zero pivots; verify no systematic drift."""
    from ko2mc.math3d import ko_to_mc_position, terrain_y_at, build_mc_height_grid
    import numpy as np

    # Castle OPD position (approximate)
    ko_x, ko_z = 863.0, 540.0
    mc_x, mc_z = ko_to_mc_position(ko_x, ko_z)

    # Flat terrain at castle height (4.74 KO units)
    n = 257
    terrain_height = 4.74
    h_grid = build_mc_height_grid(np.full((n, n), terrain_height, dtype=np.float32))
    mc_y = terrain_y_at(h_grid, mc_x, mc_z)

    shape_mc_pos = (mc_x, mc_y, mc_z)

    # Zero pivot: part anchor == shape anchor
    mc_dx, mc_dy, mc_dz = pivot_to_mc_offset((0.0, 0.0, 0.0), _identity_quat(), _unit_scale())
    part_mc_pos = (mc_x + mc_dx, mc_y + mc_dy, mc_z + mc_dz)

    assert part_mc_pos == shape_mc_pos, (
        f"Zero pivot should not shift castle anchor: {shape_mc_pos} → {part_mc_pos}"
    )


def test_pivot_print_trace_does_not_crash():
    """Smoke-test that _print_transform_trace produces output without error."""
    from io import StringIO
    import contextlib
    from unittest.mock import MagicMock
    from ko2mc.opd_parser import ShapePart, Vector3, Shape

    # Build a minimal shape with two parts (one zero-pivot, one non-zero)
    shape = Shape()
    shape.name = "test_building"
    shape.position = Vector3(400.0, 5.0, 300.0)
    shape.rotation.x, shape.rotation.y, shape.rotation.z, shape.rotation.w = 0, 0, 0, 1
    shape.scale.x, shape.scale.y, shape.scale.z = 1.0, 1.0, 1.0

    p1 = ShapePart(name="building_main.n3pmesh")
    p1.pivot = Vector3(0.0, 0.0, 0.0)
    p2 = ShapePart(name="building_annex.n3pmesh")
    p2.pivot = Vector3(10.0, 0.0, 5.0)
    shape.parts = [p1, p2]

    # Build a minimal ZoneConverter just for the method (no world/resolver needed)
    from ko2mc.zone_converter import ZoneConverter
    conv = ZoneConverter(
        resolver=MagicMock(),
        world=MagicMock(),
        debug_transform=True,
    )
    conv.map_size = 1024

    mc_pos = (161, 68, 540)
    quat   = (0.0, 0.0, 0.0, 1.0)
    scale  = (1.0, 1.0, 1.0)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        conv._print_transform_trace(shape, 0, mc_pos, quat, scale)

    output = buf.getvalue()
    assert "test_building" in output
    assert "OPD position" in output
    assert "building_main" in output
    assert "building_annex" in output
    # Annex has non-zero pivot: offset must appear in output
    assert "MC offset" in output
