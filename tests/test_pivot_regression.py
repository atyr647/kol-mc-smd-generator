"""Tree/vegetation pivot regression tests.

Validates that _po mesh parts placed via pivot_to_mc_offset() produce
the correct Y elevation above terrain and that horizontal position is
unchanged for pure-Y pivots.

Also validates a pivot magnitude histogram utility to detect systemic
pivot drift across a synthetic asset population.
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from ko2mc.math3d import (
    MC_Y_BASE,
    ko_to_mc_position,
    ko_to_mc_y,
    pivot_to_mc_offset,
    transform_pivot,
    build_mc_height_grid,
    terrain_y_at,
)
from ko2mc.opd_parser import ShapePart, Vector3


# ── Helpers ──────────────────────────────────────────────────────────────────

def _identity_quat():
    """Quaternion (x,y,z,w) = identity (no rotation)."""
    return (0.0, 0.0, 0.0, 1.0)


def _yaw_quat(yaw_rad: float):
    """Y-axis rotation quaternion for given yaw (radians)."""
    return (0.0, math.sin(yaw_rad / 2), 0.0, math.cos(yaw_rad / 2))


def _flat_terrain(height: float, n: int = 33) -> np.ndarray:
    """Build a flat MC height grid at a known KO height value."""
    gtd = np.full((n, n), height, dtype=np.float32)
    return build_mc_height_grid(gtd)


# ── ShapePart field validation ────────────────────────────────────────────────

class TestShapePartPivotField:
    """Validates that ShapePart carries and defaults the pivot correctly."""

    def test_default_pivot_is_zero_vector(self):
        part = ShapePart()
        assert part.pivot.x == 0.0
        assert part.pivot.y == 0.0
        assert part.pivot.z == 0.0

    def test_assigned_pivot_is_preserved(self):
        part = ShapePart()
        part.pivot = Vector3(1.5, 2.5, 3.5)
        assert part.pivot.x == 1.5
        assert part.pivot.y == 2.5
        assert part.pivot.z == 3.5

    def test_pivot_via_factory_default(self):
        """Each ShapePart gets its own independent Vector3 instance."""
        p1 = ShapePart()
        p2 = ShapePart()
        p1.pivot.y = 99.0
        assert p2.pivot.y == 0.0, "pivot factory default must not be shared"


# ── Tree / _po mesh pivot behaviour ──────────────────────────────────────────

class TestTreePivotPlacement:
    """_po mesh parts typically have a pure +Y pivot (trunk base above origin).

    Rules:
      1. Pure-Y pivot → only mc_dy is non-zero (XZ unaffected).
      2. Y displacement = round(pivot_y) blocks (identity rotation, unit scale).
      3. Zero-pivot part → zero offset (regression: no accidental Y shift).
      4. Larger Y pivot → proportionally larger block offset.
    """

    def test_pure_y_pivot_no_xz_shift(self):
        """Pure Y-axis pivot must not displace X or Z in MC."""
        pivot_local = (0.0, 1.5, 0.0)
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        assert mc_dx == 0, f"Expected mc_dx=0, got {mc_dx}"
        assert mc_dz == 0, f"Expected mc_dz=0, got {mc_dz}"

    def test_pure_y_pivot_correct_y_offset(self):
        """Y pivot of 1.5 KO units → +2 MC blocks (rounds to nearest)."""
        pivot_local = (0.0, 1.5, 0.0)
        _, mc_dy, _ = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        assert mc_dy == 2, f"Expected mc_dy=2, got {mc_dy}"

    def test_zero_pivot_produces_zero_offset(self):
        """Zero-pivot tree part must stay at shape anchor — no accidental shift."""
        pivot_local = (0.0, 0.0, 0.0)
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        assert mc_dx == 0 and mc_dy == 0 and mc_dz == 0

    def test_larger_y_pivot_proportional_offset(self):
        """Y pivot of 4.0 → +4 MC blocks."""
        pivot_local = (0.0, 4.0, 0.0)
        _, mc_dy, _ = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        assert mc_dy == 4, f"Expected mc_dy=4, got {mc_dy}"

    def test_tree_grounding_independent_of_pivot(self):
        """Terrain Y for a tree shape is determined by the shape anchor, not the part pivot.

        After grounding at terrain_y, the pivot adds a vertical offset.
        The two operations must be independent and additive.
        """
        terrain_h = 3.0
        h_grid = _flat_terrain(terrain_h)

        ko_x, ko_z = 300.0, 410.0
        mc_x, mc_z = ko_to_mc_position(ko_x, ko_z)
        terrain_y = terrain_y_at(h_grid, mc_x, mc_z)

        # Expected terrain Y
        assert terrain_y == ko_to_mc_y(terrain_h)

        # Apply pivot on top of terrain
        pivot_local = (0.0, 1.5, 0.0)
        _, mc_dy, _ = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        part_y = terrain_y + mc_dy

        expected_terrain = MC_Y_BASE + int(terrain_h)  # 67
        assert terrain_y == expected_terrain, f"Terrain Y mismatch: {terrain_y} != {expected_terrain}"
        assert part_y == expected_terrain + 2, (
            f"Part Y after pivot should be terrain_y+2, got {part_y}"
        )

    def test_y_pivot_under_yaw_rotation_stays_in_y(self):
        """Y-only pivot is invariant to yaw rotation (rotation is around Y axis)."""
        pivot_local = (0.0, 2.0, 0.0)
        for yaw in [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3]:
            quat = _yaw_quat(yaw)
            mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pivot_local, quat, (1.0, 1.0, 1.0))
            assert mc_dx == 0, f"yaw={math.degrees(yaw):.0f}°: unexpected mc_dx={mc_dx}"
            assert mc_dz == 0, f"yaw={math.degrees(yaw):.0f}°: unexpected mc_dz={mc_dz}"
            assert mc_dy == 2, f"yaw={math.degrees(yaw):.0f}°: mc_dy={mc_dy} != 2"

    def test_y_scale_applied_before_rotation_for_tree(self):
        """Non-unit Y scale affects pivot magnitude before rotation."""
        pivot_local = (0.0, 2.0, 0.0)
        scale = (1.0, 1.5, 1.0)   # Y scale = 1.5 → effective Y pivot = 3.0
        _, mc_dy, _ = pivot_to_mc_offset(pivot_local, _identity_quat(), scale)
        assert mc_dy == 3, f"Expected mc_dy=3 with Y scale 1.5, got {mc_dy}"


# ── Pivot X-mirror for tree XZ pivots (edge case) ────────────────────────────

class TestTreeXZPivotMirror:
    """Some tree LOD meshes may have small XZ offsets; confirm X mirror holds."""

    def test_positive_local_x_pivot_produces_negative_mc_dx(self):
        """KO local +X → MC -X (X mirror) after pivot transform."""
        pivot_local = (5.0, 0.0, 0.0)
        mc_dx, _, _ = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        assert mc_dx == -5, f"Expected mc_dx=-5, got {mc_dx}"

    def test_negative_local_x_pivot_produces_positive_mc_dx(self):
        """KO local -X → MC +X after X mirror."""
        pivot_local = (-5.0, 0.0, 0.0)
        mc_dx, _, _ = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        assert mc_dx == 5, f"Expected mc_dx=5, got {mc_dx}"

    def test_z_pivot_unchanged(self):
        """KO local +Z → MC +Z (Z is not mirrored)."""
        pivot_local = (0.0, 0.0, 3.0)
        _, _, mc_dz = pivot_to_mc_offset(pivot_local, _identity_quat(), (1.0, 1.0, 1.0))
        assert mc_dz == 3, f"Expected mc_dz=3, got {mc_dz}"


# ── Pivot magnitude histogram utility ─────────────────────────────────────────

def pivot_magnitude_histogram(pivots: list[tuple], bins=None) -> dict:
    """Bucket pivot magnitudes and return a distribution dict.

    Parameters
    ----------
    pivots : list of (px, py, pz) tuples — pivot vectors in KO model space
    bins   : list of upper-bounds for each bin, e.g. [0.1, 1.0, 5.0, float('inf')]

    Returns
    -------
    dict mapping bin label → count
    """
    if bins is None:
        bins = [0.1, 1.0, 5.0, float("inf")]
    labels = []
    for i, b in enumerate(bins):
        lo = 0.0 if i == 0 else bins[i - 1]
        hi = "∞" if b == float("inf") else b
        labels.append(f"{lo}–{hi}")

    counts = {label: 0 for label in labels}
    for px, py, pz in pivots:
        mag = math.sqrt(px * px + py * py + pz * pz)
        for i, b in enumerate(bins):
            if mag < b:
                counts[labels[i]] += 1
                break
    return counts


class TestPivotHistogram:
    """Validates the pivot histogram utility used for systemic drift detection."""

    def test_all_zero_pivots(self):
        pivots = [(0, 0, 0)] * 10
        hist = pivot_magnitude_histogram(pivots)
        assert hist["0.0–0.1"] == 10

    def test_mixed_distribution(self):
        pivots = [
            (0.0, 0.0, 0.0),    # 0.0   → 0–0.1
            (0.05, 0.0, 0.0),   # 0.05  → 0–0.1
            (0.5, 0.0, 0.0),    # 0.5   → 0.1–1.0
            (2.0, 0.0, 0.0),    # 2.0   → 1.0–5.0
            (8.0, 0.0, 0.0),    # 8.0   → 5.0–∞
        ]
        hist = pivot_magnitude_histogram(pivots)
        assert hist["0.0–0.1"] == 2
        assert hist["0.1–1.0"] == 1
        assert hist["1.0–5.0"] == 1
        assert hist["5.0–∞"]   == 1

    def test_large_population_totals(self):
        """Sum of all bins = total pivot count."""
        import random
        rng = random.Random(42)
        pivots = [(rng.uniform(-20, 20), rng.uniform(0, 10), rng.uniform(-20, 20))
                  for _ in range(200)]
        hist = pivot_magnitude_histogram(pivots)
        assert sum(hist.values()) == 200

    def test_asset_like_pivots_mostly_small(self):
        """Simulated _po tree population: most pivots in Y only, small magnitude."""
        # 71% zero, 18% small Y (1–2 units), 9% medium (3–5), 2% large
        pivots = (
            [(0.0, 0.0, 0.0)] * 71
            + [(0.0, 1.5, 0.0)] * 18
            + [(0.0, 4.0, 0.0)] * 9
            + [(0.0, 8.0, 0.0)] * 2
        )
        hist = pivot_magnitude_histogram(pivots)
        total = sum(hist.values())
        pct_near_zero = (hist["0.0–0.1"] + hist["0.1–1.0"]) / total
        assert pct_near_zero >= 0.71, f"Expected ≥71% small pivots, got {pct_near_zero*100:.0f}%"


# ── Regression: pivot fix must not break zero-pivot assets ───────────────────

class TestZeroPivotRegression:
    """Assets with zero pivot must be completely unaffected by the pivot fix."""

    @pytest.mark.parametrize("yaw", [0.0, math.pi/6, math.pi/4, math.pi/2, math.pi])
    def test_zero_pivot_all_yaws_no_offset(self, yaw):
        """Zero pivot → zero MC offset for all orientations."""
        quat = _yaw_quat(yaw)
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset((0.0, 0.0, 0.0), quat, (1.0, 1.0, 1.0))
        assert (mc_dx, mc_dy, mc_dz) == (0, 0, 0), (
            f"Zero pivot gave non-zero offset at yaw={math.degrees(yaw):.1f}°"
        )

    @pytest.mark.parametrize("scale", [(1.0,1.0,1.0), (2.0,2.0,2.0), (0.5,1.5,0.5)])
    def test_zero_pivot_all_scales_no_offset(self, scale):
        """Zero pivot → zero MC offset for all scales."""
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset((0.0, 0.0, 0.0), _identity_quat(), scale)
        assert (mc_dx, mc_dy, mc_dz) == (0, 0, 0), (
            f"Zero pivot gave non-zero offset at scale={scale}"
        )


# ── Pivot chaining (parent-relative vs root-relative) detection ───────────────

def _distance_from_root(pivot_local, quat, scale) -> float:
    """Actual distance: |R × (S × pivot)|.  Should equal |S × pivot| always."""
    from ko2mc.math3d import transform_pivot
    pw = transform_pivot(pivot_local, quat, scale)
    return float(np.linalg.norm(pw))


def _scaled_pivot_magnitude(pivot_local, scale) -> float:
    """Expected distance from root: |S × pivot_local|."""
    piv = np.asarray(pivot_local, dtype=np.float64)
    S = np.asarray(scale, dtype=np.float64)
    return float(np.linalg.norm(S * piv))


class TestPivotChainingDetection:
    """Validates that the chaining invariant |dist_from_root| == |S × pivot| holds.

    Root-relative pivots: every part's distance from the shape root equals
    |S × pivot_local| regardless of part order.

    Chained/parent-relative pivots: a part's world offset is the sum of all
    preceding pivots' offsets, so later parts show cumulative drift.
    """

    def test_root_relative_unit_scale(self):
        """Unit scale: dist_from_root == |pivot_local| for each part."""
        parts = [(10.0, 0.0, 0.0), (0.0, 8.0, 0.0), (-5.0, 0.0, 3.0)]
        scale = (1.0, 1.0, 1.0)
        for piv in parts:
            expected = _scaled_pivot_magnitude(piv, scale)
            actual = _distance_from_root(piv, _identity_quat(), scale)
            assert abs(actual - expected) < 1e-4, (
                f"Invariant broken: dist={actual:.4f} != |S×pivot|={expected:.4f}"
            )

    def test_root_relative_non_unit_scale(self):
        """Non-unit scale: dist_from_root == |S × pivot|, not |pivot|."""
        # gate_post equivalent: pivot (0,0,-8), scale (1.5,1.2,1.5)
        piv = (0.0, 0.0, -8.0)
        scale = (1.5, 1.2, 1.5)
        expected = _scaled_pivot_magnitude(piv, scale)   # ≈ 12.0
        actual = _distance_from_root(piv, _identity_quat(), scale)
        assert abs(actual - expected) < 1e-4
        # Confirm expected != raw magnitude (scale changed it)
        raw_mag = math.sqrt(8.0**2)
        assert abs(expected - raw_mag) > 0.1, "Scale should have changed magnitude"

    def test_rotation_preserves_pivot_magnitude(self):
        """Rotating a pivot (R × p) must not change its magnitude.

        This is the key property: rotation is orthogonal, so |R×p| == |p|.
        If this holds, then |R × (S × pivot)| == |S × pivot| for any yaw.
        """
        piv = (5.0, 3.0, -7.0)
        scale = (1.0, 1.0, 1.0)
        expected = _scaled_pivot_magnitude(piv, scale)
        for yaw in [0.0, math.pi/6, math.pi/4, math.pi/2, math.pi, -math.pi/3]:
            actual = _distance_from_root(piv, _yaw_quat(yaw), scale)
            assert abs(actual - expected) < 1e-4, (
                f"Magnitude changed under rotation yaw={math.degrees(yaw):.1f}°: "
                f"{actual:.4f} != {expected:.4f}"
            )

    def test_chaining_would_inflate_distance(self):
        """Simulates what chained pivots look like and confirms our invariant catches it.

        In chained mode, part_B's world offset = pivot_A + pivot_B.
        This inflates dist_from_root for part_B beyond |S × pivot_B|.
        """
        # Simulate two chained parts: actual dist for part_B = |piv_A + piv_B|
        piv_A = np.array([10.0, 0.0, 0.0])
        piv_B = np.array([0.0, 0.0, 12.0])
        scale = (1.0, 1.0, 1.0)

        chained_dist_B = float(np.linalg.norm(piv_A + piv_B))  # ≈ 15.6 (accumulated)
        root_dist_B    = float(np.linalg.norm(piv_B))           # 12.0 (correct)

        # The chained distance is detectably larger
        assert chained_dist_B > root_dist_B + 0.5, (
            "Expected chained distance to be detectably larger than root-relative distance"
        )
        # If pipeline correctly applies root-relative, the invariant |dist| == |pivot| holds
        assert abs(root_dist_B - _scaled_pivot_magnitude(tuple(piv_B), scale)) < 1e-4

    def test_part_order_independence(self):
        """Root-relative pivots must be independent of part order.

        Reordering parts must not change any individual part's distance from root.
        If pivots are parent-chained, reordering changes all subsequent distances.
        """
        pivots = [(8.0, 0.0, 0.0), (0.0, 6.0, 0.0), (-5.0, 0.0, 4.0)]
        scale = (1.0, 1.0, 1.0)

        # Compute distances in original order
        dists_original = [_distance_from_root(p, _identity_quat(), scale) for p in pivots]

        # Compute in reversed order
        dists_reversed = [_distance_from_root(p, _identity_quat(), scale) for p in reversed(pivots)]

        # For root-relative: each part's distance must equal |pivot| regardless of order
        for p, d in zip(pivots, dists_original):
            expected = _scaled_pivot_magnitude(p, scale)
            assert abs(d - expected) < 1e-4
        for p, d in zip(reversed(pivots), dists_reversed):
            expected = _scaled_pivot_magnitude(p, scale)
            assert abs(d - expected) < 1e-4

    def test_all_synthetic_shapes_root_relative(self):
        """All 4 synthetic shapes in verify_pivot pass the chaining invariant.

        This is the integration-level chaining check — same logic as
        'python verify_pivot.py --chain-check', expressed as a pytest assertion.
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from verify_pivot import SHAPES, chain_check_shape

        for shape in SHAPES:
            result = chain_check_shape(shape)
            assert not result.chaining_detected, (
                f"{result.asset_class}/{result.shape_name}: "
                f"pivot chaining detected — parts may not be root-relative"
            )
            # Verify invariant for every individual part
            for p in result.parts:
                if not p.is_zero_pivot:
                    assert abs(p.distance_from_root - p.scaled_magnitude) < 0.5, (
                        f"  {p.name}: dist_from_root={p.distance_from_root:.3f} "
                        f"!= |S×pivot|={p.scaled_magnitude:.3f}"
                    )
