"""Terrain round-trip validation tests.

Validates that the canonical height grid (h_mc = build_mc_height_grid(gtd.heights))
preserves height values and that terrain_y_at() recovers them consistently.

Transform: np.rot90(gtd_heights, k=1)  →  h_mc[i, j] = gtd_heights[j, n-1-i]
Root cause: GTD binary is X-major (outer loop = KO_X, inner = KO_Z) but the
parser loop variables are named z-outer / x-inner, so the stored array has axes
transposed: gtd_heights[parser_x, parser_z] where parser_x = KO_Z tile and
parser_z = KO_X tile.  Equivalently: height at KO tile (ko_tx, ko_tz) is stored
at gtd_heights[ko_tz, ko_tx].  rot90 CCW corrects both the transpose and the
required X-mirror (mc_tx = n-1-ko_tx) in one shot.

Tests
-----
1. build_mc_height_grid: heights are preserved (no values lost)
2. build_mc_height_grid: rot90 CCW semantics h_mc[i,j] == gtd_heights[j, n-1-i]
3. terrain_y_at: exact sample at grid tile centres matches ko_to_mc_y
4. terrain_y_at: bilinear interpolation between two known heights
5. Reconstruction error: convert h_mc back to KO heights and compare
6. Consistency: terrain_y_at(mc_x, mc_z) == ko_to_mc_y(gtd.heights[ko_tz, ko_tx])
7. ko_to_mc_y rounding: deterministic round-half-up, ≤0.5 block error
8. terrain_y_at_footprint: flat terrain, sloped terrain, footprint-min invariant
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from ko2mc.math3d import (
    MC_Y_BASE,
    KO_HEIGHT_SCALE,
    build_mc_height_grid,
    terrain_y_at,
    terrain_y_at_footprint,
    ko_to_mc_y,
    ko_to_mc_position,
    mc_to_ko_position,
)


def _make_gtd_heights(n: int = 9, seed: int = 42) -> np.ndarray:
    """Create a synthetic GTD heights array (n×n, float32)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 50.0, size=(n, n)).astype(np.float32)


# ── Grid construction ─────────────────────────────────────────────────────────

def test_build_mc_height_grid_shape():
    """Output shape matches input shape."""
    heights = _make_gtd_heights(13)
    h_mc = build_mc_height_grid(heights)
    assert h_mc.shape == heights.shape


def test_build_mc_height_grid_dtype():
    """Output is float32."""
    heights = _make_gtd_heights(5)
    h_mc = build_mc_height_grid(heights)
    assert h_mc.dtype == np.float32


def test_build_mc_height_grid_x_mirror():
    """h_mc[i, j] == gtd.heights[j, n-1-i]  for all (i, j).

    np.rot90(G, k=1) gives h_mc[i,j] = G[j, n-1-i].  This corrects both the
    GTD axis transposition (parser stored [ko_tz, ko_tx] as [ko_tx, ko_tz]) and
    the required X-mirror mc_tx = (n-1) - ko_tx, matching mc_x = map_size - ko_x.
    """
    n = 7
    heights = _make_gtd_heights(n, seed=1)
    h_mc = build_mc_height_grid(heights)
    for i in range(n):
        for j in range(n):
            assert h_mc[i, j] == pytest.approx(heights[j, n - 1 - i]), (
                f"rot90 CCW failed at [{i},{j}]: h_mc={h_mc[i,j]:.3f}, "
                f"expected heights[{j},{n-1-i}]={heights[j, n-1-i]:.3f}"
            )


def test_build_mc_height_grid_preserves_values():
    """The set of values in h_mc equals the set in gtd.heights."""
    heights = _make_gtd_heights(11, seed=7)
    h_mc = build_mc_height_grid(heights)
    np.testing.assert_array_almost_equal(
        np.sort(h_mc.ravel()),
        np.sort(heights.ravel()),
        decimal=5,
    )


def test_build_mc_height_grid_min_max():
    """min and max heights are identical between source and h_mc."""
    heights = _make_gtd_heights(17, seed=99)
    h_mc = build_mc_height_grid(heights)
    assert h_mc.min() == pytest.approx(heights.min(), abs=1e-5)
    assert h_mc.max() == pytest.approx(heights.max(), abs=1e-5)


# ── terrain_y_at ─────────────────────────────────────────────────────────────

def test_terrain_y_at_tile_centre():
    """Sampling at a tile-centre MC coordinate recovers the exact GTD height.

    A GTD tile corner (ko_tx, ko_tz) covers MC column mc_x = (n-1-ko_tx)*4.
    terrain_y_at at that exact MC column must return ko_to_mc_y(h) where
    h = true height at KO tile (ko_tx, ko_tz).

    Due to the GTD axis transposition (outer binary loop = KO_X stored as
    parser's "z" index), the height at KO tile (ko_tx, ko_tz) is stored at
    gtd_heights[ko_tz, ko_tx] — note the swapped indices.
    """
    n = 9  # 9×9 GTD → 8 tiles → 32 MC blocks per axis
    heights = _make_gtd_heights(n, seed=3)
    h_mc = build_mc_height_grid(heights)

    for ko_tx in range(n):
        for ko_tz in range(n):
            # MC tile coordinates corresponding to this KO tile corner
            mc_tx = (n - 1) - ko_tx
            mc_tz = ko_tz
            mc_x_col = mc_tx * 4
            mc_z_col = mc_tz * 4

            # Axes transposed in storage: height at KO tile (ko_tx, ko_tz)
            # is stored at heights[ko_tz, ko_tx].
            expected_y = ko_to_mc_y(float(heights[ko_tz, ko_tx]))
            got_y = terrain_y_at(h_mc, mc_x_col, mc_z_col)

            assert got_y == expected_y, (
                f"Tile [{ko_tx},{ko_tz}] (mc_col={mc_x_col},{mc_z_col}): "
                f"expected Y={expected_y} from heights[{ko_tz},{ko_tx}], got Y={got_y}"
            )


def test_terrain_y_at_bilinear_midpoint():
    """Midpoint between two tiles should interpolate heights correctly."""
    # 2×2 GTD grid designed so that h_mc has an X-step:
    #   h_mc[0, :] = 10.0   (mc_x tile 0, high side)
    #   h_mc[1, :] =  0.0   (mc_x tile 1, low side)
    #
    # With rot90 CCW: h_mc[i,j] = heights[j, n-1-i], n=2:
    #   h_mc[0,j] = heights[j, 1]  →  set heights[:,1] = 10.0
    #   h_mc[1,j] = heights[j, 0]  →  set heights[:,0] =  0.0
    heights = np.array([[0.0, 10.0],
                        [0.0, 10.0]], dtype=np.float32)
    h_mc = build_mc_height_grid(heights)

    # mc_tx goes 0..1; mc_x=0 → tx_f=0 → h_mc[0,:]=10
    # mc_x=4 would be tx_f=1.0, clamped to tx0=0, tx1=1, fx=1.0 → h_mc[1,:]=0
    # Midpoint mc_x=2 → tx_f=0.5 → blend 0.5*10 + 0.5*0 = 5
    y_mid = terrain_y_at(h_mc, mc_x=2, mc_z=0)
    expected = ko_to_mc_y(5.0)
    assert y_mid == expected, (
        f"Bilinear mid: expected Y={expected} (ko_h=5.0), got {y_mid}"
    )


def test_terrain_y_at_clamp_boundary():
    """terrain_y_at must not raise on out-of-bounds MC coordinates."""
    heights = _make_gtd_heights(5)
    h_mc = build_mc_height_grid(heights)
    world_size = (heights.shape[0] - 1) * 4

    # Edge values should clamp, not error.
    terrain_y_at(h_mc, 0, 0)
    terrain_y_at(h_mc, world_size - 1, world_size - 1)
    terrain_y_at(h_mc, -1, -1)          # below min → clamped
    terrain_y_at(h_mc, world_size + 5, 0)  # above max → clamped


# ── Reconstruction / inverse accuracy ────────────────────────────────────────

def test_terrain_reconstruction_error():
    """Convert h_mc back to KO heights; max reconstruction error < 1 block.

    This simulates exporting a Minecraft terrain and comparing to the original
    GTD heightmap. Voxel-rounding is the only expected source of error.
    """
    heights = _make_gtd_heights(33, seed=55)   # 33×33 → 128 MC blocks/axis
    h_mc = build_mc_height_grid(heights)
    n = heights.shape[0]

    errors = []
    for ko_tx in range(n):
        for ko_tz in range(n):
            mc_tx = (n - 1) - ko_tx
            mc_tz = ko_tz
            mc_x = mc_tx * 4
            mc_z = mc_tz * 4

            mc_y = terrain_y_at(h_mc, mc_x, mc_z)
            # Reconstruct KO height from MC Y.
            # GTD axes are transposed: height at KO tile (ko_tx, ko_tz) is
            # stored at heights[ko_tz, ko_tx].
            ko_h_reconstructed = (mc_y - MC_Y_BASE) / KO_HEIGHT_SCALE
            ko_h_original = float(heights[ko_tz, ko_tx])
            errors.append(abs(ko_h_reconstructed - ko_h_original))

    avg_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    # floor(x+0.5) gives symmetric ±0.5 error; old int/floor gave [0,1) bias.
    acceptable_max = 0.5 / KO_HEIGHT_SCALE + 1e-3
    assert max_err <= acceptable_max, (
        f"Terrain reconstruction max error {max_err:.4f} > {acceptable_max:.4f}"
    )
    assert avg_err < 0.5 / KO_HEIGHT_SCALE + 1e-3, (
        f"Terrain reconstruction avg error {avg_err:.4f} too large"
    )


def test_terrain_consistency_with_canonical_transform():
    """terrain_y_at(mc_x, mc_z) == ko_to_mc_y(gtd.heights[ko_tz, ko_tx]).

    This asserts that the canonical position transform and the height grid
    are consistent: for every GTD tile corner, the MC height at the
    corresponding MC column must match the KO height converted via ko_to_mc_y.

    Due to the GTD axis transposition (parser stored axes swapped), the height
    at KO tile (ko_tx, ko_tz) is at heights[ko_tz, ko_tx].

    The map_size for ko_to_mc_position must match the GTD grid:
        map_size = (n - 1) * 4
    For a 17×17 GTD grid: 16 tiles × 4 KO units = 64 KO units wide.
    """
    n = 17
    map_size = (n - 1) * 4   # = 64 KO units for a 17×17 grid
    heights = _make_gtd_heights(n, seed=77)
    h_mc = build_mc_height_grid(heights)

    mismatches = 0
    for ko_tx in range(n):
        for ko_tz in range(n):
            ko_x = float(ko_tx * 4)
            ko_z = float(ko_tz * 4)
            # Use map_size consistent with this GTD grid size
            mc_x, mc_z = ko_to_mc_position(ko_x, ko_z, map_size=map_size)

            # GTD axes transposed: height at KO tile (ko_tx, ko_tz) → heights[ko_tz, ko_tx]
            expected_y = ko_to_mc_y(float(heights[ko_tz, ko_tx]))
            got_y = terrain_y_at(h_mc, mc_x, mc_z)

            if got_y != expected_y:
                mismatches += 1

    assert mismatches == 0, (
        f"{mismatches} tile(s) showed height mismatch between "
        f"terrain_y_at and ko_to_mc_y — transform is inconsistent."
    )


# ── ko_to_mc_y: deterministic round-half-up behaviour ─────────────────────────

def test_ko_to_mc_y_integer_heights_exact():
    """Integer KO heights map to exact MC Y values with zero rounding loss."""
    for ko_y in [0, 1, 10, 64, 100, 200]:
        assert ko_to_mc_y(float(ko_y)) == MC_Y_BASE + ko_y


def test_ko_to_mc_y_sea_level():
    """KO Y=0 maps to MC_Y_BASE (=64) regardless of rounding rule."""
    assert ko_to_mc_y(0.0) == MC_Y_BASE


def test_ko_to_mc_y_no_truncation_bias():
    """Heights just below an integer boundary should round UP, not truncate down.

    int/floor of 0.9 → 0  (object sinks 0.9 blocks)
    floor(0.9+0.5) = floor(1.4) → 1  (correct)
    """
    assert ko_to_mc_y(0.9) == MC_Y_BASE + 1, (
        "0.9 should round to +1 — truncation bias would give +0 (sinking)"
    )
    assert ko_to_mc_y(0.4) == MC_Y_BASE + 0   # rounds down; error is only 0.4


def test_ko_to_mc_y_half_tie_rounds_up():
    """Exact .5 ties must round up (round-half-up, not banker's rounding).

    Python round(0.5) == 0  (banker's, rounds to even)
    math.floor(0.5+0.5) == 1  (round-half-up, always consistent)
    """
    assert ko_to_mc_y(0.5) == MC_Y_BASE + 1, (
        "0.5 tie must round up — banker's rounding would give +0"
    )
    assert ko_to_mc_y(1.5) == MC_Y_BASE + 2
    assert ko_to_mc_y(2.5) == MC_Y_BASE + 3


def test_ko_to_mc_y_max_error_half_block():
    """Max rounding error for any height must be ≤ 0.5 blocks."""
    import random
    rng = random.Random(7)
    errors = []
    for _ in range(2000):
        ko_y = rng.uniform(0, 200)
        mc_y = ko_to_mc_y(ko_y)
        errors.append(abs((mc_y - MC_Y_BASE) / KO_HEIGHT_SCALE - ko_y))
    assert max(errors) <= 0.5 + 1e-9, (
        f"Max rounding error {max(errors):.9f} > 0.5 — ko_to_mc_y is not using floor(x+0.5)"
    )


# ── terrain_y_at_footprint ────────────────────────────────────────────────────

def _flat_h_grid(height: float, n: int = 33) -> np.ndarray:
    return build_mc_height_grid(np.full((n, n), height, dtype=np.float32))


def test_footprint_zero_extent_equals_point_sample():
    """Zero half-extents must delegate directly to terrain_y_at (centre sample)."""
    h_grid = _flat_h_grid(5.0)
    mc_x, mc_z = 64, 64
    assert terrain_y_at_footprint(h_grid, mc_x, mc_z, 0, 0) == terrain_y_at(h_grid, mc_x, mc_z)


def test_footprint_small_extent_uses_centre():
    """Footprint smaller than one tile (< 4 blocks half-extent) → centre sample."""
    h_grid = _flat_h_grid(10.0)
    expected = ko_to_mc_y(10.0)
    # half_width=3 is below the tile threshold; behaviour equals centre sample
    assert terrain_y_at_footprint(h_grid, 64, 64, 3, 3) == expected


def test_footprint_flat_terrain_any_size_same_y():
    """On flat terrain every footprint size returns the same Y."""
    h_grid = _flat_h_grid(10.0)
    expected = ko_to_mc_y(10.0)
    for hw in [0, 3, 4, 8, 16, 32]:
        got = terrain_y_at_footprint(h_grid, 64, 64, hw, hw)
        assert got == expected, f"Flat terrain: half_width={hw} returned {got}, expected {expected}"


def test_footprint_min_invariant_random_terrain():
    """Core invariant: footprint_y ≤ each of the four corner samples.

    The returned value is the minimum of the four corners, so it must be
    at most equal to — never greater than — any individual corner sample.

    Note: no ordering can be asserted between footprint_y and the centre-point
    sample (terrain_y_at at the centre), because the centre may sit in a valley
    below all corners, or on a peak above them.
    """
    rng = np.random.default_rng(123)
    heights = rng.uniform(0, 50, size=(17, 17)).astype(np.float32)
    h_grid = build_mc_height_grid(heights)

    failures = []
    for mc_x in range(0, 60, 8):
        for mc_z in range(0, 60, 8):
            for hw in [4, 8, 16]:   # half-extents ≥ tile threshold → 4-corner mode
                y_fp = terrain_y_at_footprint(h_grid, mc_x, mc_z, hw, hw)
                corners = {
                    f"(-{hw},-{hw})": terrain_y_at(h_grid, mc_x - hw, mc_z - hw),
                    f"(+{hw},-{hw})": terrain_y_at(h_grid, mc_x + hw, mc_z - hw),
                    f"(-{hw},+{hw})": terrain_y_at(h_grid, mc_x - hw, mc_z + hw),
                    f"(+{hw},+{hw})": terrain_y_at(h_grid, mc_x + hw, mc_z + hw),
                }
                for label, y_corner in corners.items():
                    if y_fp > y_corner:
                        failures.append(
                            f"({mc_x},{mc_z}) hw={hw}: footprint_y={y_fp} > corner{label}={y_corner}"
                        )

    assert not failures, (
        f"terrain_y_at_footprint violated min-invariant in {len(failures)} case(s):\n"
        + "\n".join(failures[:5])
    )


def test_footprint_sloped_terrain_returns_low_corner():
    """On a step-slope, the footprint spanning the step returns the low-side Y.

    GTD grid (3×3) — heights chosen to produce a clear X-step in h_mc.

    With rot90 CCW: h_mc[i,j] = heights[j, n-1-i], n=3:
        h_mc[0,j] = heights[j, 2]
        h_mc[1,j] = heights[j, 1]
        h_mc[2,j] = heights[j, 0]

    Setting heights[:,2]=20, heights[:,0]=heights[:,1]=0 gives:
        h_mc[0,:] = 20   mc_x tiles 0–4   (high side)
        h_mc[1,:] =  0   mc_x tiles 4–8   (low  side)
        h_mc[2,:] =  0   mc_x tiles 8–...  (low  side, clamped)

    Footprint centred at mc_x=4 (the tile boundary), half_width=4:
        → spans mc_x 0 (high, y=84) to mc_x 8 (low, y=64)
        → corners at z=0:  terrain_y_at(mc_x=0) = 84,
                            terrain_y_at(mc_x=8) = 64
        → min = 64 = ko_to_mc_y(0.0)

    half_width=4 == _MC_TILE_BLOCKS threshold → triggers 4-corner mode.
    """
    heights = np.array(
        [[0.0, 0.0, 20.0],
         [0.0, 0.0, 20.0],
         [0.0, 0.0, 20.0]],
        dtype=np.float32,
    )
    h_mc = build_mc_height_grid(heights)

    # Verify setup: high side at mc_x=0, low side at mc_x=8
    assert terrain_y_at(h_mc, 0, 0) == ko_to_mc_y(20.0), "setup: mc_x=0 should be high"
    assert terrain_y_at(h_mc, 8, 0) == ko_to_mc_y(0.0),  "setup: mc_x=8 should be low"

    y_fp = terrain_y_at_footprint(h_mc, mc_x=4, mc_z=0, half_width=4, half_depth=0)
    assert y_fp == ko_to_mc_y(0.0), (
        f"Sloped terrain: footprint Y={y_fp}, expected low-side Y={ko_to_mc_y(0.0)}"
    )


def test_footprint_above_threshold_same_side_no_phantom_low():
    """Large footprint entirely on one side of a step must return that side's Y.

    This guards against the footprint corners accidentally reaching distant
    low terrain (e.g. via grid clamping or extrapolation artefacts) when the
    structure is nowhere near the slope boundary.

    Same 3×3 step grid as the previous test (heights[:,2]=20, rest=0):
        h_mc[0,:] = 20  (mc_x ≈ 0–4)   high side
        h_mc[1,:] =  0  (mc_x ≈ 4–8)   low  side

    Footprint entirely on the HIGH side:
        centre mc_x = 0, half_width = 4
        corners at mc_x = -4 (clamped to 0) and mc_x = 4
        mc_x=0  → h=20 → y = ko_to_mc_y(20)
        mc_x=4  → h= 0 → y = ko_to_mc_y( 0)   ← boundary hit by right corner!

    This exposes a real edge case: a building placed *at* mc_x=0 with a large
    footprint will have its right corner exactly on the step.  The function
    should return the minimum of what its corners actually land on; if the
    right corner clips the low side that IS valid grounding behaviour.

    Footprint entirely CLEAR of the step (centre mc_x = 0, half_width = 3):
        half_width=3 < 4 threshold → centre-sample path
        centre sample at mc_x=0 → h=20 → y = ko_to_mc_y(20)
        No phantom low point from the low side.
    """
    heights = np.array(
        [[0.0, 0.0, 20.0],
         [0.0, 0.0, 20.0],
         [0.0, 0.0, 20.0]],
        dtype=np.float32,
    )
    h_mc = build_mc_height_grid(heights)
    y_high = ko_to_mc_y(20.0)

    # Case A: below threshold — stays on centre-sample path, returns high-side Y.
    y_sub = terrain_y_at_footprint(h_mc, mc_x=0, mc_z=0, half_width=3, half_depth=0)
    assert y_sub == y_high, (
        f"Sub-threshold footprint on high side: got {y_sub}, expected {y_high}"
    )

    # Case B: at threshold — right corner hits mc_x=4 which is the step boundary.
    # terrain_y_at(mc_x=4) interpolates to h=0 (low side), so min-corner returns
    # the low-side Y.  This is CORRECT grounding: the footprint genuinely reaches
    # the step and the base must not be placed above the lower ground.
    y_at = terrain_y_at_footprint(h_mc, mc_x=0, mc_z=0, half_width=4, half_depth=0)
    y_low = ko_to_mc_y(0.0)
    assert y_at == y_low, (
        f"Threshold-width footprint touching step: got {y_at}, expected low-side {y_low}"
    )
