"""Round-trip tests for the canonical KO↔MC coordinate transform.

Every function in math3d that converts coordinates must be reversible.
Failure here means a broken transform that will corrupt every placement.

Invariants tested
-----------------
1. XZ round-trip:  ko_to_mc_position → mc_to_ko_position → same KO coords
2. Yaw round-trip: ko_to_mc_yaw      → mc_to_ko_yaw      → same yaw
3. Y round-trip:   ko_to_mc_y        → mc_to_ko_y        → same height (float)
4. Full 3-D:       ko_pos_to_mc      → manual inverse     → same KO coords
5. Axis direction: positive KO-X maps to decreasing MC-X (X mirror)
6. Z unchanged:    KO-Z == MC-Z
7. Yaw negated:    mc_yaw == -ko_yaw
"""

import math
import sys
import os

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ko2mc.math3d import (
    KO_MAP_SIZE,
    MC_Y_BASE,
    KO_HEIGHT_SCALE,
    ko_to_mc_position,
    mc_to_ko_position,
    ko_to_mc_y,
    mc_to_ko_y,
    ko_to_mc_yaw,
    mc_to_ko_yaw,
    ko_pos_to_mc,
)

TOLERANCE = 1e-4  # acceptable floating-point round-trip error


# ── XZ position round-trip ────────────────────────────────────────────────────

@pytest.mark.parametrize("ko_x, ko_z", [
    (0.0,    0.0),
    (512.0,  512.0),
    (1024.0, 1024.0),
    (100.5,  300.75),
    (0.0,    1024.0),
    (1024.0, 0.0),
    (256.0,  768.0),
    (1.0,    1.0),
    (1023.9, 1023.9),
])
def test_xz_round_trip(ko_x, ko_z):
    """ko → mc → ko must be lossless (within integer rounding)."""
    mc_x, mc_z = ko_to_mc_position(ko_x, ko_z)
    ko_x2, ko_z2 = mc_to_ko_position(mc_x, mc_z)
    # Round-trip is exact for integer inputs; float inputs lose at most 1 unit.
    assert abs(ko_x2 - int(ko_x)) < 1.0 + TOLERANCE, (
        f"XZ round-trip X failed: ko_x={ko_x} → mc_x={mc_x} → ko_x2={ko_x2}"
    )
    assert abs(ko_z2 - int(ko_z)) < 1.0 + TOLERANCE, (
        f"XZ round-trip Z failed: ko_z={ko_z} → mc_z={mc_z} → ko_z2={ko_z2}"
    )


def test_x_mirror_direction():
    """Larger KO-X must produce smaller MC-X (X is mirrored)."""
    _, _ = ko_to_mc_position(0.0, 0.0)
    mc_x_low, _  = ko_to_mc_position(100.0, 0.0)
    mc_x_high, _ = ko_to_mc_position(900.0, 0.0)
    assert mc_x_low > mc_x_high, (
        f"X mirror direction wrong: ko_x=100 → mc_x={mc_x_low}, "
        f"ko_x=900 → mc_x={mc_x_high} (expected low > high)"
    )


def test_x_mirror_boundary():
    """ko_x=0 → mc_x=map_size; ko_x=map_size → mc_x=0."""
    mc_x0, _ = ko_to_mc_position(0.0, 0.0)
    assert mc_x0 == KO_MAP_SIZE, f"ko_x=0 → mc_x expected {KO_MAP_SIZE}, got {mc_x0}"
    mc_xN, _ = ko_to_mc_position(float(KO_MAP_SIZE), 0.0)
    assert mc_xN == 0, f"ko_x=MAP_SIZE → mc_x expected 0, got {mc_xN}"


def test_z_unchanged():
    """KO-Z must equal MC-Z exactly."""
    for ko_z in (0.0, 128.0, 512.0, 1024.0, 999.9):
        _, mc_z = ko_to_mc_position(0.0, ko_z)
        assert mc_z == int(ko_z), (
            f"Z changed: ko_z={ko_z} → mc_z={mc_z} (expected {int(ko_z)})"
        )


def test_custom_map_size_round_trip():
    """Round-trip must work for non-default map sizes."""
    for size in (512, 2048, 4096):
        mc_x, mc_z = ko_to_mc_position(100.0, 200.0, map_size=size)
        ko_x2, ko_z2 = mc_to_ko_position(mc_x, mc_z, map_size=size)
        assert abs(ko_x2 - 100.0) < 1.0 + TOLERANCE
        assert abs(ko_z2 - 200.0) < 1.0 + TOLERANCE


# ── Y height round-trip ───────────────────────────────────────────────────────

@pytest.mark.parametrize("ko_y", [
    0.0, 1.0, -1.0, 10.0, 50.0, 100.0, 200.0, -10.0, 0.5, 7.3,
])
def test_y_round_trip(ko_y):
    """ko_y → mc_y → ko_y must recover original height within scale precision."""
    mc_y = ko_to_mc_y(ko_y)
    ko_y2 = mc_to_ko_y(mc_y)
    # Integer truncation in ko_to_mc_y causes at most 1/KO_HEIGHT_SCALE error.
    tolerance = 1.0 / KO_HEIGHT_SCALE + TOLERANCE
    assert abs(ko_y2 - ko_y) <= tolerance, (
        f"Y round-trip failed: ko_y={ko_y} → mc_y={mc_y} → ko_y2={ko_y2}"
    )


def test_y_baseline():
    """ko_y=0 must land at MC_Y_BASE."""
    assert ko_to_mc_y(0.0) == MC_Y_BASE, (
        f"ko_y=0 expected mc_y={MC_Y_BASE}, got {ko_to_mc_y(0.0)}"
    )


def test_y_scale():
    """Each KO unit must raise MC Y by KO_HEIGHT_SCALE blocks."""
    y0 = ko_to_mc_y(0.0)
    y1 = ko_to_mc_y(1.0 / KO_HEIGHT_SCALE)
    assert y1 == y0 + 1, (
        f"Y scale wrong: ko_y=0→{y0}, ko_y={1/KO_HEIGHT_SCALE}→{y1} (expected {y0+1})"
    )


# ── Yaw round-trip ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ko_yaw", [
    0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 4, -math.pi / 2,
    math.pi * 1.5, -math.pi, 0.001, -0.001,
])
def test_yaw_round_trip(ko_yaw):
    """ko_yaw → mc_yaw → ko_yaw must be identity."""
    mc_yaw = ko_to_mc_yaw(ko_yaw)
    ko_yaw2 = mc_to_ko_yaw(mc_yaw)
    assert abs(ko_yaw2 - ko_yaw) < TOLERANCE, (
        f"Yaw round-trip failed: ko_yaw={ko_yaw:.4f} → {mc_yaw:.4f} → {ko_yaw2:.4f}"
    )


def test_yaw_negated():
    """mc_yaw must equal -ko_yaw (X mirror negates rotation)."""
    for ko_yaw in (0.1, -0.2, math.pi / 3, -math.pi):
        mc_yaw = ko_to_mc_yaw(ko_yaw)
        assert abs(mc_yaw - (-ko_yaw)) < TOLERANCE, (
            f"Yaw not negated: ko_yaw={ko_yaw} → mc_yaw={mc_yaw} (expected {-ko_yaw})"
        )


def test_yaw_zero_preserved():
    """ko_yaw=0 → mc_yaw=0."""
    assert ko_to_mc_yaw(0.0) == 0.0
    assert mc_to_ko_yaw(0.0) == 0.0


# ── Full 3-D position round-trip ──────────────────────────────────────────────

@pytest.mark.parametrize("ko_x, ko_y, ko_z", [
    (0.0,   0.0,  0.0),
    (500.0, 10.0, 300.0),
    (1.0,   50.0, 999.0),
    (1023.0, -5.0, 512.0),
])
def test_full_3d_round_trip(ko_x, ko_y, ko_z):
    """ko_pos_to_mc → manual inverse → original KO coords (within rounding)."""
    mc_x, mc_y, mc_z = ko_pos_to_mc(ko_x, ko_y, ko_z)
    # Inverse
    ko_x2, ko_z2 = mc_to_ko_position(mc_x, mc_z)
    ko_y2 = mc_to_ko_y(mc_y)

    assert abs(ko_x2 - int(ko_x)) < 1.0 + TOLERANCE
    assert abs(ko_z2 - int(ko_z)) < 1.0 + TOLERANCE
    assert abs(ko_y2 - ko_y) <= 1.0 / KO_HEIGHT_SCALE + TOLERANCE


# ── Object placement round-trip (multi-object simulation) ─────────────────────

def test_opd_batch_round_trip():
    """Simulate OPD batch: convert all positions to MC and back, check Δ < 1."""
    objects = [
        # (ko_x, ko_z, ko_yaw) — representative KO world placements
        (100.0, 200.0, 0.0),
        (512.0, 512.0, math.pi / 4),
        (863.0, 540.0, -math.pi / 2),   # approx. Moradon castle area
        (1.0,   1.0,   0.0),
        (1023.0, 1023.0, math.pi),
    ]
    for ko_x, ko_z, ko_yaw in objects:
        mc_x, mc_z = ko_to_mc_position(ko_x, ko_z)
        mc_yaw     = ko_to_mc_yaw(ko_yaw)
        ko_x2, ko_z2 = mc_to_ko_position(mc_x, mc_z)
        ko_yaw2      = mc_to_ko_yaw(mc_yaw)

        assert abs(ko_x2 - int(ko_x)) < 1.0 + TOLERANCE, \
            f"Batch X: ko_x={ko_x} → {mc_x} → {ko_x2}"
        assert abs(ko_z2 - int(ko_z)) < 1.0 + TOLERANCE, \
            f"Batch Z: ko_z={ko_z} → {mc_z} → {ko_z2}"
        assert abs(ko_yaw2 - ko_yaw) < TOLERANCE, \
            f"Batch yaw: {ko_yaw} → {mc_yaw} → {ko_yaw2}"
