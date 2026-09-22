"""Landmark debug report generator and validator.

This module implements the debug pipeline described in the stabilization plan:

  For each known landmark (Moradon castle, harbor, main gate):
    - Accept OPD position + yaw
    - Apply canonical transforms
    - Compute terrain Y (from synthetic or real h_grid)
    - Print structured debug report

When run as a script it prints a human-readable report.
When run under pytest it validates that the transform pipeline produces
consistent, non-degenerate results for canonical landmark positions.

Known landmarks (USKO Moradon v1298 approximate world coords)
-------------------------------------------------------------
  Castle      OPD ~(863, 385, 540)
  Harbor ship OPD ~(200, 10,  700)
  Main gate   OPD ~(510, 5,   870)
"""

import math
import sys
import os
import dataclasses
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ko2mc.math3d import (
    KO_MAP_SIZE,
    MC_Y_BASE,
    ko_to_mc_position,
    mc_to_ko_position,
    ko_to_mc_y,
    ko_to_mc_yaw,
    mc_to_ko_yaw,
    build_mc_height_grid,
    terrain_y_at,
    quat_yaw,
)


# ── Known landmarks ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class Landmark:
    """A known KO world object used for debug validation."""
    name: str
    asset_hint: str          # partial mesh path for identification
    opd_x: float
    opd_y: float             # KO world Y (height above datum)
    opd_z: float
    opd_yaw: float           # radians
    opd_quat: tuple          # (x, y, z, w) from OPD rotation field

    # Expected MC values (set during transform, compared in tests)
    expected_mc_x: Optional[int] = None
    expected_mc_z: Optional[int] = None
    expected_mc_yaw: Optional[float] = None


LANDMARKS: list[Landmark] = [
    Landmark(
        name="Moradon Castle",
        asset_hint="building/castle",
        opd_x=863.0, opd_y=4.74, opd_z=540.0,
        opd_yaw=0.0,
        opd_quat=(0.0, 0.0, 0.0, 1.0),
        # Expected: mc_x = 1024-863=161, mc_z=540
        expected_mc_x=161,
        expected_mc_z=540,
        expected_mc_yaw=0.0,
    ),
    Landmark(
        name="Harbor Ship",
        asset_hint="ship/harbor",
        opd_x=200.0, opd_y=10.0, opd_z=700.0,
        opd_yaw=math.pi / 2,
        opd_quat=(0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)),
        expected_mc_x=824,
        expected_mc_z=700,
        expected_mc_yaw=-math.pi / 2,
    ),
    Landmark(
        name="Main Gate",
        asset_hint="gate/main",
        opd_x=510.0, opd_y=5.0, opd_z=870.0,
        opd_yaw=-math.pi / 4,
        opd_quat=(0.0, -math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8)),
        expected_mc_x=514,
        expected_mc_z=870,
        expected_mc_yaw=math.pi / 4,
    ),
]


# ── Debug report ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LandmarkReport:
    landmark: Landmark
    mc_x: int
    mc_z: int
    mc_yaw: float
    terrain_y: int
    transforms_applied: list[str]

    # Round-trip check
    roundtrip_ko_x: float
    roundtrip_ko_z: float
    roundtrip_ko_yaw: float
    roundtrip_dx: float
    roundtrip_dz: float
    roundtrip_dyaw: float

    def print_report(self) -> None:
        lm = self.landmark
        print(f"\n{'─'*60}")
        print(f"  LANDMARK: {lm.name}")
        print(f"  Asset:    {lm.asset_hint}")
        print(f"{'─'*60}")
        print(f"  OPD position:   ({lm.opd_x:.2f}, {lm.opd_y:.2f}, {lm.opd_z:.2f})")
        print(f"  OPD yaw:        {lm.opd_yaw:.4f} rad")
        print(f"  OPD quaternion: ({lm.opd_quat[0]:.3f}, {lm.opd_quat[1]:.3f}, "
              f"{lm.opd_quat[2]:.3f}, {lm.opd_quat[3]:.3f})")
        print(f"  MC position:    ({self.mc_x}, {self.terrain_y}, {self.mc_z})")
        print(f"  MC yaw:         {self.mc_yaw:.4f} rad")
        print(f"  Terrain Y:      {self.terrain_y}")
        print(f"  Transforms:     {', '.join(self.transforms_applied)}")
        print(f"  Round-trip ΔX:  {self.roundtrip_dx:.4f}")
        print(f"  Round-trip ΔZ:  {self.roundtrip_dz:.4f}")
        print(f"  Round-trip ΔYaw:{self.roundtrip_dyaw:.6f}")
        ok = (self.roundtrip_dx < 1.5 and
              self.roundtrip_dz < 1.5 and
              self.roundtrip_dyaw < 1e-4)
        print(f"  Round-trip OK:  {'✓' if ok else '✗ FAIL'}")


def build_landmark_report(lm: Landmark, h_grid: np.ndarray) -> LandmarkReport:
    """Run the full pipeline for one landmark and build its debug report."""
    transforms = []

    # Step 1: canonical XZ transform
    mc_x, mc_z = ko_to_mc_position(lm.opd_x, lm.opd_z)
    transforms.append("ko_to_mc_position(x,z)")

    # Step 2: canonical yaw transform
    mc_yaw = ko_to_mc_yaw(lm.opd_yaw)
    transforms.append("ko_to_mc_yaw")

    # Step 3: terrain Y
    terrain_y = terrain_y_at(h_grid, mc_x, mc_z)
    transforms.append("terrain_y_at(h_grid)")

    # Step 4: round-trip check
    rt_ko_x, rt_ko_z = mc_to_ko_position(mc_x, mc_z)
    rt_ko_yaw = mc_to_ko_yaw(mc_yaw)
    dx   = abs(rt_ko_x   - int(lm.opd_x))
    dz   = abs(rt_ko_z   - int(lm.opd_z))
    dyaw = abs(rt_ko_yaw - lm.opd_yaw)

    return LandmarkReport(
        landmark           = lm,
        mc_x               = mc_x,
        mc_z               = mc_z,
        mc_yaw             = mc_yaw,
        terrain_y          = terrain_y,
        transforms_applied = transforms,
        roundtrip_ko_x     = rt_ko_x,
        roundtrip_ko_z     = rt_ko_z,
        roundtrip_ko_yaw   = rt_ko_yaw,
        roundtrip_dx       = dx,
        roundtrip_dz       = dz,
        roundtrip_dyaw     = dyaw,
    )


def _make_flat_h_grid(n: int = 257, height: float = 4.74) -> np.ndarray:
    """Synthetic flat terrain h_grid at a given height."""
    gtd_heights = np.full((n, n), height, dtype=np.float32)
    return build_mc_height_grid(gtd_heights)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLandmarkTransforms:
    """Validate that landmark positions transform correctly."""

    @pytest.fixture
    def h_grid(self):
        return _make_flat_h_grid(n=257, height=5.0)

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_mc_x_correct(self, lm, h_grid):
        report = build_landmark_report(lm, h_grid)
        assert report.mc_x == lm.expected_mc_x, (
            f"{lm.name}: mc_x={report.mc_x}, expected {lm.expected_mc_x}"
        )

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_mc_z_correct(self, lm, h_grid):
        report = build_landmark_report(lm, h_grid)
        assert report.mc_z == lm.expected_mc_z, (
            f"{lm.name}: mc_z={report.mc_z}, expected {lm.expected_mc_z}"
        )

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_mc_yaw_correct(self, lm, h_grid):
        report = build_landmark_report(lm, h_grid)
        assert abs(report.mc_yaw - lm.expected_mc_yaw) < 1e-6, (
            f"{lm.name}: mc_yaw={report.mc_yaw:.6f}, expected {lm.expected_mc_yaw:.6f}"
        )

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_round_trip_x(self, lm, h_grid):
        report = build_landmark_report(lm, h_grid)
        assert report.roundtrip_dx < 1.5, (
            f"{lm.name}: round-trip ΔX={report.roundtrip_dx:.4f} ≥ 1.5"
        )

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_round_trip_z(self, lm, h_grid):
        report = build_landmark_report(lm, h_grid)
        assert report.roundtrip_dz < 1.5, (
            f"{lm.name}: round-trip ΔZ={report.roundtrip_dz:.4f} ≥ 1.5"
        )

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_round_trip_yaw(self, lm, h_grid):
        report = build_landmark_report(lm, h_grid)
        assert report.roundtrip_dyaw < 1e-6, (
            f"{lm.name}: round-trip ΔYaw={report.roundtrip_dyaw:.8f} ≥ 1e-6"
        )

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_terrain_y_reasonable(self, lm, h_grid):
        """Terrain Y must be within [MC_Y_BASE - 200, MC_Y_BASE + 400]."""
        report = build_landmark_report(lm, h_grid)
        assert MC_Y_BASE - 200 <= report.terrain_y <= MC_Y_BASE + 400, (
            f"{lm.name}: terrain_y={report.terrain_y} out of reasonable MC range"
        )

    @pytest.mark.parametrize("lm", LANDMARKS, ids=[lm.name for lm in LANDMARKS])
    def test_no_transform_duplication(self, lm, h_grid):
        """Each transform step must be applied exactly once."""
        report = build_landmark_report(lm, h_grid)
        assert len(report.transforms_applied) == 3, (
            f"{lm.name}: expected 3 transforms, got {len(report.transforms_applied)}: "
            f"{report.transforms_applied}"
        )

    def test_castle_height_from_gtd(self):
        """Castle terrain Y from h_grid must match ko_to_mc_y(gtd_height_at_castle)."""
        castle = next(lm for lm in LANDMARKS if "Castle" in lm.name)
        castle_gtd_h = 4.74

        # Build a synthetic GTD heights array with castle height at the correct
        # parser-index position.  The GTD binary is X-major (outer loop = KO_X,
        # inner = KO_Z) but the parser's loop variables are named "z" (outer) and
        # "x" (inner), so the stored array has axes swapped relative to KO tile
        # coordinates: gtd_heights[parser_x, parser_z] where parser_x = KO_Z tile
        # and parser_z = KO_X tile.  To set the height at KO tile (ko_tx, ko_tz)
        # we must therefore index [ko_tz, ko_tx].
        n = 257
        ko_tx = int(castle.opd_x / 4)   # tile X
        ko_tz = int(castle.opd_z / 4)   # tile Z
        gtd_heights = np.zeros((n, n), dtype=np.float32)
        gtd_heights[ko_tz, ko_tx] = castle_gtd_h   # note: [tz, tx] — axes transposed
        h_grid = build_mc_height_grid(gtd_heights)

        mc_x, mc_z = ko_to_mc_position(castle.opd_x, castle.opd_z)
        # Sample at the exact tile MC column
        mc_tx = (n - 1) - ko_tx
        mc_tz = ko_tz
        got_y = terrain_y_at(h_grid, mc_tx * 4, mc_tz * 4)
        expected_y = ko_to_mc_y(castle_gtd_h)

        assert got_y == expected_y, (
            f"Castle terrain Y: expected {expected_y} (ko_h={castle_gtd_h}), "
            f"got {got_y}"
        )


class TestLandmarkAssetClass:
    """Verify that object classes (buildings, ships, gates) behave consistently."""

    @pytest.fixture
    def h_grid(self):
        return _make_flat_h_grid(n=257, height=5.0)

    def test_distinct_landmarks_have_distinct_mc_positions(self, h_grid):
        """All landmarks must map to distinct MC positions."""
        mc_positions = set()
        for lm in LANDMARKS:
            mc_x, mc_z = ko_to_mc_position(lm.opd_x, lm.opd_z)
            mc_positions.add((mc_x, mc_z))
        assert len(mc_positions) == len(LANDMARKS), (
            "Two landmarks mapped to the same MC position — collision in transform"
        )

    def test_all_landmarks_in_mc_world_bounds(self, h_grid):
        """All landmarks must map to positive MC block coordinates."""
        for lm in LANDMARKS:
            mc_x, mc_z = ko_to_mc_position(lm.opd_x, lm.opd_z)
            assert mc_x >= 0, f"{lm.name}: mc_x={mc_x} < 0"
            assert mc_z >= 0, f"{lm.name}: mc_z={mc_z} < 0"
            assert mc_x <= KO_MAP_SIZE, f"{lm.name}: mc_x={mc_x} > MAP_SIZE"
            assert mc_z <= KO_MAP_SIZE, f"{lm.name}: mc_z={mc_z} > MAP_SIZE"

    def test_yaw_sign_convention(self, h_grid):
        """KO yaw=+π/2 must produce mc_yaw=-π/2 (X mirror negates yaw)."""
        mc_yaw = ko_to_mc_yaw(math.pi / 2)
        assert abs(mc_yaw - (-math.pi / 2)) < 1e-9


# ── Standalone report (not a pytest test) ────────────────────────────────────

def print_all_landmark_reports(h_grid: Optional[np.ndarray] = None) -> None:
    """Print full debug reports for all known landmarks.

    Call from a script or REPL to inspect transform results.
    """
    if h_grid is None:
        h_grid = _make_flat_h_grid(n=257, height=5.0)

    print("\n" + "=" * 60)
    print("  LANDMARK DEBUG REPORT")
    print("  Canonical KO→MC Transform Verification")
    print("=" * 60)

    all_ok = True
    for lm in LANDMARKS:
        report = build_landmark_report(lm, h_grid)
        report.print_report()
        ok = (report.roundtrip_dx < 1.5 and
              report.roundtrip_dz < 1.5 and
              report.roundtrip_dyaw < 1e-4)
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    print(f"  All landmarks OK: {'YES ✓' if all_ok else 'NO — CHECK ABOVE ✗'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_all_landmark_reports()
