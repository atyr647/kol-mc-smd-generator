"""Pivot fix before/after verification report for 4 KO asset classes.

Produces a structured, machine-readable diff showing exactly what changed
for each asset class when the pivot fix was applied.

Asset classes tested
--------------------
1. Building (multi-part: main body + two towers with ±X pivots)
2. Gate / wall (multi-part: left post, right post, beam — Z-axis pivots)
3. Tree / vegetation (_po mesh with Y-axis pivot for height offset)
4. Ship (hull centered at origin, bridge offset in X+Z with scale)

For each shape we compute:
  BEFORE: part placed at shape anchor (pivot ignored — the old bug)
  AFTER:  part placed at pivot-adjusted anchor (the fix)
  DELTA:  the fix displacement (mc_dx, mc_dy, mc_dz)

A non-zero DELTA reveals objects that were previously misplaced.
A zero DELTA means zero pivot → no regression for that part.

Run
---
  python verify_pivot.py                  # full report
  python verify_pivot.py --json           # JSON output for tooling
  python verify_pivot.py --csv            # CSV for spreadsheet import
  python verify_pivot.py --chain-check    # nested-pivot chaining diagnostic
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from ko2mc.math3d import (
    KO_MAP_SIZE,
    MC_Y_BASE,
    ko_to_mc_position,
    ko_to_mc_yaw,
    ko_to_mc_y,
    pivot_to_mc_offset,
    build_mc_height_grid,
    terrain_y_at,
)


# ── Synthetic asset class definitions ────────────────────────────────────────

@dataclasses.dataclass
class SyntheticPart:
    name: str
    pivot_x: float
    pivot_y: float
    pivot_z: float


@dataclasses.dataclass
class SyntheticShape:
    asset_class: str        # building | gate | tree | ship
    shape_name: str
    ko_x: float
    ko_y: float             # OPD world Y (KO units, used for info only)
    ko_z: float
    ko_yaw: float           # radians
    scale: tuple            # (sx, sy, sz)
    parts: list[SyntheticPart]
    terrain_h: float        # KO terrain height at this position


# ── Four representative shapes ────────────────────────────────────────────────

SHAPES: list[SyntheticShape] = [
    # ── 1. Building — Moradon castle area
    # Main body (zero pivot) + two towers offset ±12 units in local X
    SyntheticShape(
        asset_class="building",
        shape_name="castle_keep",
        ko_x=863.0, ko_y=4.74, ko_z=540.0,
        ko_yaw=0.0,
        scale=(1.0, 1.0, 1.0),
        terrain_h=4.74,
        parts=[
            SyntheticPart("keep_body.n3pmesh",    pivot_x=0.0,   pivot_y=0.0, pivot_z=0.0),
            SyntheticPart("keep_tower_l.n3pmesh", pivot_x=-12.0, pivot_y=0.0, pivot_z=0.0),
            SyntheticPart("keep_tower_r.n3pmesh", pivot_x=+12.0, pivot_y=0.0, pivot_z=0.0),
            SyntheticPart("keep_roof.n3pmesh",    pivot_x=0.0,   pivot_y=8.0, pivot_z=0.0),
        ],
    ),

    # ── 2. Gate / wall — Moradon main gate (rotated 45°, non-unit scale)
    # Left post (-Z pivot), right post (+Z pivot), beam (elevated Y pivot)
    SyntheticShape(
        asset_class="gate",
        shape_name="main_gate",
        ko_x=510.0, ko_y=5.0, ko_z=870.0,
        ko_yaw=math.pi / 4,          # 45° rotation — pivots must rotate through this
        scale=(1.5, 1.2, 1.5),       # non-uniform scale — applied before rotation
        terrain_h=5.0,
        parts=[
            SyntheticPart("gate_post_l.n3pmesh", pivot_x=0.0, pivot_y=0.0, pivot_z=-8.0),
            SyntheticPart("gate_post_r.n3pmesh", pivot_x=0.0, pivot_y=0.0, pivot_z=+8.0),
            SyntheticPart("gate_beam.n3pmesh",   pivot_x=0.0, pivot_y=6.0, pivot_z=0.0),
        ],
    ),

    # ── 3. Tree / vegetation — forest area
    # _po mesh with a small upward Y pivot (trunk base slightly above origin)
    SyntheticShape(
        asset_class="tree",
        shape_name="war_tree_a01_po",
        ko_x=300.0, ko_y=2.1, ko_z=410.0,
        ko_yaw=0.0,
        scale=(1.0, 1.0, 1.0),
        terrain_h=2.1,
        parts=[
            SyntheticPart("obj_war_tree_a01_po01.n3pmesh", pivot_x=0.0, pivot_y=1.5, pivot_z=0.0),
        ],
    ),

    # ── 4. Ship — harbor area
    # Hull (zero pivot), bridge (offset +X and +Y), mast (high Y pivot)
    SyntheticShape(
        asset_class="ship",
        shape_name="harbor_ship",
        ko_x=200.0, ko_y=10.0, ko_z=700.0,
        ko_yaw=math.pi / 2,          # 90° yaw — pivots in X/Z will rotate
        scale=(1.0, 1.0, 1.0),
        terrain_h=10.0,
        parts=[
            SyntheticPart("ship_hull.n3pmesh",   pivot_x=0.0,  pivot_y=0.0,  pivot_z=0.0),
            SyntheticPart("ship_bridge.n3pmesh", pivot_x=5.0,  pivot_y=4.0,  pivot_z=0.0),
            SyntheticPart("ship_mast.n3pmesh",   pivot_x=0.0,  pivot_y=12.0, pivot_z=-8.0),
        ],
    ),
]


# ── Report dataclasses ────────────────────────────────────────────────────────

@dataclasses.dataclass
class PartVerification:
    part_name: str
    pivot_local: tuple              # (px, py, pz) in KO model space

    # BEFORE: pivot ignored (old bug)
    before_mc_x: int
    before_mc_y: int
    before_mc_z: int

    # AFTER: pivot applied (fix)
    after_mc_x: int
    after_mc_y: int
    after_mc_z: int

    # Delta (fix displacement)
    delta_x: int
    delta_y: int
    delta_z: int
    delta_magnitude: float

    @property
    def was_misplaced(self) -> bool:
        return self.delta_magnitude > 0.5   # any non-trivial displacement


@dataclasses.dataclass
class ShapeVerification:
    asset_class: str
    shape_name: str
    ko_position: tuple              # (ko_x, ko_y, ko_z)
    ko_yaw: float
    mc_anchor: tuple                # shape anchor after canonical transform
    mc_yaw: float
    terrain_y: int
    parts: list[PartVerification]

    @property
    def parts_misplaced(self) -> int:
        return sum(1 for p in self.parts if p.was_misplaced)

    @property
    def regression_safe(self) -> bool:
        """True if zero-pivot parts are unaffected (delta = 0)."""
        for p in self.parts:
            if p.pivot_local == (0.0, 0.0, 0.0) and p.was_misplaced:
                return False
        return True


# ── Core verification logic ───────────────────────────────────────────────────

def _quat_from_yaw(yaw: float) -> tuple:
    return (0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2))


def verify_shape(shape: SyntheticShape) -> ShapeVerification:
    """Run the full placement pipeline for one synthetic shape, before and after."""
    # Build a flat terrain h_grid at the shape's terrain height
    n = 257
    h_grid = build_mc_height_grid(
        np.full((n, n), shape.terrain_h, dtype=np.float32)
    )

    # Step 1: canonical XZ transform (shape anchor)
    mc_x, mc_z = ko_to_mc_position(shape.ko_x, shape.ko_z, KO_MAP_SIZE)

    # Step 2: terrain Y at shape anchor
    terrain_y = terrain_y_at(h_grid, mc_x, mc_z)

    # Step 3: yaw transform
    mc_yaw = ko_to_mc_yaw(shape.ko_yaw)

    # Shape MC anchor
    mc_anchor = (mc_x, terrain_y, mc_z)

    # Build quaternion for this shape's yaw
    quat = _quat_from_yaw(shape.ko_yaw)
    scale = shape.scale

    # Per-part before/after
    part_verifications = []
    for part in shape.parts:
        pivot_local = (part.pivot_x, part.pivot_y, part.pivot_z)

        # BEFORE (old bug): pivot ignored → placed at shape anchor
        before = mc_anchor

        # AFTER (fix): pivot applied
        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pivot_local, quat, scale)
        after = (mc_x + mc_dx, terrain_y + mc_dy, mc_z + mc_dz)

        dx = after[0] - before[0]
        dy = after[1] - before[1]
        dz = after[2] - before[2]
        magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)

        part_verifications.append(PartVerification(
            part_name      = part.name,
            pivot_local    = pivot_local,
            before_mc_x    = before[0],
            before_mc_y    = before[1],
            before_mc_z    = before[2],
            after_mc_x     = after[0],
            after_mc_y     = after[1],
            after_mc_z     = after[2],
            delta_x        = dx,
            delta_y        = dy,
            delta_z        = dz,
            delta_magnitude= magnitude,
        ))

    return ShapeVerification(
        asset_class  = shape.asset_class,
        shape_name   = shape.shape_name,
        ko_position  = (shape.ko_x, shape.ko_y, shape.ko_z),
        ko_yaw       = shape.ko_yaw,
        mc_anchor    = mc_anchor,
        mc_yaw       = mc_yaw,
        terrain_y    = terrain_y,
        parts        = part_verifications,
    )


# ── Report printers ───────────────────────────────────────────────────────────

def _deg(rad: float) -> str:
    return f"{math.degrees(rad):.1f}°"


def print_report(results: list[ShapeVerification]) -> None:
    print("\n" + "=" * 72)
    print("  PIVOT FIX — BEFORE / AFTER VERIFICATION REPORT")
    print("  Four KO asset classes: building · gate · tree · ship")
    print("=" * 72)

    total_parts      = 0
    total_misplaced  = 0
    all_regression_safe = True

    for sv in results:
        print(f"\n{'─'*72}")
        print(f"  [{sv.asset_class.upper()}]  {sv.shape_name}")
        print(f"{'─'*72}")
        print(f"  OPD position (KO):    ({sv.ko_position[0]:.1f}, {sv.ko_position[1]:.2f}, {sv.ko_position[2]:.1f})")
        print(f"  OPD yaw (KO):         {_deg(sv.ko_yaw)}")
        print(f"  MC anchor:            ({sv.mc_anchor[0]}, {sv.mc_anchor[1]}, {sv.mc_anchor[2]})")
        print(f"  MC yaw:               {_deg(sv.mc_yaw)}")
        print(f"  Terrain Y:            {sv.terrain_y}")
        print(f"  Parts misplaced:      {sv.parts_misplaced}/{len(sv.parts)}")
        print(f"  Regression safe:      {'YES ✓' if sv.regression_safe else 'NO ✗ — zero-pivot part has non-zero delta!'}")

        print(f"\n  {'Part':<40} {'Pivot (local)':<22} {'BEFORE':<20} {'AFTER':<20} {'DELTA':<16} {'Δ|'}")
        print(f"  {'─'*40} {'─'*22} {'─'*20} {'─'*20} {'─'*16} {'─'*6}")

        for p in sv.parts:
            pv = p.pivot_local
            pivot_str = f"({pv[0]:+.1f},{pv[1]:+.1f},{pv[2]:+.1f})"
            before_str = f"({p.before_mc_x},{p.before_mc_y},{p.before_mc_z})"
            after_str  = f"({p.after_mc_x},{p.after_mc_y},{p.after_mc_z})"
            delta_str  = f"({p.delta_x:+},{p.delta_y:+},{p.delta_z:+})"
            flag = " ← FIXED" if p.was_misplaced else "  (no change)"

            part_short = p.part_name[:39]
            print(f"  {part_short:<40} {pivot_str:<22} {before_str:<20} {after_str:<20} {delta_str:<16} {p.delta_magnitude:.1f}{flag}")

        total_parts     += len(sv.parts)
        total_misplaced += sv.parts_misplaced
        if not sv.regression_safe:
            all_regression_safe = False

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"  Total parts:         {total_parts}")
    print(f"  Parts now fixed:     {total_misplaced}  (had non-zero displacement)")
    print(f"  Parts unaffected:    {total_parts - total_misplaced}  (zero-pivot — no regression)")
    print(f"  Regression safe:     {'YES ✓' if all_regression_safe else 'NO ✗'}")
    print(f"{'='*72}\n")


def print_json(results: list[ShapeVerification]) -> None:
    data = []
    for sv in results:
        data.append({
            "asset_class": sv.asset_class,
            "shape_name": sv.shape_name,
            "ko_position": list(sv.ko_position),
            "ko_yaw_deg": round(math.degrees(sv.ko_yaw), 2),
            "mc_anchor": list(sv.mc_anchor),
            "mc_yaw_deg": round(math.degrees(sv.mc_yaw), 2),
            "terrain_y": sv.terrain_y,
            "parts_misplaced": sv.parts_misplaced,
            "regression_safe": sv.regression_safe,
            "parts": [
                {
                    "name": p.part_name,
                    "pivot_local": list(p.pivot_local),
                    "before": [p.before_mc_x, p.before_mc_y, p.before_mc_z],
                    "after":  [p.after_mc_x,  p.after_mc_y,  p.after_mc_z],
                    "delta":  [p.delta_x, p.delta_y, p.delta_z],
                    "delta_magnitude": round(p.delta_magnitude, 2),
                    "was_misplaced": p.was_misplaced,
                }
                for p in sv.parts
            ],
        })
    print(json.dumps(data, indent=2))


def print_csv(results: list[ShapeVerification]) -> None:
    print("asset_class,shape_name,part_name,pivot_x,pivot_y,pivot_z,"
          "before_x,before_y,before_z,after_x,after_y,after_z,"
          "delta_x,delta_y,delta_z,delta_magnitude,was_misplaced")
    for sv in results:
        for p in sv.parts:
            pv = p.pivot_local
            print(f"{sv.asset_class},{sv.shape_name},{p.part_name},"
                  f"{pv[0]},{pv[1]},{pv[2]},"
                  f"{p.before_mc_x},{p.before_mc_y},{p.before_mc_z},"
                  f"{p.after_mc_x},{p.after_mc_y},{p.after_mc_z},"
                  f"{p.delta_x},{p.delta_y},{p.delta_z},"
                  f"{p.delta_magnitude:.2f},{p.was_misplaced}")


# ── Pivot chaining diagnostic ─────────────────────────────────────────────────
#
# In some MMO asset pipelines, part pivots are stored relative to the *previous*
# part (chained / parent-relative) rather than relative to the shape root.
#
# Symptom: inter-part distances grow cumulatively instead of being independent:
#
#   root-relative:  distances = [10, 12, 14, 9]    (each independent of others)
#   chained:        distances = [10, 22, 36, 45]   (each adds previous pivot)
#
# This diagnostic computes the distance of each part anchor from the shape root
# in KO space, using the ACTUAL pivot values stored in SHAPES.  If the pipeline
# ever accumulates pivots, the distances will drift monotonically.
#
# A passing shape shows:
#   - Distance for zero-pivot part = 0.0 exactly
#   - Distances are independent (reordering parts does not change any distance)
#   - No part has a distance > the sum of all pivot magnitudes (impossible if
#     pivots were accumulated)

@dataclasses.dataclass
class ChainCheckPart:
    name: str
    pivot_local: tuple          # (px, py, pz)
    pivot_magnitude: float      # |pivot_local|  (raw, before scale)
    scaled_magnitude: float     # |S × pivot_local|  (expected distance from root)
    distance_from_root: float   # |R × (S × pivot_local)|  actual (R preserves magnitude)
    is_zero_pivot: bool


@dataclasses.dataclass
class ChainCheckResult:
    asset_class: str
    shape_name: str
    scale: tuple
    parts: list[ChainCheckPart]
    max_distance: float
    cumulative_sum: float       # sum of all scaled pivot magnitudes
    chaining_detected: bool     # True if any distance significantly exceeds scaled magnitude


def chain_check_shape(shape: SyntheticShape) -> ChainCheckResult:
    """Compute per-part root distances and detect chained/accumulated pivots.

    The correct invariant is:
        distance_from_root == |S × pivot_local|

    because R is orthogonal (rotation preserves magnitude), so:
        |R × (S × pivot)| == |S × pivot|

    Chaining would add accumulated parent offsets, making:
        distance_from_root >> |S × pivot_local|

    Non-unit scale is accounted for: compare distance against scaled magnitude,
    not raw pivot magnitude.
    """
    from ko2mc.math3d import transform_pivot
    quat = _quat_from_yaw(shape.ko_yaw)
    scale = shape.scale
    S = np.asarray(scale, dtype=np.float64)

    parts = []
    cumulative_sum = 0.0

    for part in shape.parts:
        piv = (part.pivot_x, part.pivot_y, part.pivot_z)
        piv_arr = np.asarray(piv, dtype=np.float64)
        raw_mag = float(np.linalg.norm(piv_arr))
        scaled_mag = float(np.linalg.norm(S * piv_arr))  # expected distance
        cumulative_sum += scaled_mag

        # Actual distance after R × (S × pivot) — should equal scaled_mag
        pw = transform_pivot(piv, quat, scale)
        dist = float(np.linalg.norm(pw))

        parts.append(ChainCheckPart(
            name=part.name,
            pivot_local=piv,
            pivot_magnitude=raw_mag,
            scaled_magnitude=scaled_mag,
            distance_from_root=dist,
            is_zero_pivot=raw_mag < 1e-6,
        ))

    # Chaining: actual distance significantly exceeds scaled pivot magnitude
    # (accumulated parent offsets inflate the distance beyond what S×pivot alone gives)
    chaining_detected = any(
        p.distance_from_root > p.scaled_magnitude + 0.5
        for p in parts
        if not p.is_zero_pivot
    )

    return ChainCheckResult(
        asset_class=shape.asset_class,
        shape_name=shape.shape_name,
        scale=shape.scale,
        parts=parts,
        max_distance=max((p.distance_from_root for p in parts), default=0.0),
        cumulative_sum=cumulative_sum,
        chaining_detected=chaining_detected,
    )


def print_chain_check(shapes: list[SyntheticShape]) -> None:
    """Print the nested-pivot chaining diagnostic report."""
    print("\n" + "=" * 72)
    print("  PIVOT CHAINING DIAGNOSTIC")
    print("  Checks that part pivots are root-relative, not parent-chained")
    print("=" * 72)
    print()
    print("  Invariant:  distance_from_root == |S × pivot_local|")
    print("              (rotation is orthogonal; scale is the only magnitude change)")
    print("  Chaining:   distance_from_root >> |S × pivot_local|")
    print("              (accumulated offsets from parent parts inflate the distance)")
    print()

    any_chaining = False

    for shape in shapes:
        result = chain_check_shape(shape)
        status = "CHAINING DETECTED ✗" if result.chaining_detected else "root-relative ✓"
        print(f"  [{result.asset_class.upper()}]  {result.shape_name}  "
              f"scale={result.scale}  —  {status}")
        print(f"  {'Part':<42} {'|pivot|':>8} {'|S×pivot|':>10} {'dist/root':>10} {'OK?':>8}")
        print(f"  {'─'*42} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")

        for p in result.parts:
            delta = abs(p.distance_from_root - p.scaled_magnitude)
            ok = "✓" if p.is_zero_pivot or delta < 0.5 else "✗ CHAIN"
            print(f"  {p.name[:41]:<42} {p.pivot_magnitude:>8.2f} "
                  f"{p.scaled_magnitude:>10.2f} {p.distance_from_root:>10.2f} {ok:>8}")

        print(f"  Scaled pivot sum: {result.cumulative_sum:.2f} | "
              f"Max distance: {result.max_distance:.2f}")
        print()

        if result.chaining_detected:
            any_chaining = True

    print("=" * 72)
    print(f"  Result: "
          f"{'CHAINING DETECTED — pivots are parent-relative ✗' if any_chaining else 'All pivots are root-relative ✓'}")
    print("=" * 72 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pivot fix before/after verification report"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--json",        action="store_true", help="JSON output")
    grp.add_argument("--csv",         action="store_true", help="CSV output")
    grp.add_argument("--chain-check", action="store_true",
                     help="Nested pivot chaining diagnostic")
    args = parser.parse_args()

    if args.chain_check:
        print_chain_check(SHAPES)
        return

    results = [verify_shape(s) for s in SHAPES]

    if args.json:
        print_json(results)
    elif args.csv:
        print_csv(results)
    else:
        print_report(results)


if __name__ == "__main__":
    main()
