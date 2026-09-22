"""OPD-driven zone-to-Minecraft voxel converter.

Reads a Knight Online .opd file, resolves each shape's mesh parts via
AssetResolver, applies OPD transforms, voxelizes, and places the results
into a Minecraft world.

Anchor semantics
----------------
Each OPD shape has a position (world-space KO units), rotation quaternion,
and scale vector.  The mesh's local origin (0, 0, 0) maps to the OPD world
position after KO→MC coordinate conversion.

The voxelizer sets grid.origin = bbox_min − padding*voxel_size (model-local
space, 1 KO unit = 1 MC block).  Placement offsets use math.floor() — NOT
int() — to correctly handle negative origins (common: meshes centered at local
origin extend into negative X/Z):

    int(-1.63) = -1  ← WRONG: places voxels 1 block too high/right
    floor(-1.63) = -2  ← CORRECT

    mc_anchor_x, mc_anchor_y, mc_anchor_z = ko_pos_to_mc(shape.position)
    offset_x = mc_anchor_x + floor(grid.origin[0])
    offset_y = mc_anchor_y + floor(grid.origin[1])
    offset_z = mc_anchor_z + floor(grid.origin[2])

Height scale (KO_HEIGHT_SCALE from math3d, currently 1.0) is applied only to
the world-position Y (shape.position.y), NOT to model-local geometry.

Placement statuses
------------------
  placed          — all parts resolved, voxelized, and placed
  partial         — at least one part placed; others failed
  missing_mesh    — zero parts resolved by AssetResolver
  parse_failed    — mesh path found but BinaryParseError on parse
  voxelize_failed — mesh parsed but voxelizer raised an exception
  skipped         — filtered out by --limit / --name-filter / --radius

Debug markers
-------------
When --debug-markers is enabled, a single marker block is placed at the
MC anchor for every shape that could not be placed:
  orange_wool     — missing_mesh
  red_wool        — parse_failed
  magenta_wool    — voxelize_failed

Usage
-----
    # Full zone, default voxel_size=1.0
    python -m ko2mc.zone_converter moradon.opd \\
        --asset-roots /path/to/1886-Client /path/to/ko-assets-1298 \\
        -o /tmp/moradon_world

    # First 200 shapes, debug markers on
    python -m ko2mc.zone_converter moradon.opd --limit 200 --debug-markers \\
        --asset-roots ... -o /tmp/moradon_world

    # Shapes whose names contain "tree"
    python -m ko2mc.zone_converter moradon.opd --name-filter tree \\
        --asset-roots ... -o /tmp/moradon_world

    # Shapes within 100 KO units of (860, 540)
    python -m ko2mc.zone_converter moradon.opd --radius 100 --center 860 540 \\
        --asset-roots ... -o /tmp/moradon_world
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .asset_resolver import AssetResolver
from .binary_reader import BinaryParseError
from .math3d import (
    MC_Y_BASE,
    KO_HEIGHT_SCALE,
    apply_transform,
    ko_pos_to_mc,
    ko_to_mc_position,
    ko_to_mc_y,
    ko_to_mc_yaw,
    build_mc_height_grid,
    terrain_y_at as _terrain_y_at_canonical,
    terrain_y_at_footprint as _terrain_y_at_footprint,
    quat_yaw,
    pivot_to_mc_offset,
    transform_pivot,
)
from .mc_world import MinecraftWorld
from .n3pmesh_parser import parse_n3pmesh
from .opd_parser import Shape, parse_opd
from .voxelizer import place_voxel_grid, voxelize_mesh

logger = logging.getLogger(__name__)


# ── Billboard impostor detection ──────────────────────────────────────────────
# KO uses two suffixes for tree/bush meshes:
#   _ip<N>  — 2D impostor/billboard sprite (flat quad, useless to voxelize)
#   _po<N>  — 3D polygon mesh (voxelizable, handled normally)
# Source: KO engine naming convention; confirmed from OPD placement report.
_BILLBOARD_RE = re.compile(r"_ip\d*", re.IGNORECASE)


def _is_billboard_impostor(part_name: str) -> bool:
    """Return True if this mesh part is a 2D billboard impostor (not a 3D mesh).

    KO part names are full relative paths, e.g.:
        object\\obj_war_tree_a01_ip01.n3pmesh
    The _ip<N> segment appears before the extension, so we search anywhere in
    the string rather than anchoring to end-of-string.
    """
    return bool(_BILLBOARD_RE.search(part_name))


def _shape_is_tree(shape) -> bool:
    """Return True if this OPD shape is a tree/bush (has any billboard impostor part).

    KO tree shapes have _ip<N> billboard parts alongside a _po 3D polygon part.
    We detect at the shape level so we place exactly ONE oak tree per shape and
    skip voxelizing both the billboard and polygon parts.
    """
    return any(_is_billboard_impostor(part.name) for part in shape.parts)


# ── Oak tree voxel template ───────────────────────────────────────────────────
# (dx, dy, dz, block_name) relative to the tree base (mc_pos with origin offset).
# Mirrors a natural Minecraft oak tree: 5-block trunk + canopy.
# Leaves use persistent=true so they don't decay without adjacent logs.
_OAK_LEAVES = "minecraft:oak_leaves[persistent=true,distance=1,waterlogged=false]"
_OAK_LOG    = "minecraft:oak_log"

def _build_oak_tree_template() -> list[tuple[int, int, int, str]]:
    blocks = []
    # Trunk: 5 blocks (y+0 to y+4)
    for y in range(5):
        blocks.append((0, y, 0, _OAK_LOG))
    # Canopy y+3 and y+4: 3x3 ring (skip trunk column)
    for y in (3, 4):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue  # trunk
                blocks.append((dx, y, dz, _OAK_LEAVES))
    # Canopy y+5: full 3x3 (no trunk here)
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            blocks.append((dx, 5, dz, _OAK_LEAVES))
    # Crown y+6: cardinal cross
    for dx, dz in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        blocks.append((dx, 6, dz, _OAK_LEAVES))
    return blocks

_OAK_TREE_TEMPLATE: list[tuple[int, int, int, str]] = _build_oak_tree_template()


# ── Terrain height grid (shared between terrain and building placement) ────────

def _compute_h_grid(gtd) -> np.ndarray:
    """Return the canonical MC-oriented height grid from GTD data.

    Delegates to math3d.build_mc_height_grid which is the single authority for
    this transform.  Both build_terrain() and _terrain_y_at() use this result,
    ensuring one shared height system with no scaling mismatch.

    h_mc[mc_tx, mc_tz] == gtd.heights[ko_tx, ko_tz]
    where mc_tx = (n-1) - ko_tx  (consistent with ko_to_mc_position X mirror).
    """
    return build_mc_height_grid(gtd.heights)


# ── Vegetation detection and block mapping ────────────────────────────────────
# Returns None  → use normal voxelization with shape-appropriate block
# Returns 'skip' → do not place anything for this shape
# Returns str   → voxelize but use this block name (overrides stone_bricks)

_VEGE_SKIP = re.compile(
    r"(_grass|_gras|_flower|zz_fx|_fx_plus|_mist|_smoke|_steam|_spark)",
    re.IGNORECASE,
)
_VEGE_LEAVES = re.compile(
    r"(bush|shrub|wabum|dumbul|gass_puls|_co_reed|_mora_plant|_co_plant)",
    re.IGNORECASE,
)
_VEGE_LIGHT = re.compile(
    r"(_light|_lamp|_torch|_lantern)",
    re.IGNORECASE,
)

_TEX_WOOD = re.compile(r"(wood|plank|timber|log|tree|bark)", re.IGNORECASE)
_TEX_STONE = re.compile(r"(stone|rock|wall|brick|castle|ruin)", re.IGNORECASE)
_TEX_GLASS = re.compile(r"(glass|window)", re.IGNORECASE)
_TEX_METAL = re.compile(r"(metal|iron|steel|gate|chain)", re.IGNORECASE)
_TEX_SAND = re.compile(r"(sand|desert|dune)", re.IGNORECASE)
_TEX_DIRT = re.compile(r"(dirt|mud|soil|grass)", re.IGNORECASE)


def _vegetation_action(shape_name: str) -> Optional[str]:
    """Return placement action for vegetation-type shapes, or None for normal."""
    if _VEGE_SKIP.search(shape_name):
        return "skip"
    if _VEGE_LEAVES.search(shape_name):
        return _OAK_LEAVES
    if _VEGE_LIGHT.search(shape_name):
        return "minecraft:sea_lantern"
    return None


# ── Smart block name for voxelized structures ──────────────────────────────────

def _block_name_for_shape(shape_name: str) -> str:
    """Return an appropriate MC block for the given shape, based on its name."""
    name = shape_name.lower()
    if any(k in name for k in ("wood", "plank", "board", "log", "timber")):
        return "minecraft:oak_planks"
    if any(k in name for k in ("light", "lamp", "torch", "lantern", "glow")):
        return "minecraft:sea_lantern"
    if any(k in name for k in ("water", "pond", "lake", "sea", "river")):
        return "minecraft:water"
    # Default: stone_bricks fits KO's stone architecture
    return "minecraft:stone_bricks"


# Palette selection sources recorded in the audit report.
PALETTE_TEXTURE          = "texture"                   # a texture name matched a rule
PALETTE_FALLBACK_NO_TEX  = "fallback_no_texture"       # part has no textures → shape-name rule
PALETTE_FALLBACK_UNKNOWN = "fallback_unknown_texture"  # textures present, none matched → shape-name rule
PALETTE_FALLBACK_SOURCES = (PALETTE_FALLBACK_NO_TEX, PALETTE_FALLBACK_UNKNOWN)

_TEXTURE_RULES = (
    (_TEX_GLASS, "minecraft:glass"),
    (_TEX_METAL, "minecraft:iron_block"),
    (_TEX_WOOD, "minecraft:oak_planks"),
    (_TEX_SAND, "minecraft:sandstone"),
    (_TEX_DIRT, "minecraft:dirt"),
    (_TEX_STONE, "minecraft:stone_bricks"),
)


def _block_for_part_with_source(shape_name: str, texture_names: list[str]) -> tuple[str, str]:
    """Return (block, palette_source) for a part.

    palette_source is PALETTE_TEXTURE when a texture rule matched, otherwise
    one of PALETTE_FALLBACK_SOURCES (the shape-name heuristic was used).
    """
    if not texture_names:
        return _block_name_for_shape(shape_name), PALETTE_FALLBACK_NO_TEX

    joined = " ".join(texture_names).lower()
    for pattern, block in _TEXTURE_RULES:
        if pattern.search(joined):
            return block, PALETTE_TEXTURE
    return _block_name_for_shape(shape_name), PALETTE_FALLBACK_UNKNOWN


def _block_name_for_part(shape_name: str, texture_names: list[str]) -> str:
    """Texture-driven block selection with shape-name fallback.

    Prioritizes OPD part texture names when available, then falls back to
    shape-level heuristics.  This improves material variety for multipart
    structures whose part names are generic but texture names are descriptive.
    """
    return _block_for_part_with_source(shape_name, texture_names)[0]


# ── Voxelization modes ────────────────────────────────────────────────────────
# Map the --voxel-mode policy onto voxelize_mesh() capabilities:
#   surface — SAT surface hits, then strip fully enclosed voxels (hollow shell)
#   hybrid  — raw SAT occupancy: thick/overlapping geometry is kept as-is,
#             enclosed air is NOT filled
#   solid   — SAT surface + flood-fill of enclosed interior air (old --fill)
VOXEL_MODES = ("surface", "hybrid", "solid")


def voxel_mode_options(mode: str) -> dict:
    """Return voxelize_mesh() keyword arguments for a voxel mode."""
    if mode == "surface":
        return {"fill": False, "surface_only": True}
    if mode == "hybrid":
        return {"fill": False, "surface_only": False}
    if mode == "solid":
        return {"fill": True, "surface_only": False}
    raise ValueError(f"unknown voxel mode {mode!r}; expected one of {VOXEL_MODES}")


# ── Grounding quality metrics ─────────────────────────────────────────────────
# Terrain is sampled on a grid across the shape footprint and compared with the
# chosen base Y.  A sample within ±_GROUND_TOLERANCE blocks counts as support;
# terrain higher than that buries the base, lower leaves it floating.
_GROUND_TOLERANCE = 1
_GROUND_SAMPLE_FRACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def grounding_quality(
    h_grid: Optional[np.ndarray],
    mc_x: float,
    mc_z: float,
    base_y: int,
    half_x: float,
    half_z: float,
) -> Optional[dict]:
    """Return support/buried/floating ratios for a footprint, or None without terrain."""
    if h_grid is None:
        return None
    support = buried = floating = 0
    for fx in _GROUND_SAMPLE_FRACTIONS:
        for fz in _GROUND_SAMPLE_FRACTIONS:
            ty = _terrain_y_at_canonical(h_grid, mc_x + fx * half_x, mc_z + fz * half_z)
            d = ty - base_y
            if d > _GROUND_TOLERANCE:
                buried += 1
            elif d < -_GROUND_TOLERANCE:
                floating += 1
            else:
                support += 1
    n = support + buried + floating
    return {
        "support_ratio": round(support / n, 4),
        "buried_ratio": round(buried / n, 4),
        "floating_ratio": round(floating / n, 4),
        "samples": n,
    }


# ── Placement-policy classifier ───────────────────────────────────────────────
# Shapes fall into two classes:
#
#   TERRAIN_GROUNDED — base snapped to terrain surface via _ground_shape().
#     Buildings, walls, gates, ruins, platforms, rocks, terrain props.
#
#   WORLD_Y — placed at their authored OPD Y coordinate via ko_to_mc_y().
#     Boats, chains, suspended bridges, hanging decorations, floating props.
#     Their Y was set by level designers in world space, not terrain-relative.
#
# Primary signal: name-pattern match.
# Secondary signal: |OPD_Y_mc − terrain_Y| delta.  A large delta (> 8 blocks)
# on an unrecognised shape suggests intentional above/below-ground placement.

_FREEPOS_RE = re.compile(
    r"(bat_ship|_ship|_boat|nangan|_chain|_rope|_bridge|_hang|_suspend"
    r"|_wagun|_wagon|bat_El_ship|obj_el_ship|obj_bat_e_ship|obj_bat_ship"
    r"|_flag_b|_banner|_pollen|_balloon|water_obj|_fish|_barrel_f)",
    re.IGNORECASE,
)
"""Shapes whose authored world Y must be preserved (not terrain-grounded)."""

# The delta between OPD_Y and terrain_Y is computed and logged for diagnostics
# but NO LONGER used as a grounding decision branch (removed after Track A audit
# proved it caused incorrect world_y selection for lampposts at delta=13 and
# trees on hills at delta=12-19).  Grounding is now determined solely by
# explicit name-pattern classification (_FREEPOS_RE).
_FREEPOS_DELTA_LOGGED: bool = True   # keep for diagnostic visibility

# Minimum scale factor for a shape to be treated as a large structure
# (uses footprint-aware terrain sampling instead of centre-point).
_LARGE_STRUCTURE_SCALE: float = 2.0


def _grounding_mode(
    shape_name: str,
    ko_opd_y: float,
    terrain_mc_y: int,
) -> tuple:
    """Return (mode, reason) for this shape.

    mode   : 'terrain' → snap base to terrain surface (buildings, walls, etc.)
             'world_y' → use OPD Y directly (boats, chains, hanging props, etc.)
    reason : human-readable string explaining which rule fired (for diagnostics).

    Grounding is determined solely by name-pattern classification (_FREEPOS_RE).
    The OPD_Y vs terrain_Y delta is computed for diagnostic visibility but is
    NOT used as a decision branch.  The delta heuristic was removed after
    Track A proved it caused incorrect world_y selection for terrain-placed
    assets (lampposts delta=13, hillside trees delta=12-19) whose large delta
    was due to terrain variation, not genuine free-positioning.
    """
    opd_mc_y = ko_to_mc_y(ko_opd_y)
    delta = abs(opd_mc_y - terrain_mc_y)

    if _FREEPOS_RE.search(shape_name):
        return "world_y", f"name_pattern (delta={delta})"

    return "terrain", f"terrain_default (delta={delta})"


# ── X-flipped voxel placement ─────────────────────────────────────────────────

def _place_voxel_grid_x_flipped(
    grid,
    world,
    off_x: int,
    off_y: int,
    off_z: int,
    block_name: str = "minecraft:stone_bricks",
) -> int:
    """Write all filled voxels from grid into a MinecraftWorld with X axis mirrored.

    Because the canonical KO→MC world transform mirrors the X axis
    (mc_x = map_size - ko_x), local model X offsets must be SUBTRACTED from
    the MC anchor rather than added.  Y and Z are unaffected.

        wx = off_x - ix   (X mirrored)
        wy = off_y + iy
        wz = off_z + iz

    off_x must already be computed as  mc_anchor_x - int(grid.origin[0]).

    Returns: number of blocks placed.
    """
    import numpy as np
    filled_indices = np.argwhere(grid.data > 0)  # (K, 3) — z, y, x
    placed = 0
    for iz, iy, ix in filled_indices:
        wx = off_x - int(ix)   # X mirror
        wy = off_y + int(iy)
        wz = off_z + int(iz)
        world.set_block(wx, wy, wz, block_name)
        placed += 1
    return placed


# ── Status types ──────────────────────────────────────────────────────────────

class ShapeStatus(str, Enum):
    PLACED          = "placed"
    PARTIAL         = "partial"
    MISSING_MESH    = "missing_mesh"
    PARSE_FAILED    = "parse_failed"
    VOXELIZE_FAILED = "voxelize_failed"
    SKIPPED         = "skipped"


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class PartResult:
    part_name: str
    status: str          # placed | missing | parse_failed | voxelize_failed
    blocks_placed: int = 0
    error: Optional[str] = None
    resolver_strategy: Optional[str] = None
    resolver_candidate_count: int = 0
    resolver_ambiguous: bool = False
    palette_source: Optional[str] = None
    textures: list = field(default_factory=list)


@dataclass
class ShapeResult:
    shape_name: str
    shape_index: int
    status: ShapeStatus
    ko_position: tuple   # (x, y, z) KO world units
    mc_position: tuple   # (x, y, z) MC block coords
    part_results: list   # list[PartResult]
    blocks_placed: int = 0
    error: Optional[str] = None
    grounding_mode: Optional[str] = None
    grounding_reason: Optional[str] = None
    terrain_mc_y: Optional[int] = None
    opd_mc_y: Optional[int] = None
    grounding_delta: Optional[int] = None
    grounding_quality: Optional[dict] = None  # support/buried/floating ratios
    partial_reason_counts: dict = field(default_factory=dict)


@dataclass
class PlacementReport:
    opd_path: str
    world_path: str
    total_shapes: int
    processed: int
    counts: dict             # ShapeStatus.value → int
    shapes: list             # list[ShapeResult]
    missing_refs: list       # deduplicated refs AssetResolver couldn't find
    audit_report_level: str = "standard"
    voxel_mode: str = "surface"
    unknown_textures: dict = field(default_factory=dict)  # texture name → part count
    terrain_palette: Optional[dict] = None                # from build_terrain(stats=...)

    # Unknown-texture list length in "standard" reports ("detailed" lists all).
    STANDARD_UNKNOWN_TEXTURE_LIMIT = 25

    def palette_summary(self) -> dict:
        sources: dict[str, int] = {}
        for r in self.shapes:
            for p in r.part_results:
                if p.palette_source:
                    sources[p.palette_source] = sources.get(p.palette_source, 0) + 1
        total = sum(sources.values())
        fallback = sum(v for k, v in sources.items() if k in PALETTE_FALLBACK_SOURCES)
        ranked = sorted(self.unknown_textures.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.audit_report_level != "detailed":
            ranked = ranked[: self.STANDARD_UNKNOWN_TEXTURE_LIMIT]
        return {
            "sources": sources,
            "parts_evaluated": total,
            "fallback_rate": round(fallback / total, 4) if total else None,
            "unknown_texture_count": len(self.unknown_textures),
            "unknown_textures": [{"texture": t, "parts": n} for t, n in ranked],
        }

    def grounding_summary(self) -> dict:
        qs = [r.grounding_quality for r in self.shapes if r.grounding_quality]
        if not qs:
            return {"shapes_evaluated": 0}

        def _mean(key: str) -> float:
            return round(sum(q[key] for q in qs) / len(qs), 4)

        return {
            "shapes_evaluated": len(qs),
            "mean_support_ratio": _mean("support_ratio"),
            "mean_buried_ratio": _mean("buried_ratio"),
            "mean_floating_ratio": _mean("floating_ratio"),
            "mostly_floating_shapes": sum(1 for q in qs if q["floating_ratio"] > 0.5),
            "mostly_buried_shapes": sum(1 for q in qs if q["buried_ratio"] > 0.5),
        }

    def to_dict(self) -> dict:
        detailed = self.audit_report_level == "detailed"
        return {
            "opd_path": self.opd_path,
            "world_path": self.world_path,
            "total_shapes": self.total_shapes,
            "processed": self.processed,
            "counts": self.counts,
            "missing_refs": self.missing_refs,
            "shapes": [
                {
                    "name": r.shape_name,
                    "index": r.shape_index,
                    "status": r.status.value,
                    "ko_position": list(r.ko_position),
                    "mc_position": list(r.mc_position),
                    "blocks_placed": r.blocks_placed,
                    "grounding": {
                        "mode": r.grounding_mode,
                        "reason": r.grounding_reason,
                        "terrain_mc_y": r.terrain_mc_y,
                        "opd_mc_y": r.opd_mc_y,
                        "delta": r.grounding_delta,
                        **({k: v for k, v in r.grounding_quality.items()
                            if detailed or k != "samples"}
                           if r.grounding_quality else {}),
                    },
                    "partial_reason_counts": r.partial_reason_counts,
                    "parts": [
                        {
                            "name": p.part_name,
                            "status": p.status,
                            "blocks_placed": p.blocks_placed,
                            "error": p.error,
                            "resolver": {
                                "strategy": p.resolver_strategy,
                                "candidate_count": p.resolver_candidate_count,
                                "ambiguous": p.resolver_ambiguous,
                            },
                            "palette_source": p.palette_source,
                            **({"textures": list(p.textures)} if detailed else {}),
                        }
                        for p in r.part_results
                    ],
                    "error": r.error,
                }
                for r in self.shapes
            ],
            "audit_report_level": self.audit_report_level,
            "voxel_mode": self.voxel_mode,
            "palette": self.palette_summary(),
            "terrain_palette": self.terrain_palette,
            "grounding_summary": self.grounding_summary(),
        }

    def print_summary(self) -> None:
        print(f"\n=== Placement Report ===")
        print(f"  OPD:        {self.opd_path}")
        print(f"  World:      {self.world_path}")
        print(f"  Total shapes in OPD: {self.total_shapes}")
        print(f"  Processed:  {self.processed}")
        for status, count in sorted(self.counts.items()):
            print(f"    {status:<20} {count}")
        total_blocks = sum(r.blocks_placed for r in self.shapes)
        print(f"  Total blocks placed: {total_blocks:,}")
        print(f"  Unresolved mesh refs: {len(self.missing_refs)}")
        pal = self.palette_summary()
        if pal["parts_evaluated"]:
            print(f"  Palette fallback rate: {pal['fallback_rate']:.1%} "
                  f"of {pal['parts_evaluated']} parts; "
                  f"{pal['unknown_texture_count']} unknown texture(s)")
        if self.terrain_palette:
            print(f"  Terrain texture fallback rate: "
                  f"{self.terrain_palette['fallback_rate']:.1%} "
                  f"({len(self.terrain_palette['unmapped_tex_ids'])} unmapped id(s))")
        g = self.grounding_summary()
        if g["shapes_evaluated"]:
            print(f"  Grounding: support={g['mean_support_ratio']:.2f} "
                  f"buried={g['mean_buried_ratio']:.2f} "
                  f"floating={g['mean_floating_ratio']:.2f} "
                  f"(mostly floating: {g['mostly_floating_shapes']}, "
                  f"mostly buried: {g['mostly_buried_shapes']})")


# ── ZoneConverter ─────────────────────────────────────────────────────────────

_MARKER_MISSING        = "minecraft:orange_wool"
_MARKER_PARSE_FAILED   = "minecraft:red_wool"
_MARKER_VOXEL_FAILED   = "minecraft:magenta_wool"

# Transform-chain visualization markers (--debug-transform)
_MARKER_SHAPE_ANCHOR   = "minecraft:red_wool"       # shape OPD anchor (canonical)
_MARKER_PART_ANCHOR    = "minecraft:lime_wool"       # per-part anchor after pivot


class ZoneConverter:
    """Converts OPD shapes to Minecraft blocks using mesh voxelization."""

    def __init__(
        self,
        resolver: AssetResolver,
        world: MinecraftWorld,
        voxel_size: float = 1.0,
        block_name: str = "minecraft:stone_bricks",
        fill: bool = False,
        debug_markers: bool = False,
        debug_transform: bool = False,
        debug_grounding: bool = False,
        debug_voxelize: bool = False,
        map_size: int = 1024,
        h_grid: Optional[np.ndarray] = None,
        audit_report_level: str = "standard",
        voxel_mode: Optional[str] = None,
    ) -> None:
        # voxel_mode supersedes the legacy `fill` flag (fill=True ≡ "solid").
        if voxel_mode is None:
            voxel_mode = "solid" if fill else "surface"
        self._voxel_kwargs = voxel_mode_options(voxel_mode)
        self.voxel_mode = voxel_mode
        self.resolver = resolver
        self.world = world
        self.voxel_size = voxel_size
        self.block_name = block_name
        self.fill = self._voxel_kwargs["fill"]
        self.debug_markers = debug_markers
        self.debug_transform = debug_transform  # print pivot/anchor trace per shape
        self.debug_grounding = debug_grounding  # print grounding decision per shape
        self.debug_voxelize = debug_voxelize    # Track B: voxelization consistency audit
        self.map_size = map_size
        self.h_grid = h_grid  # canonical MC-oriented height grid from GTD
        self.audit_report_level = audit_report_level
        self._footprint_cache: dict[tuple, tuple[float, float]] = {}
        # Texture names that matched no palette rule → number of parts using them.
        self.unknown_textures: Counter = Counter()

    def _terrain_y_at(self, mc_x: int, mc_z: int) -> int:
        """Return MC Y of terrain surface at (mc_x, mc_z) via bilinear interp.

        Delegates to math3d.terrain_y_at() using the canonical h_grid built by
        _compute_h_grid().  Falls back to MC_Y_BASE when no terrain is loaded.
        """
        if self.h_grid is None:
            return MC_Y_BASE
        return _terrain_y_at_canonical(self.h_grid, mc_x, mc_z)

    def _ground_shape(
        self,
        mc_x: int,
        mc_z: int,
        scale_x: float,
        scale_z: float,
        half_ext_x: Optional[float] = None,
        half_ext_z: Optional[float] = None,
    ) -> int:
        """Return the MC Y to use as the base of a shape's placement.

        Grounding policy
        ----------------
        Small props (trees, lamps, poles, tiny decor): the footprint fits
        within a single terrain tile, so a centre bilinear sample is accurate
        and consistent.

        Large structures (keeps, gates, walls, houses): the footprint can span
        multiple terrain tiles.  Anchoring to only the centre-point may bury
        one corner into a higher tile.  The lowest of the four footprint corners
        is used so the base of the structure never sinks below terrain.

        The tile-span threshold used internally by terrain_y_at_footprint()
        is _MC_TILE_BLOCKS (4 blocks).  convert_shape() passes the real
        transformed-mesh footprint from _estimate_shape_half_extent(); the
        scale-based estimate below (≈8 blocks × scale) is only used when a
        caller omits the extents.
        """
        if self.h_grid is None:
            return MC_Y_BASE
        # Estimate half-extent: typical KO building half-width at unit scale
        # is roughly 8 KO units, which maps to 8 MC blocks.  Scale linearly.
        if half_ext_x is None or half_ext_z is None:
            half_ext = max(float(scale_x), float(scale_z)) * 8.0
            half_ext_x = half_ext
            half_ext_z = half_ext
        return _terrain_y_at_footprint(self.h_grid, mc_x, mc_z, half_ext_x, half_ext_z)

    def _estimate_shape_half_extent(self, shape, quat: tuple, scale: tuple) -> tuple[float, float]:
        """Estimate shape footprint half extents in KO/MC units from transformed meshes.

        Falls back to a scale heuristic when mesh data cannot be resolved/parsed.
        """
        cache_key = (
            shape.name.lower(),
            tuple(round(float(v), 4) for v in quat),
            tuple(round(float(v), 4) for v in scale),
            tuple(p.name.lower() for p in shape.parts),
        )
        if cache_key in self._footprint_cache:
            return self._footprint_cache[cache_key]

        min_x = float("inf")
        max_x = float("-inf")
        min_z = float("inf")
        max_z = float("-inf")
        seen = 0

        for part in shape.parts:
            mesh_path = self.resolver.resolve(part.name)
            if mesh_path is None:
                continue
            try:
                mesh = parse_n3pmesh(mesh_path)
                verts_t = apply_transform(mesh.vertices, quat, scale)
            except Exception:
                continue

            pivot_local = (part.pivot.x, part.pivot.y, part.pivot.z)
            piv = transform_pivot(pivot_local, quat, scale)
            x = verts_t[:, 0] + float(piv[0])
            z = verts_t[:, 2] + float(piv[2])
            min_x = min(min_x, float(x.min()))
            max_x = max(max_x, float(x.max()))
            min_z = min(min_z, float(z.min()))
            max_z = max(max_z, float(z.max()))
            seen += 1

        if seen == 0:
            half = max(float(scale[0]), float(scale[2])) * 8.0
            out = (half, half)
        else:
            out = (max(abs(min_x), abs(max_x)), max(abs(min_z), abs(max_z)))

        self._footprint_cache[cache_key] = out
        return out

    # ── Public ────────────────────────────────────────────────────────────────

    def convert_shape(self, shape: Shape, shape_index: int) -> ShapeResult:
        """Convert a single OPD shape to MC blocks and return its result."""
        ko_pos = (shape.position.x, shape.position.y, shape.position.z)

        # Step 1 — transform XZ position via the ONE canonical function.
        mc_x, mc_z = ko_to_mc_position(ko_pos[0], ko_pos[2], self.map_size)

        quat   = (shape.rotation.x, shape.rotation.y,
                  shape.rotation.z, shape.rotation.w)
        scale  = (shape.scale.x, shape.scale.y, shape.scale.z)
        half_ext_x, half_ext_z = self._estimate_shape_half_extent(shape, quat, scale)

        # Step 2 — choose grounding policy then compute mc_y.
        #
        # TERRAIN_GROUNDED (buildings, walls, gates, …):
        #   base snapped to terrain surface via footprint-aware bilinear sample.
        #
        # WORLD_Y (boats, chains, suspended/hanging props, …):
        #   authored OPD Y preserved via ko_to_mc_y(), because level designers
        #   placed these in world space, not terrain-relative.
        terrain_mc_y = self._ground_shape(
            mc_x, mc_z, shape.scale.x, shape.scale.z,
            half_ext_x=half_ext_x, half_ext_z=half_ext_z,
        )
        mode, mode_reason = _grounding_mode(shape.name, ko_pos[1], terrain_mc_y)

        if mode == "world_y":
            mc_y = ko_to_mc_y(ko_pos[1])
        else:
            mc_y = terrain_mc_y

        mc_pos = (mc_x, mc_y, mc_z)

        # Optional grounding-decision trace (--debug-grounding / debug_grounding=True)
        # TRACK A: placement policy audit — see which rule fired and why.
        if self.debug_grounding:
            opd_mc_y = ko_to_mc_y(ko_pos[1])
            delta = abs(opd_mc_y - terrain_mc_y)
            print(
                f"[GROUNDING] {shape.name!r:50s}"
                f"  opd_ko_y={ko_pos[1]:+8.2f}  opd_mc_y={opd_mc_y:+5d}"
                f"  terrain_mc_y={terrain_mc_y:+5d}  delta={delta:4d}"
                f"  mode={mode:9s}  reason={mode_reason!r:35s}"
                f"  final_base_y={mc_y:+5d}"
                f"  n_parts={len(shape.parts)}"
            )

        # Optional full transform trace (--debug-transform / debug_transform=True)
        if self.debug_transform:
            self._print_transform_trace(shape, shape_index, mc_pos, quat, scale)

        # ── Vegetation: skip or replace with native MC block ─────────────────
        veg = _vegetation_action(shape.name)
        if veg == "skip":
            pr = PartResult(part_name="<skipped>", status="skipped")
            pr.blocks_placed = 0
            part_results = [pr]
            status = ShapeStatus.SKIPPED
            return ShapeResult(
                shape_name=shape.name, shape_index=shape_index,
                status=status, ko_position=ko_pos, mc_position=mc_pos,
                part_results=part_results, blocks_placed=0,
            )
        elif veg is not None:
            # Voxelize with vegetation block instead of stone_bricks.
            # Vegetation shapes typically have zero pivots; _part_mc_pos is
            # a no-op for those, so calling it here is safe and consistent.
            part_results = []
            for part in shape.parts:
                if _is_billboard_impostor(part.name):
                    continue  # skip billboard parts
                mc_part_pos = self._part_mc_pos(mc_pos, part, quat, scale)
                pr = self._convert_part(part.name, quat, scale, mc_part_pos,
                                        block_name=veg)
                part_results.append(pr)
            if not part_results:
                pr = PartResult(part_name="<skipped>", status="skipped")
                part_results = [pr]

        # ── Trees: faithful voxelization of the _po mesh ─────────────────────
        elif _shape_is_tree(shape):
            # Tree _po meshes may carry a pivot; let _place_faithful_tree
            # handle pivot application per-part.
            part_results = self._place_faithful_tree(shape, mc_pos, quat, scale)

        # ── Normal shapes ─────────────────────────────────────────────────────
        else:
            # Step 3 — apply OPD part pivot (per-part, in local model space).
            # The pivot shifts the mesh anchor relative to the shape's world
            # position.  It is applied AFTER terrain grounding (separate steps).
            # See _part_mc_pos() for the full formula.
            part_results = []
            for part in shape.parts:
                mc_part_pos = self._part_mc_pos(mc_pos, part, quat, scale)
                blk, palette_source = _block_for_part_with_source(shape.name, part.textures)
                if palette_source == PALETTE_FALLBACK_UNKNOWN:
                    for tex in part.textures:
                        if tex not in self.unknown_textures:
                            logger.info("Unknown texture %r (part %r): palette fallback to %s",
                                        tex, part.name, blk)
                        self.unknown_textures[tex] += 1
                # TRACK A: log per-part anchor so we can verify all parts of one
                # OPD object share the same base Y and don't independently reground.
                if self.debug_grounding:
                    ppx, ppy, ppz = mc_part_pos
                    piv = part.pivot
                    print(
                        f"  [PART]  {part.name!r:48s}"
                        f"  pivot_local=({piv.x:.2f},{piv.y:.2f},{piv.z:.2f})"
                        f"  anchor=({ppx:+5d},{ppy:+5d},{ppz:+5d})"
                    )
                pr = self._convert_part(part.name, quat, scale, mc_part_pos,
                                        block_name=blk)
                pr.palette_source = palette_source
                pr.textures = list(part.textures)
                part_results.append(pr)

        status = self._aggregate_status(part_results)

        result = ShapeResult(
            shape_name    = shape.name,
            shape_index   = shape_index,
            status        = status,
            ko_position   = ko_pos,
            mc_position   = mc_pos,
            part_results  = part_results,
            blocks_placed = sum(p.blocks_placed for p in part_results),
            grounding_mode=mode,
            grounding_reason=mode_reason,
            terrain_mc_y=terrain_mc_y,
            opd_mc_y=ko_to_mc_y(ko_pos[1]),
            grounding_delta=abs(ko_to_mc_y(ko_pos[1]) - terrain_mc_y),
            grounding_quality=grounding_quality(
                self.h_grid, mc_x, mc_z, mc_y, half_ext_x, half_ext_z),
            partial_reason_counts=self._partial_reason_counts(part_results),
        )

        if self.debug_markers and status not in (
            ShapeStatus.PLACED, ShapeStatus.PARTIAL, ShapeStatus.SKIPPED
        ):
            marker = _MARKER_MISSING
            if status == ShapeStatus.PARSE_FAILED:
                marker = _MARKER_PARSE_FAILED
            elif status == ShapeStatus.VOXELIZE_FAILED:
                marker = _MARKER_VOXEL_FAILED
            mx, my, mz = mc_pos
            self.world.set_block(mx, my, mz, marker)

        # ── Transform chain visualization markers ─────────────────────────────
        # When --debug-transform is on, place colored markers that show the
        # three steps of the placement chain for every non-skipped shape:
        #
        #   Red wool   (mc_pos + 3)  = shape OPD anchor after canonical XZ+Y transform
        #   Lime wool  (part_pos + 1) = per-part anchor after pivot offset
        #
        # These two should overlap for shapes with zero pivots.
        # A systematic gap between them reveals a pivot that was previously ignored.
        if self.debug_transform and status != ShapeStatus.SKIPPED:
            mx, my, mz = mc_pos
            # Shape anchor marker (red wool, 3 blocks up so it's visible above terrain)
            self.world.set_block(mx, my + 3, mz, _MARKER_SHAPE_ANCHOR)
            # Per-part pivot markers (lime wool, 1 block up)
            for part in shape.parts:
                if _is_billboard_impostor(part.name):
                    continue
                px, py, pz = self._part_mc_pos(mc_pos, part, quat, scale)
                if (px, py, pz) != (mx, my, mz):
                    # Only mark if pivot actually shifted the anchor
                    self.world.set_block(px, py + 1, pz, _MARKER_PART_ANCHOR)

        return result

    def build_report(
        self,
        opd_path: str,
        world_path: str,
        total_shapes: int,
        results: list,
    ) -> PlacementReport:
        counts = {s.value: 0 for s in ShapeStatus}
        for r in results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1

        return PlacementReport(
            opd_path     = opd_path,
            world_path   = world_path,
            total_shapes = total_shapes,
            processed    = len(results),
            counts       = {k: v for k, v in counts.items() if v > 0},
            shapes       = results,
            missing_refs = list(dict.fromkeys(self.resolver.missing_refs)),
            audit_report_level=self.audit_report_level,
            voxel_mode   = self.voxel_mode,
            unknown_textures=dict(self.unknown_textures),
        )

    @staticmethod
    def _partial_reason_counts(part_results: list[PartResult]) -> dict:
        counts: dict[str, int] = {}
        for p in part_results:
            counts[p.status] = counts.get(p.status, 0) + 1
            if p.resolver_ambiguous:
                counts["resolver_ambiguous"] = counts.get("resolver_ambiguous", 0) + 1
        return counts

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _part_mc_pos(
        self,
        shape_mc_pos: tuple,
        part,
        quat: tuple,
        scale: tuple,
    ) -> tuple:
        """Compute the per-part MC anchor by applying the OPD part pivot.

        In KO, each mesh part has a *pivot* — a local-space offset that tells
        the engine where the part's mesh origin sits relative to the shape's
        world origin.  The KO engine applies:

            part_world = shape_world_pos + R × (S × pivot_local)

        where R is the shape rotation matrix and S is the shape scale vector.

        This function converts that offset to MC space via pivot_to_mc_offset(),
        which mirrors the X component (matching the canonical mc_x = map_size − ko_x
        world transform).  Then it adds the offset to the shape's MC anchor.

        Grounding (terrain Y) is computed once at the SHAPE level and stays fixed.
        The pivot's Y component shifts the part up/down relative to the terrain
        surface but does NOT trigger a new terrain re-sample.

        Returns the pivot-adjusted MC anchor (mc_x, mc_y, mc_z) for this part.
        """
        pivot_local = (part.pivot.x, part.pivot.y, part.pivot.z)

        # Fast path: zero pivot (most parts) — no math needed.
        if pivot_local == (0.0, 0.0, 0.0):
            return shape_mc_pos

        mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pivot_local, quat, scale)
        mx, my, mz = shape_mc_pos
        return (mx + mc_dx, my + mc_dy, mz + mc_dz)

    def _print_transform_trace(
        self,
        shape,
        shape_index: int,
        shape_mc_pos: tuple,
        quat: tuple,
        scale: tuple,
    ) -> None:
        """Print a per-shape transform audit to stdout.

        Outputs the full chain from OPD coords through each transform step to
        the final per-part MC anchor, plus the pivot values.  Intended for
        landmark debugging (--debug-transform flag).

        This trace makes it possible to verify that:
          1. The canonical position transform is applied correctly.
          2. The pivot is in local model space and is correctly rotated.
          3. Grounding and pivot handling are separate steps.
          4. The X mirror is applied once (in pivot_to_mc_offset).
        """
        import math as _math
        ko_x, ko_y, ko_z = shape.position.x, shape.position.y, shape.position.z
        mc_x, mc_y, mc_z = shape_mc_pos
        ko_yaw = quat_yaw(quat)
        mc_yaw = -ko_yaw  # canonical yaw transform (X mirror negates)

        print(f"\n{'─'*64}")
        print(f"  SHAPE [{shape_index}] {shape.name!r}")
        print(f"{'─'*64}")
        print(f"  OPD position (KO):  ({ko_x:.2f}, {ko_y:.2f}, {ko_z:.2f})")
        print(f"  OPD yaw (KO):       {ko_yaw:.4f} rad  ({_math.degrees(ko_yaw):.1f}°)")
        print(f"  MC anchor (XZ):     ({mc_x}, {mc_z})   [mc_x = {self.map_size} - {int(ko_x)}]")
        print(f"  Terrain Y:          {mc_y}")
        print(f"  MC yaw:             {mc_yaw:.4f} rad  ({_math.degrees(mc_yaw):.1f}°)")
        print(f"  Parts: {len(shape.parts)}")

        for i, part in enumerate(shape.parts):
            pv = part.pivot
            pivot_local = (pv.x, pv.y, pv.z)
            mc_part_pos = self._part_mc_pos(shape_mc_pos, part, quat, scale)
            px, py, pz = mc_part_pos
            if pivot_local == (0.0, 0.0, 0.0):
                pivot_note = "(zero pivot — no offset)"
            else:
                mc_dx, mc_dy, mc_dz = pivot_to_mc_offset(pivot_local, quat, scale)
                pivot_note = (
                    f"KO pivot=({pv.x:.3f},{pv.y:.3f},{pv.z:.3f}) → "
                    f"MC offset=({mc_dx},{mc_dy},{mc_dz})"
                )
            print(f"    part[{i}] {part.name!r}")
            print(f"           {pivot_note}")
            print(f"           anchor → ({px}, {py}, {pz})")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _convert_part(
        self,
        part_name: str,
        quat: tuple,
        scale: tuple,
        mc_pos: tuple,
        block_name: Optional[str] = None,
    ) -> PartResult:
        pr = PartResult(part_name=part_name, status="missing")

        # 1. Resolve mesh path
        mesh_path, rmeta = self.resolver.resolve_with_meta(part_name)
        pr.resolver_strategy = rmeta.get("strategy")
        pr.resolver_candidate_count = int(rmeta.get("candidate_count", 0))
        pr.resolver_ambiguous = bool(rmeta.get("ambiguous", False))
        if mesh_path is None:
            logger.debug("Missing mesh: %s", part_name)
            return pr

        # 2. Parse mesh
        try:
            mesh = parse_n3pmesh(mesh_path)
        except BinaryParseError as exc:
            pr.status = "parse_failed"
            pr.error  = str(exc)
            logger.warning("Parse failed [%s]: %s", part_name, exc)
            return pr
        except Exception as exc:
            pr.status = "parse_failed"
            pr.error  = f"{type(exc).__name__}: {exc}"
            logger.warning("Parse error [%s]: %s", part_name, exc)
            return pr

        # 3. Apply OPD transform (rotation + scale) in model-local space
        try:
            verts_t = apply_transform(mesh.vertices, quat, scale)
            grid    = voxelize_mesh(
                verts_t, mesh.indices, self.voxel_size, **self._voxel_kwargs
            )
        except Exception as exc:
            pr.status = "voxelize_failed"
            pr.error  = f"{type(exc).__name__}: {exc}"
            logger.warning("Voxelize failed [%s]: %s", part_name, exc)
            return pr

        # TRACK B: voxelization consistency audit (--debug-voxelize).
        # Exposes grid.origin float vs int-truncation vs floor, bbox, voxel count.
        # Use this to compare two "identical" meshes and find grid-alignment divergence.
        if self.debug_voxelize:
            import math as _math
            bmin = verts_t[:, :3].min(axis=0)
            bmax = verts_t[:, :3].max(axis=0)
            orig_f = grid.origin  # float32 (3,)
            orig_int   = (int(orig_f[0]),           int(orig_f[1]),           int(orig_f[2]))
            orig_floor = (_math.floor(orig_f[0]),   _math.floor(orig_f[1]),   _math.floor(orig_f[2]))
            frac = (orig_f[0] - orig_floor[0],      orig_f[1] - orig_floor[1], orig_f[2] - orig_floor[2])
            phase_error = tuple(orig_int[i] - orig_floor[i] for i in range(3))  # 0 or 1
            print(
                f"\n[VOXELIZE] {part_name!r}"
                f"\n  quat=({quat[0]:.4f},{quat[1]:.4f},{quat[2]:.4f},{quat[3]:.4f})"
                f"  scale=({scale[0]:.3f},{scale[1]:.3f},{scale[2]:.3f})"
                f"\n  bbox_min=({bmin[0]:+.3f},{bmin[1]:+.3f},{bmin[2]:+.3f})"
                f"  bbox_max=({bmax[0]:+.3f},{bmax[1]:+.3f},{bmax[2]:+.3f})"
                f"\n  grid_origin_float=({orig_f[0]:+.4f},{orig_f[1]:+.4f},{orig_f[2]:+.4f})"
                f"\n  origin_int   =({orig_int[0]:+d},{orig_int[1]:+d},{orig_int[2]:+d})"
                f"  origin_floor =({orig_floor[0]:+d},{orig_floor[1]:+d},{orig_floor[2]:+d})"
                f"\n  frac=(x={frac[0]:.4f},y={frac[1]:.4f},z={frac[2]:.4f})"
                f"  would_have_erred_if_int_not_floor={phase_error}  ← (1,0,0) means floor fixed 1-block X shift"
                f"\n  grid_dims={grid.nx}x{grid.ny}x{grid.nz}"
                f"  voxels={grid.count_filled()}"
                f"  mc_anchor={mc_pos}"
            )

        # 4. Place in MC world.
        #    Anchor: mesh local origin (0,0,0) → mc_pos (mc_y = terrain Y).
        #    grid.origin = bbox_min − padding*voxel_size in model-local space.
        #
        #    X-axis mirror:  mc_x = map_size - ko_x means local KO X offsets
        #    must be SUBTRACTED from mc_x (not added).
        #        off_x = mc_x - origin[0]
        #        wx    = off_x - ix
        #
        #    Y / Z are not mirrored; offsets add normally:
        #        off_y = mc_y_adj + origin[1],  wy = off_y + iy
        #        off_z = mc_z    + origin[2],  wz = off_z + iz
        #
        #    Floating-mesh correction: if bbox_min_y > 0 the mesh sits entirely
        #    above its local origin → off_y would be ABOVE terrain surface.
        #    Pull the anchor down so the mesh bottom aligns with terrain.
        #
        #    IMPORTANT: use math.floor(), NOT int(), for grid.origin.
        #    int(-1.63) = -1 (truncates toward zero) but the correct floor is -2.
        #    Using int() on negative origins (typical for meshes centered at local
        #    origin) shifts all voxels 1 block in the wrong direction in X and Z.
        #    Confirmed by Track B diagnostic: phase_error_int_vs_floor=(1,0,1)
        #    for every instance of every mesh with a centered local origin.
        mc_x, mc_y, mc_z = mc_pos
        origin_x = math.floor(grid.origin[0])
        origin_y = math.floor(grid.origin[1])
        origin_z = math.floor(grid.origin[2])

        if origin_y > 0:
            mc_y_adj = mc_y - origin_y
        else:
            mc_y_adj = mc_y

        # X is mirrored: subtract origin and iterate in reverse.
        off_x = mc_x - origin_x
        off_y = mc_y_adj + origin_y
        off_z = mc_z + origin_z

        blk = block_name if block_name is not None else self.block_name
        placed = _place_voxel_grid_x_flipped(
            grid, self.world, off_x, off_y, off_z,
            block_name=blk,
        )

        pr.status        = "placed"
        pr.blocks_placed = placed
        return pr

    def _place_faithful_tree(
        self,
        shape,
        mc_pos: tuple,
        quat: tuple,
        scale: tuple,
    ) -> list:
        """Voxelize the tree's _po (3D polygon) mesh and place with log/leaf blocks.

        Lower 30 % of voxel height = oak_log (trunk), upper 70 % = oak_leaves.
        If no _po part exists, fall back to the oak template.
        Skips all _ip (billboard) parts.
        """
        po_parts = [p for p in shape.parts
                    if "_po" in p.name.lower()
                    and not _is_billboard_impostor(p.name)]

        if not po_parts:
            return [self._place_tree_template(mc_pos)]

        results = []
        for part in po_parts:
            # Apply pivot per-part (same as normal shapes).
            mc_part_pos = self._part_mc_pos(mc_pos, part, quat, scale)
            pr = self._convert_part_tree(part.name, quat, scale, mc_part_pos)
            results.append(pr)
        return results

    def _convert_part_tree(
        self,
        part_name: str,
        quat: tuple,
        scale: tuple,
        mc_pos: tuple,
    ) -> PartResult:
        """Voxelize a tree _po mesh and place voxels as log (trunk) or leaves (canopy)."""
        pr = PartResult(part_name=part_name, status="missing")

        mesh_path = self.resolver.resolve(part_name)
        if mesh_path is None:
            return self._place_tree_template(mc_pos)

        try:
            mesh = parse_n3pmesh(mesh_path)
        except Exception as exc:
            pr.status = "parse_failed"
            pr.error  = str(exc)
            return pr

        try:
            verts_t = apply_transform(mesh.vertices, quat, scale)
            grid    = voxelize_mesh(verts_t, mesh.indices, self.voxel_size,
                                    **self._voxel_kwargs)
        except Exception as exc:
            pr.status = "voxelize_failed"
            pr.error  = str(exc)
            return pr

        # Classify voxels: lower 30 % of height → log, rest → leaves
        filled = np.argwhere(grid.data > 0)  # (K, 3) — z, y, x
        if len(filled) == 0:
            return PartResult(part_name=part_name, status="placed")

        min_iy  = int(filled[:, 1].min())
        max_iy  = int(filled[:, 1].max())
        height  = max(1, max_iy - min_iy)
        trunk_top = min_iy + max(1, int(height * 0.30))

        mc_x, mc_y, mc_z = mc_pos
        origin_x = math.floor(grid.origin[0])   # floor, not int (see _convert_part)
        origin_y = math.floor(grid.origin[1])
        origin_z = math.floor(grid.origin[2])
        mc_y_adj = (mc_y - origin_y) if origin_y > 0 else mc_y
        # X-axis mirror: subtract origin and negate ix (see _convert_part).
        off_x = mc_x - origin_x
        off_y = mc_y_adj + origin_y
        off_z = mc_z + origin_z

        placed = 0
        for iz, iy, ix in filled:
            blk = _OAK_LOG if int(iy) <= trunk_top else _OAK_LEAVES
            self.world.set_block(
                off_x - int(ix),
                off_y + int(iy),
                off_z + int(iz),
                blk,
            )
            placed += 1

        pr.status        = "placed"
        pr.blocks_placed = placed
        return pr

    def _place_tree_template(self, mc_pos: tuple) -> PartResult:
        """Fallback: place the oak template when no _po mesh is available."""
        mc_x, mc_y, mc_z = mc_pos
        placed = 0
        for dx, dy, dz, block in _OAK_TREE_TEMPLATE:
            self.world.set_block(mc_x + dx, mc_y + dy, mc_z + dz, block)
            placed += 1
        pr = PartResult(part_name="<oak_tree>", status="placed")
        pr.blocks_placed = placed
        return pr

    @staticmethod
    def _aggregate_status(part_results: list) -> ShapeStatus:
        if not part_results:
            return ShapeStatus.MISSING_MESH

        statuses = {p.status for p in part_results}

        if statuses == {"placed"}:
            return ShapeStatus.PLACED
        if statuses == {"missing"}:
            return ShapeStatus.MISSING_MESH
        if statuses == {"parse_failed"}:
            return ShapeStatus.PARSE_FAILED
        if statuses == {"voxelize_failed"}:
            return ShapeStatus.VOXELIZE_FAILED
        if "placed" in statuses:
            return ShapeStatus.PARTIAL
        # Mix of non-placed outcomes: prefer most informative status
        if "parse_failed" in statuses:
            return ShapeStatus.PARSE_FAILED
        if "voxelize_failed" in statuses:
            return ShapeStatus.VOXELIZE_FAILED
        return ShapeStatus.MISSING_MESH


# ── Terrain builder ───────────────────────────────────────────────────────────

# KO MakeMoveTable threshold: if max-min height across a tile's 4 corners >= this,
# the tile is a cliff (impassable). Source: GameTerrain.cpp NOTMOVE_HEIGHT = 10.
KO_NOTMOVE_HEIGHT: float = 10.0

# Surface block selection — driven strictly by GTD height + slope data.
# No global water fill: KO water is per-pond (PondMesh objects), not sea level.
_SURF_CLIFF   = "minecraft:stone"       # slope >= NOTMOVE_HEIGHT (steep cliff)
_SURF_LOW     = "minecraft:gravel"      # h < 0 (below datum, low-lying ground)
_SURF_COASTAL = "minecraft:sand"        # h in [0, 2) (near-ground level)
_SURF_LAND    = "minecraft:grass_block" # otherwise
_FILL_DIRT    = "minecraft:dirt"
_FILL_STONE   = "minecraft:stone"
_FILL_BEDROCK = "minecraft:bedrock"

# GTD texture-id fallback buckets (best-effort; map-specific IDs can vary).
_TEXID_BLOCKS = {
    0: "minecraft:grass_block",
    1: "minecraft:dirt",
    2: "minecraft:sand",
    3: "minecraft:stone",
    4: "minecraft:gravel",
    5: "minecraft:cobblestone",
    6: "minecraft:snow_block",
    7: "minecraft:coarse_dirt",
    8: "minecraft:sandstone",
    9: "minecraft:clay",
    10: "minecraft:podzol",
    11: "minecraft:moss_block",
    12: "minecraft:mud",
    13: "minecraft:packed_mud",
    14: "minecraft:terracotta",
    15: "minecraft:red_sand",
}


def _terrain_surface_block(ko_h: float, slope: float, tex_id: int) -> str:
    """Choose terrain block using texture-id first, then slope/height overrides."""
    if slope >= KO_NOTMOVE_HEIGHT:
        return _SURF_CLIFF
    if ko_h < 0.0:
        return _SURF_LOW
    base = _TEXID_BLOCKS.get(int(tex_id) % 16, _SURF_LAND)
    if ko_h < 2.0 and base in {"minecraft:grass_block", "minecraft:dirt"}:
        return _SURF_COASTAL
    return base


def build_terrain(
    gtd,
    world: "MinecraftWorld",
    fill_depth: int = 8,
    stats: Optional[dict] = None,
) -> int:
    """Place GTD heightmap terrain into a MinecraftWorld.

    Coordinate mapping:
        GTD tile (tx, tz) spans KO world [tx*4, (tx+1)*4) in X and Z.
        Each MC block (mx, mz) maps to fractional tile position (mx/4, mz/4).
        Heights are bilinearly interpolated across the 4 tile corners.

    Height conversion (consistent with ko_pos_to_mc for object placement):
        mc_y = int(MC_Y_BASE + ko_h * KO_HEIGHT_SCALE)
        With MC_Y_BASE=64 and KO_HEIGHT_SCALE=1.0: 1 KO unit = 1 MC block vertically.

    Block selection (data-driven from GTD, matches KO MakeMoveTable):
        1. slope >= KO_NOTMOVE_HEIGHT (10.0) → stone (cliff, GameTerrain.cpp)
        2. ko_h < 0                          → gravel (low-lying area below datum)
        3. ko_h < 2.0                        → sand (near-ground level)
        4. otherwise                         → grass_block

    Note: KO water is per-pond PondMesh objects, not a global sea level.
    No water fill is applied here.

    If ``stats`` is given it is filled with terrain palette telemetry:
    ``unique_tex_ids``, ``unmapped_tex_ids`` ({tex_id: block columns} for IDs
    with no entry in _TEXID_BLOCKS, which fall back to the ``id % 16``
    bucket) and ``fallback_rate`` (share of columns using such IDs).

    Returns total surface blocks placed.
    """
    n = gtd.heightmap_size        # 257 for a 1024-unit map  (USKO Moradon v1298)
    world_size = (n - 1) * 4      # 1024 MC blocks wide

    print(f"  Building terrain {world_size}×{world_size} from {n}×{n} GTD vertices "
          f"(bilinear interp, scale={KO_HEIGHT_SCALE}, base_y={MC_Y_BASE})...")

    # Use the canonical MC-oriented height grid (single source of truth).
    # Verified: castle mc_x=315 → tx=78, h_mc[78,96]=4.74 = OPD_Y=4.74 ✓
    h = build_mc_height_grid(gtd.heights)

    # ── Precompute per-GTD-tile slope (max-min of 4 corner heights) ─────────
    # Matches GameTerrain.cpp MakeMoveTable: NOTMOVE_HEIGHT=10 → impassable cliff
    # Shape: (n-1, n-1) — one entry per tile
    h00 = h[:-1, :-1]
    h10 = h[1:,  :-1]
    h01 = h[:-1, 1:]
    h11 = h[1:,  1:]
    tile_slope = (
        np.maximum(np.maximum(h00, h10), np.maximum(h01, h11)) -
        np.minimum(np.minimum(h00, h10), np.minimum(h01, h11))
    )  # (n-1, n-1)

    # ── Bilinear interpolation: per-MC-block fractional tile coordinates ─────
    # Each MC block (mx, mz): fractional tile position = (mx/4, mz/4)
    mx_arr = np.arange(world_size, dtype=np.float32)
    mz_arr = np.arange(world_size, dtype=np.float32)

    tx_f = mx_arr / 4.0                           # (world_size,)
    tz_f = mz_arr / 4.0

    tx0 = np.floor(tx_f).astype(np.int32)         # lower tile corner X
    tz0 = np.floor(tz_f).astype(np.int32)         # lower tile corner Z
    tx1 = np.minimum(tx0 + 1, n - 1)             # upper tile corner X (clamped)
    tz1 = np.minimum(tz0 + 1, n - 1)             # upper tile corner Z (clamped)
    fx = (tx_f - tx0).astype(np.float32)          # blend fraction X  [0, 1)
    fz = (tz_f - tz0).astype(np.float32)          # blend fraction Z  [0, 1)

    # H[mx, mz] = bilinear blend of 4 surrounding GTD vertices
    H00 = h[np.ix_(tx0, tz0)]                     # (world_size, world_size)
    H10 = h[np.ix_(tx1, tz0)]
    H01 = h[np.ix_(tx0, tz1)]
    H11 = h[np.ix_(tx1, tz1)]

    fx2d = fx[:, np.newaxis]                       # (world_size, 1)
    fz2d = fz[np.newaxis, :]                       # (1, world_size)

    ko_h_grid = (
        H00 * (1.0 - fx2d) * (1.0 - fz2d) +
        H10 * fx2d          * (1.0 - fz2d) +
        H01 * (1.0 - fx2d) * fz2d          +
        H11 * fx2d          * fz2d
    )  # (world_size, world_size) float32

    # Height conversion consistent with ko_pos_to_mc (uses MC_Y_BASE + h * KO_HEIGHT_SCALE)
    mc_y_grid = (MC_Y_BASE + ko_h_grid * KO_HEIGHT_SCALE).astype(np.int32)

    # ── Per-MC-block slope + texture id (tile the block belongs to) ───────────
    # tile index for block mx = tx0[mx], clamped to valid tile range [0, n-2]
    tx0c = np.minimum(tx0, n - 2)
    tz0c = np.minimum(tz0, n - 2)
    slope_grid = tile_slope[np.ix_(tx0c, tz0c)]  # (world_size, world_size) float32
    tex_grid = gtd.texture_ids[np.ix_(tx0c, tz0c)]

    if stats is not None:
        ids, counts = np.unique(tex_grid, return_counts=True)
        unmapped = {int(i): int(c) for i, c in zip(ids, counts) if int(i) not in _TEXID_BLOCKS}
        total = int(counts.sum())
        for tid, cnt in sorted(unmapped.items()):
            logger.info("Unmapped GTD texture id %d (%d columns): fallback bucket %d",
                        tid, cnt, tid % 16)
        stats.update({
            "unique_tex_ids": int(len(ids)),
            "unmapped_tex_ids": {str(k): v for k, v in sorted(unmapped.items())},
            "fallback_rate": round(sum(unmapped.values()) / total, 4) if total else 0.0,
        })

    # ── Write blocks ─────────────────────────────────────────────────────────
    # Buildings were placed before terrain (buildings-first ordering).
    # Skip surface/fill positions that are already occupied by building blocks.
    # No terrain carving: with h_grid-based building Y, buildings sit at the
    # same height as terrain — carving is neither needed nor correct.
    placed = 0
    for mx in range(world_size):
        for mz in range(world_size):
            ko_h  = float(ko_h_grid[mx, mz])
            mc_y  = int(mc_y_grid[mx, mz])
            slope = float(slope_grid[mx, mz])
            tex_id = int(tex_grid[mx, mz])
            surf = _terrain_surface_block(ko_h, slope, tex_id)

            # Place surface (skip if a building block is already there)
            if not world.has_block(mx, mc_y, mz):
                world.set_block(mx, mc_y, mz, surf)

            # Fill below surface
            for dy in range(1, fill_depth + 1):
                fy = mc_y - dy
                if fy < -64:
                    break
                if world.has_block(mx, fy, mz):
                    continue
                if dy <= 3:
                    world.set_block(mx, fy, mz, _FILL_DIRT)
                elif dy == fill_depth:
                    world.set_block(mx, fy, mz, _FILL_BEDROCK)
                else:
                    world.set_block(mx, fy, mz, _FILL_STONE)
            placed += 1

    print(f"  Terrain: {placed:,} surface blocks placed.")
    return placed


# ── Subset filters ────────────────────────────────────────────────────────────

def _passes_filters(
    shape: Shape,
    name_filter: Optional[str],
    center: Optional[tuple],
    radius: Optional[float],
) -> bool:
    """Return True if shape passes all active subset filters."""
    if name_filter and name_filter.lower() not in shape.name.lower():
        return False
    if center is not None and radius is not None:
        cx, cz = center
        dx = shape.position.x - cx
        dz = shape.position.z - cz
        if (dx * dx + dz * dz) ** 0.5 > radius:
            return False
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OPD-driven zone-to-Minecraft voxel converter"
    )
    parser.add_argument("opd_file", help=".opd input file")
    parser.add_argument(
        "--asset-roots",
        nargs="+",
        metavar="DIR",
        default=[],
        help=(
            "Asset root directories in priority order (highest first). "
            "Example: /path/to/1886-Client /path/to/ko-assets-1298"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        metavar="DIR",
        required=True,
        help="Output directory for the Minecraft world",
    )
    parser.add_argument(
        "--world-name",
        metavar="NAME",
        default=None,
        help="Minecraft world name (default: derived from OPD filename)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        metavar="SIZE",
        help="Voxel side length in KO units (default: 1.0)",
    )
    parser.add_argument(
        "--block",
        default="minecraft:stone_bricks",
        metavar="NAME",
        help="Minecraft block for placed voxels (default: stone_bricks)",
    )
    parser.add_argument(
        "--voxel-mode",
        choices=VOXEL_MODES,
        default=None,
        help=(
            "Voxelization policy: 'surface' = hollow shell (default), "
            "'hybrid' = raw SAT occupancy without shell stripping or fill, "
            "'solid' = flood-fill enclosed interiors"
        ),
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Alias for --voxel-mode solid (flood-fill interior voxels)",
    )
    parser.add_argument(
        "--debug-markers",
        action="store_true",
        help="Place marker blocks at unresolved shape positions",
    )
    parser.add_argument(
        "--debug-transform",
        action="store_true",
        help=(
            "Print transform chain (OPD pos → pivot → MC anchor) for each shape "
            "and place colored in-world markers: red=shape anchor, lime=part pivot anchor"
        ),
    )
    parser.add_argument(
        "--debug-grounding",
        action="store_true",
        help=(
            "TRACK A — Print grounding decision for every shape: "
            "name, OPD_Y, terrain_mc_Y, delta, mode (terrain/world_y), "
            "rule that fired, final base Y, per-part anchors"
        ),
    )
    parser.add_argument(
        "--debug-voxelize",
        action="store_true",
        help=(
            "TRACK B — Print voxelization audit for every mesh part: "
            "quat/scale, bbox, grid.origin (float/int/floor), fractional phase, "
            "phase_error (int vs floor), voxel count. "
            "Use with --name-filter to compare two copies of the same asset."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N shapes (after other filters)",
    )
    parser.add_argument(
        "--name-filter",
        metavar="SUBSTRING",
        default=None,
        help="Only process shapes whose name contains SUBSTRING (case-insensitive)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        metavar="R",
        help="Only process shapes within R KO units of --center (XZ plane)",
    )
    parser.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("X", "Z"),
        default=None,
        help="Center point (KO world X Z) for --radius filter",
    )
    parser.add_argument(
        "--report",
        metavar="FILE",
        default=None,
        help="Write JSON placement report to FILE (default: <world_dir>/placement_report.json)",
    )
    parser.add_argument(
        "--gtd",
        metavar="FILE",
        default=None,
        help=(
            "GTD terrain file to place before objects. "
            "If omitted, auto-detected as <opd_file>.gtd if it exists."
        ),
    )
    parser.add_argument(
        "--no-terrain",
        action="store_true",
        help="Skip terrain generation even if a GTD file is found.",
    )
    parser.add_argument(
        "--terrain-fill-depth",
        type=int,
        default=8,
        metavar="N",
        help="Number of fill blocks below terrain surface (default: 8)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--audit-report-level",
        choices=("standard", "detailed"),
        default="standard",
        help="Detail level for placement report diagnostics (default: standard)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Validate radius/center ───────────────────────────────────────────────
    if (args.radius is None) != (args.center is None):
        parser.error("--radius and --center must both be specified or neither")
    center = tuple(args.center) if args.center else None

    # ── Voxel mode (--fill is the legacy spelling of --voxel-mode solid) ─────
    if args.fill and args.voxel_mode not in (None, "solid"):
        parser.error("--fill conflicts with --voxel-mode " + args.voxel_mode)
    voxel_mode = args.voxel_mode or ("solid" if args.fill else "surface")

    # ── Asset resolver ───────────────────────────────────────────────────────
    if not args.asset_roots:
        print(
            "WARNING: No --asset-roots specified. Mesh resolution will fail for all shapes.\n"
            "         Use --debug-markers to see placement positions anyway.",
            file=sys.stderr,
        )
    print(f"Indexing assets from {len(args.asset_roots)} root(s)...")
    resolver = AssetResolver(args.asset_roots)
    print(f"  {resolver.total_files():,} files indexed.")

    # ── Parse OPD ───────────────────────────────────────────────────────────
    print(f"\nParsing OPD: {args.opd_file}")
    opd = parse_opd(args.opd_file)
    total_shapes = len(opd.shapes)

    # ── Build Minecraft world ────────────────────────────────────────────────
    world_name = args.world_name or Path(args.opd_file).stem
    out_root   = Path(args.output)
    world_dir  = str(out_root / world_name)

    print(f"\nCreating world '{world_name}' at: {world_dir}")
    world = MinecraftWorld(world_dir, world_name)

    # ── Parse GTD early (needed for building height lookups) ────────────────
    # h_grid must be available before processing shapes so that each building's
    # mc_y is derived from the same height array used by build_terrain.
    gtd       = None
    h_grid    = None
    gtd_path  = args.gtd
    if not args.no_terrain:
        if gtd_path is None:
            candidate = str(Path(args.opd_file).with_suffix(".gtd"))
            if Path(candidate).exists():
                gtd_path = candidate
        if gtd_path and Path(gtd_path).exists():
            from .gtd_parser import parse_gtd
            print(f"\nParsing terrain: {gtd_path}")
            gtd    = parse_gtd(gtd_path)
            h_grid = _compute_h_grid(gtd)
            print(f"  Terrain grid: {gtd.heightmap_size}×{gtd.heightmap_size}, "
                  f"h range [{gtd.heights.min():.1f}, {gtd.heights.max():.1f}] KO units")
        else:
            print("\nNo GTD terrain file found — building heights will use MC_Y_BASE.")

    # ── Converter ───────────────────────────────────────────────────────────
    converter = ZoneConverter(
        resolver        = resolver,
        world           = world,
        voxel_size      = args.voxel_size,
        block_name      = args.block,
        voxel_mode      = voxel_mode,
        debug_markers   = args.debug_markers,
        debug_transform = args.debug_transform,
        debug_grounding = args.debug_grounding,
        debug_voxelize  = args.debug_voxelize,
        map_size        = int(opd.map_width),
        h_grid          = h_grid,
        audit_report_level=args.audit_report_level,
    )

    # ── Process shapes FIRST (before terrain) ────────────────────────────────
    # Buildings are placed before terrain so that terrain fill does not bury
    # building blocks. build_terrain() skips positions already occupied.
    results = []
    skipped = 0
    processed = 0

    print(
        f"\nProcessing {total_shapes} shapes "
        f"(voxel_size={args.voxel_size}, voxel_mode={voxel_mode})..."
    )

    for i, shape in enumerate(opd.shapes):
        # Subset filters
        if not _passes_filters(shape, args.name_filter, center, args.radius):
            skipped += 1
            continue

        if args.limit is not None and processed >= args.limit:
            skipped += (total_shapes - i)
            break

        processed += 1
        if processed % 100 == 0:
            print(f"  [{processed}/{total_shapes}] Processing shape '{shape.name}'...")

        result = converter.convert_shape(shape, i)
        results.append(result)

    # ── Terrain SECOND (does not overwrite building blocks) ──────────────────
    terrain_stats: Optional[dict] = None
    if not args.no_terrain:
        if gtd is not None:
            print("\nPlacing terrain (skipping positions occupied by buildings)...")
            terrain_stats = {}
            build_terrain(gtd, world, fill_depth=args.terrain_fill_depth,
                          stats=terrain_stats)
        else:
            print("\nNo GTD terrain file found (use --gtd FILE to specify one).")

    # Skipped shapes get a lightweight result (not written to report body, just counted)
    # We record the count in the summary only.

    print(f"\nAll shapes processed. Saving world...")
    world.save()
    print(f"  World saved: {world_dir}")

    # ── Placement report ─────────────────────────────────────────────────────
    report = converter.build_report(
        opd_path     = str(Path(args.opd_file).resolve()),
        world_path   = str(Path(world_dir).resolve()),
        total_shapes = total_shapes,
        results      = results,
    )
    report.terrain_palette = terrain_stats
    # Add skipped count
    if skipped > 0:
        report.counts[ShapeStatus.SKIPPED.value] = skipped

    report.print_summary()

    report_path = args.report or str(out_root / world_name / "placement_report.json")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"\nPlacement report: {report_path}")
    print(f"\nDone. Copy '{world_name}' to Minecraft saves/ to inspect in-game.")


if __name__ == "__main__":
    main()
