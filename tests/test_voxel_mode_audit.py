"""Tests for --voxel-mode, palette-fallback telemetry and grounding metrics."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ko2mc.math3d import MC_Y_BASE
from ko2mc.opd_parser import Shape, ShapePart, Vector3
from ko2mc.voxelizer import voxelize_mesh
from ko2mc.zone_converter import (
    PALETTE_FALLBACK_NO_TEX,
    PALETTE_FALLBACK_UNKNOWN,
    PALETTE_TEXTURE,
    VOXEL_MODES,
    ZoneConverter,
    _block_for_part_with_source,
    build_terrain,
    grounding_quality,
    voxel_mode_options,
)


def _cube(lo: float, hi: float):
    c = [(x, y, z) for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)]
    idx = {p: i for i, p in enumerate(c)}
    quads = [
        [(lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo)],
        [(lo, lo, hi), (hi, lo, hi), (hi, hi, hi), (lo, hi, hi)],
        [(lo, lo, lo), (hi, lo, lo), (hi, lo, hi), (lo, lo, hi)],
        [(lo, hi, lo), (hi, hi, lo), (hi, hi, hi), (lo, hi, hi)],
        [(lo, lo, lo), (lo, hi, lo), (lo, hi, hi), (lo, lo, hi)],
        [(hi, lo, lo), (hi, hi, lo), (hi, hi, hi), (hi, lo, hi)],
    ]
    tris = []
    for q in quads:
        a, b, c2, d = (idx[p] for p in q)
        tris += [a, b, c2, a, c2, d]
    return np.array(c, dtype=np.float32), np.array(tris, dtype=np.uint16)


def _nested_cubes():
    """Three nested cubes one voxel apart: a 3-voxel-thick wall around a cavity."""
    parts = [_cube(0.0, 8.0), _cube(1.0, 7.0), _cube(2.0, 6.0)]
    verts, idx, off = [], [], 0
    for v, i in parts:
        verts.append(v)
        idx.append(i + off)
        off += len(v)
    return np.vstack(verts), np.concatenate(idx)


# ── voxel modes ────────────────────────────────────────────────────────────────

def test_voxel_mode_options_mapping():
    assert voxel_mode_options("surface") == {"fill": False, "surface_only": True}
    assert voxel_mode_options("hybrid") == {"fill": False, "surface_only": False}
    assert voxel_mode_options("solid") == {"fill": True, "surface_only": False}
    with pytest.raises(ValueError):
        voxel_mode_options("bogus")


def test_voxel_modes_produce_increasing_occupancy():
    verts, idx = _nested_cubes()
    counts = {
        m: voxelize_mesh(verts, idx, 1.0, **voxel_mode_options(m)).count_filled()
        for m in VOXEL_MODES
    }
    # surface strips voxels enclosed by other voxels; hybrid keeps raw SAT hits;
    # solid additionally fills enclosed air.
    assert counts["surface"] < counts["hybrid"] < counts["solid"]


def test_converter_voxel_mode_and_legacy_fill_flag():
    assert ZoneConverter(resolver=MagicMock(), world=MagicMock()).voxel_mode == "surface"
    conv = ZoneConverter(resolver=MagicMock(), world=MagicMock(), fill=True)
    assert conv.voxel_mode == "solid" and conv.fill is True
    conv = ZoneConverter(resolver=MagicMock(), world=MagicMock(), voxel_mode="hybrid")
    assert conv.voxel_mode == "hybrid" and conv.fill is False


# ── palette telemetry ─────────────────────────────────────────────────────────

def test_block_for_part_reports_palette_source():
    assert _block_for_part_with_source("obj_x", ["wall_stone.dxt"]) == (
        "minecraft:stone_bricks", PALETTE_TEXTURE)
    assert _block_for_part_with_source("obj_wood_hut", []) == (
        "minecraft:oak_planks", PALETTE_FALLBACK_NO_TEX)
    blk, src = _block_for_part_with_source("obj_x", ["zz_unknown_07.dxt"])
    assert src == PALETTE_FALLBACK_UNKNOWN
    assert blk == "minecraft:stone_bricks"


class _MissingResolver:
    """Resolver stub that never finds a mesh."""
    missing_refs: list = []

    def resolve(self, name):
        return None

    def resolve_with_meta(self, name):
        return None, {"strategy": None, "candidate_count": 0, "ambiguous": False}


def _flat_h_grid(ko_height: float = 0.0, n: int = 65):
    return np.full((n, n), ko_height, dtype=np.float32)


def test_report_includes_palette_fallback_and_grounding_metrics():
    conv = ZoneConverter(resolver=_MissingResolver(), world=MagicMock(),
                         h_grid=_flat_h_grid(), map_size=256)
    shape = Shape(name="obj_building", position=Vector3(100.0, 0.0, 100.0))
    shape.parts = [
        ShapePart(name="a.n3pmesh", textures=["mystery_tex.dxt"]),
        ShapePart(name="b.n3pmesh", textures=["castle_wall.dxt"]),
        ShapePart(name="c.n3pmesh"),
    ]
    result = conv.convert_shape(shape, 0)
    assert conv.unknown_textures == {"mystery_tex.dxt": 1}
    assert result.grounding_quality["support_ratio"] == 1.0

    report = conv.build_report("x.opd", "world", 1, [result])
    d = json.loads(json.dumps(report.to_dict()))
    assert d["voxel_mode"] == "surface"
    assert d["palette"]["parts_evaluated"] == 3
    assert d["palette"]["fallback_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert d["palette"]["unknown_textures"] == [{"texture": "mystery_tex.dxt", "parts": 1}]
    assert d["grounding_summary"]["shapes_evaluated"] == 1
    assert d["grounding_summary"]["mean_support_ratio"] == 1.0
    g = d["shapes"][0]["grounding"]
    assert {"support_ratio", "buried_ratio", "floating_ratio"} <= g.keys()
    assert "samples" not in g  # standard level omits sample counts
    assert "textures" not in d["shapes"][0]["parts"][0]

    report.audit_report_level = "detailed"
    d = report.to_dict()
    assert d["shapes"][0]["grounding"]["samples"] == 25
    assert d["shapes"][0]["parts"][0]["textures"] == ["mystery_tex.dxt"]


# ── grounding metrics ─────────────────────────────────────────────────────────

def test_grounding_quality_ratios():
    h = _flat_h_grid(ko_height=10.0)
    ground = MC_Y_BASE + 10
    assert grounding_quality(None, 50, 50, ground, 4, 4) is None
    q = grounding_quality(h, 50, 50, ground, 8, 8)
    assert q["support_ratio"] == 1.0 and q["samples"] == 25
    assert grounding_quality(h, 50, 50, ground - 5, 8, 8)["buried_ratio"] == 1.0
    assert grounding_quality(h, 50, 50, ground + 5, 8, 8)["floating_ratio"] == 1.0


def test_grounding_quality_on_slope_is_mixed():
    n = 65
    # Height rises along MC X: 0 at x=0 up to 64 KO units at the far edge.
    h = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, n))
    base = MC_Y_BASE + 20  # terrain height at mc_x = 80 (tile 20)
    q = grounding_quality(h, 80, 80, base, 16, 4)
    assert q["support_ratio"] > 0
    assert q["buried_ratio"] > 0
    assert q["floating_ratio"] > 0
    assert q["support_ratio"] + q["buried_ratio"] + q["floating_ratio"] == pytest.approx(1.0)


# ── terrain palette telemetry ─────────────────────────────────────────────────

def test_build_terrain_reports_unmapped_texture_ids():
    n = 3
    gtd = SimpleNamespace(
        heightmap_size=n,
        heights=np.zeros((n, n), dtype=np.float32),
        texture_ids=np.array([[0, 20], [20, 20]], dtype=np.int32),
    )
    world = MagicMock()
    world.has_block.return_value = True  # skip actual placement
    stats: dict = {}
    build_terrain(gtd, world, fill_depth=1, stats=stats)
    assert stats["unmapped_tex_ids"] == {"20": 48}
    assert stats["fallback_rate"] == pytest.approx(0.75)
    assert stats["unique_tex_ids"] == 2
