from pathlib import Path

from ko2mc.asset_resolver import AssetResolver
from ko2mc.zone_converter import _block_name_for_part, _terrain_surface_block, KO_NOTMOVE_HEIGHT


def test_block_name_for_part_prefers_textures():
    blk = _block_name_for_part("obj_castle_wall", ["wall_stone_a.dxt"])
    assert blk == "minecraft:stone_bricks"

    blk = _block_name_for_part("obj_generic", ["window_glass_01.dxt"])
    assert blk == "minecraft:glass"


def test_terrain_surface_block_uses_texture_id_before_height_defaults():
    # tex_id=6 maps to snow, slope/height are normal land.
    assert _terrain_surface_block(ko_h=30.0, slope=1.0, tex_id=6) == "minecraft:snow_block"

    # steep slope always stays cliff regardless of texture id.
    assert _terrain_surface_block(ko_h=30.0, slope=KO_NOTMOVE_HEIGHT + 1.0, tex_id=6) == "minecraft:stone"


def test_resolve_with_meta_reports_basename_ambiguity(tmp_path: Path):
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    (r1 / "Object").mkdir(parents=True)
    (r2 / "Object").mkdir(parents=True)
    (r1 / "Object" / "dup.n3pmesh").write_text("a", encoding="utf-8")
    (r2 / "Object" / "dup.n3pmesh").write_text("b", encoding="utf-8")

    resolver = AssetResolver([str(r1), str(r2)])
    path, meta = resolver.resolve_with_meta("dup.n3pmesh")

    assert path is not None
    assert meta["strategy"] == "basename"
    assert meta["candidate_count"] >= 2
    assert meta["ambiguous"] is True
