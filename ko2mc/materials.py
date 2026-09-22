"""Terrain material classification.

The GTD file tells us which texture set each tile uses (for example
'map_mora_brick01' or 'map_el_ngrass03'). We don't ship the KO texture
images, so we classify each set by keywords in its name. Every material has:

  - a reference colour used to draw the KO preview, and
  - the Minecraft blocks used for the surface and the layers under it.

To change how a texture becomes a block, edit MATERIALS / TEXTURE_RULES below,
or pass a JSON file with --texture-map (see README).
"""

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    ko_color: tuple[int, int, int]  # approximate colour of the KO texture
    surface: str                    # Minecraft block on top
    subsurface: str                 # a few blocks under the surface
    deep: str = "minecraft:stone"   # everything below


MATERIALS = {
    "grass": Material("grass", (96, 128, 58), "minecraft:grass_block", "minecraft:dirt"),
    "dark_grass": Material("dark_grass", (70, 100, 48), "minecraft:moss_block", "minecraft:dirt"),
    "dirt": Material("dirt", (126, 101, 70), "minecraft:coarse_dirt", "minecraft:dirt"),
    "path": Material("path", (150, 124, 88), "minecraft:dirt_path", "minecraft:dirt"),
    "stone": Material("stone", (128, 124, 116), "minecraft:stone", "minecraft:stone"),
    "cobble": Material("cobble", (112, 108, 100), "minecraft:cobblestone", "minecraft:stone"),
    "brick": Material("brick", (150, 140, 124), "minecraft:stone_bricks", "minecraft:stone"),
    "sand": Material("sand", (206, 186, 136), "minecraft:sand", "minecraft:sandstone", "minecraft:sandstone"),
    "light_stone": Material("light_stone", (190, 190, 186), "minecraft:calcite", "minecraft:stone"),
    "snow": Material("snow", (236, 240, 245), "minecraft:snow_block", "minecraft:dirt"),
    "cave": Material("cave", (92, 84, 76), "minecraft:tuff", "minecraft:stone", "minecraft:deepslate"),
    "hidden": Material("hidden", (60, 60, 60), "minecraft:stone", "minecraft:stone"),
}

DEFAULT_MATERIAL = "grass"

# (regex, material) checked in order against the texture set name.
TEXTURE_RULES = [
    (r"invisible", "hidden"),
    (r"co_nsnow", "light_stone"),
    (r"snow|winter|ice", "snow"),
    (r"desert|sand|beach", "sand"),
    (r"brick|tile|pave", "brick"),
    (r"co_stone0[1-3]|road", "cobble"),
    (r"stone|rock|cliff", "stone"),
    (r"dun_|dungeon|cave", "cave"),
    (r"nground|ground|dirt|soil|mud", "dirt"),
    (r"ka_ngrass|ka_gr|dark", "dark_grass"),
    (r"grass|_gr\d|spring|field|jong", "grass"),
]

WATER_BLOCK = "minecraft:water"
WATER_KO_COLOR = (58, 96, 128)


def load_texture_map(path: str) -> None:
    """Prepend user rules from a JSON file: {"regex": "material_or_block", ...}.

    A value can be a material name (e.g. "sand") or a Minecraft block id
    (e.g. "minecraft:red_sand"), which creates a new material on the fly.
    """
    with open(path, encoding="utf-8") as f:
        user = json.load(f)
    rules = []
    for pattern, target in user.items():
        if target not in MATERIALS:
            block = target if ":" in target else f"minecraft:{target}"
            MATERIALS[target] = Material(target, (128, 128, 128), block, "minecraft:dirt")
        rules.append((pattern, target))
    TEXTURE_RULES[:0] = rules


def classify_texture(name: str) -> str:
    """Return the material name for a KO texture set name."""
    name = name.lower()
    if not name:
        return DEFAULT_MATERIAL
    for pattern, material in TEXTURE_RULES:
        if re.search(pattern, name):
            return material
    return DEFAULT_MATERIAL


def material_grid(gtd):
    """Return (material_names, index_grid) for every heightmap vertex of a GTDFile.

    index_grid[x, z] indexes into material_names.
    """
    import numpy as np

    names = list(MATERIALS)
    lookup = np.full(1024, names.index(DEFAULT_MATERIAL), dtype=np.uint8)
    for i, tex in enumerate(gtd.tile_textures[:1023]):
        lookup[i] = names.index(classify_texture(tex))
    return names, lookup[np.clip(gtd.tex1, 0, 1023)]
