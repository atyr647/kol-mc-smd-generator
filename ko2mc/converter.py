"""Converts Knight Online map data to a Minecraft world.

Coordinate systems
------------------
KO (Direct3D, left-handed): x = east, y = up, z = NORTH, units = meters.
Minecraft:                  x = east, y = up, z = SOUTH, units = blocks.

So the KO z axis is flipped, otherwise the whole map comes out mirrored.
With `scale` blocks per 4 m KO tile:

    mc_x = ko_x * scale / 4
    mc_z = (map_size_m - ko_z) * scale / 4
    mc_y = y_offset + ko_y * vertical_scale     (vertical_scale defaults to scale / 4,
                                                 so hills keep their real proportions)

The terrain is sampled per Minecraft column with bilinear interpolation (like
the KO client renders it), filled solid down to bedrock, textured by the tile
texture names, and flooded where KO ponds/rivers are. Buildings and walls come
from the server collision mesh in the .opd file; trees, bushes, rocks, lamps,
etc. are placed from the object list.
"""

import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np

from . import materials
from .gtd_parser import UNIT_DISTANCE, GTDFile, parse_gtd
from .mc_world import MAX_Y, MIN_Y, MinecraftWorld
from .opd_parser import (
    EVENT_TYPE_NAMES,
    OBJECT_ANVIL,
    OBJECT_ARTIFACT,
    OBJECT_BARRICADE,
    OBJECT_BIND,
    OBJECT_FLAG_LEVER,
    OBJECT_GATE,
    OBJECT_GATE2,
    OBJECT_GATE_LEVER,
    OBJECT_REMOVE_BIND,
    OBJECT_WARP_GATE,
    OPDFile,
    Shape,
    parse_opd,
)

# Keep the map inside the buildable height range with some headroom.
TERRAIN_Y_MIN = MIN_Y + 16   # lowest allowed terrain surface
TERRAIN_Y_MAX = MAX_Y - 40   # highest allowed terrain surface (room for trees/buildings)
PREFERRED_ZERO_Y = 64        # KO height 0 goes here when the map fits

UNDERWATER_SURFACE = {
    "grass": "minecraft:dirt", "dark_grass": "minecraft:dirt", "snow": "minecraft:dirt",
    "path": "minecraft:dirt",
}

# Collision-mesh voxels closer than this (meters) to invisible "boundary walls" are skipped.
_COLLISION_Y_LIMIT = 4000.0


@dataclass
class CoordMap:
    """Converts between KO meters and Minecraft block coordinates."""
    scale: int                 # blocks per KO tile (4 m)
    map_size_m: float
    vertical_scale: float      # blocks per KO meter (vertical)
    y_offset: float            # mc_y of KO height 0

    @property
    def blocks_per_meter(self) -> float:
        return self.scale / UNIT_DISTANCE

    @property
    def size_blocks(self) -> int:
        return int(round(self.map_size_m * self.blocks_per_meter))

    def x(self, ko_x):
        return np.asarray(ko_x) * self.blocks_per_meter

    def z(self, ko_z):
        return (self.map_size_m - np.asarray(ko_z)) * self.blocks_per_meter

    def y(self, ko_y):
        return self.y_offset + np.asarray(ko_y) * self.vertical_scale

    def to_json(self) -> dict:
        return {"scale": self.scale, "map_size_m": self.map_size_m,
                "vertical_scale": self.vertical_scale, "y_offset": self.y_offset}

    @staticmethod
    def for_map(gtd: GTDFile, scale: int, vertical_scale: float | None = None) -> "CoordMap":
        v = vertical_scale if vertical_scale else scale / UNIT_DISTANCE
        hmin, hmax = float(gtd.heights.min()), float(gtd.heights.max())
        room = TERRAIN_Y_MAX - TERRAIN_Y_MIN
        if (hmax - hmin) * v > room:
            new_v = room / (hmax - hmin)
            print(f"  Note: terrain is {hmax - hmin:.0f} m tall; squashing vertical scale "
                  f"{v:.3f} -> {new_v:.3f} blocks/m so it fits in Minecraft's height limit.")
            v = new_v
        offset = PREFERRED_ZERO_Y
        if offset + hmax * v > TERRAIN_Y_MAX:
            offset = TERRAIN_Y_MAX - hmax * v
        if offset + hmin * v < TERRAIN_Y_MIN:
            offset = TERRAIN_Y_MIN - hmin * v
        return CoordMap(scale, gtd.size_meters, v, float(offset))


def _bilinear(grid: np.ndarray, fx: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """Sample grid[x, z] at fractional indices (broadcast arrays)."""
    n = grid.shape[0]
    fx = np.clip(fx, 0, n - 1.0001)
    fz = np.clip(fz, 0, n - 1.0001)
    x0 = fx.astype(np.int32)
    z0 = fz.astype(np.int32)
    tx = fx - x0
    tz = fz - z0
    return (grid[x0, z0] * (1 - tx) * (1 - tz) + grid[x0 + 1, z0] * tx * (1 - tz)
            + grid[x0, z0 + 1] * (1 - tx) * tz + grid[x0 + 1, z0 + 1] * tx * tz)


class TerrainModel:
    """Per-column terrain data on the Minecraft grid, indexed [mc_z, mc_x]."""

    def __init__(self, gtd: GTDFile, cm: CoordMap, world: MinecraftWorld, texture_pack=None, library=None):
        self.cm = cm
        size = cm.size_blocks
        s = cm.scale
        cols = np.arange(size) + 0.5
        # KO heightmap indices for each MC column centre
        fx = cols / s                       # along x
        fz = (size - cols) / s              # MC z row j -> KO z index (flipped)
        FX, FZ = np.meshgrid(fx, fz)        # shape [z, x]
        heights = _bilinear(gtd.heights, FX, FZ)
        self.top = np.floor(cm.y(heights)).astype(np.int32)   # top solid block y
        self.top = np.clip(self.top, MIN_Y + 1, MAX_Y - 1)

        names, mat_grid = materials.material_grid(gtd)
        n = gtd.heightmap_size
        tx = np.clip(np.floor(FX).astype(np.int32), 0, n - 1)
        tz = np.clip(np.floor(FZ).astype(np.int32), 0, n - 1)
        self.material = mat_grid[tx, tz]
        self.material_names = names
        self.tex = gtd.tex1[tx, tz]

        # KO textures via the resource pack: tile texture index -> note block state
        self.col_custom = None   # per column KO ground block (note block state), 0 = none
        self.col_below = None
        if texture_pack is not None and library is not None:
            self._assign_textures(gtd, world, texture_pack, library)

        # Water level (top water block y) per column, or MIN_Y if none
        self.water_top = np.full((size, size), MIN_Y, dtype=np.int32)
        for mesh in gtd.water:
            self._rasterize_water(mesh.triangles())
        self.water_top[self.water_top <= self.top] = MIN_Y

        # Block ids per material
        mats = [materials.MATERIALS[m] for m in names]
        self.surface_ids = np.array([world.block_id(m.surface) for m in mats], dtype=np.uint16)
        self.wet_surface_ids = np.array(
            [world.block_id(UNDERWATER_SURFACE.get(m.name, m.surface)) for m in mats], dtype=np.uint16)
        self.sub_ids = np.array([world.block_id(m.subsurface) for m in mats], dtype=np.uint16)
        self.deep_ids = np.array([world.block_id(m.deep) for m in mats], dtype=np.uint16)
        self.bedrock = world.block_id("minecraft:bedrock")
        self.water = world.block_id(materials.WATER_BLOCK)

    def _assign_textures(self, gtd, world, pack, library):
        """Give every column a piece of the real KO ground (base + overlay texture,
        rotated like KO does), grouped into as many textures as there are note block states."""
        from .ko_ground import GroundBuilder
        from .ko_textures import MAX_CUSTOM, custom_state
        gb = GroundBuilder(gtd, library, 1.0)   # the pack applies the brightness
        missing = sorted({gtd.tile_files[i] for i in np.unique(gb.t1)
                          if i < len(gtd.tile_files) and int(i) not in gb.found})
        s = self.cm.scale
        group, reps = gb.block_textures(s, MAX_CUSTOM)
        if group is None:
            print("  KO textures: none of this map's ground textures were found")
            return
        ids, below = [], []
        for g, img in enumerate(reps):
            slot = pack.add(img, f"ground piece {g}") if img is not None else None
            if slot is None:
                ids.append(0)
                below.append(0)
                continue
            state, under = custom_state(slot)
            ids.append(world.block_id(state))
            below.append(world.block_id(under))
        ids = np.array(ids + [0], np.uint16)          # index -1 -> 0 (no KO texture)
        below = np.array(below + [0], np.uint16)
        size = self.cm.size_blocks
        n = gtd.heightmap_size - 1
        i = np.arange(size)
        tx = np.clip(i // s, 0, n - 1)
        tz = np.clip((size - 1 - i) // s, 0, n - 1)    # MC row j -> KO tile z
        sub_x = i % s
        sub_z = (i - (size - (tz + 1) * s)) % s          # rows run north -> south in a tile
        g = group[tx[None, :], tz[:, None], sub_x[None, :], sub_z[:, None]]   # [z, x]
        self.col_custom = ids[g]
        self.col_below = below[g]
        print(f"  KO ground: {len(reps)} block textures from {len(np.unique(g))} pieces "
              f"(scale {s}: each KO tile = {s}x{s} blocks)")
        if missing:
            print(f"  Missing: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")

    def _rasterize_water(self, tris_ko: np.ndarray):
        cm = self.cm
        size = cm.size_blocks
        xs = cm.x(tris_ko[:, :, 0])
        zs = cm.z(tris_ko[:, :, 2])
        ys = cm.y(tris_ko[:, :, 1])
        for (ax, bx, cx), (az, bz, cz), (ay, by, cy) in zip(xs, zs, ys):
            i0 = max(int(math.floor(min(ax, bx, cx))), 0)
            i1 = min(int(math.ceil(max(ax, bx, cx))), size - 1)
            j0 = max(int(math.floor(min(az, bz, cz))), 0)
            j1 = min(int(math.ceil(max(az, bz, cz))), size - 1)
            if i1 < i0 or j1 < j0:
                continue
            det = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
            if abs(det) < 1e-9:
                continue
            px, pz = np.meshgrid(np.arange(i0, i1 + 1) + 0.5, np.arange(j0, j1 + 1) + 0.5)
            w1 = ((bz - cz) * (px - cx) + (cx - bx) * (pz - cz)) / det
            w2 = ((cz - az) * (px - cx) + (ax - cx) * (pz - cz)) / det
            w3 = 1 - w1 - w2
            inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6)
            level = np.floor(w1 * ay + w2 * by + w3 * cy).astype(np.int32)
            region = self.water_top[j0:j1 + 1, i0:i1 + 1]
            np.maximum(region, np.where(inside, level, MIN_Y), out=region)

    def fill_chunk(self, cx: int, cz: int, out: np.ndarray):
        size = self.cm.size_blocks
        x0, z0 = cx * 16, cz * 16
        if x0 >= size or z0 >= size or x0 + 16 <= 0 or z0 + 16 <= 0:
            return
        xa, xb = max(x0, 0), min(x0 + 16, size)
        za, zb = max(z0, 0), min(z0 + 16, size)
        top = self.top[za:zb, xa:xb]
        mat = self.material[za:zb, xa:xb]
        wtop = self.water_top[za:zb, xa:xb]
        wet = wtop > top

        ys = (np.arange(out.shape[0]) + MIN_Y)[:, None, None]
        view = out[:, za - z0:zb - z0, xa - x0:xb - x0]
        surface = np.where(wet, self.wet_surface_ids[mat], self.surface_ids[mat])
        if self.col_custom is not None:
            custom = self.col_custom[za:zb, xa:xb]
            surface = np.where(custom > 0, custom, surface)
            sub = np.where(ys == top - 1, np.where(custom > 0, self.col_below[za:zb, xa:xb],
                                                   self.sub_ids[mat]), self.sub_ids[mat])
        else:
            sub = self.sub_ids[mat]
        view[:] = np.where(ys < top - 3, self.deep_ids[mat], np.where(ys < top, sub, surface))
        view[ys > top] = 0
        water = (ys > top) & (ys <= wtop)
        view[water] = self.water
        view[0] = self.bedrock

    def top_at(self, x: int, z: int) -> int:
        size = self.cm.size_blocks
        return int(self.top[min(max(z, 0), size - 1), min(max(x, 0), size - 1)])


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------

LEAVES = "minecraft:{}_leaves[persistent=true]"

# (regex, kind) checked in order against the lower-case object name
OBJECT_RULES = [
    (r"fx|smoke|fog|smog|collisioncube|alpha|effect", None),
    (r"xmas|snowtree|chim|conifer|pine|fir\b|spruce", "tree_spruce"),
    (r"palm", "tree_palm"),
    (r"tree|_tr_|_tr\d|trred|dtree|bigtree", "tree"),
    (r"dumbul|bush|_ip\d|_ip$|leaf", "bush"),
    (r"sunflower", "sunflower"),
    (r"flower|flw|pollen", "flower"),
    (r"reed|cactus", "reed"),
    (r"grass|gass|gras|plant|dunkul|vine", "grass"),
    (r"mushrom|mushroom", "mushroom"),
    (r"pumpkin", "pumpkin"),
    (r"ston|rock|boulder|mineral|iceston", "rock"),
    (r"lamp|light|garo|toch|torch|tourou|torou", "lamp"),
    (r"fire|bonfire|firpot", "fire"),
    (r"flag", "flag"),
    (r"fence", "fence"),
]

_FLOWERS = ["minecraft:poppy", "minecraft:dandelion", "minecraft:cornflower",
            "minecraft:oxeye_daisy", "minecraft:azure_bluet", "minecraft:allium"]

# Collision mesh material by nearest object name
COLLISION_RULES = [
    (r"tree|_tr_|dumbul|bush|reed|grass|gass|plant|flower", None),  # vegetation handled as objects
    (r"ston|rock|mountain|cliff|mineral|sandhil|rockwal", ("minecraft:stone", "minecraft:stone")),
    (r"wood|bridge|board|fence|box|ship|cart|wagun|bench|table|tent", ("minecraft:oak_planks", "minecraft:spruce_planks")),
    (r"snw|ice", ("minecraft:packed_ice", "minecraft:snow_block")),
    (r"sand", ("minecraft:sandstone", "minecraft:smooth_sandstone")),
    (r"zip|house|haus|home|shop|bill|warehouse|room", ("minecraft:stone_bricks", "minecraft:spruce_planks")),
    (r"wal|castl|catl|gate|tower|tow_|fort|pillar|post", ("minecraft:stone_bricks", "minecraft:polished_andesite")),
]
DEFAULT_COLLISION_BLOCKS = ("minecraft:stone_bricks", "minecraft:polished_andesite")


def classify_object(name: str) -> str | None:
    n = name.lower()
    for pattern, kind in OBJECT_RULES:
        if re.search(pattern, n):
            return kind
    return None


def _hash01(*vals) -> float:
    """Deterministic pseudo random number in [0, 1) from numbers."""
    h = 2166136261
    for v in vals:
        h = ((h ^ (int(v * 1000) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return (h % 10007) / 10007.0


class ObjectPlacer:
    def __init__(self, world: MinecraftWorld, terrain: TerrainModel, cm: CoordMap):
        self.world = world
        self.terrain = terrain
        self.cm = cm
        self.counts: dict[str, int] = {}

    def _base(self, shape: Shape) -> tuple[int, int, int]:
        cm = self.cm
        x = int(math.floor(float(cm.x(shape.position.x))))
        z = int(math.floor(float(cm.z(shape.position.z))))
        ground = self.terrain.top_at(x, z) + 1
        y = int(math.floor(float(cm.y(shape.position.y)))) + 1
        # Objects near or below the ground sit on it; objects on bridges/roofs keep their height.
        if y < ground + 2:
            y = ground
        return x, y, z

    def size_blocks(self, meters: float, minimum: int = 1) -> int:
        return max(minimum, int(round(meters * self.cm.blocks_per_meter)))

    def place(self, shape: Shape):
        if shape.is_event_object:
            self._place_event(shape)
            return
        kind = classify_object(shape.name)
        if kind is None:
            return
        x, y, z = self._base(shape)
        size = max(abs(shape.scale.x), abs(shape.scale.y), abs(shape.scale.z), 0.3)
        r = _hash01(shape.position.x, shape.position.z)
        w = self.world
        if kind.startswith("tree"):
            self._tree(x, y, z, size, kind, r)
        elif kind == "bush":
            rad = self.size_blocks(1.2 * size) - 1
            self._blob(x, y + (0 if rad == 0 else rad - 1), z, rad, LEAVES.format("oak" if r < 0.6 else "azalea"))
        elif kind == "grass":
            if self.cm.scale >= 2 or r < 0.3:
                w.set_block(x, y, z, "minecraft:short_grass" if r < 0.7 else "minecraft:fern")
        elif kind == "flower":
            w.set_block(x, y, z, _FLOWERS[int(r * len(_FLOWERS))])
        elif kind == "sunflower":
            w.set_block(x, y, z, "minecraft:sunflower[half=lower]")
            w.set_block(x, y + 1, z, "minecraft:sunflower[half=upper]")
        elif kind == "reed":
            for dy in range(self.size_blocks(2.0 * size)):
                w.set_block(x, y + dy, z, "minecraft:sugar_cane")
        elif kind == "mushroom":
            w.set_block(x, y, z, "minecraft:red_mushroom" if r < 0.5 else "minecraft:brown_mushroom")
        elif kind == "pumpkin":
            w.set_block(x, y, z, "minecraft:pumpkin")
        elif kind == "rock":
            rad = self.size_blocks(1.5 * size) - 1
            self._blob(x, y - 1, z, rad, "minecraft:stone" if r < 0.5 else "minecraft:andesite",
                       squash=0.7)
        elif kind == "lamp":
            h = self.size_blocks(3.5 * size, 2)
            for dy in range(h - 1):
                w.set_block(x, y + dy, z, "minecraft:dark_oak_fence")
            w.set_block(x, y + h - 1, z, "minecraft:lantern")
        elif kind == "fire":
            w.set_block(x, y, z, "minecraft:campfire")
        elif kind == "flag":
            h = self.size_blocks(5.0 * size, 2)
            for dy in range(h - 1):
                w.set_block(x, y + dy, z, "minecraft:oak_fence")
            w.set_block(x, y + h - 1, z, "minecraft:white_banner[rotation=0]")
        elif kind == "fence":
            w.set_block(x, y, z, "minecraft:oak_fence")
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _tree(self, x, y, z, size, kind, r):
        w = self.world
        height = self.size_blocks((9.0 + 4.0 * r) * min(size, 3.0), 3)
        if kind == "tree_spruce":
            wood, leaves = "minecraft:spruce_log[axis=y]", LEAVES.format("spruce")
        elif kind == "tree_palm":
            wood, leaves = "minecraft:jungle_log[axis=y]", LEAVES.format("jungle")
        else:
            wood = "minecraft:oak_log[axis=y]" if r < 0.7 else "minecraft:birch_log[axis=y]"
            leaves = LEAVES.format("oak" if r < 0.7 else "birch")
        if height <= 3:
            # Tiny trees at small scales: one trunk block and a leaf cap
            w.set_block(x, y, z, wood)
            self._blob(x, y + 1, z, 1 if height == 3 else 0, leaves)
            return
        if kind == "tree_spruce":
            for dy in range(1, height):
                rad = max(0, int((height - dy) * 0.35 + 0.5)) if dy >= 2 else 0
                if rad:
                    self._disc(x, y + dy, z, rad, leaves)
            w.set_block(x, y + height, z, leaves)
        elif kind == "tree_palm":
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
                w.set_block(x + dx, y + height - 1, z + dz, leaves)
            w.set_block(x, y + height, z, leaves)
        else:
            rad = max(1, height // 3)
            self._blob(x, y + height - rad // 2, z, rad, leaves)
        w.set_blocks([x] * (height - 1), np.arange(y, y + height - 1), [z] * (height - 1), wood)

    def _disc(self, x, y, z, rad, block):
        d = np.arange(-rad, rad + 1)
        dx, dz = np.meshgrid(d, d)
        m = dx * dx + dz * dz <= rad * rad + rad
        self.world.set_blocks(x + dx[m], np.full(m.sum(), y), z + dz[m], block)

    def _blob(self, x, y, z, rad, block, squash=1.0):
        if rad <= 0:
            self.world.set_block(x, y, z, block)
            return
        d = np.arange(-rad, rad + 1)
        dx, dy, dz = np.meshgrid(d, d, d, indexing="ij")
        m = dx * dx + (dy / squash) ** 2 + dz * dz <= rad * rad + 0.5
        self.world.set_blocks(x + dx[m], y + dy[m], z + dz[m], block)

    def _place_event(self, shape: Shape):
        x, y, z = self._base(shape)
        w = self.world
        t = shape.event_type
        if t == OBJECT_WARP_GATE:
            for dx in range(-1, 2):
                for dz in range(-1, 2):
                    if dx or dz:
                        w.set_block(x + dx, y - 1, z + dz, "minecraft:end_portal_frame[eye=true,facing=north]")
            w.set_block(x, y - 1, z, "minecraft:crying_obsidian")
            w.set_block(x, y, z, "minecraft:end_rod")
        elif t in (OBJECT_GATE, OBJECT_GATE2):
            w.fill_box(x - 2, y, z, x + 2, y + 3, z, "minecraft:iron_bars")
        elif t == OBJECT_BARRICADE:
            w.fill_box(x - 2, y, z, x + 2, y + 1, z, "minecraft:oak_fence")
        elif t in (OBJECT_BIND, OBJECT_REMOVE_BIND):
            w.fill_box(x - 1, y - 1, z - 1, x + 1, y - 1, z + 1, "minecraft:gold_block")
            w.set_block(x, y, z, "minecraft:respawn_anchor[charges=4]")
        elif t == OBJECT_ANVIL:
            w.set_block(x, y, z, "minecraft:anvil[facing=north]")
        elif t == OBJECT_ARTIFACT:
            w.fill_box(x - 1, y - 1, z - 1, x + 1, y - 1, z + 1, "minecraft:iron_block")
            w.set_block(x, y, z, "minecraft:beacon")
        elif t in (OBJECT_GATE_LEVER, OBJECT_FLAG_LEVER):
            w.set_block(x, y, z, "minecraft:lever[face=floor,facing=north,powered=false]")
        else:
            w.set_block(x, y, z, "minecraft:glowstone")
        self.counts["event"] = self.counts.get("event", 0) + 1


# ---------------------------------------------------------------------------
# Collision mesh (buildings, walls, bridges)
# ---------------------------------------------------------------------------

def collision_triangles(opd: OPDFile) -> np.ndarray:
    """Return usable collision triangles (n, 3, 3) in KO meters."""
    if not opd.collision_vertices:
        return np.zeros((0, 3, 3), dtype=np.float32)
    v = np.array([(p.x, p.y, p.z) for p in opd.collision_vertices], dtype=np.float32)
    tris = v[: len(v) // 3 * 3].reshape(-1, 3, 3)
    ok = np.all(np.isfinite(tris), axis=(1, 2)) & (np.abs(tris[:, :, 1]).max(axis=1) < _COLLISION_Y_LIMIT)
    return tris[ok]


def _nearest_shape_names(opd: OPDFile, points_xz: np.ndarray, cell: float = 16.0) -> list[str]:
    """Name of the nearest (non-effect) object for each point, using a grid lookup."""
    shapes = [s for s in opd.shapes if not re.search(r"fx|smoke|fog|zz_", s.name.lower())]
    if not shapes:
        return [""] * len(points_xz)
    pos = np.array([(s.position.x, s.position.z) for s in shapes], dtype=np.float32)
    keys = np.floor(pos / cell).astype(np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (kx, kz) in enumerate(keys):
        buckets.setdefault((int(kx), int(kz)), []).append(i)
    names = []
    for px, pz in points_xz:
        kx, kz = int(px // cell), int(pz // cell)
        best, best_d = -1, 1e18
        for ring in range(0, 4):
            for dx in range(-ring, ring + 1):
                for dz in range(-ring, ring + 1):
                    if max(abs(dx), abs(dz)) != ring:
                        continue
                    for i in buckets.get((kx + dx, kz + dz), ()):
                        d = (pos[i, 0] - px) ** 2 + (pos[i, 1] - pz) ** 2
                        if d < best_d:
                            best, best_d = i, d
            if best >= 0:
                break
        names.append(shapes[best].name.lower() if best >= 0 else "")
    return names


def voxelize_collision(opd: OPDFile, terrain: TerrainModel, world: MinecraftWorld, cm: CoordMap):
    tris = collision_triangles(opd)
    if len(tris) == 0:
        return 0
    names = _nearest_shape_names(opd, tris[:, :, [0, 2]].mean(axis=1))
    blocks = []
    for n in names:
        chosen = DEFAULT_COLLISION_BLOCKS
        for pattern, pair in COLLISION_RULES:
            if re.search(pattern, n):
                chosen = pair
                break
        blocks.append(chosen)

    # To Minecraft coordinates
    mc = np.stack([cm.x(tris[:, :, 0]), cm.y(tris[:, :, 1]), cm.z(tris[:, :, 2])], axis=-1)
    e1 = mc[:, 1] - mc[:, 0]
    e2 = mc[:, 2] - mc[:, 0]
    normal = np.cross(e1, e2)
    nlen = np.linalg.norm(normal, axis=1) + 1e-9
    flat = np.abs(normal[:, 1]) / nlen > 0.7
    edge = np.max(np.stack([np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1),
                            np.linalg.norm(mc[:, 2] - mc[:, 1], axis=1)]), axis=0)
    steps = np.clip(np.ceil(edge / 0.45).astype(np.int32), 1, 400)

    size = cm.size_blocks
    placed = 0
    all_x, all_y, all_z, all_id = [], [], [], []
    for k in np.unique(steps):
        idx = np.flatnonzero(steps == k)
        a, b = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        m = a + b <= k
        bary = np.stack([a[m], b[m]], axis=1).astype(np.float32) / k   # (P, 2)
        for chunk in np.array_split(idx, max(1, len(idx) * len(bary) // 2_000_000 + 1)):
            t = mc[chunk]
            pts = (t[:, None, 0] + bary[None, :, 0:1] * (t[:, None, 1] - t[:, None, 0])
                   + bary[None, :, 1:2] * (t[:, None, 2] - t[:, None, 0]))
            vox = np.floor(pts).astype(np.int32)
            ids = np.array([0 if blocks[i] is None else
                            world.block_id(blocks[i][1] if flat[i] else blocks[i][0])
                            for i in chunk], dtype=np.uint16)
            use = np.repeat(ids > 0, len(bary))
            ids = np.repeat(ids, len(bary))
            vox = vox.reshape(-1, 3)
            inside = (vox[:, 0] >= 0) & (vox[:, 0] < size) & (vox[:, 2] >= 0) & (vox[:, 2] < size)
            keep = use & inside
            vox, ids = vox[keep], ids[keep]
            above = vox[:, 1] > terrain.top[vox[:, 2], vox[:, 0]]
            vox, ids = vox[above], ids[above]
            all_x.append(vox[:, 0])
            all_y.append(vox[:, 1])
            all_z.append(vox[:, 2])
            all_id.append(ids)
    if all_x:
        xs, ys, zs, ids = (np.concatenate(a) for a in (all_x, all_y, all_z, all_id))
        key = (xs.astype(np.int64) * 4096 + zs) * 1024 + (ys - MIN_Y)
        _, first = np.unique(key, return_index=True)
        world.set_blocks(xs[first], ys[first], zs[first], ids[first])
        placed = len(first)
    return placed


# Objects that become single Minecraft plants instead of models with --simple-plants
_SIMPLE_KINDS = {"grass", "flower", "sunflower", "reed", "mushroom"}


# Flat, walkable surfaces this close above the ground are filled solid underneath
# (steps, platforms, floors), so they don't float and can be walked on.
FILL_BELOW_STEPS = 6
_KEY_Y = 1024
_KEY_Z = 8192


def _key(x, y, z):
    return (x.astype(np.int64) * _KEY_Z + z) * _KEY_Y + (y - MIN_Y)


def _unkey(k):
    y = k % _KEY_Y + MIN_Y
    xz = k // _KEY_Y
    return xz // _KEY_Z, y, xz % _KEY_Z


def _close_diagonal_gaps(keys: np.ndarray, ids: np.ndarray):
    """Make a voxel surface solid: where two blocks touch only at an edge, add a block
    between them, so walls and roofs have no see-through diagonal gaps."""
    if len(keys) == 0:
        return keys, ids
    order = np.argsort(keys)
    keys, ids = keys[order], ids[order]
    x, y, z = _unkey(keys)

    def has(xx, yy, zz):
        k = _key(xx, yy, zz)
        pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        return keys[pos] == k

    add_k, add_i = [], []
    for (a, b) in (((1, 0, 0), (0, 0, 1)), ((1, 0, 0), (0, 0, -1)),
                   ((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, -1, 0)),
                   ((0, 0, 1), (0, 1, 0)), ((0, 0, 1), (0, -1, 0))):
        dx, dy, dz = a[0] + b[0], a[1] + b[1], a[2] + b[2]
        diag = has(x + dx, y + dy, z + dz)
        gap = diag & ~has(x + a[0], y + a[1], z + a[2]) & ~has(x + b[0], y + b[1], z + b[2])
        if gap.any():
            add_k.append(_key(x[gap] + a[0], y[gap] + a[1], z[gap] + a[2]))
            add_i.append(ids[gap])
    if add_k:
        keys = np.concatenate([keys] + add_k)
        ids = np.concatenate([ids] + add_i)
    return keys, ids


def voxelize_models(opd: OPDFile, terrain: TerrainModel, world: MinecraftWorld, cm: CoordMap,
                    library, simple_plants: bool = False) -> tuple[int, set]:
    """Build objects from their KO 3D models. Returns (blocks placed, ids of shapes built).

    - Walls and other steep surfaces become full blocks.
    - Flat walkable surfaces (floors, stair treads) snap to half-block heights: a bottom
      slab or a full block, so steps are even and can be walked up. Low ones are filled
      solid down to the ground.
    - Each texture is reduced to its 1-3 main colours, so a wall gets a few consistent
      building blocks rather than speckles.
    - Diagonal gaps in walls and roofs are closed.
    """
    from . import ko_models as km

    size = cm.size_blocks
    PRIO_FULL, PRIO_SLAB = 2, 1
    rec_k, rec_b, rec_p = [], [], []
    region_cache = {}
    built, missing = set(), 0

    def region_blocks(tname, tex, alpha, diffuse):
        key = (tname, alpha)
        if key not in region_cache:
            if tex is None:
                labels = None
                rgb = np.array([[int(c * 255) for c in diffuse[:3]]], np.float64).clip(40, 230)
            else:
                labels, rgb = km.texture_regions(tex)
            full, slab = [], []
            for c in rgb:
                green = c[1] > c[0] * 1.05 and c[1] > c[2] * 1.05
                if alpha and green:
                    n = km.LEAF_PALETTE.names[km.LEAF_PALETTE.nearest(c[None])[0]]
                    fid = world.block_id(km.block_state(n))
                    full.append(fid)
                    slab.append(fid)
                    continue
                n = km.SOLID_PALETTE.names[km.SOLID_PALETTE.nearest(c[None])[0]]
                full.append(world.block_id(km.block_state(n)))
                sn = n if n in km.SLABS else km.SLAB_PALETTE.names[km.SLAB_PALETTE.nearest(c[None])[0]]
                slab.append(world.block_id(km.slab_state(sn)))
            region_cache[key] = (labels, np.array(full, np.uint16), np.array(slab, np.uint16))
        return region_cache[key]

    for i, shape in enumerate(opd.shapes):
        name = shape.name.lower()
        if re.search(r"fx|smoke|fog|smog|collisioncube|alpha", name) or not shape.parts:
            continue
        if simple_plants and classify_object(shape.name) in _SIMPLE_KINDS and not shape.is_event_object:
            continue
        if library.missing(shape):
            missing += 1
            continue
        mirrored = shape.scale.x * shape.scale.y * shape.scale.z < 0
        for tris, uvs, tex, part in library.shape_parts(shape):
            if part.dest_blend == 2:          # additive glow effects, not solid
                continue
            mc = np.stack([cm.x(tris[..., 0]), cm.y(tris[..., 1]), cm.z(tris[..., 2])], -1)
            alpha = bool(part.render_flags & km.RF_ALPHABLENDING) or (
                tex is not None and (tex[..., 3] < 128).mean() > 0.05)
            tname = part.textures[0].lower() if part.textures else ""
            labels, full_ids, slab_ids = region_blocks(tname, tex, alpha, part.diffuse)
            pts, tri, tx, ty = km.sample_part(mc, uvs, tex, alpha)
            if len(pts) == 0:
                continue
            region = labels[ty, tx] if labels is not None else np.zeros(len(pts), np.int64)
            # outward normals point down after the z flip, hence the minus
            nrm = -np.cross(mc[:, 1] - mc[:, 0], mc[:, 2] - mc[:, 0])
            if mirrored:
                nrm = -nrm
            ny = nrm[:, 1] / (np.linalg.norm(nrm, axis=1) + 1e-9)
            flat = (np.abs(ny) > 0.7)[tri] & (not alpha)
            up = (ny > 0.7)[tri] & (not alpha)

            vx = np.floor(pts[:, 0]).astype(np.int64)
            vz = np.floor(pts[:, 2]).astype(np.int64)
            inside = (vx >= 0) & (vx < size) & (vz >= 0) & (vz < size)
            ground = np.full(len(pts), MAX_Y, np.int64)
            ground[inside] = terrain.top[vz[inside], vx[inside]]

            # walls: plain voxels
            wall = inside & ~up
            wy = np.floor(pts[:, 1]).astype(np.int64)
            wall &= wy >= ground
            wk, wb = _close_diagonal_gaps(_key(vx[wall], wy[wall], vz[wall]), full_ids[region[wall]])
            rec_k.append(wk); rec_b.append(wb); rec_p.append(np.full(len(wk), PRIO_FULL, np.int8))

            # walkable tops: snap to half blocks
            top = inside & up
            if top.any():
                q = np.round(pts[top, 1] * 2) / 2
                half = (q % 1) != 0
                ty_ = np.where(half, np.floor(q), q - 1).astype(np.int64)
                tx_, tz_, g = vx[top], vz[top], ground[top]
                rid = region[top]
                keep = ty_ > g
                keep_slab = keep & half
                keep_full = keep & ~half
                rec_k.append(_key(tx_[keep_full], ty_[keep_full], tz_[keep_full]))
                rec_b.append(full_ids[rid[keep_full]]); rec_p.append(np.full(keep_full.sum(), PRIO_FULL, np.int8))
                rec_k.append(_key(tx_[keep_slab], ty_[keep_slab], tz_[keep_slab]))
                rec_b.append(slab_ids[rid[keep_slab]]); rec_p.append(np.full(keep_slab.sum(), PRIO_SLAB, np.int8))
                # fill steps/platforms down to the ground
                low = keep & (ty_ - g <= FILL_BELOW_STEPS)
                if low.any():
                    cols = np.unique(np.stack([tx_[low], tz_[low], ty_[low], g[low], rid[low]], 1), axis=0)
                    depth = cols[:, 2] - cols[:, 3] - 1
                    rep = np.repeat(np.arange(len(cols)), np.maximum(depth, 0))
                    if len(rep):
                        off = np.arange(len(rep)) - np.repeat(np.cumsum(np.maximum(depth, 0)) - np.maximum(depth, 0),
                                                              np.maximum(depth, 0))
                        fy = cols[rep, 3] + 1 + off
                        rec_k.append(_key(cols[rep, 0], fy, cols[rep, 1]))
                        rec_b.append(full_ids[cols[rep, 4]])
                        rec_p.append(np.full(len(rep), PRIO_FULL, np.int8))
        built.add(i)
        if len(built) % 2000 == 0:
            print(f"    {len(built)} objects built...")

    if not rec_k:
        return 0, built
    keys = np.concatenate(rec_k)
    blocks = np.concatenate(rec_b).astype(np.int64)
    prio = np.concatenate(rec_p).astype(np.int64)
    # count votes per (voxel, block, priority), then keep the best per voxel:
    # full blocks beat slabs, then the most common block wins
    combo = np.stack([keys, blocks, prio], 1)
    uc, counts = np.unique(combo, axis=0, return_counts=True)
    order = np.lexsort((-counts, -uc[:, 2], uc[:, 0]))
    uc = uc[order]
    first = np.r_[True, uc[1:, 0] != uc[:-1, 0]]
    win = uc[first]
    x, y, z = _unkey(win[:, 0])
    world.set_blocks(x, y, z, win[:, 1].astype(np.uint16))
    if missing:
        print(f"  {missing} objects have model files missing; using simple stand-ins for them")
    return len(win), built


def convert_map(gtd_path: str, opd_path: str | None, output_dir: str,
                world_name: str = "KnightOnline", scale: int = 4,
                vertical_scale: float | None = None, objects: bool = True,
                buildings: bool = True, ko_textures: str | None = None,
                pack_resolution: int = 32, pack_brightness: float = 1.3,
                ko_models: str | None = None, simple_plants: bool = False) -> str:
    """Convert KO map files to a Minecraft world.

    Args:
        gtd_path: Path to .gtd file.
        opd_path: Path to .opd file (optional).
        output_dir: Output directory for the Minecraft world.
        world_name: Name for the Minecraft world.
        scale: MC blocks per KO tile (4 = 1 block per meter, true size).
        vertical_scale: MC blocks per KO meter vertically (default: same as horizontal).
        objects: Place trees, rocks, lamps, event markers etc.
        buildings: Voxelize the collision mesh (buildings, walls, bridges).
        ko_textures: Folder with the KO client's .gtt terrain textures (Data/dtex).
            If given, a resource pack with the real KO ground textures is made.
        pack_resolution: Pixel size of each block texture in the resource pack.
        pack_brightness: Multiplier for KO texture colours (KO draws terrain brighter than stored).
        ko_models: Folder with the KO client's Object files (.n3pmesh + .dxt). If given,
            objects and buildings are built from their real 3D models.
        simple_plants: With models, still use single Minecraft plants for grass/flowers/reeds.

    Returns:
        Path to the generated world directory.
    """
    print("Knight Online -> Minecraft Converter")
    print("=" * 50)

    print(f"\nParsing GTD: {gtd_path}")
    gtd = parse_gtd(gtd_path)

    opd = None
    if opd_path and os.path.exists(opd_path):
        print(f"\nParsing OPD: {opd_path}")
        try:
            opd = parse_opd(opd_path)
        except Exception as e:
            print(f"  Warning: Failed to parse OPD file: {e}")
            print("  Continuing with terrain only...")

    world_dir = os.path.join(output_dir, world_name)
    world = MinecraftWorld(world_dir, world_name)
    cm = CoordMap.for_map(gtd, scale, vertical_scale)
    size = cm.size_blocks

    print(f"\nBuilding terrain ({size}x{size} blocks, scale={scale}, "
          f"vertical {cm.vertical_scale:.3f} blocks/m)...")
    pack = library = None
    if ko_textures:
        from .ko_textures import TexturePack, TextureLibrary
        library = TextureLibrary(ko_textures)
        pack = TexturePack(world_name, pack_resolution, pack_brightness)
    terrain = TerrainModel(gtd, cm, world, pack, library)
    world.set_terrain(terrain.fill_chunk, (0, 0, size - 1, size - 1))
    water_cols = int((terrain.water_top > terrain.top).sum())
    print(f"  Surface Y range {terrain.top.min()}..{terrain.top.max()}, {water_cols} water columns")

    built = set()
    if opd and ko_models and (objects or buildings):
        from .ko_models import ModelLibrary
        print("\nBuilding objects from KO 3D models...")
        n, built = voxelize_models(opd, terrain, world, cm, ModelLibrary(ko_models), simple_plants)
        print(f"  Placed {n} blocks for {len(built)} objects")
    elif opd and buildings:
        print("\nVoxelizing collision mesh (buildings, walls, bridges)...")
        n = voxelize_collision(opd, terrain, world, cm)
        print(f"  Placed {n} blocks from {opd.collision_face_count} collision faces")

    spawn = (size // 2, None, size // 2)
    if opd and objects:
        print(f"\nPlacing {len(opd.shapes) - len(built)} simple objects...")
        placer = ObjectPlacer(world, terrain, cm)
        for i, shape in enumerate(opd.shapes):
            if i not in built:
                placer.place(shape)
        print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(placer.counts.items())))
        binds = [s for s in opd.shapes if s.is_event_object and s.event_type in (OBJECT_BIND, OBJECT_WARP_GATE)]
        if binds:
            s = binds[0]
            spawn = (int(cm.x(s.position.x)) + 3, int(math.floor(float(cm.y(s.position.y)))) + 2,
                     int(cm.z(s.position.z)) + 3)
    sx, sy, sz = spawn
    world.spawn = (sx, max(sy or 0, terrain.top_at(sx, sz) + 2), sz)

    print("\nSaving Minecraft world...")
    world.save()

    pack_path = None
    if pack is not None and pack.images:
        # resources.zip inside a world folder is applied automatically in singleplayer
        pack_path = os.path.join(world_dir, "resources.zip")
        pack.write(pack_path)
        pack.write(os.path.join(output_dir, f"{world_name}_KO_textures.zip"))

    info = {
        "tool": "ko2mc",
        "gtd": os.path.abspath(gtd_path),
        "opd": os.path.abspath(opd_path) if opd else None,
        "coords": cm.to_json(),
        "size_blocks": size,
        "spawn": world.spawn,
        "resource_pack": "resources.zip" if pack_path else None,
        "ko_textures": os.path.abspath(ko_textures) if ko_textures else None,
        "ko_models": os.path.abspath(ko_models) if ko_models else None,
        "pack_brightness": pack_brightness,
    }
    with open(os.path.join(world_dir, "ko2mc.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 50}")
    print("Conversion complete!")
    print(f"  World: {world_dir}")
    print(f"  Terrain: {gtd.heightmap_size - 1}x{gtd.heightmap_size - 1} KO tiles -> {size}x{size} blocks")
    if opd:
        events = [s for s in opd.shapes if s.is_event_object]
        print(f"  Objects: {len(opd.shapes)} ({len(events)} events)")
        for s in events[:10]:
            print(f"    {EVENT_TYPE_NAMES.get(s.event_type, 'Event')}: MC "
                  f"{int(cm.x(s.position.x))} {int(cm.y(s.position.y))} {int(cm.z(s.position.z))}")
    print(f"  Spawn: {world.spawn}")
    print(f"  KO (x, z) -> MC: x = x_ko * {cm.blocks_per_meter:g}, "
          f"z = ({cm.map_size_m:g} - z_ko) * {cm.blocks_per_meter:g}")
    if pack_path:
        print(f"  KO textures: resource pack inside the world (resources.zip); a copy for your")
        print(f"  resourcepacks folder is {os.path.join(output_dir, world_name + '_KO_textures.zip')}")
    print(f"\nTo use: Copy '{world_name}' folder to your Minecraft saves directory.")
    print("  Windows: %appdata%/.minecraft/saves/")
    print("  Linux:   ~/.minecraft/saves/")
    print("  macOS:   ~/Library/Application Support/minecraft/saves/")
    print(f"To preview: python -m ko2mc.preview mc \"{world_dir}\"")

    return world_dir
