"""Build KO objects (houses, walls, trees...) from their real 3D models.

Every object in an .opd lists its parts: a mesh file (.n3pmesh), a texture
(.dxt) and a pivot. The models live in the KO client's Object folder. For each
object we:

  1. load the part meshes and place them with the object's scale, rotation and
     position (the same order the KO client uses: scale, rotate, translate),
  2. sample points densely over every triangle and look up the texture colour
     at each point (see-through pixels of leaves/fences are skipped),
  3. turn the points into blocks, picking for each block the Minecraft block
     whose average colour is closest to the KO texture colour there.

.n3pmesh layout (little-endian), checked against all 5147 files in the client:
    int32 name_len, char[name_len] name
    int32 collapses, index_changes, max_vertices, max_indices, min_vertices, min_indices
    max_vertices x (x, y, z, nx, ny, nz, u, v) float32
    max_indices x uint16 triangle list
    collapses x 24 bytes, index_changes x int32, int32 lod_count, lod_count x 8 bytes
"""

import math
import os
import re
import struct

import numpy as np

from .ko_textures import read_n3_textures

# Render flags from the KO client's material (__Material::nRenderFlags)
RF_ALPHABLENDING = 1
RF_DOUBLESIDED = 4

# Average colours of Minecraft 1.20 blocks (measured from the vanilla textures)
BLOCK_COLORS = {
    "stone": (126, 126, 126), "cobblestone": (128, 127, 128), "mossy_cobblestone": (110, 118, 95),
    "stone_bricks": (122, 122, 122), "mossy_stone_bricks": (115, 121, 105),
    "cracked_stone_bricks": (118, 118, 118), "smooth_stone": (159, 159, 159),
    "andesite": (136, 136, 137), "polished_andesite": (132, 135, 134), "diorite": (189, 188, 189),
    "polished_diorite": (193, 193, 195), "granite": (149, 103, 86),
    "polished_granite": (154, 107, 89), "deepslate_bricks": (71, 71, 71),
    "polished_deepslate": (72, 73, 73), "cobbled_deepslate": (77, 77, 81), "tuff": (108, 109, 103),
    "calcite": (223, 224, 221), "bricks": (151, 98, 83), "mud_bricks": (137, 104, 79),
    "packed_mud": (142, 107, 80), "sandstone": (216, 203, 156),
    "smooth_sandstone": (224, 214, 170), "cut_sandstone": (218, 206, 160),
    "red_sandstone": (187, 99, 29), "smooth_red_sandstone": (181, 98, 31),
    "quartz_block": (236, 230, 223), "smooth_quartz": (237, 230, 224),
    "prismarine": (98, 162, 146), "prismarine_bricks": (99, 172, 158),
    "dark_prismarine": (52, 92, 76), "nether_bricks": (44, 22, 26),
    "red_nether_bricks": (70, 7, 9), "blackstone": (42, 36, 41),
    "polished_blackstone_bricks": (48, 43, 50), "end_stone_bricks": (218, 224, 162),
    "terracotta": (152, 94, 68), "dirt": (134, 96, 67), "coarse_dirt": (119, 86, 59),
    "mud": (60, 57, 61), "clay": (161, 166, 179), "gravel": (132, 127, 127),
    "sand": (219, 207, 163), "snow_block": (249, 254, 254), "packed_ice": (142, 180, 250),
    "obsidian": (15, 11, 25), "iron_block": (220, 220, 220), "gold_block": (246, 208, 62),
    "copper_block": (192, 108, 80), "exposed_copper": (161, 126, 104),
    "weathered_copper": (108, 153, 110), "oxidized_copper": (82, 163, 133),
    "bone_block": (229, 226, 208), "hay_block": (166, 136, 38), "dried_kelp_block": (38, 49, 30),
    "basalt": (73, 73, 78), "oak_planks": (162, 131, 79), "spruce_planks": (115, 85, 49),
    "birch_planks": (192, 175, 121), "jungle_planks": (160, 115, 81),
    "acacia_planks": (168, 90, 50), "dark_oak_planks": (67, 43, 20),
    "mangrove_planks": (118, 54, 49), "cherry_planks": (227, 179, 173),
    "crimson_planks": (101, 49, 71), "warped_planks": (43, 105, 99),
    "bamboo_planks": (193, 173, 80), "oak_log": (109, 85, 51), "spruce_log": (59, 38, 17),
    "birch_log": (217, 215, 210), "jungle_log": (85, 68, 25), "acacia_log": (103, 97, 87),
    "dark_oak_log": (60, 47, 26), "white_terracotta": (210, 178, 161),
    "white_concrete": (207, 213, 214), "white_wool": (234, 236, 237),
    "orange_terracotta": (162, 84, 38), "orange_concrete": (224, 97, 1),
    "orange_wool": (241, 118, 20), "magenta_terracotta": (150, 88, 109),
    "magenta_concrete": (169, 48, 159), "magenta_wool": (190, 69, 180),
    "light_blue_terracotta": (113, 109, 138), "light_blue_concrete": (36, 137, 199),
    "light_blue_wool": (58, 175, 217), "yellow_terracotta": (186, 133, 35),
    "yellow_concrete": (241, 175, 21), "yellow_wool": (249, 198, 40),
    "lime_terracotta": (104, 118, 53), "lime_concrete": (94, 169, 24), "lime_wool": (112, 185, 26),
    "pink_terracotta": (162, 78, 79), "pink_concrete": (214, 101, 143),
    "pink_wool": (238, 141, 172), "gray_terracotta": (58, 42, 36), "gray_concrete": (55, 58, 62),
    "gray_wool": (63, 68, 72), "light_gray_terracotta": (135, 107, 98),
    "light_gray_concrete": (125, 125, 115), "light_gray_wool": (142, 142, 135),
    "cyan_terracotta": (87, 91, 91), "cyan_concrete": (21, 119, 136), "cyan_wool": (21, 138, 145),
    "purple_terracotta": (118, 70, 86), "purple_concrete": (100, 32, 156),
    "purple_wool": (122, 42, 173), "blue_terracotta": (74, 60, 91), "blue_concrete": (45, 47, 143),
    "blue_wool": (53, 57, 157), "brown_terracotta": (77, 51, 36), "brown_concrete": (96, 60, 32),
    "brown_wool": (114, 72, 41), "green_terracotta": (76, 83, 42), "green_concrete": (73, 91, 36),
    "green_wool": (85, 110, 28), "red_terracotta": (143, 61, 47), "red_concrete": (142, 33, 33),
    "red_wool": (161, 39, 35), "black_terracotta": (37, 23, 16), "black_concrete": (8, 10, 15),
    "black_wool": (21, 21, 26),
}
LEAF_COLORS = {
    "oak_leaves": (67, 97, 27), "jungle_leaves": (73, 103, 26), "dark_oak_leaves": (70, 101, 28),
    "spruce_leaves": (48, 76, 48), "birch_leaves": (66, 85, 43), "azalea_leaves": (90, 115, 44),
    "flowering_azalea_leaves": (100, 111, 61), "cherry_leaves": (229, 173, 194),
}
# Blocks that need a direction; we always place them upright.
_AXIS_BLOCKS = ("_log", "basalt", "hay_block", "bone_block")


def block_state(name: str) -> str:
    if name.endswith("_leaves"):
        return f"minecraft:{name}[persistent=true]"
    if any(name.endswith(a) for a in _AXIS_BLOCKS):
        return f"minecraft:{name}[axis=y]"
    return f"minecraft:{name}"


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255) -> CIE Lab, for perceptual colour matching."""
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


class Palette:
    """Nearest-colour lookup from RGB to a Minecraft block."""

    def __init__(self, colors: dict):
        self.names = list(colors)
        self.lab = _to_lab(np.array([colors[n] for n in self.names]))

    def nearest(self, rgb: np.ndarray) -> np.ndarray:
        lab = _to_lab(rgb)
        d = ((lab[:, None, :] - self.lab[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)


# Wool looks fuzzy and odd on buildings; everything else is fair game.
SOLID_PALETTE = Palette({k: v for k, v in BLOCK_COLORS.items() if not k.endswith("_wool")})
LEAF_PALETTE = Palette(LEAF_COLORS)


def parse_n3pmesh(data: bytes):
    """Return (vertices (N, 8) float32, triangle indices (M, 3) int)."""
    (n,) = struct.unpack_from("<i", data, 0)
    off = 4 + n
    _nc, _tic, max_v, max_i, _mnv, _mni = struct.unpack_from("<6i", data, off)
    off += 24
    verts = np.frombuffer(data, "<f4", max_v * 8, off).reshape(max_v, 8)
    off += max_v * 32
    idx = np.frombuffer(data, "<u2", max_i, off).astype(np.int32)
    idx = idx[: len(idx) // 3 * 3].reshape(-1, 3)
    idx = idx[(idx < max_v).all(1)]
    return verts, idx


def quat_rotate(q, v: np.ndarray) -> np.ndarray:
    """Rotate row vectors v (N, 3) by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return v @ r.T


class ModelLibrary:
    """Finds and caches meshes and textures from the KO client's Object folder."""

    def __init__(self, folder: str):
        self.folder = folder
        self.files = {}
        for root, _, names in os.walk(folder):
            for n in names:
                self.files.setdefault(n.lower(), os.path.join(root, n))
        self._meshes = {}
        self._textures = {}
        print(f"  KO models: {len(self.files)} files found in {folder}")

    @staticmethod
    def _key(path: str) -> str:
        return os.path.basename(path.replace("\\", "/")).lower()

    def mesh(self, name: str):
        k = self._key(name)
        if k not in self._meshes:
            path = self.files.get(k)
            try:
                self._meshes[k] = parse_n3pmesh(open(path, "rb").read()) if path else None
            except (struct.error, ValueError):
                self._meshes[k] = None
        return self._meshes[k]

    def texture(self, name: str):
        k = self._key(name)
        if k not in self._textures:
            path = self.files.get(k)
            texs = read_n3_textures(open(path, "rb").read()) if path else []
            self._textures[k] = texs[0].rgba if texs else None
        return self._textures[k]

    def shape_parts(self, shape):
        """Yield (triangles_ko (T,3,3), uvs (T,3,2), texture rgba or None, part) in KO meters."""
        q = (shape.rotation.x, shape.rotation.y, shape.rotation.z, shape.rotation.w)
        scale = np.array([shape.scale.x, shape.scale.y, shape.scale.z])
        pos = np.array([shape.position.x, shape.position.y, shape.position.z])
        for part in shape.parts:
            m = self.mesh(part.name)
            if m is None or len(m[1]) == 0:
                continue
            verts, idx = m
            pivot = np.array([part.pivot.x, part.pivot.y, part.pivot.z]) if part.pivot else 0.0
            p = (verts[:, :3] + pivot) * scale
            p = quat_rotate(q, p) + pos
            tex = self.texture(part.textures[0]) if part.textures else None
            yield p[idx], verts[:, 6:8][idx], tex, part

    def missing(self, shape) -> bool:
        return any(self.mesh(p.name) is None for p in shape.parts)


def sample_part(tris_mc: np.ndarray, uvs: np.ndarray, tex, alpha_test: bool, spacing: float = 0.45):
    """Sample points on triangles (Minecraft coords). Returns (points (P,3), rgb (P,3))."""
    e = np.stack([tris_mc[:, 1] - tris_mc[:, 0], tris_mc[:, 2] - tris_mc[:, 0],
                  tris_mc[:, 2] - tris_mc[:, 1]])
    edge = np.linalg.norm(e, axis=2).max(0)
    steps = np.clip(np.ceil(edge / spacing).astype(np.int32), 1, 300)
    pts_all, col_all = [], []
    for k in np.unique(steps):
        sel = np.flatnonzero(steps == k)
        a, b = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        m = a + b <= k
        w1 = a[m].astype(np.float32) / k
        w2 = b[m].astype(np.float32) / k
        w0 = 1 - w1 - w2
        t = tris_mc[sel]
        pts = (w0[None, :, None] * t[:, None, 0] + w1[None, :, None] * t[:, None, 1]
               + w2[None, :, None] * t[:, None, 2]).reshape(-1, 3)
        uv = uvs[sel]
        tc = (w0[None, :, None] * uv[:, None, 0] + w1[None, :, None] * uv[:, None, 1]
              + w2[None, :, None] * uv[:, None, 2]).reshape(-1, 2)
        if tex is not None:
            h, w = tex.shape[:2]
            tx = np.floor(np.mod(tc[:, 0], 1.0) * w).astype(np.int32) % w
            ty = np.floor(np.mod(tc[:, 1], 1.0) * h).astype(np.int32) % h
            texel = tex[ty, tx]
            keep = texel[:, 3] >= 128 if alpha_test else np.ones(len(texel), bool)
            pts_all.append(pts[keep])
            col_all.append(texel[keep, :3])
        else:
            pts_all.append(pts)
            col_all.append(np.full((len(pts), 3), 128, np.uint8))
    if not pts_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(pts_all), np.concatenate(col_all)
