"""Minecraft block appearance for the previews.

For a preview that looks exactly like the game, the real block textures are
read from a Minecraft client .jar. Mojang doesn't allow redistributing them,
so they are NOT included in this project. Instead we:

  1. use --mc-jar PATH if given, else
  2. look for an installed Minecraft (e.g. ~/.minecraft/versions/1.20.4/1.20.4.jar), else
  3. with --download-textures, download the official client jar from Mojang
     into ~/.cache/ko2mc (the same file the launcher downloads), else
  4. fall back to built-in approximate colours (still blocky, just not textured).

The result is a BlockAppearance: a texture atlas plus, for every block state
in the world palette, which atlas tile to use on top/side/bottom, the tint
colour and how to draw it (cube, cross-shaped plant, liquid, see-through).
"""

import io
import json
import os
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field

import numpy as np

MC_VERSION = "1.20.4"
TILE = 16

# Plains biome colours (what the converter writes)
GRASS_TINT = (145, 189, 89)
FOLIAGE_TINT = (119, 171, 47)
WATER_TINT = (63, 118, 228)
FIXED_FOLIAGE = {"birch_leaves": (128, 167, 85), "spruce_leaves": (97, 153, 97)}

# Approximate average colours used without a jar (and for unknown textures)
FALLBACK_COLORS = {
    "grass_block_top": (127, 178, 56), "grass_block_side": (134, 96, 67), "dirt": (134, 96, 67),
    "coarse_dirt": (119, 85, 59), "dirt_path_top": (148, 122, 65), "stone": (125, 125, 125),
    "cobblestone": (122, 122, 122), "stone_bricks": (122, 121, 122), "sand": (219, 207, 163),
    "sandstone": (216, 203, 155), "sandstone_top": (223, 214, 170), "smooth_sandstone": (223, 214, 170),
    "snow": (249, 254, 254), "calcite": (223, 224, 220), "tuff": (108, 109, 102),
    "deepslate": (80, 80, 82), "moss_block": (89, 109, 45), "bedrock": (85, 85, 85),
    "water_still": (63, 118, 228), "oak_log": (109, 85, 50), "oak_log_top": (151, 121, 73),
    "birch_log": (216, 215, 210), "birch_log_top": (193, 179, 135), "spruce_log": (58, 37, 16),
    "spruce_log_top": (108, 80, 46), "jungle_log": (85, 67, 25), "jungle_log_top": (149, 109, 70),
    "oak_leaves": (144, 144, 144), "birch_leaves": (130, 129, 130), "spruce_leaves": (126, 126, 126),
    "jungle_leaves": (156, 154, 143), "azalea_leaves": (90, 115, 44), "oak_planks": (162, 130, 78),
    "spruce_planks": (115, 85, 49), "polished_andesite": (132, 134, 133), "andesite": (136, 136, 136),
    "packed_ice": (141, 180, 250), "snow_block": (249, 254, 254), "gold_block": (246, 208, 61),
    "iron_block": (220, 220, 220), "glowstone": (171, 131, 84), "short_grass": (130, 130, 130),
    "fern": (124, 124, 124), "poppy": (200, 40, 30), "dandelion": (245, 220, 40),
    "cornflower": (70, 110, 220), "oxeye_daisy": (230, 230, 220), "azure_bluet": (200, 220, 230),
    "allium": (170, 100, 220), "sugar_cane": (148, 192, 101), "lantern": (180, 140, 70),
    "campfire_log_lit": (110, 80, 50), "white_banner": (240, 240, 240), "oak_fence": (162, 130, 78),
    "dark_oak_planks": (66, 43, 20), "pumpkin_side": (198, 118, 24), "pumpkin_top": (198, 118, 24),
    "red_mushroom": (200, 40, 40), "brown_mushroom": (150, 110, 80), "end_portal_frame_top": (91, 120, 97),
    "crying_obsidian": (32, 10, 60), "end_rod": (240, 230, 220), "respawn_anchor_top": (60, 20, 100),
    "anvil": (68, 68, 68), "beacon": (117, 220, 215), "iron_bars": (136, 139, 135), "lever": (110, 110, 110),
    "sunflower_front": (245, 200, 40), "gravel": (136, 126, 126), "mud": (60, 57, 60), "clay": (160, 166, 179),
}

TINTED = {"grass_block_top", "grass_block_side_overlay", "short_grass", "fern", "tall_grass_top",
          "tall_grass_bottom", "oak_leaves", "jungle_leaves", "acacia_leaves", "dark_oak_leaves",
          "mangrove_leaves", "vine", "sugar_cane", "birch_leaves", "spruce_leaves",
          "water_still", "water_flow", "lily_pad"}

# Render kinds
CUBE, CUTOUT, CROSS, LIQUID = 0, 1, 2, 3
CUTOUT_BLOCKS = ("leaves", "glass", "iron_bars", "fence", "lantern", "campfire", "banner",
                 "lever", "anvil", "end_rod", "beacon", "portal_frame", "respawn_anchor")
CROSS_BLOCKS = ("short_grass", "fern", "tall_grass", "poppy", "dandelion", "cornflower", "daisy",
                "bluet", "allium", "tulip", "sugar_cane", "mushroom", "sapling", "sunflower",
                "rose", "lilac", "peony", "dead_bush", "torch", "lantern")


@dataclass
class BlockLook:
    kind: int
    top: int
    side: int
    bottom: int
    tint_top: tuple = (255, 255, 255)
    tint_side: tuple = (255, 255, 255)
    color: tuple = (128, 128, 128)   # average top colour, for 2D maps


@dataclass
class BlockAppearance:
    atlas: np.ndarray                 # (H, W, 4) uint8 RGBA
    tiles_per_row: int
    looks: list[BlockLook] = field(default_factory=list)
    textured: bool = False
    source: str = ""


def find_client_jar(explicit: str | None = None, download: bool = False) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(os.environ.get("APPDATA", ""), ".minecraft", "versions"),
        os.path.join(home, ".minecraft", "versions"),
        os.path.join(home, "Library", "Application Support", "minecraft", "versions"),
    ]
    found = []
    for base in candidates:
        if os.path.isdir(base):
            for ver in os.listdir(base):
                jar = os.path.join(base, ver, ver + ".jar")
                if os.path.exists(jar) and ver[:1].isdigit():
                    found.append(jar)
    exact = [j for j in found if os.path.basename(j) == MC_VERSION + ".jar"]
    if exact:
        return exact[0]
    cache = os.path.join(home, ".cache", "ko2mc", f"client-{MC_VERSION}.jar")
    if os.path.exists(cache):
        return cache
    if found:
        return sorted(found)[-1]
    if download:
        return download_client_jar(cache)
    return None


def download_client_jar(dest: str) -> str | None:
    print(f"  Downloading Minecraft {MC_VERSION} client jar from Mojang (textures only, ~24 MB)...")
    try:
        manifest = json.load(urllib.request.urlopen(
            "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json", timeout=30))
        ver = next(v for v in manifest["versions"] if v["id"] == MC_VERSION)
        info = json.load(urllib.request.urlopen(ver["url"], timeout=30))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(info["downloads"]["client"]["url"], dest + ".part")
        os.replace(dest + ".part", dest)
        return dest
    except Exception as e:  # network errors etc.
        print(f"  Could not download textures ({e}); using plain colours.")
        return None


def _asset(ref: str, kind: str, ext: str) -> str:
    """'ko2mc:block/t0' -> 'assets/ko2mc/<kind>/block/t0.<ext>' (default namespace minecraft)."""
    ns, _, path = ref.rpartition(":")
    return f"assets/{ns or 'minecraft'}/{kind}/{path}.{ext}"


class _Zips:
    """Files looked up in resource packs first, then the Minecraft jar."""

    def __init__(self, paths):
        self.zips = [zipfile.ZipFile(p) for p in paths if p]

    def read(self, name):
        for z in self.zips:
            try:
                return z.read(name)
            except KeyError:
                pass
        raise KeyError(name)


class _JarModels:
    """Resolves block state -> textures using blockstates/models JSON."""

    def __init__(self, zf: "_Zips"):
        self.zf = zf
        self._models = {}

    def _json(self, path):
        try:
            return json.loads(self.zf.read(path))
        except KeyError:
            return None

    def model(self, ref: str) -> dict:
        ref = ref if ":" in ref else "minecraft:" + ref
        if ref in self._models:
            return self._models[ref]
        m = self._json(_asset(ref, "models", "json")) or {}
        textures, parents = {}, []
        cur = m
        while cur:
            parents.append(cur.get("parent", ""))
            for k, v in cur.get("textures", {}).items():
                textures.setdefault(k, v)
            p = cur.get("parent")
            cur = self._json(_asset(p if ":" in p else "minecraft:" + p, "models", "json")) if p else None
        # resolve '#refs'
        for _ in range(5):
            for k, v in list(textures.items()):
                if isinstance(v, str) and v.startswith("#"):
                    textures[k] = textures.get(v[1:], v)
        result = {"textures": textures, "parents": parents}
        self._models[ref] = result
        return result

    def state_model(self, name: str, props: dict) -> dict:
        bs = self._json(_asset("minecraft:" + name, "blockstates", "json"))
        if not bs:
            return {"textures": {}, "parents": []}
        if "variants" in bs:
            best, best_score = None, -1
            for key, var in bs["variants"].items():
                conds = dict(kv.split("=") for kv in key.split(",") if "=" in kv)
                if any(props.get(k, v) != v for k, v in conds.items()):
                    score = -1
                else:
                    score = len(conds)
                if best is None or score > best_score:
                    best, best_score = var, score
            var = best[0] if isinstance(best, list) else best
            return self.model(var["model"])
        for part in bs.get("multipart", []):
            apply = part["apply"]
            apply = apply[0] if isinstance(apply, list) else apply
            return self.model(apply["model"])
        return {"textures": {}, "parents": []}


def _tex_name(ref: str) -> str:
    """Short name for vanilla block textures ('stone'), full 'ns:path' for others."""
    ref = str(ref).removeprefix("minecraft:")
    return ref.removeprefix("block/") if ":" not in ref else ref


def _kind_for(name: str, parents: list[str]) -> int:
    short = name.split(":")[-1]
    if short in ("water", "lava") or short.endswith("water"):
        return LIQUID
    if any("cross" in p for p in parents) or any(k in short for k in CROSS_BLOCKS):
        return CROSS
    if any(k in short for k in CUTOUT_BLOCKS):
        return CUTOUT
    return CUBE


def build_appearance(palette: list[str], jar_path: str | None,
                     pack_paths: list[str] | None = None) -> BlockAppearance:
    """Create the atlas and per-block looks for a world palette.

    pack_paths: resource packs (e.g. the KO texture pack) that override the jar.
    """
    from PIL import Image

    paths = [p for p in (pack_paths or []) if p and os.path.exists(p)] + ([jar_path] if jar_path else [])
    zf = _Zips(paths) if paths else None
    models = _JarModels(zf) if zf else None
    tiles: dict[str, int] = {}
    images: list[np.ndarray] = []

    def tile_for(tex: str) -> int:
        tex = _tex_name(tex)
        if tex in tiles:
            return tiles[tex]
        img = None
        if zf:
            try:
                ref = tex if ":" in tex else f"minecraft:block/{tex}"
                raw = zf.read(_asset(ref, "textures", "png"))
                im = Image.open(io.BytesIO(raw)).convert("RGBA")
                im = im.crop((0, 0, im.width, im.width))
                img = np.array(im.resize((TILE, TILE), Image.BOX if im.width > TILE else Image.NEAREST))
            except KeyError:
                img = None
        if img is None:
            img = _fallback_texture(tex)
        if tex in TINTED:
            tint = (WATER_TINT if tex.startswith("water") else
                    GRASS_TINT if tex in ("grass_block_top", "grass_block_side_overlay", "short_grass",
                                          "fern", "tall_grass_top", "tall_grass_bottom", "sugar_cane")
                    else FIXED_FOLIAGE.get(tex, FOLIAGE_TINT))
            if jar_path or tex.startswith("water"):
                img = img.copy()
                img[..., :3] = (img[..., :3].astype(np.float32) * np.array(tint) / 255).astype(np.uint8)
        tiles[tex] = len(images)
        images.append(img)
        return tiles[tex]

    looks = []
    for state in palette:
        name, props = _split_state(state)
        short = name.split(":")[-1]
        if short in ("air", "cave_air", "void_air"):
            looks.append(BlockLook(CUBE, 0, 0, 0))
            continue
        tex, parents = {}, []
        if models:
            m = models.state_model(short, props)
            tex, parents = m["textures"], m["parents"]
        kind = _kind_for(name, parents)

        def pick(*keys, default=None):
            for k in keys:
                if k in tex and not str(tex[k]).startswith("#"):
                    return tex[k]
            return default

        if kind == LIQUID:
            top = side = bottom = f"{short}_still"
        elif kind == CROSS:
            top = side = bottom = pick("cross", "plant", "texture", "lantern", "particle", default=short)
            if short == "sunflower":
                top = side = bottom = "sunflower_front" if props.get("half") == "upper" else "sunflower_bottom"
        else:
            top = pick("top", "end", "all", "up", "texture", "particle", default=_guess_top(short))
            side = pick("side", "all", "north", "texture", "wall", "particle", default=_guess_side(short))
            bottom = pick("bottom", "end", "all", "down", "texture", "particle", default=_guess_top(short))
        t, s, b = tile_for(top), tile_for(side), tile_for(bottom)
        if short == "grass_block" and jar_path:
            s = _grass_side(zf, tiles, images, tile_for)
        img = images[t]
        alpha = img[..., 3:4].astype(np.float32) / 255
        color = tuple(int(c) for c in (img[..., :3] * alpha).sum((0, 1)) / max(alpha.sum(), 1))
        looks.append(BlockLook(kind, t, s, b, color=color))

    per_row = 32
    rows = (len(images) + per_row - 1) // per_row or 1
    atlas = np.zeros((rows * TILE, per_row * TILE, 4), dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, per_row)
        atlas[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = im
    return BlockAppearance(atlas, per_row, looks, textured=jar_path is not None,
                           source=", ".join(os.path.basename(p) for p in paths) or "built-in colours")


def _grass_side(zf, tiles, images, tile_for) -> int:
    """grass_block_side with its tinted overlay composited on top (as the game draws it)."""
    key = "grass_block_side+overlay"
    if key in tiles:
        return tiles[key]
    base = images[tile_for("grass_block_side")].astype(np.float32)
    over = images[tile_for("grass_block_side_overlay")].astype(np.float32)
    a = over[..., 3:4] / 255
    out = base.copy()
    out[..., :3] = base[..., :3] * (1 - a) + over[..., :3] * a
    tiles[key] = len(images)
    images.append(out.astype(np.uint8))
    return tiles[key]


def _split_state(state: str):
    if "[" not in state:
        return state, {}
    name, rest = state[:-1].split("[", 1)
    return name, dict(kv.split("=", 1) for kv in rest.split(",") if "=" in kv)


def _guess_top(short: str) -> str:
    if short.endswith("_log"):
        return short + "_top"
    return {"grass_block": "grass_block_top", "snow_block": "snow", "dirt_path": "dirt_path_top",
            "sandstone": "sandstone_top", "pumpkin": "pumpkin_top"}.get(short, short)


def _guess_side(short: str) -> str:
    return {"grass_block": "grass_block_side", "snow_block": "snow", "dirt_path": "dirt_path_side",
            "pumpkin": "pumpkin_side"}.get(short, short)


def _fallback_texture(tex: str) -> np.ndarray:
    """A 16x16 noisy tile in the block's approximate colour."""
    base = FALLBACK_COLORS.get(tex)
    if base is None:
        for k, v in FALLBACK_COLORS.items():
            if tex.startswith(k) or k.startswith(tex):
                base = v
                break
    if base is None:
        base = (150, 150, 150)
    if tex in ("oak_leaves", "jungle_leaves", "birch_leaves", "spruce_leaves"):
        base = FIXED_FOLIAGE.get(tex, FOLIAGE_TINT)
    if tex in ("short_grass", "fern", "sugar_cane"):
        base = GRASS_TINT
    rng = np.random.default_rng(zlib.crc32(tex.encode()))
    noise = rng.normal(1.0, 0.08, (TILE, TILE, 1))
    img = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    img[..., :3] = np.clip(np.array(base) * noise, 0, 255)
    img[..., 3] = 255
    if tex == "grass_block_side":
        img[:4, :, :3] = np.clip(np.array(GRASS_TINT) * noise[:4], 0, 255)
    if any(k in tex for k in ("leaves",)):
        img[..., 3] = np.where(rng.random((TILE, TILE)) < 0.2, 0, 255)
    if any(k in tex for k in CROSS_BLOCKS):
        yy, xx = np.mgrid[0:TILE, 0:TILE]
        stem = (np.abs(xx - 7.5) < 1.5) | ((yy < 8) & (np.abs(xx - 7.5) < 4))
        img[..., 3] = np.where(stem, 255, 0)
    return img
