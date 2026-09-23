"""Knight Online sky and water -> Minecraft (resource pack + world datapack).

KO draws its sky from a .n3sky file (Misc/Sky/<zone>.n3sky): sky, fog, sun,
cloud and light colours keyed to sunrise / noon / sunset / midnight, plus the
textures for the sun disk, glow, cloud layers and moon phases. Water surfaces
in the .gtd name their texture (Misc/river/*.dxt).

Vanilla Minecraft can't draw a textured sky dome, but it gets close:

  * sky and fog colours: a custom biome (world datapack) with KO's noon sky
    colour overhead and KO's fog colour at the horizon; Minecraft darkens it
    for night by itself
  * sun, moon phases and clouds: the resource pack replaces Minecraft's
    sun.png, moon_phases.png and clouds.png with KO's
  * water: the pack's water texture is KO's (animated by scrolling, like KO
    scrolls its water UVs); Minecraft tints water per biome, so every KO water
    texture gets its own biome whose water colour is that texture's colour

n3sky layout: 10 x (int32 length, name), int32 count, then count records of
(int32 length, name, int32 type, int32 time of day in seconds, D3DCOLOR
(B, G, R, A), 4 bytes (a second colour or a number), float32).
"""

import io
import json
import os
import re
import struct

import numpy as np

SKY_TYPES = {0: "sky", 1: "fog", 2: "stars", 3: "moon_phase", 4: "sun", 5: "glow", 6: "flare",
             7: "cloud1", 8: "cloud2", 10: "light0", 11: "light1", 12: "light2"}
WATER_FRAMES = 32
DATAPACK_FORMAT = 26           # Minecraft 1.20.3 / 1.20.4


class KOSky:
    def __init__(self, path: str):
        d = open(path, "rb").read()
        pos = 0

        def string():
            nonlocal pos
            n, = struct.unpack_from("<i", d, pos)
            s = d[pos + 4:pos + 4 + n].split(b"\0")[0].decode("latin-1")
            pos += 4 + n
            return s

        self.path = path
        self.textures = [string() for _ in range(10)]
        count, = struct.unpack_from("<i", d, pos)
        pos += 4
        self.keys = []                 # (time s, kind, (r, g, b), raw second value, float, phase)
        for _ in range(count):
            phase = string().split(" - ")[0].strip().lower()        # sunrise / noon / sunset / midnight
            kind, t = struct.unpack_from("<ii", d, pos)
            b, g, r, a = d[pos + 8:pos + 12]
            extra = d[pos + 12:pos + 16]
            f, = struct.unpack_from("<f", d, pos + 16)
            pos += 20
            self.keys.append((t, SKY_TYPES.get(kind, str(kind)), (r, g, b), extra, f, phase))

    def color(self, kind: str, phase: str = "noon") -> tuple[int, int, int]:
        """The colour of `kind` for a phase of the day (sunrise, noon, sunset, midnight)."""
        ks = [k for k in self.keys if k[1] == kind]
        if not ks:
            return (128, 160, 255)
        return next((k[2] for k in ks if k[5] == phase), ks[0][2])


def find_sky(misc_dir: str, map_name: str) -> str | None:
    """<misc>/Sky/<map>.n3sky, trying shorter names, then default.n3sky."""
    sky_dir = next((os.path.join(misc_dir, d) for d in os.listdir(misc_dir) if d.lower() == "sky"), None) \
        if misc_dir and os.path.isdir(misc_dir) else None
    if not sky_dir:
        return None
    files = {f.lower(): os.path.join(sky_dir, f) for f in os.listdir(sky_dir)}
    name = map_name.lower()
    cands = [name, re.sub(r"[_\d]+$", "", name), re.sub(r"_.*$", "", name), "default"]
    for c in cands:
        if c and f"{c}.n3sky" in files:
            return files[f"{c}.n3sky"]
    return None


def find_misc(*client_dirs) -> str | None:
    """The KO client's Misc folder next to a DTex/Object folder."""
    for d in client_dirs:
        if not d:
            continue
        for base in (d, os.path.dirname(os.path.abspath(d))):
            for n in os.listdir(base) if os.path.isdir(base) else []:
                if n.lower() == "misc" and os.path.isdir(os.path.join(base, n)):
                    return os.path.join(base, n)
    return None


def _file(folder, name):
    """Case-insensitive file lookup; name may contain KO-style backslash folders."""
    parts = name.replace("\\", "/").split("/")
    cur = folder
    for p in parts:
        if not os.path.isdir(cur):
            return None
        match = next((n for n in os.listdir(cur) if n.lower() == p.lower()), None)
        if match is None:
            return None
        cur = os.path.join(cur, match)
    return cur


def _png(img: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()


def _load(path, size=None):
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    if size:
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im).astype(np.float32)


# ---------------------------------------------------------------------------
# resource pack images
# ---------------------------------------------------------------------------

def sky_pack_files(sky: KOSky, misc_dir: str) -> dict[str, bytes]:
    """assets/minecraft/textures/environment/{sun,moon_phases,clouds}.png from KO's sky textures."""
    out = {}
    tex = {os.path.basename(t.replace("\\", "/")).lower(): t for t in sky.textures}

    def path(short):
        t = tex.get(short)
        return _file(misc_dir, t.split("\\", 1)[1] if t and "\\" in t else (t or "")) if t else None

    # sun: KO's glow + disk, tinted by the noon sun colour. Minecraft draws the sun additively,
    # so black is see-through
    disk, glow = path("sundisk.bmp"), path("sunglow.bmp")
    if disk and glow:
        n = 64
        g = _load(glow, (n, n))[..., :3].mean(-1)
        dk = _load(disk, (n // 2, n // 2))[..., :3].mean(-1)
        lum = g * 0.6
        o = n // 4
        lum[o:o + n // 2, o:o + n // 2] = np.maximum(lum[o:o + n // 2, o:o + n // 2], dk)
        c = np.array(sky.color("sun"), np.float32) / 255
        rgb = np.clip(lum[..., None] * (0.35 + 0.65 * c), 0, 255)
        out["assets/minecraft/textures/environment/sun.png"] = _png(
            np.dstack([rgb, np.full((n, n), 255)]).astype(np.uint8))
    # moon: KO has 24 phases (6 x 4); Minecraft 8 (4 x 2): full, waning gibbous, last quarter,
    # waning crescent, new, waxing crescent, first quarter, waxing gibbous
    ph = path("phases.tga")
    if ph:
        img = _load(ph)
        h, w = img.shape[:2]
        cell = min(h // 4, w // 6)
        cells = [img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] for r in range(4) for c in range(6)]
        lit = [c[..., :3].mean(-1) * c[..., 3] / 255 for c in cells]
        frac = np.array([l.sum() for l in lit])
        frac = frac / max(frac.max(), 1)
        xs = np.arange(cell) - cell / 2
        side = np.array([(l.sum(0) * xs).sum() / max(l.sum(), 1) for l in lit])   # >0: lit on the right
        size = 64
        from PIL import Image
        sheet = np.zeros((2 * size, 4 * size, 4), np.uint8)
        for i in range(8):
            a = i * np.pi / 4
            want = (1 + np.cos(a)) / 2
            want_side = 0 if i in (0, 4) else (-1 if i < 4 else 1)
            score = np.abs(frac - want) + 0.3 * np.abs(np.sign(side) - want_side) * (want_side != 0)
            if i == 4:
                score = frac                                   # new moon: the darkest one
            best = cells[int(score.argmin())]
            im = np.asarray(Image.fromarray(best.astype(np.uint8)).resize((size, size), Image.BILINEAR))
            rgb = im[..., :3].astype(np.float32) * im[..., 3:4] / 255
            sheet[(i // 4) * size:(i // 4 + 1) * size, (i % 4) * size:(i % 4 + 1) * size] = \
                np.dstack([rgb, np.full((size, size), 255)]).astype(np.uint8)
        out["assets/minecraft/textures/environment/moon_phases.png"] = _png(sheet)
    # clouds: Minecraft builds clouds from clouds.png (one pixel = 12 x 12 blocks); use KO's
    # puffy cloud layer, 4 x 4 tiles so a KO cloud is about 100 blocks across
    puffs = path("puffs.tga") or path("wisps.tga")
    if puffs:
        a = _load(puffs, (64, 64))
        cover = a[..., 3] / 255 * a[..., :3].mean(-1) / 255
        mask = np.tile(cover > 0.35, (4, 4))
        img = np.zeros((256, 256, 4), np.uint8)
        img[mask] = 255
        out["assets/minecraft/textures/environment/clouds.png"] = _png(img)
    return out


def water_pack_files(textures: dict[str, np.ndarray]) -> tuple[dict[str, bytes], dict[str, tuple]]:
    """Animated water textures from the most used KO water texture.

    textures: KO water texture name -> RGBA, most used first. Returns (pack files,
    water tint per texture name) - the pack's water is greyscale and each KO water
    texture's colour comes back through the biome tint."""
    if not textures:
        return {}, {}
    main = next(iter(textures.values())).astype(np.float32)
    n = main.shape[0]
    lum = main[..., :3] @ np.array([0.3, 0.55, 0.15], np.float32)
    gray = np.clip(lum / max(lum.mean(), 1) * 190, 0, 255)
    frames = []
    for f in range(WATER_FRAMES):
        s = int(round(f * n / WATER_FRAMES))
        frames.append(np.roll(gray, (s, s // 2), axis=(0, 1)))       # KO scrolls its water UVs
    still = np.concatenate(frames, 0)
    still = np.dstack([still, still, still, np.full(still.shape, 190)]).astype(np.uint8)
    flow = np.concatenate([np.roll(gray, s, axis=0) for s in
                           (np.arange(WATER_FRAMES) * n // WATER_FRAMES)], 0)
    flow = np.dstack([flow, flow, flow, np.full(flow.shape, 190)]).astype(np.uint8)
    meta = json.dumps({"animation": {"frametime": 3, "interpolate": True}}).encode()
    files = {
        "assets/minecraft/textures/block/water_still.png": _png(still),
        "assets/minecraft/textures/block/water_still.png.mcmeta": meta,
        "assets/minecraft/textures/block/water_flow.png": _png(flow),
        "assets/minecraft/textures/block/water_flow.png.mcmeta": meta,
    }
    tints = {}
    for name, t in textures.items():
        mean = t[..., :3].reshape(-1, 3).astype(np.float32).mean(0)
        tints[name] = tuple(int(v) for v in np.clip(mean * 255 / 190 * 1.1, 0, 255))
    return files, tints


# ---------------------------------------------------------------------------
# biomes (world datapack)
# ---------------------------------------------------------------------------

def _rgb(c):
    return (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])


def biome_json(sky: KOSky | None, water: tuple, grass=(110, 160, 60)) -> bytes:
    sky_c = sky.color("sky") if sky else (120, 167, 255)
    fog_c = sky.color("fog") if sky else (192, 216, 255)
    return json.dumps({
        "has_precipitation": False,
        "temperature": 0.7,
        "downfall": 0.3,
        "effects": {
            "sky_color": _rgb(sky_c),
            "fog_color": _rgb(fog_c),
            "water_color": _rgb(water),
            "water_fog_color": _rgb(tuple(int(v * 0.45) for v in water)),
            "grass_color": _rgb(grass),
            "foliage_color": _rgb(grass),
            "mood_sound": {"sound": "minecraft:ambient.cave", "tick_delay": 6000,
                           "block_search_extent": 8, "offset": 2.0},
        },
        "spawners": {},
        "spawn_costs": {},
        "carvers": {},
        "features": [],
    }, indent=2).encode()


def datapack(map_name: str, sky: KOSky | None, tints: dict[str, tuple]) -> tuple[dict[str, bytes], list[str]]:
    """World datapack with one KO biome per water texture. Returns (files, biome ids) - the first
    biome is the map's own (land, and water with the most used texture)."""
    slug = re.sub(r"[^a-z0-9_]", "_", map_name.lower())
    files = {"pack.mcmeta": json.dumps({"pack": {"pack_format": DATAPACK_FORMAT,
                                                 "description": f"Knight Online sky and water for {map_name}"}},
                                       indent=2).encode()}
    ids = []
    names = list(tints) or ["water"]
    for i, n in enumerate(names):
        bid = slug if i == 0 else f"{slug}_{re.sub(r'[^a-z0-9_]', '_', os.path.splitext(n.lower())[0])}"
        files[f"data/ko2mc/worldgen/biome/{bid}.json"] = biome_json(sky, tints.get(n, (63, 118, 228)))
        ids.append(f"ko2mc:{bid}")
    return files, ids


def biome_grid(water_body: np.ndarray, n_biomes: int) -> np.ndarray:
    """Biome per 4 x 4 column cell [z, x]: the water body it's in or nearest to.

    water_body: per column [z, x], index of the KO water texture there, -1 for dry land."""
    size = water_body.shape[0]
    cells = (size + 3) // 4
    pad = np.full((cells * 4, cells * 4), -1, np.int32)
    pad[:size, :size] = water_body
    blk = pad.reshape(cells, 4, cells, 4).transpose(0, 2, 1, 3).reshape(cells, cells, 16)
    grid = np.full((cells, cells), -1, np.int32)
    for i in range(n_biomes):
        grid[((blk == i).sum(-1) > 0) & (grid < 0)] = i
    if (grid < 0).all():
        return np.zeros((cells, cells), np.int32)
    # grow every water body's area over dry land (nearest body), a few cells per step
    while (grid < 0).any():
        g = np.pad(grid, 1, constant_values=-1)
        nb = np.stack([g[:-2, 1:-1], g[2:, 1:-1], g[1:-1, :-2], g[1:-1, 2:]])
        best = nb.max(0)
        fill = (grid < 0) & (best >= 0)
        if not fill.any():
            break
        grid[fill] = best[fill]
    grid[grid < 0] = 0
    return grid
