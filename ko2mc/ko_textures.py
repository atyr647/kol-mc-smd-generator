"""Knight Online terrain textures -> Minecraft resource pack.

KO terrain tiles are painted with textures stored in the client's
Data/dtex/*.gtt files. A .gtt holds several textures ("tiles") one after the
other, each in the N3Texture format:

    int32  name_length, char[name_length] name      (CN3BaseFileAccess)
    char   id[4]  = 'N','T','F', version
    int32  width, height
    uint32 D3DFORMAT (DXT1/DXT3/DXT5 FourCC or 16/32-bit RGB formats)
    int32  has_mipmaps
    pixel data (mip chain, sometimes followed by an uncompressed copy)

The .gtd says which (gtt file, tile number) every terrain tile uses.

Minecraft has no free blocks for hundreds of custom textures, so we use the
standard custom-block trick: note block states. A note block has 16
instruments x 25 notes, and a resource pack can give each state its own model
and texture. Minecraft recalculates a note block's instrument from the block
under it, so the converter puts the matching "instrument block" directly
underneath (e.g. stone for basedrum); that keeps the texture stable when
players build next to it. Right-clicking a note block still changes its note
(and therefore its texture), so this is best for exploring, not survival play.
"""

import io
import json
import os
import zipfile
from dataclasses import dataclass

import numpy as np

PACK_FORMAT = 22          # Minecraft 1.20.3 / 1.20.4
NAMESPACE = "ko2mc"

# instrument -> a block that produces it when placed under a note block
INSTRUMENT_BLOCKS = [
    ("harp", "minecraft:dirt"),
    ("basedrum", "minecraft:stone"),
    ("snare", "minecraft:sand"),        # safe: it always sits on solid ground
    ("hat", "minecraft:glass"),
    ("bass", "minecraft:oak_planks"),
    ("flute", "minecraft:clay"),
    ("bell", "minecraft:gold_block"),
    ("guitar", "minecraft:white_wool"),
    ("chime", "minecraft:packed_ice"),
    ("xylophone", "minecraft:bone_block"),
    ("iron_xylophone", "minecraft:iron_block"),
    ("cow_bell", "minecraft:soul_sand"),
    ("didgeridoo", "minecraft:pumpkin"),
    ("bit", "minecraft:emerald_block"),
    ("banjo", "minecraft:hay_block"),
    ("pling", "minecraft:glowstone"),
]
NOTES = 25
ALL_INSTRUMENTS = [i for i, _ in INSTRUMENT_BLOCKS] + [
    "zombie", "skeleton", "creeper", "dragon", "wither_skeleton", "piglin", "custom_head"]
MAX_CUSTOM = len(INSTRUMENT_BLOCKS) * NOTES * 2   # powered=false/true


def custom_state(i: int) -> tuple[str, str]:
    """Note block state + the block that must sit under it, for custom texture #i."""
    powered, rest = divmod(i, len(INSTRUMENT_BLOCKS) * NOTES)
    inst, note = divmod(rest, NOTES)
    name, below = INSTRUMENT_BLOCKS[inst]
    state = f"minecraft:note_block[instrument={name},note={note},powered={'true' if powered else 'false'}]"
    return state, below


# ---------------------------------------------------------------------------
# N3Texture / GTT decoding
# ---------------------------------------------------------------------------

_FOURCC = {0x31545844: ("DXT1", 1, 8), 0x32545844: ("DXT2", 2, 16), 0x33545844: ("DXT3", 2, 16),
           0x34545844: ("DXT4", 3, 16), 0x35545844: ("DXT5", 3, 16)}
_RGB = {20: ("R8G8B8", 3), 21: ("A8R8G8B8", 4), 22: ("X8R8G8B8", 4), 23: ("R5G6B5", 2),
        24: ("X1R5G5B5", 2), 25: ("A1R5G5B5", 2), 26: ("A4R4G4B4", 2)}


@dataclass
class N3Texture:
    width: int
    height: int
    fmt: str
    offset: int
    rgba: np.ndarray   # (h, w, 4) uint8


def _decode_level(data: bytes, off: int, w: int, h: int, fmt: int) -> tuple[np.ndarray, int]:
    from PIL import Image
    if fmt in _FOURCC:
        _, n, block = _FOURCC[fmt]
        size = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block
        raw = data[off:off + size]
        if len(raw) < size:
            raise ValueError("truncated")
        img = Image.frombytes("RGBA", (w, h), raw, "bcn", n)
        return np.array(img), size
    name, bpp = _RGB[fmt]
    size = w * h * bpp
    raw = data[off:off + size]
    if len(raw) < size:
        raise ValueError("truncated")
    if bpp == 2:
        v = np.frombuffer(raw, "<u2").reshape(h, w).astype(np.uint32)
        if name == "R5G6B5":
            r, g, b, a = (v >> 11) & 31, (v >> 5) & 63, v & 31, np.full_like(v, 1)
            rgba = [r * 255 // 31, g * 255 // 63, b * 255 // 31, a * 255]
        elif name == "A4R4G4B4":
            rgba = [((v >> 8) & 15) * 17, ((v >> 4) & 15) * 17, (v & 15) * 17, ((v >> 12) & 15) * 17]
        else:
            a = (v >> 15) & 1 if name == "A1R5G5B5" else np.ones_like(v)
            rgba = [((v >> 10) & 31) * 255 // 31, ((v >> 5) & 31) * 255 // 31, (v & 31) * 255 // 31, a * 255]
        return np.stack(rgba, -1).astype(np.uint8), size
    v = np.frombuffer(raw, np.uint8).reshape(h, w, bpp)
    out = np.empty((h, w, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = v[..., 2], v[..., 1], v[..., 0]   # BGR(A) in memory
    out[..., 3] = v[..., 3] if name == "A8R8G8B8" else 255
    return out, size


def read_n3_textures(data: bytes) -> list[N3Texture]:
    """Find and decode every N3Texture in a .gtt/.dxt file, in file order."""
    out = []
    pos = 0
    while True:
        hit = data.find(b"NTF", pos)
        if hit < 0 or hit + 20 > len(data):
            break
        ver = data[hit + 3]
        w, h, fmt, _mip = np.frombuffer(data, "<i4", 4, hit + 4)
        w, h, fmt = int(w), int(h), int(fmt) & 0xFFFFFFFF
        ok = (1 <= ver <= 9 and 4 <= w <= 4096 and 4 <= h <= 4096 and (w & (w - 1)) == 0
              and (h & (h - 1)) == 0 and (fmt in _FOURCC or fmt in _RGB))
        if not ok:
            pos = hit + 1
            continue
        try:
            rgba, size = _decode_level(data, hit + 20, w, h, fmt)
        except ValueError:
            pos = hit + 1
            continue
        name = _FOURCC[fmt][0] if fmt in _FOURCC else _RGB[fmt][0]
        out.append(N3Texture(w, h, name, hit, rgba))
        pos = hit + 20 + size
    return out


class TextureLibrary:
    """Finds and caches KO textures from a folder (e.g. the client's Data/dtex)."""

    def __init__(self, folder: str):
        self.folder = folder
        self.files = {}
        for root, _, names in os.walk(folder):
            for n in names:
                self.files.setdefault(n.lower(), os.path.join(root, n))
        self._cache = {}
        print(f"  KO textures: {len(self.files)} files found in {folder}")

    def textures_in(self, filename: str) -> list[N3Texture]:
        key = os.path.basename(filename.replace("\\", "/")).lower()
        if key not in self._cache:
            path = self.files.get(key)
            if path is None:
                self._cache[key] = None
            else:
                with open(path, "rb") as f:
                    self._cache[key] = read_n3_textures(f.read())
        return self._cache[key]

    def tile(self, gtt_name: str, index: int) -> np.ndarray | None:
        texs = self.textures_in(gtt_name)
        if not texs or not 0 <= index < len(texs):
            return None
        return texs[index].rgba


# ---------------------------------------------------------------------------
# Resource pack
# ---------------------------------------------------------------------------

class TexturePack:
    """Collects custom block textures and writes a Minecraft resource pack."""

    def __init__(self, title: str, resolution: int = 32, brightness: float = 1.3):
        self.title = title
        self.resolution = resolution
        # The KO client lights terrain with "modulate 2x" (then darkens it with its
        # colour map), so stored textures are darker than they look in game.
        self.brightness = brightness
        self.images: list[np.ndarray] = []
        self.names: list[str] = []

    def add(self, rgba: np.ndarray, label: str) -> int | None:
        if len(self.images) >= MAX_CUSTOM:
            return None
        self.images.append(rgba)
        self.names.append(label)
        return len(self.images) - 1

    def average_color(self, i: int) -> tuple[int, int, int]:
        return tuple(int(c) for c in self.images[i][..., :3].reshape(-1, 3).mean(0))

    def _png(self, rgba: np.ndarray) -> bytes:
        from PIL import Image
        rgb = np.clip(rgba[..., :3].astype(np.float32) * self.brightness, 0, 255).astype(np.uint8)
        im = Image.fromarray(rgb, "RGB")   # terrain is opaque
        r = self.resolution
        if im.width != r or im.height != r:
            im = im.resize((r, r), Image.BOX if im.width > r else Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def write(self, path: str):
        variants = {}
        files = {}
        for i, (img, label) in enumerate(zip(self.images, self.names)):
            state, _ = custom_state(i)
            props = state[state.index("[") + 1:-1]
            model = f"{NAMESPACE}:block/t{i}"
            variants[props] = {"model": model}
            files[f"assets/{NAMESPACE}/models/block/t{i}.json"] = json.dumps(
                {"parent": "minecraft:block/cube_all", "textures": {"all": f"{NAMESPACE}:block/t{i}"}})
            files[f"assets/{NAMESPACE}/textures/block/t{i}.png"] = self._png(img)
        # every other note block state keeps the normal look (all states must be listed)
        for inst in ALL_INSTRUMENTS:
            for note in range(NOTES):
                for powered in ("false", "true"):
                    variants.setdefault(f"instrument={inst},note={note},powered={powered}",
                                        {"model": "minecraft:block/note_block"})
        files["assets/minecraft/blockstates/note_block.json"] = json.dumps({"variants": variants}, indent=1)
        files["pack.mcmeta"] = json.dumps({"pack": {
            "pack_format": PACK_FORMAT,
            "description": f"Knight Online terrain textures for {self.title} (ko2mc)"}}, indent=2)
        files["ko2mc_textures.txt"] = "\n".join(
            f"t{i}\t{custom_state(i)[0]}\t{n}" for i, n in enumerate(self.names))
        if self.images:
            files["pack.png"] = self._png(self.images[0])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        print(f"  wrote resource pack {path} ({len(self.images)} KO textures)")


def tile_texture_key(gtd, tex_idx: int) -> tuple[str, int] | None:
    """(gtt file name, tile number inside it) for a tile texture index of a GTDFile."""
    if not 0 <= tex_idx < len(gtd.tile_files):
        return None
    return gtd.tile_files[tex_idx], gtd.tile_subindex[tex_idx]
