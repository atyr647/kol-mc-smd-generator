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
underneath (e.g. netherrack for basedrum); that keeps the texture stable when
players build next to it. Ground textures are grouped by look per instrument,
and the instrument block gets its group's look, so terrace edges match. Right-clicking a note block still changes its note
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

# instrument -> a block that produces it when placed under a note block. These
# blocks show on terrace edges, so the pack gives them a KO ground look too
# (see TexturePack.set_under); they're picked so a converted world uses them
# nowhere else. (pling needs glowstone, which would light up the ground, so it's left out.)
INSTRUMENT_BLOCKS = [
    ("harp", "minecraft:rooted_dirt"),
    ("basedrum", "minecraft:netherrack"),
    ("snare", "minecraft:light_gray_concrete_powder"),   # safe: always sits on solid ground
    ("hat", "minecraft:glass"),
    ("bass", "minecraft:bookshelf"),
    ("flute", "minecraft:clay"),
    ("bell", "minecraft:gold_block"),
    ("guitar", "minecraft:white_wool"),
    ("chime", "minecraft:packed_ice"),
    ("xylophone", "minecraft:bone_block[axis=y]"),
    ("iron_xylophone", "minecraft:iron_block"),
    ("cow_bell", "minecraft:soul_sand"),
    ("didgeridoo", "minecraft:pumpkin"),
    ("bit", "minecraft:emerald_block"),
    ("banjo", "minecraft:hay_block[axis=y]"),
]
NOTES = 25
SLOTS_PER_INSTRUMENT = NOTES * 2                   # powered=false/true
ALL_INSTRUMENTS = [i for i, _ in INSTRUMENT_BLOCKS] + [
    "pling", "zombie", "skeleton", "creeper", "dragon", "wither_skeleton", "piglin", "custom_head"]
MAX_CUSTOM = len(INSTRUMENT_BLOCKS) * SLOTS_PER_INSTRUMENT


def custom_state(i: int) -> tuple[str, str]:
    """Note block state + the block that must sit under it, for custom texture #i.

    Slots are grouped by instrument: 0-49 harp, 50-99 basedrum, ..."""
    inst, j = divmod(i, SLOTS_PER_INSTRUMENT)
    powered, note = divmod(j, NOTES)
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
    """Collects custom block textures and writes a Minecraft resource pack.

    Ground textures use note block states (add()); objects use the host states in
    custom_blocks.py (add_solid(), add_foliage(), add_plant(), set_stairs(), set_slab()).
    """

    def __init__(self, title: str, resolution: int = 32, brightness: float = 1.3):
        self.title = title
        self.resolution = resolution
        # The KO client lights terrain with "modulate 2x" (then darkens it with its
        # colour map), so stored textures are darker than they look in game.
        self.brightness = brightness
        self.images: dict[int, np.ndarray] = {}   # ground: note block slot -> image
        self.names: dict[int, str] = {}
        self.under: dict[str, np.ndarray] = {}    # instrument block state -> image
        from . import custom_blocks as cb
        self._solid_slots = cb.solid_slots()
        self._foliage_slots = cb.foliage_slots()
        self._plant_slots = cb.plant_slots()
        self.solid: list[np.ndarray] = []         # object textures (already final colours)
        self.foliage: list[np.ndarray] = []
        self.plants: list[tuple[np.ndarray, int]] = []
        self.stairs: dict[str, np.ndarray] = {}
        self.slabs: dict[str, np.ndarray] = {}
        self.extra: dict[str, bytes] = {}         # other pack files (KO sky, water; see ko_sky.py)

    @property
    def empty(self) -> bool:
        return not (self.images or self.solid or self.foliage or self.plants or self.extra)

    # ---- ground ----
    def add(self, rgba: np.ndarray, label: str, slot: int | None = None) -> int | None:
        """Ground texture; slot picks the note block state (see custom_state), default next free."""
        if slot is None:
            slot = next((i for i in range(MAX_CUSTOM) if i not in self.images), None)
        if slot is None or not 0 <= slot < MAX_CUSTOM:
            return None
        self.images[slot] = rgba
        self.names[slot] = label
        return slot

    def set_under(self, block_state: str, rgba: np.ndarray):
        """Ground look for an instrument block (shows on terrace edges)."""
        self.under[block_state] = rgba

    # ---- objects ----
    @property
    def max_solid(self) -> int:
        return len(self._solid_slots)

    @property
    def max_foliage(self) -> int:
        return len(self._foliage_slots)

    @property
    def max_plants(self) -> int:
        return len(self._plant_slots)

    def add_solid(self, rgba: np.ndarray) -> str | None:
        if len(self.solid) >= self.max_solid:
            return None
        self.solid.append(rgba)
        return self._solid_slots[len(self.solid) - 1][0]

    def add_foliage(self, rgba: np.ndarray) -> str | None:
        if len(self.foliage) >= self.max_foliage:
            return None
        self.foliage.append(rgba)
        return self._foliage_slots[len(self.foliage) - 1][0]

    def add_plant(self, rgba: np.ndarray, height_px: int = 16) -> str | None:
        if len(self.plants) >= self.max_plants:
            return None
        self.plants.append((rgba, int(np.clip(height_px, 4, 32))))
        return self._plant_slots[len(self.plants) - 1][0]

    def set_stairs(self, block_type: str, rgba: np.ndarray):
        self.stairs[block_type] = rgba

    def set_slab(self, block_type: str, rgba: np.ndarray):
        self.slabs[block_type] = rgba

    def average_color(self, i: int) -> tuple[int, int, int]:  # i = ground slot
        return tuple(int(c) for c in self.images[i][..., :3].reshape(-1, 3).mean(0))

    def _png(self, rgba: np.ndarray, brighten: bool = True, alpha: bool = False) -> bytes:
        from PIL import Image
        f = self.brightness if brighten else 1.0
        rgb = np.clip(rgba[..., :3].astype(np.float32) * f, 0, 255).astype(np.uint8)
        if alpha:
            a = rgba[..., 3] if rgba.shape[-1] == 4 else np.full(rgb.shape[:2], 255, np.uint8)
            im = Image.fromarray(np.dstack([rgb, np.where(a >= 128, 255, 0).astype(np.uint8)]), "RGBA")
        else:
            im = Image.fromarray(rgb, "RGB")
        r = self.resolution
        if im.width != r or im.height != r:
            im = im.resize((r, r), Image.NEAREST if alpha or im.width < r else Image.BOX)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def write(self, path: str):
        from . import custom_blocks as cb
        ns = NAMESPACE
        files = {}
        states: dict[str, dict] = {}          # host block -> variant key -> model

        def cube(name, tex, parent="minecraft:block/cube_all"):
            files[f"assets/{ns}/models/block/{name}.json"] = json.dumps(
                {"parent": parent, "textures": {"all": f"{ns}:block/{tex}"}})
            return f"{ns}:block/{name}"

        # ground: note block states with the instrument block underneath
        for i, img in sorted(self.images.items()):
            state, _ = custom_state(i)
            props = state[state.index("[") + 1:-1]
            states.setdefault("note_block", {})[props] = {"model": cube(f"t{i}", f"t{i}")}
            files[f"assets/{ns}/textures/block/t{i}.png"] = self._png(img)
        # instrument blocks under the ground: their group's look on every face
        under_hosts = set()
        for st, img in self.under.items():
            b = st.split(":")[1].split("[")[0]
            under_hosts.add(b)
            m = cube(f"u_{b}", f"u_{b}")
            files[f"assets/{ns}/textures/block/u_{b}.png"] = self._png(img)
            for k in (cb.all_block_states(b) if b in cb.UNDER_STATES else [""]):
                states.setdefault(b, {})[k] = {"model": m}
        # solid object blocks
        for i, img in enumerate(self.solid):
            _, host, keys = self._solid_slots[i]
            m = cube(f"o{i}", f"o{i}")
            files[f"assets/{ns}/textures/block/o{i}.png"] = self._png(img, brighten=False)
            for k in keys:
                states.setdefault(host, {})[k] = {"model": m}
        # foliage: leaf blocks (drawn see-through), no biome tint
        for i, img in enumerate(self.foliage):
            _, host, keys = self._foliage_slots[i]
            m = cube(f"f{i}", f"f{i}")
            files[f"assets/{ns}/textures/block/f{i}.png"] = self._png(img, brighten=False, alpha=True)
            for k in keys:
                states.setdefault(host, {})[k] = {"model": m}
        # plants: tripwire drawn as two crossed sprites
        for i, (img, h) in enumerate(self.plants):
            _, host, keys = self._plant_slots[i]
            t = f"{ns}:block/p{i}"
            files[f"assets/{ns}/textures/block/p{i}.png"] = self._png(img, brighten=False, alpha=True)
            plane = {"uv": [0, 0, 16, 16], "texture": "#cross"}
            files[f"assets/{ns}/models/block/p{i}.json"] = json.dumps({
                "ambientocclusion": False, "textures": {"particle": t, "cross": t},
                "elements": [
                    {"from": [0.8, 0, 8], "to": [15.2, h, 8], "shade": False,
                     "rotation": {"origin": [8, 8, 8], "axis": "y", "angle": 45, "rescale": True},
                     "faces": {"north": plane, "south": plane}},
                    {"from": [8, 0, 0.8], "to": [8, h, 15.2], "shade": False,
                     "rotation": {"origin": [8, 8, 8], "axis": "y", "angle": 45, "rescale": True},
                     "faces": {"west": plane, "east": plane}}]})
            for k in keys:
                states.setdefault(host, {})[k] = {"model": f"{ns}:block/p{i}"}
        # stairs and slabs: re-texture the vanilla models (keeps all rotations)
        for t, img in self.stairs.items():
            tex = f"{ns}:block/st_{t}"
            files[f"assets/{ns}/textures/block/st_{t}.png"] = self._png(img, brighten=False)
            for suffix, parent in (("", "stairs"), ("_inner", "inner_stairs"), ("_outer", "outer_stairs")):
                files[f"assets/minecraft/models/block/{t}_stairs{suffix}.json"] = json.dumps(
                    {"parent": f"minecraft:block/{parent}", "textures": {"bottom": tex, "top": tex, "side": tex}})
        for t, img in self.slabs.items():
            tex = f"{ns}:block/sl_{t}"
            files[f"assets/{ns}/textures/block/sl_{t}.png"] = self._png(img, brighten=False)
            for suffix, parent in (("_slab", "slab"), ("_slab_top", "slab_top")):
                files[f"assets/minecraft/models/block/{t}{suffix}.json"] = json.dumps(
                    {"parent": f"minecraft:block/{parent}", "textures": {"bottom": tex, "top": tex, "side": tex}})

        # complete blockstates files: every state of a host must be listed
        for host, variants in states.items():
            if host == "note_block":
                for inst in ALL_INSTRUMENTS:
                    for note in range(NOTES):
                        for powered in ("false", "true"):
                            variants.setdefault(f"instrument={inst},note={note},powered={powered}",
                                                {"model": "minecraft:block/note_block"})
            elif host in under_hosts:
                pass                                   # every state already listed
            elif host.endswith("_slab"):
                # only the double slab carries a KO block look; half slabs keep their models
                for half, suffix in (("bottom", ""), ("top", "_top")):
                    for w in ("false", "true"):
                        variants.setdefault(f"type={half},waterlogged={w}",
                                            {"model": f"minecraft:block/{host}{suffix}"})
            else:
                fallback = next(iter(variants.values()))
                for k in cb.all_block_states(host):
                    variants.setdefault(k, fallback)
            files[f"assets/minecraft/blockstates/{host}.json"] = json.dumps({"variants": variants})

        files.update(self.extra)
        files["pack.mcmeta"] = json.dumps({"pack": {
            "pack_format": PACK_FORMAT,
            "description": f"Knight Online textures for {self.title} (ko2mc)"}}, indent=2)
        files["ko2mc_textures.txt"] = "\n".join(
            [f"t{i}\t{custom_state(i)[0]}\t{n}" for i, n in sorted(self.names.items())]
            + [f"u\t{b}" for b in self.under]
            + [f"o{i}\t{self._solid_slots[i][0]}" for i in range(len(self.solid))]
            + [f"f{i}\t{self._foliage_slots[i][0]}" for i in range(len(self.foliage))]
            + [f"p{i}\t{self._plant_slots[i][0]}" for i in range(len(self.plants))])
        icon = next(iter(self.images.values())) if self.images else (self.solid[0] if self.solid else None)
        if icon is not None:
            files["pack.png"] = self._png(icon, brighten=bool(self.images))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        print(f"  wrote resource pack {path} ({len(self.images)} ground, {len(self.solid)} building, "
              f"{len(self.foliage)} foliage, {len(self.plants)} plant, "
              f"{len(self.stairs) + len(self.slabs)} step textures)")


def tile_texture_key(gtd, tex_idx: int) -> tuple[str, int] | None:
    """(gtt file name, tile number inside it) for a tile texture index of a GTDFile."""
    if not 0 <= tex_idx < len(gtd.tile_files):
        return None
    return gtd.tile_files[tex_idx], gtd.tile_subindex[tex_idx]
