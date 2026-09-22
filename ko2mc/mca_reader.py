"""Read blocks back out of a Minecraft Java world (Anvil .mca region files).

Used by the previewer, so what you see in the preview is exactly what was
written to disk (and what Minecraft will load). Works with any 1.18+ world,
not just ones made by ko2mc.
"""

import gzip
import math
import os
import re
import struct
import zlib

import numpy as np

from . import nbt

MIN_Y = -64
HEIGHT = 384


def _unpack_indices(longs: np.ndarray, palette_size: int) -> np.ndarray:
    bits = max(4, math.ceil(math.log2(palette_size)))
    per_long = 64 // bits
    u = longs.astype(">i8").view(">u8").astype(np.uint64)
    shifts = np.arange(per_long, dtype=np.uint64) * np.uint64(bits)
    mask = np.uint64((1 << bits) - 1)
    vals = (u[:, None] >> shifts[None, :]) & mask
    return vals.ravel()[:4096].astype(np.int32)


def _state_name(entry: dict) -> str:
    name = entry.get("Name", "minecraft:air")
    props = entry.get("Properties")
    if props:
        name += "[" + ",".join(f"{k}={v}" for k, v in sorted(props.items())) + "]"
    return name


class WorldReader:
    """Loads chunks from a world folder. Block ids refer to self.palette."""

    def __init__(self, world_dir: str):
        self.world_dir = world_dir
        self.region_dir = os.path.join(world_dir, "region")
        if not os.path.isdir(self.region_dir):
            raise FileNotFoundError(f"No region folder in {world_dir}")
        self.palette: list[str] = ["minecraft:air"]
        self._ids = {"minecraft:air": 0}
        self.regions = {}
        for f in os.listdir(self.region_dir):
            m = re.match(r"r\.(-?\d+)\.(-?\d+)\.mca$", f)
            if m:
                self.regions[(int(m.group(1)), int(m.group(2)))] = os.path.join(self.region_dir, f)
        self._headers = {}

    def level_info(self) -> dict:
        path = os.path.join(self.world_dir, "level.dat")
        if not os.path.exists(path):
            return {}
        with gzip.open(path, "rb") as f:
            return nbt.decode(f.read()).get("Data", {})

    def _id(self, state: str) -> int:
        i = self._ids.get(state)
        if i is None:
            i = len(self.palette)
            self.palette.append(state)
            self._ids[state] = i
        return i

    def _header(self, rx, rz):
        if (rx, rz) not in self._headers:
            path = self.regions.get((rx, rz))
            if path is None:
                self._headers[(rx, rz)] = None
            else:
                with open(path, "rb") as f:
                    self._headers[(rx, rz)] = struct.unpack(">1024I", f.read(4096))
        return self._headers[(rx, rz)]

    def chunk_positions(self) -> list[tuple[int, int]]:
        out = []
        for (rx, rz) in self.regions:
            hdr = self._header(rx, rz)
            for i, loc in enumerate(hdr):
                if loc:
                    out.append((rx * 32 + (i & 31), rz * 32 + (i >> 5)))
        return sorted(out)

    def read_chunk_nbt(self, cx: int, cz: int) -> dict | None:
        rx, rz = cx >> 5, cz >> 5
        hdr = self._header(rx, rz)
        if not hdr:
            return None
        loc = hdr[(cx & 31) + (cz & 31) * 32]
        if not loc:
            return None
        with open(self.regions[(rx, rz)], "rb") as f:
            f.seek((loc >> 8) * 4096)
            length, comp = struct.unpack(">iB", f.read(5))
            raw = f.read(length - 1)
        if comp == 2:
            raw = zlib.decompress(raw)
        elif comp == 1:
            raw = gzip.decompress(raw)
        elif comp != 3:
            raise ValueError(f"Unsupported chunk compression {comp}")
        return nbt.decode(raw)

    def read_chunk(self, cx: int, cz: int) -> np.ndarray | None:
        """Return blocks as a (384, 16, 16) [y, z, x] array of palette ids, or None."""
        root = self.read_chunk_nbt(cx, cz)
        if root is None:
            return None
        out = np.zeros((HEIGHT, 16, 16), dtype=np.uint16)
        for sec in root.get("sections", []):
            sy = sec.get("Y", 0)
            base = (sy * 16) - MIN_Y
            if base < 0 or base >= HEIGHT:
                continue
            states = sec.get("block_states")
            if not states:
                continue
            pal = [self._id(_state_name(e)) for e in states.get("palette", [])]
            if not pal:
                continue
            lut = np.array(pal, dtype=np.uint16)
            data = states.get("data")
            if data is None or len(pal) == 1:
                out[base:base + 16] = lut[0]
            else:
                idx = _unpack_indices(np.asarray(data), len(pal))
                out[base:base + 16] = lut[np.clip(idx, 0, len(pal) - 1)].reshape(16, 16, 16)
        return out
