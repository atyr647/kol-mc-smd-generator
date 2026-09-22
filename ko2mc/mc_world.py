"""Minecraft Java Edition world writer (Anvil region format, 1.20.4).

The world is built from two layers:

  1. A *terrain provider*: a function that, given a chunk position, fills a
     dense (384, 16, 16) array of block ids for that chunk. Terrain is
     generated chunk by chunk while saving, so even huge maps use little memory.
  2. *Placed blocks*: individual blocks (trees, buildings, markers...) set with
     set_block()/set_blocks(). They are drawn on top of the terrain, and later
     placements win over earlier ones.

Block names may include block state properties, Minecraft-style:
  "minecraft:oak_leaves[persistent=true]"
"""

import gzip
import math
import os
import struct
import time
import zlib

import numpy as np

from . import nbt

MC_DATA_VERSION = 3700      # Minecraft 1.20.4
MC_VERSION_NAME = "1.20.4"
MIN_Y = -64
MAX_Y = 319
HEIGHT = MAX_Y - MIN_Y + 1  # 384
MIN_SECTION_Y = MIN_Y >> 4  # -4
NUM_SECTIONS = HEIGHT // 16  # 24

AIR = 0


def parse_block_state(state: str) -> tuple[str, dict]:
    """'minecraft:oak_log[axis=y]' -> ('minecraft:oak_log', {'axis': 'y'})"""
    if "[" not in state:
        return state, {}
    name, props = state[:-1].split("[", 1)
    out = {}
    for kv in props.split(","):
        if kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return name, out


def _pack_indices(indices: np.ndarray, palette_size: int) -> np.ndarray:
    """Pack 4096 palette indices into the 1.16+ long array (entries never span longs)."""
    bits = max(4, math.ceil(math.log2(palette_size)))
    per_long = 64 // bits
    n_longs = math.ceil(4096 / per_long)
    padded = np.zeros(n_longs * per_long, dtype=np.uint64)
    padded[:4096] = indices
    padded = padded.reshape(n_longs, per_long)
    shifts = (np.arange(per_long, dtype=np.uint64) * np.uint64(bits))
    longs = np.bitwise_or.reduce(padded << shifts, axis=1)
    return longs.view(np.int64)


class MinecraftWorld:
    """Collects blocks and writes a Minecraft world folder."""

    def __init__(self, world_dir: str, world_name: str = "KnightOnline"):
        self.world_dir = world_dir
        self.world_name = world_name
        self.palette: list[str] = ["minecraft:air"]
        self._palette_index = {"minecraft:air": 0}
        self.terrain_provider = None
        self.bounds = None  # (min_x, min_z, max_x, max_z) in blocks, inclusive
        self.spawn = (0, 100, 0)
        self.biome = "minecraft:plains"
        self._pending: list[tuple[np.ndarray, ...]] = []
        self._single: list[tuple[int, int, int, int]] = []

    # ---- palette ---------------------------------------------------------

    def block_id(self, block: str) -> int:
        idx = self._palette_index.get(block)
        if idx is None:
            idx = len(self.palette)
            self.palette.append(block)
            self._palette_index[block] = idx
        return idx

    # ---- block placement -------------------------------------------------

    def set_terrain(self, provider, bounds):
        """provider(cx, cz, out) fills out[y - MIN_Y, z, x] with block ids."""
        self.terrain_provider = provider
        self.bounds = bounds

    def set_block(self, x: int, y: int, z: int, block: str):
        if MIN_Y <= y <= MAX_Y:
            self._single.append((x, y, z, self.block_id(block)))

    def set_blocks(self, xs, ys, zs, block):
        """Place many blocks at once. `block` is a name or an array of ids."""
        xs = np.asarray(xs, dtype=np.int32).ravel()
        ys = np.asarray(ys, dtype=np.int32).ravel()
        zs = np.asarray(zs, dtype=np.int32).ravel()
        if isinstance(block, str):
            ids = np.full(len(xs), self.block_id(block), dtype=np.uint16)
        else:
            ids = np.asarray(block, dtype=np.uint16).ravel()
        keep = (ys >= MIN_Y) & (ys <= MAX_Y)
        self._flush_single()
        self._pending.append((xs[keep], ys[keep], zs[keep], ids[keep]))

    def fill_box(self, x0, y0, z0, x1, y1, z1, block: str):
        """Fill an inclusive box."""
        xs, ys, zs = np.meshgrid(np.arange(min(x0, x1), max(x0, x1) + 1),
                                 np.arange(min(y0, y1), max(y0, y1) + 1),
                                 np.arange(min(z0, z1), max(z0, z1) + 1), indexing="ij")
        self.set_blocks(xs, ys, zs, block)

    def _flush_single(self):
        if self._single:
            a = np.array(self._single, dtype=np.int32)
            self._single = []
            self._pending.append((a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.uint16)))

    def _placed_by_chunk(self):
        """Group placed blocks by chunk, keeping placement order (last wins)."""
        self._flush_single()
        if not self._pending:
            return {}, None
        xs = np.concatenate([p[0] for p in self._pending])
        ys = np.concatenate([p[1] for p in self._pending])
        zs = np.concatenate([p[2] for p in self._pending])
        ids = np.concatenate([p[3] for p in self._pending])
        self._pending = []
        key = (xs >> 4).astype(np.int64) * 1_000_003 + (zs >> 4)
        order = np.argsort(key, kind="stable")
        key = key[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], len(key)]
        groups = {}
        for s, e in zip(starts, ends):
            o = order[s:e]
            groups[(int(xs[o[0]] >> 4), int(zs[o[0]] >> 4))] = o
        return groups, (xs, ys, zs, ids)

    # ---- saving ------------------------------------------------------------

    def save(self):
        """Write region files and level.dat."""
        region_dir = os.path.join(self.world_dir, "region")
        os.makedirs(region_dir, exist_ok=True)
        for f in os.listdir(region_dir):
            if f.endswith(".mca"):
                os.remove(os.path.join(region_dir, f))

        groups, placed = self._placed_by_chunk()

        chunks = set(groups)
        if self.bounds:
            x0, z0, x1, z1 = self.bounds
            for cx in range(x0 >> 4, (x1 >> 4) + 1):
                for cz in range(z0 >> 4, (z1 >> 4) + 1):
                    chunks.add((cx, cz))

        regions: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cx, cz in chunks:
            regions.setdefault((cx >> 5, cz >> 5), []).append((cx, cz))

        print(f"  Saving {len(chunks)} chunks across {len(regions)} region files "
              f"({len(self.palette)} block types)...")
        palette_nbt = [self._palette_entry(b) for b in self.palette]
        blocks = np.zeros((HEIGHT, 16, 16), dtype=np.uint16)
        done = 0
        for (rx, rz), members in sorted(regions.items()):
            payloads = {}
            for cx, cz in members:
                blocks.fill(AIR)
                if self.terrain_provider:
                    self.terrain_provider(cx, cz, blocks)
                o = groups.get((cx, cz))
                if o is not None:
                    xs, ys, zs, ids = placed
                    blocks[ys[o] - MIN_Y, zs[o] & 15, xs[o] & 15] = ids[o]
                payloads[(cx, cz)] = self._chunk_bytes(cx, cz, blocks, palette_nbt)
                done += 1
            self._write_region(region_dir, rx, rz, payloads)
            print(f"    region r.{rx}.{rz}.mca ({done}/{len(chunks)} chunks)")

        self._write_level_dat()
        print(f"  World saved to {self.world_dir}")

    @staticmethod
    def _palette_entry(block: str) -> dict:
        name, props = parse_block_state(block)
        entry = {"Name": name}
        if props:
            entry["Properties"] = props
        return entry

    def _chunk_bytes(self, cx: int, cz: int, blocks: np.ndarray, palette_nbt) -> bytes:
        sections = []
        for si in range(NUM_SECTIONS):
            sec = blocks[si * 16:(si + 1) * 16]  # (y, z, x) -> index y*256 + z*16 + x
            flat = sec.ravel()
            first = flat[0]
            if (flat == first).all():
                states = {"palette": nbt.List(nbt.TAG_COMPOUND, [palette_nbt[first]])}
            else:
                uniq, inverse = np.unique(flat, return_inverse=True)
                states = {
                    "palette": nbt.List(nbt.TAG_COMPOUND, [palette_nbt[u] for u in uniq]),
                    "data": _pack_indices(inverse.astype(np.uint64), len(uniq)),
                }
            sections.append({
                "Y": nbt.Byte(MIN_SECTION_Y + si),
                "block_states": states,
                "biomes": {"palette": nbt.List(nbt.TAG_STRING, [self.biome])},
            })
        root = {
            "DataVersion": nbt.Int(MC_DATA_VERSION),
            "xPos": nbt.Int(cx),
            "yPos": nbt.Int(MIN_SECTION_Y),
            "zPos": nbt.Int(cz),
            "Status": "minecraft:full",
            "LastUpdate": nbt.Long(0),
            "InhabitedTime": nbt.Long(0),
            "sections": nbt.List(nbt.TAG_COMPOUND, sections),
            "block_entities": nbt.List(nbt.TAG_COMPOUND, []),
            "Heightmaps": {},
            "PostProcessing": nbt.List(nbt.TAG_LIST, []),
            "structures": {"References": {}, "starts": {}},
        }
        return nbt.encode(root)

    @staticmethod
    def _write_region(region_dir: str, rx: int, rz: int, payloads: dict):
        header = bytearray(8192)
        body = []
        sector = 2
        now = int(time.time())
        for (cx, cz), raw in payloads.items():
            data = zlib.compress(raw, 6)
            blob = struct.pack(">iB", len(data) + 1, 2) + data
            blob += b"\x00" * (-len(blob) % 4096)
            count = len(blob) // 4096
            if count > 255:
                raise ValueError(f"Chunk {cx},{cz} is too large for a region file")
            idx = (cx & 31) + (cz & 31) * 32
            struct.pack_into(">I", header, idx * 4, (sector << 8) | count)
            struct.pack_into(">I", header, 4096 + idx * 4, now)
            body.append(blob)
            sector += count
        with open(os.path.join(region_dir, f"r.{rx}.{rz}.mca"), "wb") as f:
            f.write(header)
            for b in body:
                f.write(b)

    def _write_level_dat(self):
        sx, sy, sz = self.spawn
        flat_overworld = {
            "type": "minecraft:overworld",
            "generator": {
                "type": "minecraft:flat",
                "settings": {
                    "biome": "minecraft:the_void",
                    "features": nbt.Byte(0),
                    "lakes": nbt.Byte(0),
                    "layers": nbt.List(nbt.TAG_COMPOUND, [
                        {"block": "minecraft:air", "height": nbt.Int(1)}]),
                    "structure_overrides": nbt.List(nbt.TAG_STRING, []),
                },
            },
        }
        data = {
            "DataVersion": nbt.Int(MC_DATA_VERSION),
            "version": nbt.Int(19133),
            "Version": {"Id": nbt.Int(MC_DATA_VERSION), "Name": MC_VERSION_NAME,
                        "Series": "main", "Snapshot": nbt.Byte(0)},
            "LevelName": self.world_name,
            "SpawnX": nbt.Int(sx), "SpawnY": nbt.Int(sy), "SpawnZ": nbt.Int(sz),
            "SpawnAngle": nbt.Float(0.0),
            "GameType": nbt.Int(1),  # Creative
            "hardcore": nbt.Byte(0),
            "allowCommands": nbt.Byte(1),
            "initialized": nbt.Byte(1),
            "Difficulty": nbt.Byte(0),  # Peaceful
            "DifficultyLocked": nbt.Byte(0),
            "Time": nbt.Long(6000),
            "DayTime": nbt.Long(6000),
            "LastPlayed": nbt.Long(int(time.time() * 1000)),
            "raining": nbt.Byte(0), "rainTime": nbt.Int(0),
            "thundering": nbt.Byte(0), "thunderTime": nbt.Int(0),
            "clearWeatherTime": nbt.Int(0),
            "GameRules": {"doDaylightCycle": "false", "doWeatherCycle": "false",
                          "doMobSpawning": "false", "randomTickSpeed": "0"},
            "DragonFight": {"NeedsStateScanning": nbt.Byte(1), "DragonKilled": nbt.Byte(0),
                            "PreviouslyKilled": nbt.Byte(0)},
            "DataPacks": {"Enabled": nbt.List(nbt.TAG_STRING, ["vanilla"]),
                          "Disabled": nbt.List(nbt.TAG_STRING, [])},
            "WorldGenSettings": {
                "seed": nbt.Long(0),
                "generate_features": nbt.Byte(0),
                "bonus_chest": nbt.Byte(0),
                "dimensions": {
                    "minecraft:overworld": flat_overworld,
                    "minecraft:the_nether": {
                        "type": "minecraft:the_nether",
                        "generator": {"type": "minecraft:noise", "settings": "minecraft:nether",
                                      "biome_source": {"type": "minecraft:multi_noise",
                                                       "preset": "minecraft:nether"}},
                    },
                    "minecraft:the_end": {
                        "type": "minecraft:the_end",
                        "generator": {"type": "minecraft:noise", "settings": "minecraft:end",
                                      "biome_source": {"type": "minecraft:the_end"}},
                    },
                },
            },
        }
        os.makedirs(self.world_dir, exist_ok=True)
        with gzip.open(os.path.join(self.world_dir, "level.dat"), "wb") as f:
            f.write(nbt.encode({"Data": data}))
