"""Minecraft world generator using the Anvil region file format.

Generates a Minecraft Java Edition world from block data.
Targets Minecraft 1.21.4 (data version 4189, chunk format with sections at Y=-64 to 319).
"""

import gzip
import io
import math
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Optional

# Minecraft data version for 1.21.4
MC_DATA_VERSION = 4189
MIN_SECTION_Y = -4  # Y=-64 in section coords
MAX_SECTION_Y = 19   # Y=319 in section coords


def _write_nbt_tag(buf: io.BytesIO, tag_type: int, name: Optional[str], value):
    """Write an NBT tag to the buffer."""
    if name is not None:
        buf.write(struct.pack(">bH", tag_type, len(name)))
        buf.write(name.encode("utf-8"))
    else:
        # Inside a list, no type/name header
        pass

    if tag_type == 1:  # TAG_Byte
        buf.write(struct.pack(">b", value))
    elif tag_type == 2:  # TAG_Short
        buf.write(struct.pack(">h", value))
    elif tag_type == 3:  # TAG_Int
        buf.write(struct.pack(">i", value))
    elif tag_type == 4:  # TAG_Long
        buf.write(struct.pack(">q", value))
    elif tag_type == 5:  # TAG_Float
        buf.write(struct.pack(">f", value))
    elif tag_type == 6:  # TAG_Double
        buf.write(struct.pack(">d", value))
    elif tag_type == 7:  # TAG_Byte_Array
        buf.write(struct.pack(">i", len(value)))
        buf.write(bytes(value))
    elif tag_type == 8:  # TAG_String
        encoded = value.encode("utf-8")
        buf.write(struct.pack(">H", len(encoded)))
        buf.write(encoded)
    elif tag_type == 11:  # TAG_Int_Array
        buf.write(struct.pack(">i", len(value)))
        for v in value:
            buf.write(struct.pack(">i", v))
    elif tag_type == 12:  # TAG_Long_Array
        buf.write(struct.pack(">i", len(value)))
        for v in value:
            buf.write(struct.pack(">q", v))


def _pack_block_states(indices: list[int], palette_size: int) -> list[int]:
    """Pack block state indices into a long array for MC chunk sections.

    Uses the compacted format: bits per entry = max(4, ceil(log2(palette_size))).
    Each long holds floor(64 / bits_per_entry) entries, entries don't span longs.
    """
    if palette_size <= 1:
        # Single block type, no block states array needed in modern format
        # but we still provide it for compatibility
        bits_per_entry = 4
    else:
        bits_per_entry = max(4, math.ceil(math.log2(palette_size)))

    entries_per_long = 64 // bits_per_entry
    num_longs = math.ceil(4096 / entries_per_long)
    longs = []

    for long_idx in range(num_longs):
        val = 0
        for entry_idx in range(entries_per_long):
            block_idx = long_idx * entries_per_long + entry_idx
            if block_idx < 4096:
                idx = indices[block_idx]
                val |= (idx & ((1 << bits_per_entry) - 1)) << (entry_idx * bits_per_entry)
        # Convert to signed 64-bit
        if val >= (1 << 63):
            val -= (1 << 64)
        longs.append(val)

    return longs


@dataclass
class ChunkSection:
    """A 16x16x16 section of blocks within a chunk."""
    y: int  # Section Y coordinate
    palette: list[str] = field(default_factory=lambda: ["minecraft:air"])
    blocks: list[int] = field(default_factory=lambda: [0] * 4096)

    def set_block(self, x: int, y: int, z: int, block_name: str):
        """Set a block at local coordinates (0-15)."""
        if block_name not in self.palette:
            self.palette.append(block_name)
        idx = self.palette.index(block_name)
        self.blocks[y * 256 + z * 16 + x] = idx

    def has_block(self, x: int, y: int, z: int) -> bool:
        """Return True if a non-air block exists at local coordinates (0-15)."""
        return self.blocks[y * 256 + z * 16 + x] != 0

    def is_empty(self) -> bool:
        return all(b == 0 for b in self.blocks)


class Chunk:
    """A 16x16 column of blocks."""

    def __init__(self, cx: int, cz: int):
        self.cx = cx
        self.cz = cz
        self.sections: dict[int, ChunkSection] = {}

    def get_section(self, section_y: int) -> ChunkSection:
        if section_y not in self.sections:
            self.sections[section_y] = ChunkSection(y=section_y)
        return self.sections[section_y]

    def set_block(self, x: int, y: int, z: int, block_name: str):
        """Set a block at chunk-local x,z and world y."""
        section_y = y >> 4
        local_y = y & 0xF
        section = self.get_section(section_y)
        section.set_block(x, local_y, z, block_name)

    def has_block(self, x: int, y: int, z: int) -> bool:
        """Return True if a non-air block exists at chunk-local x,z and world y."""
        section_y = y >> 4
        local_y = y & 0xF
        if section_y not in self.sections:
            return False
        return self.sections[section_y].has_block(x, local_y, z)

    def to_nbt_bytes(self) -> bytes:
        """Serialize this chunk to NBT bytes (uncompressed; region writer compresses)."""
        buf = io.BytesIO()

        # Root compound tag
        _write_nbt_tag(buf, 10, "", None)

        _write_nbt_tag(buf, 3, "DataVersion", MC_DATA_VERSION)
        _write_nbt_tag(buf, 3, "xPos", self.cx)
        _write_nbt_tag(buf, 3, "yPos", MIN_SECTION_Y)
        _write_nbt_tag(buf, 3, "zPos", self.cz)
        _write_nbt_tag(buf, 4, "LastUpdate", 0)
        _write_nbt_tag(buf, 4, "InhabitedTime", 0)
        _write_nbt_tag(buf, 8, "Status", "minecraft:full")
        # isLightOn=0 → MC recalculates sky/block light on first load (correct brightness)
        _write_nbt_tag(buf, 1, "isLightOn", 0)

        # ── sections ──────────────────────────────────────────────────────────
        sections_to_write = []
        for sy in range(MIN_SECTION_Y, MAX_SECTION_Y + 1):
            if sy in self.sections and not self.sections[sy].is_empty():
                sections_to_write.append(self.sections[sy])
            else:
                sections_to_write.append(ChunkSection(y=sy))

        buf.write(struct.pack(">bH", 9, len("sections")))
        buf.write(b"sections")
        buf.write(struct.pack(">bi", 10, len(sections_to_write)))

        for section in sections_to_write:
            _write_nbt_tag(buf, 1, "Y", section.y)

            # block_states
            _write_nbt_tag(buf, 10, "block_states", None)
            buf.write(struct.pack(">bH", 9, len("palette")))
            buf.write(b"palette")
            buf.write(struct.pack(">bi", 10, len(section.palette)))
            for block_entry in section.palette:
                # Support "block_name[key=val,key=val]" for block states
                if "[" in block_entry:
                    name_part, props_str = block_entry.rstrip("]").split("[", 1)
                    props = dict(kv.split("=") for kv in props_str.split(",") if "=" in kv)
                else:
                    name_part = block_entry
                    props = {}
                _write_nbt_tag(buf, 8, "Name", name_part)
                if props:
                    _write_nbt_tag(buf, 10, "Properties", None)
                    for pk, pv in props.items():
                        _write_nbt_tag(buf, 8, pk, pv)
                    buf.write(b"\x00")  # end Properties
                buf.write(b"\x00")  # end block compound
            if len(section.palette) > 1:
                longs = _pack_block_states(section.blocks, len(section.palette))
                _write_nbt_tag(buf, 12, "data", longs)
            buf.write(b"\x00")  # end block_states

            # biomes — single-entry palette (plains)
            _write_nbt_tag(buf, 10, "biomes", None)
            buf.write(struct.pack(">bH", 9, len("palette")))
            buf.write(b"palette")
            buf.write(struct.pack(">bi", 8, 1))
            enc = "minecraft:plains".encode("utf-8")
            buf.write(struct.pack(">H", len(enc)))
            buf.write(enc)
            buf.write(b"\x00")  # end biomes

            buf.write(b"\x00")  # end section compound

        # ── Heightmaps (empty — MC recalculates) ──────────────────────────────
        _write_nbt_tag(buf, 10, "Heightmaps", None)
        buf.write(b"\x00")

        # ── block_entities / ticks / PostProcessing (all empty) ───────────────
        def _empty_list(name: str, elem_type: int) -> None:
            n = name.encode("utf-8")
            buf.write(struct.pack(">bH", 9, len(n)))
            buf.write(n)
            buf.write(struct.pack(">bi", elem_type, 0))

        _empty_list("block_entities", 10)   # list of compounds
        _empty_list("block_ticks",    10)   # list of compounds
        _empty_list("fluid_ticks",    10)   # list of compounds
        _empty_list("PostProcessing", 9)    # list of lists

        # ── structures (empty References + starts) ────────────────────────────
        _write_nbt_tag(buf, 10, "structures", None)
        _write_nbt_tag(buf, 10, "References", None)
        buf.write(b"\x00")
        _write_nbt_tag(buf, 10, "starts", None)
        buf.write(b"\x00")
        buf.write(b"\x00")  # end structures

        buf.write(b"\x00")  # end root compound

        return buf.getvalue()


class MinecraftWorld:
    """Manages a Minecraft world (region files)."""

    def __init__(self, world_dir: str, world_name: str = "KnightOnline"):
        self.world_dir = world_dir
        self.world_name = world_name
        self.chunks: dict[tuple[int, int], Chunk] = {}

    def set_block(self, x: int, y: int, z: int, block_name: str):
        """Set a block at world coordinates."""
        cx = x >> 4
        cz = z >> 4
        key = (cx, cz)
        if key not in self.chunks:
            self.chunks[key] = Chunk(cx, cz)
        self.chunks[key].set_block(x & 0xF, y, z & 0xF, block_name)

    def has_block(self, x: int, y: int, z: int) -> bool:
        """Return True if a non-air block exists at world coordinates."""
        key = (x >> 4, z >> 4)
        if key not in self.chunks:
            return False
        return self.chunks[key].has_block(x & 0xF, y, z & 0xF)

    def save(self):
        """Write all chunks to region files and create level.dat."""
        region_dir = os.path.join(self.world_dir, "region")
        os.makedirs(region_dir, exist_ok=True)

        # Group chunks by region
        regions: dict[tuple[int, int], list[Chunk]] = {}
        for (cx, cz), chunk in self.chunks.items():
            rx = cx >> 5
            rz = cz >> 5
            key = (rx, rz)
            if key not in regions:
                regions[key] = []
            regions[key].append(chunk)

        print(f"  Saving {len(self.chunks)} chunks across {len(regions)} region files...")

        for (rx, rz), chunks in regions.items():
            self._write_region(region_dir, rx, rz, chunks)

        self._write_level_dat()
        print(f"  World saved to {self.world_dir}")

    def _write_region(self, region_dir: str, rx: int, rz: int, chunks: list[Chunk]):
        """Write a .mca region file."""
        filepath = os.path.join(region_dir, f"r.{rx}.{rz}.mca")

        # Region file: 8KiB header (4KiB locations + 4KiB timestamps) + chunk data
        locations = [0] * 1024  # offset(3 bytes) + sector_count(1 byte)
        timestamps = [0] * 1024

        chunk_data_parts = []
        current_sector = 2  # First 2 sectors are header

        for chunk in chunks:
            local_x = chunk.cx & 31
            local_z = chunk.cz & 31
            idx = local_x + local_z * 32

            # Serialize chunk
            nbt_data = chunk.to_nbt_bytes()
            compressed = zlib.compress(nbt_data)

            # Chunk data: length(4) + compression_type(1) + data
            chunk_bytes = struct.pack(">iB", len(compressed) + 1, 2) + compressed

            # Pad to 4KiB sectors
            padded_len = math.ceil(len(chunk_bytes) / 4096) * 4096
            chunk_bytes = chunk_bytes.ljust(padded_len, b"\x00")
            sector_count = padded_len // 4096

            locations[idx] = (current_sector << 8) | (sector_count & 0xFF)
            timestamps[idx] = int(time.time())

            chunk_data_parts.append(chunk_bytes)
            current_sector += sector_count

        with open(filepath, "wb") as f:
            # Write location table
            for loc in locations:
                f.write(struct.pack(">I", loc))
            # Write timestamp table
            for ts in timestamps:
                f.write(struct.pack(">I", ts))
            # Write chunk data
            for data in chunk_data_parts:
                f.write(data)

    def _write_level_dat(self):
        """Write a level.dat compatible with Minecraft 1.21.4."""
        buf = io.BytesIO()

        def _str(name, value):
            _write_nbt_tag(buf, 8, name, value)

        def _int(name, value):
            _write_nbt_tag(buf, 3, name, value)

        def _long(name, value):
            _write_nbt_tag(buf, 4, name, value)

        def _byte(name, value):
            _write_nbt_tag(buf, 1, name, value)

        def _float(name, value):
            _write_nbt_tag(buf, 5, name, value)

        def _double(name, value):
            _write_nbt_tag(buf, 6, name, value)

        def _compound(name):
            _write_nbt_tag(buf, 10, name, None)

        def _end():
            buf.write(b"\x00")

        # Root compound
        _compound("")

        # Data compound
        _compound("Data")

        _int("DataVersion", MC_DATA_VERSION)

        # Version compound — required by 1.21.4 to accept the world
        _compound("Version")
        _int("Id", MC_DATA_VERSION)
        _str("Name", "1.21.4")
        _str("Series", "main")
        _byte("Snapshot", 0)
        _end()  # end Version

        _str("LevelName", self.world_name)
        _int("version", 19133)           # Anvil format marker
        _byte("initialized", 1)
        _byte("WasModded", 0)

        # Spawn point
        _int("SpawnX", 900)
        _int("SpawnY", 70)
        _int("SpawnZ", 550)
        _float("SpawnAngle", 0.0)

        # Game settings
        _int("GameType", 1)              # Creative
        _byte("hardcore", 0)
        _byte("allowCommands", 1)
        _byte("Difficulty", 0)          # Peaceful
        _byte("DifficultyLocked", 0)
        _long("Time", 6000)
        _long("DayTime", 6000)
        _long("LastPlayed", int(time.time() * 1000))
        _byte("raining", 0)
        _int("rainTime", 0)
        _byte("thundering", 0)
        _int("thunderTime", 0)

        # World border (defaults)
        _double("BorderCenterX", 0.0)
        _double("BorderCenterZ", 0.0)
        _double("BorderSize", 59999968.0)
        _double("BorderSizeLerpTarget", 59999968.0)
        _long("BorderSizeLerpTime", 0)
        _double("BorderSafeZone", 5.0)
        _double("BorderDamagePerBlock", 0.2)
        _double("BorderWarningBlocks", 5.0)
        _double("BorderWarningTime", 15.0)

        # Wandering trader
        _int("WanderingTraderSpawnChance", 25)
        _int("WanderingTraderSpawnDelay", 24000)

        # CustomBossEvents (empty compound)
        _compound("CustomBossEvents")
        _end()

        # DragonFight
        _compound("DragonFight")
        _byte("NeedsStateScanning", 1)
        _byte("DragonKilled", 0)
        _byte("PreviouslyKilled", 0)
        buf.write(struct.pack(">bH", 11, len("Gateways")))
        buf.write(b"Gateways")
        buf.write(struct.pack(">i", 0))  # empty int array
        _end()

        # GameRules (all defaults)
        _compound("GameRules")
        for name, val in [
            ("announceAdvancements", "true"),
            ("commandBlockOutput", "true"),
            ("disableElytraMovementCheck", "false"),
            ("disableRaids", "false"),
            ("doDaylightCycle", "false"),
            ("doEntityDrops", "true"),
            ("doFireTick", "false"),
            ("doImmediateRespawn", "false"),
            ("doInsomnia", "false"),
            ("doLimitedCrafting", "false"),
            ("doMobLoot", "true"),
            ("doMobSpawning", "false"),
            ("doPatrolSpawning", "false"),
            ("doTileDrops", "true"),
            ("doTraderSpawning", "false"),
            ("doWardenSpawning", "false"),
            ("doWeatherCycle", "false"),
            ("drowningDamage", "true"),
            ("fallDamage", "true"),
            ("fireDamage", "true"),
            ("forgiveDeadPlayers", "true"),
            ("freezeDamage", "true"),
            ("keepInventory", "true"),
            ("logAdminCommands", "true"),
            ("maxCommandChainLength", "65536"),
            ("maxEntityCramming", "24"),
            ("mobGriefing", "false"),
            ("naturalRegeneration", "true"),
            ("playersSleepingPercentage", "100"),
            ("randomTickSpeed", "3"),
            ("reducedDebugInfo", "false"),
            ("sendCommandFeedback", "true"),
            ("showDeathMessages", "true"),
            ("snowAccumulationHeight", "1"),
            ("spawnRadius", "10"),
            ("spectatorsGenerateChunks", "false"),
            ("universalAnger", "false"),
        ]:
            _str(name, val)
        _end()  # end GameRules

        # WorldGenSettings — void overworld so MC doesn't generate terrain
        _compound("WorldGenSettings")
        _long("seed", 0)
        _byte("bonus_chest", 0)
        _byte("generate_features", 0)

        _compound("dimensions")

        # Overworld — flat/void (no layers = void)
        _compound("minecraft:overworld")
        _str("type", "minecraft:overworld")
        _compound("generator")
        _str("type", "minecraft:flat")
        _compound("settings")
        _str("biome", "minecraft:the_void")
        _byte("features", 0)
        _byte("lakes", 0)
        # layers: empty list
        buf.write(struct.pack(">bH", 9, len("layers")))
        buf.write(b"layers")
        buf.write(struct.pack(">bi", 10, 0))
        _end()  # end settings
        _end()  # end generator
        _end()  # end minecraft:overworld

        # Nether
        _compound("minecraft:the_nether")
        _str("type", "minecraft:the_nether")
        _compound("generator")
        _str("type", "minecraft:noise")
        _str("settings", "minecraft:nether")
        _compound("biome_source")
        _str("type", "minecraft:multi_noise")
        _str("preset", "minecraft:nether")
        _end()
        _end()  # end generator
        _end()  # end minecraft:the_nether

        # End
        _compound("minecraft:the_end")
        _str("type", "minecraft:the_end")
        _compound("generator")
        _str("type", "minecraft:noise")
        _str("settings", "minecraft:end")
        _compound("biome_source")
        _str("type", "minecraft:the_end")
        _end()
        _end()  # end generator
        _end()  # end minecraft:the_end

        _end()  # end dimensions
        _end()  # end WorldGenSettings

        # ServerBrands (list of strings)
        buf.write(struct.pack(">bH", 9, len("ServerBrands")))
        buf.write(b"ServerBrands")
        buf.write(struct.pack(">bi", 8, 1))
        enc = "vanilla".encode("utf-8")
        buf.write(struct.pack(">H", len(enc)))
        buf.write(enc)

        _end()  # end Data compound
        _end()  # end root compound

        level_dat_path = os.path.join(self.world_dir, "level.dat")
        with gzip.open(level_dat_path, "wb") as f:
            f.write(buf.getvalue())
