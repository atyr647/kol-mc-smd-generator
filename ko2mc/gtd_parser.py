"""Parser for Knight Online .gtd (Game Terrain Data) files.

GTD files are the client-side terrain files. Binary layout (little-endian),
matching CN3Terrain::Load in the KO client:

  - uint32 string_size
  - If string_size > 2: NEW format
      - char[string_size] map_name (encrypted)
      - uint32 unknown
  - Else: OLD format
      - uint32 string_size (re-read)
      - char[string_size] map_name
  - uint32 heightmap_size (N, e.g. 257 = 256 tiles + 1)
  - N*N x MAPDATA (x-major: for x in 0..N, for z in 0..N):
      - float  height (meters)
      - uint32 bitfield:  bIsTileFull:1, Tex1Dir:5, Tex2Dir:5, Tex1Idx:10, Tex2Idx:10
  - ((N-1)/8)^2 x (float middle_y, float radius)   patch bounding info
  - N*N x uint8 grass attribute
  - char[260] grass file name
  - int32 tile texture count, int32 tile texture source count
  - source_count x char[260] source file names (dtex\\*.gtt)
  - tile_texture_count x (int16 source_index, int16 tile_index)
  - int32 light map count (+ light map data)
  - river data, pond data (water surfaces)

One heightmap vertex is placed every 4 meters (the "unit distance").
Tex1Idx is an index into the tile texture list; 1023 means "no texture".
"""

import re
import struct
from dataclasses import dataclass, field

import numpy as np

UNIT_DISTANCE = 4.0  # meters between heightmap vertices
NO_TEXTURE = 1023
MAX_PATH = 260
PATCH_TILE_SIZE = 8

# Water vertex (__VertexXyzNormalColorT2): pos(3f) normal(3f) color(u32) uv1(2f) uv2(2f)
_WATER_VERTEX_SIZE = 44
_UP_NORMAL = struct.pack("<3f", 0.0, 1.0, 0.0)


@dataclass
class WaterMesh:
    """A pond or river surface, stored as a grid of vertices (rows of `width`)."""
    texture: str
    width: int
    vertices: np.ndarray  # (count, 3) float32 x, y, z in KO meters

    @property
    def rows(self) -> int:
        return len(self.vertices) // self.width

    def triangles(self) -> np.ndarray:
        """Return (n, 3, 3) triangles covering the grid."""
        w, r = self.width, self.rows
        v = self.vertices[: w * r].reshape(r, w, 3)
        a, b = v[:-1, :-1], v[:-1, 1:]
        c, d = v[1:, :-1], v[1:, 1:]
        t1 = np.stack([a, b, c], axis=2).reshape(-1, 3, 3)
        t2 = np.stack([b, d, c], axis=2).reshape(-1, 3, 3)
        return np.concatenate([t1, t2])


@dataclass
class GTDFile:
    """Parsed GTD terrain data."""
    map_name: str = ""
    heightmap_size: int = 0
    heights: np.ndarray | None = None       # [x, z] float32, meters
    texture_ids: np.ndarray | None = None   # [x, z] raw MAPDATA bitfield
    tex1: np.ndarray | None = None          # [x, z] tile texture index (1023 = none)
    tex2: np.ndarray | None = None          # [x, z] blend texture index (1023 = none)
    tile_textures: list[str] = field(default_factory=list)  # tile texture index -> source name
    tile_subindex: list[int] = field(default_factory=list)  # tile texture index -> index in .gtt
    water: list[WaterMesh] = field(default_factory=list)

    @property
    def size_meters(self) -> float:
        return (self.heightmap_size - 1) * UNIT_DISTANCE

    def get_height(self, x: int, z: int) -> float:
        if x < 0 or x >= self.heightmap_size or z < 0 or z >= self.heightmap_size:
            return 0.0
        return float(self.heights[x, z])

    def height_at(self, px: float, pz: float) -> float:
        """Bilinear terrain height at a KO world position (meters)."""
        n = self.heightmap_size
        fx = min(max(px / UNIT_DISTANCE, 0.0), n - 1.001)
        fz = min(max(pz / UNIT_DISTANCE, 0.0), n - 1.001)
        x0, z0 = int(fx), int(fz)
        tx, tz = fx - x0, fz - z0
        h = self.heights
        return float(h[x0, z0] * (1 - tx) * (1 - tz) + h[x0 + 1, z0] * tx * (1 - tz)
                     + h[x0, z0 + 1] * (1 - tx) * tz + h[x0 + 1, z0 + 1] * tx * tz)

    def texture_name(self, tex_idx: int) -> str:
        """Source texture set name for a tile texture index, e.g. 'map_mora_brick01'."""
        if 0 <= tex_idx < len(self.tile_textures):
            return self.tile_textures[tex_idx]
        return ""


def _clean_texture_name(raw: bytes) -> str:
    name = raw.split(b"\x00")[0].decode("latin-1").strip().lower()
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    # 'map_mora_brick01_3.gtt' -> 'map_mora_brick01'
    return re.sub(r"_\d+\.gtt$", "", name).removesuffix(".gtt")


def _find_water(tail: bytes, map_size_m: float) -> list[WaterMesh]:
    """Locate pond/river vertex grids in the trailing water section.

    The river and pond headers differ between client versions, but both store
    vertices as 44-byte records with an up-facing normal, so we scan for runs
    of those records instead of relying on the headers.
    """
    meshes = []
    pos = 0
    while True:
        hit = tail.find(_UP_NORMAL, pos)
        if hit < 0:
            break
        start = hit - 12
        if start < 0:
            pos = hit + 1
            continue
        count = 0
        while (start + (count + 1) * _WATER_VERTEX_SIZE <= len(tail)
               and tail[start + count * 44 + 12:start + count * 44 + 24] == _UP_NORMAL):
            count += 1
        if count < 4:
            pos = hit + 1
            continue

        rec = np.frombuffer(tail, dtype=np.dtype([
            ("pos", "<f4", 3), ("normal", "<f4", 3), ("color", "<u4"),
            ("uv1", "<f4", 2), ("uv2", "<f4", 2)]), count=count, offset=start)
        verts = rec["pos"].astype(np.float32)
        pos = start + count * _WATER_VERTEX_SIZE

        if not np.all(np.isfinite(verts)) or np.abs(verts).max() > map_size_m * 4 + 1000:
            continue

        width = _guess_grid_width(tail, start, rec)
        if width < 2 or count // width < 2:
            continue

        # Texture name is either just before (ponds) or just after (rivers) the vertices.
        m = re.search(rb"([\x20-\x7e]{2,}\.dxt)\x00", tail[max(0, start - 64):start]) \
            or re.search(rb"([\x20-\x7e]{2,}\.dxt)\x00", tail[pos:pos + 64])
        texture = m.group(1).decode() if m else "water"
        meshes.append(WaterMesh(texture=texture, width=width, vertices=verts))
    return meshes


def _guess_grid_width(tail: bytes, start: int, rec: np.ndarray) -> int:
    count = len(rec)
    # Ponds: header ... int32 vertex_count, int32 width, int32 name_len, name ... vertices
    m = re.search(rb"[\x20-\x7e]{2,}\.dxt\x00$", tail[max(0, start - 64):start])
    if m:
        name_len = len(m.group(0))
        hdr = start - name_len - 12
        if hdr >= 0:
            vc, w, nl = struct.unpack_from("<iii", tail, hdr)
            if vc == count and nl == name_len and 1 < w <= count and count % w == 0:
                return w
    # Rivers: rows of vertices where the first texture coordinate restarts each row.
    u = rec["uv1"][:, 0]
    for w in range(2, min(count - 1, 64) + 1):
        if count % w == 0 and u[w] == u[0] and u[1] != u[0]:
            return w
    return 0


def parse_gtd(filepath: str, verbose: bool = True) -> GTDFile:
    """Parse a .gtd file and return terrain data."""
    gtd = GTDFile()

    with open(filepath, "rb") as fp:
        data = fp.read()

    off = 0

    def read(fmt):
        nonlocal off
        vals = struct.unpack_from(fmt, data, off)
        off += struct.calcsize(fmt)
        return vals

    (string_size,) = read("<I")
    if string_size > 2:
        # New format: encrypted name followed by an unknown value
        gtd.map_name = data[off:off + string_size].decode("latin-1").rstrip("\x00")
        off += string_size + 4
    else:
        # Old format - re-read string size
        (string_size,) = read("<I")
        gtd.map_name = data[off:off + string_size].decode("latin-1").rstrip("\x00")
        off += string_size

    (n,) = read("<I")
    gtd.heightmap_size = n

    mapdata = np.frombuffer(data, dtype=[("h", "<f4"), ("bits", "<u4")], count=n * n, offset=off)
    off += n * n * 8
    # File order is x-major (index = x * N + z), as in the KO client's CN3Terrain.
    # (Reading it z-major, like SMDExporter's CGameTerrain does, transposes the map:
    # objects then float/sink by up to ~15 m and the terrain looks rotated + mirrored.)
    gtd.heights = mapdata["h"].reshape(n, n).astype(np.float32)
    bits = mapdata["bits"].reshape(n, n).copy()
    gtd.texture_ids = bits
    gtd.tex1 = ((bits >> 11) & 0x3FF).astype(np.int32)
    gtd.tex2 = ((bits >> 21) & 0x3FF).astype(np.int32)

    try:
        patches = (n - 1) // PATCH_TILE_SIZE
        off += patches * patches * 8          # patch middle y / radius
        off += n * n                          # grass attributes
        off += MAX_PATH                       # grass file name
        num_tex, num_src = read("<ii")
        sources = []
        for _ in range(num_src):
            sources.append(_clean_texture_name(data[off:off + MAX_PATH]))
            off += MAX_PATH
        pairs = np.frombuffer(data, dtype="<i2", count=num_tex * 2, offset=off).reshape(-1, 2)
        off += num_tex * 4
        gtd.tile_textures = [sources[s] if 0 <= s < len(sources) else "" for s, _ in pairs]
        gtd.tile_subindex = [int(t) for _, t in pairs]
        gtd.water = _find_water(data[off:], gtd.size_meters)
    except (struct.error, ValueError, IndexError) as e:
        if verbose:
            print(f"  GTD: could not read texture/water section ({e}); using heights only")

    if verbose:
        h = gtd.heights
        print(f"  GTD: map='{gtd.map_name}', heightmap={n}x{n} ({gtd.size_meters:.0f}m square)")
        print(f"  GTD: height range [{h.min():.1f}, {h.max():.1f}] m, "
              f"{len(set(gtd.tile_textures))} texture sets, {len(gtd.water)} water surfaces")

    return gtd
