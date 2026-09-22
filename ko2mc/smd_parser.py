"""Knight Online .smd server map file parser.

Parses the binary .smd format written by GameserverMap.cpp + N3ShapeMgr.cpp.

Binary layout (little-endian):
  uint32  hm_size           -- heightmap grid side (257 for 1024-unit map)
  float32 unit_dist          -- KO units per tile (4.0)
  float32[hm_size*hm_size]  -- height values, row-major (tx, tz)

  -- CN3ShapeMgr collision data --
  float32 map_width
  float32 map_length
  int32   face_count
  Vector3[face_count*3]     -- collision vertices (12 bytes per Vector3)

  -- Cell grid: MAX_CELL_MAIN x MAX_CELL_MAIN (64x64) --
  For each cell (z-major, z outer / x inner):
    uint32 bExist
    if bExist != 0:
      __CellMain:
        int32  nShapeCount
        uint16[nShapeCount]   shape indices
        __CellSub[4][4]:      (z outer, x inner)
          int32  nCCPolyCount
          uint32[nCCPolyCount*3]  face index triples

  -- Events (not parsed, skipped) --

  -- Movement table --
  int16[(hm_size+1) * (hm_size+1)]  move_table  (row-major, tx outer)
    0 = impassable (wall / steep slope / water)
    1 = passable
    2+ = event cell (warp gate, etc.)

Usage:
  from ko2mc.smd_parser import parse_smd
  smd = parse_smd("moradon_1869.smd")
  print(smd.hm_size, smd.unit_dist)
  print(smd.heights[128, 128])   # KO height at vertex (128,128)
  print(smd.move_table[64, 96])  # move flag at tile (64,96)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class SMDFile:
    path: str
    hm_size: int          # number of vertices per side (e.g. 257)
    unit_dist: float      # KO units per tile (4.0)
    heights: np.ndarray   # float32 (hm_size, hm_size)
    map_width: float      # collision map width (e.g. 1024.0)
    map_length: float     # collision map length
    face_count: int       # collision face count
    # move_table[tx, tz] where tx,tz ∈ [0, hm_size]:
    # 0 = impassable, 1 = passable, 2+ = event trigger
    move_table: np.ndarray  # int16 (hm_size+1, hm_size+1)


def parse_smd(path: Union[str, Path]) -> SMDFile:
    """Parse a Knight Online .smd server map file."""
    path = str(path)
    with open(path, "rb") as f:
        data = f.read()

    pos = 0

    def read_u32():
        nonlocal pos
        v, = struct.unpack_from("<I", data, pos)
        pos += 4
        return v

    def read_i32():
        nonlocal pos
        v, = struct.unpack_from("<i", data, pos)
        pos += 4
        return v

    def read_f32():
        nonlocal pos
        v, = struct.unpack_from("<f", data, pos)
        pos += 4
        return v

    def skip(n: int):
        nonlocal pos
        pos += n

    # ── Terrain header ────────────────────────────────────────────────────────
    hm_size   = read_u32()
    unit_dist = read_f32()

    n_heights = hm_size * hm_size
    heights_flat = np.frombuffer(data, dtype="<f4", count=n_heights, offset=pos).copy()
    heights = heights_flat.reshape(hm_size, hm_size)
    pos += n_heights * 4

    # ── Collision data (N3ShapeMgr::LoadCollisionData) ────────────────────────
    map_width  = read_f32()
    map_length = read_f32()
    face_count = read_i32()

    # Skip collision vertices: face_count * 3 vertices * 12 bytes each
    skip(face_count * 3 * 12)

    # ── Cell grid ─────────────────────────────────────────────────────────────
    CELL_MAIN_SIZE   = 16        # spatial: 16 KO units per cell
    CELL_MAIN_DEVIDE = 4         # sub-cells per side
    MAX_CELL_MAIN    = int(4096 / CELL_MAIN_SIZE)  # = 256?? No...

    # Actually MAX_CELL_MAIN = 4096 / CELL_MAIN_SIZE where CELL_MAIN_SIZE=16 = 256
    # But map is 1024 wide, so grid = 1024/16 = 64 cells per side
    # The C++ loops: for fZ in 0..map_length step CELL_MAIN_SIZE
    n_cells_z = int(map_length / CELL_MAIN_SIZE)   # 64
    n_cells_x = int(map_width  / CELL_MAIN_SIZE)   # 64

    for _cz in range(n_cells_z):
        for _cx in range(n_cells_x):
            b_exist = read_u32()
            if not b_exist:
                continue
            # __CellMain::Load
            n_shapes = read_i32()
            if n_shapes > 0:
                skip(n_shapes * 2)   # uint16 shape indices
            # 4×4 sub-cells
            for _sz in range(CELL_MAIN_DEVIDE):
                for _sx in range(CELL_MAIN_DEVIDE):
                    n_poly = read_i32()
                    if n_poly > 0:
                        skip(n_poly * 3 * 4)   # uint32 face index triples

    # ── Skip events (variable length) ────────────────────────────────────────
    # The movement table is exactly (hm_size+1)^2 * 2 bytes at the end.
    # We locate it by seeking from the end.
    mt_size = (hm_size + 1) * (hm_size + 1) * 2   # bytes
    mt_offset = len(data) - mt_size

    if mt_offset < pos:
        raise ValueError(
            f"SMD parse error: computed movement table offset {mt_offset} < "
            f"current pos {pos} (file may have unexpected format)"
        )

    # ── Movement table ────────────────────────────────────────────────────────
    # int16[(hm_size+1)*(hm_size+1)], row-major tx outer
    mt_flat = np.frombuffer(data, dtype="<i2", count=(hm_size + 1) ** 2,
                            offset=mt_offset).copy()
    move_table = mt_flat.reshape(hm_size + 1, hm_size + 1)

    return SMDFile(
        path       = path,
        hm_size    = hm_size,
        unit_dist  = unit_dist,
        heights    = heights,
        map_width  = map_width,
        map_length = map_length,
        face_count = face_count,
        move_table = move_table,
    )


def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Parse KO .smd server map file")
    ap.add_argument("smd_file")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    smd = parse_smd(args.smd_file)
    print(f"SMD: {args.smd_file}")
    print(f"  hm_size={smd.hm_size}  unit_dist={smd.unit_dist}")
    print(f"  map={smd.map_width}x{smd.map_length}  faces={smd.face_count}")
    print(f"  height range: [{smd.heights.min():.2f}, {smd.heights.max():.2f}]")
    mt = smd.move_table
    n = mt.size
    passable  = int((mt == 1).sum())
    blocked   = int((mt == 0).sum())
    events    = int((mt >= 2).sum())
    print(f"  move_table {mt.shape}: passable={passable} ({passable/n*100:.1f}%) "
          f"blocked={blocked} ({blocked/n*100:.1f}%) events={events}")
    if args.stats:
        vals, counts = np.unique(mt, return_counts=True)
        for v, c in zip(vals[:20], counts[:20]):
            print(f"    value {v}: {c}")


if __name__ == "__main__":
    main()
