"""Parser for Knight Online .n3pmesh (progressive mesh) files.

Binary layout (little-endian, verified against N3PMeshConverter Main.cpp:268-438
and KnightOnline-master src/N3Base/N3PMesh.cpp):

  int32              nameLen
  char[nameLen]      name string

  int32              numCollapses
  int32              totalIndexChanges
  int32              maxNumVertices
  int32              maxNumIndices
  int32              minNumVertices
  int32              minNumIndices

  Vertex[maxNumVertices]         32 bytes each: x,y,z,nx,ny,nz,u,v  (8 × float32)
  uint16[maxNumIndices]          triangle index list

  EdgeCollapse[numCollapses]     24 bytes each (MSVC: 5 × int32 + bool + 3 pad)
  int32[totalIndexChanges]       index delta list

  int32              LODCtrlValueCount
  LODCtrlValue[count]            8 bytes each (float32 threshold + int32 index)

Only the max-LOD vertices and indices are retained; the collapse/LOD data
is read past and discarded.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .binary_reader import BinaryParseError, BinaryReader
from .debug_export import export_obj, print_mesh_stats

# EdgeCollapse: 5 × int32 (20 bytes) + bool (1 byte) + 3 pad = 24 bytes
_EDGE_COLLAPSE_BYTES = 24

# LODCtrlValue: float32 (4) + int32 (4) = 8 bytes
_LOD_CTRL_VALUE_BYTES = 8

# Sanity caps — reject obviously corrupt files early
_MAX_VERTICES = 1_000_000
_MAX_INDICES = 3_000_000
_MAX_COLLAPSES = 2_000_000


@dataclass
class N3PMesh:
    name: str
    vertices: np.ndarray   # (N, 8) float32: x,y,z,nx,ny,nz,u,v
    indices: np.ndarray    # (M,)  uint16, triangle list
    min_vertices: int
    max_vertices: int
    min_indices: int
    max_indices: int


def parse_n3pmesh(source) -> N3PMesh:
    """Parse an .n3pmesh file and return the max-LOD mesh.

    source: file path (str / Path) or bytes buffer.
    Raises BinaryParseError on malformed data.
    """
    r = BinaryReader(source)

    name = r.read_len_string()

    num_collapses = r.read_i32()
    total_index_changes = r.read_i32()
    max_verts = r.read_i32()
    max_indices = r.read_i32()
    min_verts = r.read_i32()
    min_indices = r.read_i32()

    if not (0 <= num_collapses <= _MAX_COLLAPSES):
        raise BinaryParseError(f"Implausible numCollapses {num_collapses}")
    if not (0 <= max_verts <= _MAX_VERTICES):
        raise BinaryParseError(f"Implausible maxNumVertices {max_verts}")
    if not (0 <= max_indices <= _MAX_INDICES):
        raise BinaryParseError(f"Implausible maxNumIndices {max_indices}")

    # Max-LOD vertex array: N × 8 float32 (x, y, z, nx, ny, nz, u, v)
    raw_verts = r.read_f32_array(max_verts * 8)
    vertices = raw_verts.reshape((max_verts, 8))

    # Max-LOD index list (uint16 triangle list)
    indices = r.read_u16_array(max_indices)

    # Skip EdgeCollapse array
    r.skip(num_collapses * _EDGE_COLLAPSE_BYTES)

    # Skip index change array
    r.skip(total_index_changes * 4)

    # Skip LOD control values
    lod_count = r.read_i32()
    r.skip(lod_count * _LOD_CTRL_VALUE_BYTES)

    return N3PMesh(
        name=name,
        vertices=vertices,
        indices=indices,
        min_vertices=min_verts,
        max_vertices=max_verts,
        min_indices=min_indices,
        max_indices=max_indices,
    )


def main():
    parser = argparse.ArgumentParser(description="Parse a .n3pmesh file and export to OBJ")
    parser.add_argument("mesh_file", help=".n3pmesh input file")
    parser.add_argument("-o", "--output", help="Output .obj file path")
    args = parser.parse_args()

    try:
        mesh = parse_n3pmesh(args.mesh_file)
    except BinaryParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    print_mesh_stats(mesh)

    if args.output:
        export_obj(mesh.vertices, mesh.indices, args.output)
        print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
