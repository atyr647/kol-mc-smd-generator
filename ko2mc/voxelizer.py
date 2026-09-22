"""Mesh voxelizer: converts triangle meshes to 3D block grids.

VoxelGrid storage: uint16 ndarray indexed [z, y, x]
  0   = empty (air)
  1   = surface voxel (default fill value)
  2+  = future block palette IDs

Surface voxelization uses the Separating Axis Theorem (SAT) for correct
triangle-AABB intersection. Every voxel whose AABB overlaps a triangle is
marked, including voxels that only share an edge or vertex with a thin face.
This reliably captures walls, beams, arches, fences, and roof edges.

Optional flood-fill solidification is available (default OFF). Surface-only
is safer until mesh watertightness is verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── VoxelGrid ─────────────────────────────────────────────────────────────────

@dataclass
class VoxelGrid:
    """A 3D grid of uint16 palette IDs in [z, y, x] index order.

    origin: world-space position of the (0,0,0) voxel *corner* (not center).
    voxel_size: side length of one voxel in world units.
    """

    data: np.ndarray          # uint16, shape (nz, ny, nx)
    origin: np.ndarray        # float32 (3,) — world position of corner [0,0,0]
    voxel_size: float

    @property
    def nx(self) -> int:
        return int(self.data.shape[2])

    @property
    def ny(self) -> int:
        return int(self.data.shape[1])

    @property
    def nz(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple:
        return self.data.shape  # (nz, ny, nx)

    def world_to_voxel(self, wx: float, wy: float, wz: float) -> tuple:
        """Convert world coords to fractional voxel indices."""
        ox, oy, oz = self.origin
        return (
            (wx - ox) / self.voxel_size,
            (wy - oy) / self.voxel_size,
            (wz - oz) / self.voxel_size,
        )

    def voxel_to_world_center(self, ix: int, iy: int, iz: int) -> tuple:
        """Return the world-space center of a voxel."""
        ox, oy, oz = self.origin
        return (
            ox + (ix + 0.5) * self.voxel_size,
            oy + (iy + 0.5) * self.voxel_size,
            oz + (iz + 0.5) * self.voxel_size,
        )

    def get(self, ix: int, iy: int, iz: int) -> int:
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            return int(self.data[iz, iy, ix])
        return 0

    def set(self, ix: int, iy: int, iz: int, value: int = 1) -> None:
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            self.data[iz, iy, ix] = value

    def count_filled(self) -> int:
        return int((self.data > 0).sum())

    def print_stats(self) -> None:
        filled = self.count_filled()
        total = self.nx * self.ny * self.nz
        pct = 100 * filled / total if total else 0
        print(f"VoxelGrid: {self.nx}×{self.ny}×{self.nz} voxels "
              f"(voxel_size={self.voxel_size})")
        print(f"  Filled: {filled}/{total} ({pct:.1f}%)")
        print(f"  Origin: ({self.origin[0]:.3f}, {self.origin[1]:.3f}, {self.origin[2]:.3f})")


# ── SAT triangle-AABB intersection ────────────────────────────────────────────

def _tri_aabb_sat_batch(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    centers: np.ndarray,
    half: float = 0.5,
) -> np.ndarray:
    """Test a triangle against N axis-aligned unit voxels.

    All coordinates in normalised voxel units (voxel side = 1).

    v0, v1, v2: (3,) — triangle vertices
    centers:    (N, 3) — voxel centers
    half:       float — half-voxel side length (0.5 for unit voxels)

    Returns: (N,) bool — True where triangle overlaps the voxel AABB.

    Implements the 13-axis SAT test:
      3 box face normals + 1 triangle face normal + 9 edge×axis crosses.
    """
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # Vertices relative to each voxel center: (N, 3)
    r0 = v0 - centers
    r1 = v1 - centers
    r2 = v2 - centers

    sep = np.zeros(len(centers), dtype=bool)

    def _check(p0, p1, p2, r):
        """True where min(p) > r or max(p) < -r (separation found)."""
        mn = np.minimum(np.minimum(p0, p1), p2)
        mx = np.maximum(np.maximum(p0, p1), p2)
        return (mn > r) | (mx < -r)

    # ── 3 box face normal axes (X, Y, Z) ─────────────────────────────────────
    sep |= _check(r0[:, 0], r1[:, 0], r2[:, 0], half)
    sep |= _check(r0[:, 1], r1[:, 1], r2[:, 1], half)
    sep |= _check(r0[:, 2], r1[:, 2], r2[:, 2], half)
    if sep.all():
        return ~sep

    # ── Triangle face normal ──────────────────────────────────────────────────
    n = np.cross(e0, e1)
    n_len = float(np.dot(n, n)) ** 0.5
    if n_len > 1e-10:
        r_n = half * (abs(n[0]) + abs(n[1]) + abs(n[2]))
        d = float(v0 @ n)
        proj = centers @ n        # (N,)
        sep |= np.abs(proj - d) > r_n
    if sep.all():
        return ~sep

    # ── 9 edge × box-axis cross products ─────────────────────────────────────
    for e in (e0, e1, e2):
        for ai in range(3):
            # Compute cross product with canonical axis ai
            ax = np.zeros(3, dtype=np.float64)
            ax[(ai + 1) % 3] = -e[(ai + 2) % 3]
            ax[(ai + 2) % 3] = e[(ai + 1) % 3]
            # Same as: ax = np.cross(e, canonical[ai])
            ax_len = float(np.dot(ax, ax)) ** 0.5
            if ax_len < 1e-10:
                continue
            p0 = r0 @ ax
            p1 = r1 @ ax
            p2 = r2 @ ax
            r_a = half * (abs(ax[0]) + abs(ax[1]) + abs(ax[2]))
            sep |= _check(p0, p1, p2, r_a)
            if sep.all():
                break
        if sep.all():
            break

    return ~sep


# ── Surface shell extraction ──────────────────────────────────────────────────

def _extract_surface_shell(data: np.ndarray) -> np.ndarray:
    """Return only surface voxels: filled cells with ≥1 air neighbour (6-conn).

    Pads the grid with one layer of air on each face so boundary voxels are
    always considered surface.  Interior voxels fully surrounded by other filled
    voxels are removed.  This converts solid geometry (e.g. thick walls, filled
    pyramid interiors, arena seating blocks) into a hollow shell, matching how
    game-art meshes are visually rendered in the original engine.

    data: uint16 ndarray shaped (nz, ny, nx).  Non-zero = filled.
    Returns same shape and dtype; interior voxels replaced with 0.
    """
    if data.size == 0:
        return data
    padded = np.pad(data, 1, mode='constant', constant_values=0)
    filled = padded > 0
    has_air = (
        ~filled[:-2, 1:-1, 1:-1] | ~filled[2:,  1:-1, 1:-1] |  # z neighbours
        ~filled[1:-1, :-2, 1:-1] | ~filled[1:-1, 2:,  1:-1] |  # y neighbours
        ~filled[1:-1, 1:-1, :-2] | ~filled[1:-1, 1:-1, 2:  ]   # x neighbours
    )
    surface = (data > 0) & has_air
    return np.where(surface, data, np.uint16(0))


# ── Flood fill (interior solidification) ──────────────────────────────────────

def _flood_fill_exterior(data: np.ndarray) -> np.ndarray:
    """Return a boolean mask of voxels reachable from the grid boundary.

    Uses 6-connectivity BFS starting from all boundary air voxels.
    """
    from collections import deque

    nz, ny, nx = data.shape
    visited = np.zeros((nz, ny, nx), dtype=bool)
    queue = deque()

    def _enqueue(x, y, z):
        if (0 <= x < nx and 0 <= y < ny and 0 <= z < nz
                and data[z, y, x] == 0 and not visited[z, y, x]):
            visited[z, y, x] = True
            queue.append((x, y, z))

    # Seed from all six face boundaries
    for z in range(nz):
        for y in range(ny):
            _enqueue(0, y, z)
            _enqueue(nx - 1, y, z)
    for z in range(nz):
        for x in range(nx):
            _enqueue(x, 0, z)
            _enqueue(x, ny - 1, z)
    for y in range(ny):
        for x in range(nx):
            _enqueue(x, y, 0)
            _enqueue(x, y, nz - 1)

    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            _enqueue(x + dx, y + dy, z + dz)

    return visited


# ── Main voxelization ──────────────────────────────────────────────────────────

def voxelize_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    voxel_size: float,
    padding: int = 1,
    fill: bool = False,
    surface_only: bool = True,
    block_id: int = 1,
) -> VoxelGrid:
    """Voxelize a triangle mesh into a VoxelGrid.

    vertices:     (N, 3+) float32 — only first 3 columns (x,y,z) are used
    indices:      (M,)    uint16  — triangle list, M divisible by 3
    voxel_size:   float           — world-unit side length of each voxel
    padding:      int             — extra voxels of empty space around bbox (default 1)
    fill:         bool            — flood-fill interior after surface voxelisation (default False)
    surface_only: bool            — strip fully-surrounded interior voxels after SAT
                                    (default True).  Converts solid meshes (thick walls,
                                    arena seating, filled pyramids) into hollow shells
                                    matching the game engine's visual surface rendering.
    block_id:     int             — uint16 value to write into hit voxels

    Returns a VoxelGrid with the mesh's AABB (+ padding) as bounds.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if len(indices) % 3 != 0:
        raise ValueError(f"indices length {len(indices)} is not divisible by 3")

    pos = np.asarray(vertices[:, :3], dtype=np.float64)

    if len(pos) == 0 or len(indices) == 0:
        empty = np.zeros((1, 1, 1), dtype=np.uint16)
        return VoxelGrid(data=empty, origin=np.zeros(3, dtype=np.float32),
                         voxel_size=voxel_size)

    bbox_min = pos.min(axis=0)
    bbox_max = pos.max(axis=0)

    origin = bbox_min - padding * voxel_size

    extent = (bbox_max - bbox_min) + 2 * padding * voxel_size
    grid_dims = np.maximum(np.ceil(extent / voxel_size).astype(int), 1)
    nx, ny, nz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])

    data = np.zeros((nz, ny, nx), dtype=np.uint16)
    n_tris = len(indices) // 3

    for ti in range(n_tris):
        i0 = int(indices[ti * 3])
        i1 = int(indices[ti * 3 + 1])
        i2 = int(indices[ti * 3 + 2])

        # Vertices in voxel-unit space (each voxel = 1 unit)
        v0 = (pos[i0] - origin) / voxel_size
        v1 = (pos[i1] - origin) / voxel_size
        v2 = (pos[i2] - origin) / voxel_size

        # Triangle AABB → candidate voxel index range
        t_min = np.floor(np.minimum(np.minimum(v0, v1), v2)).astype(int)
        t_max = np.ceil(np.maximum(np.maximum(v0, v1), v2)).astype(int)

        t_min = np.maximum(t_min, 0)
        t_max = np.minimum(t_max, np.array([nx - 1, ny - 1, nz - 1]))

        if np.any(t_min > t_max):
            continue

        # Enumerate candidate voxels
        xs = np.arange(t_min[0], t_max[0] + 1, dtype=np.int32)
        ys = np.arange(t_min[1], t_max[1] + 1, dtype=np.int32)
        zs = np.arange(t_min[2], t_max[2] + 1, dtype=np.int32)

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        cands = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        # Voxel centers in voxel-unit space
        centers = cands.astype(np.float64) + 0.5

        hits = _tri_aabb_sat_batch(v0, v1, v2, centers, half=0.5)

        if hits.any():
            hi = cands[hits]
            data[hi[:, 2], hi[:, 1], hi[:, 0]] = block_id

    grid = VoxelGrid(
        data=data,
        origin=np.array(origin, dtype=np.float32),
        voxel_size=float(voxel_size),
    )

    if fill:
        exterior = _flood_fill_exterior(data)
        # Any air voxel NOT reachable from outside → interior → fill it
        interior_air = (data == 0) & (~exterior)
        data[interior_air] = block_id

    if surface_only and not fill:
        # Strip voxels fully surrounded by other filled voxels.
        # This converts solid geometry into a surface shell matching how game-art
        # meshes are rendered by the KO engine (only surfaces, no filled volumes).
        data = _extract_surface_shell(data)
        grid = VoxelGrid(data=data, origin=grid.origin, voxel_size=grid.voxel_size)

    return grid


# ── World placement helper ────────────────────────────────────────────────────

def place_voxel_grid(
    grid: VoxelGrid,
    world,
    offset_x: int = 0,
    offset_y: int = 64,
    offset_z: int = 0,
    block_name: str = "minecraft:stone_bricks",
) -> int:
    """Write all filled voxels from grid into a MinecraftWorld.

    offset_x/y/z: world block coordinates of the grid's (0,0,0) voxel corner.
    block_name:   Minecraft block ID string for filled voxels (palette ID 1).
    Returns: number of blocks placed.
    """
    placed = 0
    filled_indices = np.argwhere(grid.data > 0)  # (K, 3) — z, y, x

    for iz, iy, ix in filled_indices:
        wx = offset_x + int(ix)
        wy = offset_y + int(iy)
        wz = offset_z + int(iz)
        world.set_block(wx, wy, wz, block_name)
        placed += 1

    return placed
