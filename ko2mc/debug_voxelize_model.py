"""Single-model voxelization test harness.

Loads a .n3pmesh, optionally applies rotation and scale, voxelizes at one
or more voxel sizes, and writes each result into a flat Minecraft test world.

Usage:
    python -m ko2mc.debug_voxelize_model <mesh.n3pmesh> [options]

Examples:
    # Identity, 1-unit voxels, write MC world
    python -m ko2mc.debug_voxelize_model model.n3pmesh -o /tmp/test_world

    # Multiple voxel sizes
    python -m ko2mc.debug_voxelize_model model.n3pmesh -v 0.5 1.0 2.0 -o /tmp/test_world

    # 90° yaw rotation around Y, scale 2× on all axes
    python -m ko2mc.debug_voxelize_model model.n3pmesh --quat 0 0.707 0 0.707 --scale 2 2 2

    # Export transformed OBJ before voxelization
    python -m ko2mc.debug_voxelize_model model.n3pmesh --obj /tmp/transformed.obj

    # Flood-fill interior (default off)
    python -m ko2mc.debug_voxelize_model model.n3pmesh --fill
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .binary_reader import BinaryParseError
from .debug_export import export_obj, print_mesh_stats
from .math3d import apply_transform
from .mc_world import MinecraftWorld
from .n3pmesh_parser import parse_n3pmesh
from .voxelizer import VoxelGrid, place_voxel_grid, voxelize_mesh


# ── Flat world builder ────────────────────────────────────────────────────────

def _make_flat_world(world_dir: str, world_name: str) -> MinecraftWorld:
    """Create a MinecraftWorld with a flat grass/stone ground plane at Y=64."""
    world = MinecraftWorld(world_dir, world_name)
    ground_y = 64

    # 64×64 ground plate centered at origin — enough room for any single model
    for x in range(-32, 32):
        for z in range(-32, 32):
            world.set_block(x, ground_y, z, "minecraft:grass_block")
            world.set_block(x, ground_y - 1, z, "minecraft:dirt")
            world.set_block(x, ground_y - 2, z, "minecraft:stone")
            world.set_block(x, ground_y - 3, z, "minecraft:bedrock")

    return world


# ── Quaternion from axis-angle helpers ────────────────────────────────────────

def _quat_identity():
    return (0.0, 0.0, 0.0, 1.0)


def _quat_from_axis_angle(ax, ay, az, angle_deg: float):
    """Build a unit quaternion from axis-angle (degrees)."""
    angle_rad = np.deg2rad(angle_deg)
    half = angle_rad / 2.0
    length = (ax * ax + ay * ay + az * az) ** 0.5
    if length < 1e-10:
        return _quat_identity()
    ax, ay, az = ax / length, ay / length, az / length
    s = np.sin(half)
    return (ax * s, ay * s, az * s, np.cos(half))


# ── Core voxelization + placement ────────────────────────────────────────────

def voxelize_and_report(
    mesh,
    voxel_size: float,
    rotation_quat,
    scale,
    fill: bool,
) -> VoxelGrid:
    """Apply transform, voxelize, and print stats. Returns VoxelGrid."""
    # Apply rotation + scale to full 8-column vertex array
    verts_t = apply_transform(mesh.vertices, rotation_quat, scale)
    grid = voxelize_mesh(verts_t, mesh.indices, voxel_size, fill=fill)

    print(f"\n  voxel_size={voxel_size}")
    grid.print_stats()
    return grid


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-model .n3pmesh voxelization test harness"
    )
    parser.add_argument("mesh_file", help=".n3pmesh input file")
    parser.add_argument(
        "-v", "--voxel-sizes",
        nargs="+",
        type=float,
        default=[1.0],
        metavar="SIZE",
        help="One or more voxel sizes to test (default: 1.0)",
    )
    parser.add_argument(
        "--quat",
        nargs=4,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "W"),
        help="Rotation quaternion x y z w (default: identity 0 0 0 1)",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
        metavar="DEG",
        help="Convenience: rotate N degrees around Y axis (overrides --quat)",
    )
    parser.add_argument(
        "--scale",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
        help="Scale factors sx sy sz (default: 1 1 1)",
    )
    parser.add_argument(
        "--obj",
        metavar="FILE",
        help="Export transformed mesh to OBJ before voxelizing",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Flood-fill interior voxels (default: surface-only)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="DIR",
        help="Output directory for Minecraft test worlds",
    )
    parser.add_argument(
        "--block",
        default="minecraft:stone_bricks",
        help="Block name to use for voxels (default: minecraft:stone_bricks)",
    )
    args = parser.parse_args()

    # ── Parse mesh ──────────────────────────────────────────────────────────
    print(f"Mesh: {args.mesh_file}")
    try:
        mesh = parse_n3pmesh(args.mesh_file)
    except BinaryParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    print_mesh_stats(mesh)

    # ── Build transform ────────────────────────────────────────────────────
    if args.yaw is not None:
        quat = _quat_from_axis_angle(0, 1, 0, args.yaw)
        print(f"Rotation: Y yaw {args.yaw}° → quat {tuple(round(v, 4) for v in quat)}")
    elif args.quat is not None:
        quat = tuple(args.quat)
        print(f"Rotation: quat {quat}")
    else:
        quat = _quat_identity()
        print("Rotation: identity")

    scale = tuple(args.scale)
    print(f"Scale: {scale}")

    # ── Export transformed OBJ ─────────────────────────────────────────────
    if args.obj:
        verts_t = apply_transform(mesh.vertices, quat, scale)
        export_obj(verts_t, mesh.indices, args.obj)
        print(f"\nTransformed OBJ exported: {args.obj}")

    # ── Voxelize at each size ──────────────────────────────────────────────
    print(f"\nVoxelizing (fill={'on' if args.fill else 'off'})...")
    grids = {}
    for vs in args.voxel_sizes:
        grids[vs] = voxelize_and_report(mesh, vs, quat, scale, args.fill)

    # ── Write MC test worlds ───────────────────────────────────────────────
    if args.output:
        out_root = Path(args.output)
        mesh_stem = Path(args.mesh_file).stem

        for vs, grid in grids.items():
            vs_tag = f"{vs:.2f}".replace(".", "_")
            world_name = f"{mesh_stem}_vs{vs_tag}"
            world_dir = str(out_root / world_name)

            print(f"\nBuilding world '{world_name}'...")
            world = _make_flat_world(world_dir, world_name)

            # Place object above ground, centered at (0, 65, 0)
            placed = place_voxel_grid(
                grid,
                world,
                offset_x=-(grid.nx // 2),
                offset_y=65,
                offset_z=-(grid.nz // 2),
                block_name=args.block,
            )
            world.save()

            print(f"  Placed {placed} blocks → {world_dir}")
            print(f"  Copy '{world_name}' to Minecraft saves/ to inspect in-game.")

    print("\nDone.")


if __name__ == "__main__":
    main()
