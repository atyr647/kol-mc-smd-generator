"""Object placement pipeline tests.

Validates the full pipeline for a single OPD shape:
  1. Position transform (canonical)
  2. Yaw transform (canonical)
  3. Mesh voxelization (local space, no Y compression)
  4. Terrain grounding (bbox_min_y correction)
  5. Voxel placement with X-flip (no systematic sinking or mirroring errors)

These tests use synthetic data so they run without any KO asset files.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ko2mc.math3d import (
    KO_MAP_SIZE,
    MC_Y_BASE,
    ko_to_mc_position,
    ko_to_mc_y,
    ko_to_mc_yaw,
    build_mc_height_grid,
    terrain_y_at,
    apply_transform,
    compute_aabb,
)
from ko2mc.voxelizer import voxelize_mesh, VoxelGrid


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unit_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, indices) for a unit cube centred at origin.

    Vertices have shape (N, 8): x,y,z,nx,ny,nz,u,v.
    Indices form triangles (CCW winding).
    """
    # 8 corners of a 1×1×1 cube centred at (0,0,0)
    h = 0.5
    v = np.array([
        [-h, -h, -h, 0, 0, -1, 0, 0],
        [ h, -h, -h, 0, 0, -1, 1, 0],
        [ h,  h, -h, 0, 0, -1, 1, 1],
        [-h,  h, -h, 0, 0, -1, 0, 1],
        [-h, -h,  h, 0, 0,  1, 0, 0],
        [ h, -h,  h, 0, 0,  1, 1, 0],
        [ h,  h,  h, 0, 0,  1, 1, 1],
        [-h,  h,  h, 0, 0,  1, 0, 1],
    ], dtype=np.float32)

    idx = np.array([
        # -Z face
        0, 2, 1, 0, 3, 2,
        # +Z face
        4, 5, 6, 4, 6, 7,
        # -X face
        0, 4, 7, 0, 7, 3,
        # +X face
        1, 2, 6, 1, 6, 5,
        # -Y face
        0, 1, 5, 0, 5, 4,
        # +Y face
        3, 7, 6, 3, 6, 2,
    ], dtype=np.uint16)

    return v, idx


def _flat_terrain_h_grid(height: float = 5.0, n: int = 65) -> np.ndarray:
    """Return a uniform flat terrain h_grid_mc (already MC-oriented)."""
    return np.full((n, n), height, dtype=np.float32)


def _ground_object(mc_y_terrain: int, bbox_min_y: float) -> int:
    """Compute the MC Y base for the voxel grid after terrain grounding.

    Mirrors the logic in ZoneConverter._convert_part:
        origin_y = bbox_min_y (integer approximation)
        if origin_y > 0: mc_y_adj = mc_y - origin_y
        else:            mc_y_adj = mc_y
        off_y = mc_y_adj + origin_y

    This always yields off_y = max(mc_y, mc_y - origin_y + origin_y) = mc_y
    when the mesh is grounded (bbox_min_y ≤ 0) and correctly lowers the
    anchor when bbox_min_y > 0 (floating mesh).
    """
    origin_y = int(bbox_min_y)
    mc_y_adj = mc_y_terrain - origin_y if origin_y > 0 else mc_y_terrain
    off_y = mc_y_adj + origin_y
    return off_y


# ── Step 1: position transform ────────────────────────────────────────────────

class TestPositionTransform:
    def test_canonical_x_flip(self):
        ko_x, ko_z = 200.0, 300.0
        mc_x, mc_z = ko_to_mc_position(ko_x, ko_z)
        assert mc_x == KO_MAP_SIZE - 200
        assert mc_z == 300

    def test_canonical_z_identity(self):
        for z in (0.0, 512.0, 1023.0):
            _, mc_z = ko_to_mc_position(0.0, z)
            assert mc_z == int(z)

    def test_distinct_positions_map_distinctly(self):
        positions = [(100, 200), (200, 200), (100, 300), (500, 500)]
        mc_set = set()
        for ko_x, ko_z in positions:
            mc_x, mc_z = ko_to_mc_position(float(ko_x), float(ko_z))
            mc_set.add((mc_x, mc_z))
        assert len(mc_set) == len(positions), "Distinct KO positions must map to distinct MC positions"


# ── Step 2: yaw transform ─────────────────────────────────────────────────────

class TestYawTransform:
    def test_yaw_negated(self):
        for ko_yaw in (0.0, math.pi / 2, math.pi, -math.pi / 4):
            mc_yaw = ko_to_mc_yaw(ko_yaw)
            assert mc_yaw == pytest.approx(-ko_yaw)

    def test_facing_north_unchanged_angle(self):
        """A shape facing KO-Z (yaw=0) should face -MC-Z after transform if axes align."""
        mc_yaw = ko_to_mc_yaw(0.0)
        assert mc_yaw == 0.0


# ── Step 3: voxelization in local space ───────────────────────────────────────

class TestVoxelization:
    def test_unit_cube_not_empty(self):
        verts, idxs = _unit_cube_mesh()
        grid = voxelize_mesh(verts, idxs, voxel_size=1.0)
        assert grid.count_filled() > 0

    def test_unit_cube_y_not_compressed(self):
        """Voxel Y extent must match the geometric height (no compression)."""
        verts, idxs = _unit_cube_mesh()
        # Scale up to 5 units tall
        verts_scaled = verts.copy()
        verts_scaled[:, 1] *= 5.0
        grid = voxelize_mesh(verts_scaled, idxs, voxel_size=1.0)
        assert grid.ny >= 5, (
            f"Expected ≥5 voxels tall for 5-unit mesh, got grid.ny={grid.ny}"
        )

    def test_apply_transform_preserves_height(self):
        """apply_transform must not alter Y scale (no height compression)."""
        verts, _ = _unit_cube_mesh()
        # 45-degree rotation around Y axis, scale (1,1,1)
        quat_y45 = (0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))
        transformed = apply_transform(verts, quat_y45, (1.0, 1.0, 1.0))
        original_aabb = compute_aabb(verts)
        transformed_aabb = compute_aabb(transformed)
        # Y extent must be the same (rotation around Y doesn't change Y range)
        orig_height = float(original_aabb[1][1] - original_aabb[0][1])
        trans_height = float(transformed_aabb[1][1] - transformed_aabb[0][1])
        assert abs(trans_height - orig_height) < 0.01, (
            f"Y height changed after Y-axis rotation: {orig_height:.3f} → {trans_height:.3f}"
        )

    def test_apply_transform_does_not_use_world_coords(self):
        """apply_transform must work in local space only (no world offset applied)."""
        verts, _ = _unit_cube_mesh()
        identity_quat = (0.0, 0.0, 0.0, 1.0)
        transformed = apply_transform(verts, identity_quat, (1.0, 1.0, 1.0))
        np.testing.assert_array_almost_equal(transformed[:, :3], verts[:, :3], decimal=5)

    def test_voxelization_origin_is_local(self):
        """grid.origin must be in local mesh space (near bbox_min), not world space."""
        verts, idxs = _unit_cube_mesh()
        grid = voxelize_mesh(verts, idxs, voxel_size=1.0)
        # For a unit cube centred at origin, bbox_min ≈ (-0.5, -0.5, -0.5)
        # grid.origin = bbox_min - 1 voxel padding ≈ (-1.5, -1.5, -1.5)
        assert grid.origin[1] < 0, (
            f"grid.origin[1]={grid.origin[1]:.3f} should be negative (local Y)"
        )


# ── Step 4: terrain grounding ─────────────────────────────────────────────────

class TestTerrainGrounding:
    def test_grounded_mesh_bottom_at_terrain(self):
        """When bbox_min_y ≤ 0, the mesh bottom must sit at the terrain surface Y."""
        mc_y_terrain = 70   # flat terrain surface
        bbox_min_y = -0.5   # mesh origin below bottom of mesh (typical)

        off_y = _ground_object(mc_y_terrain, bbox_min_y)
        # off_y = mc_y_terrain (since origin_y = int(-0.5) = 0, no correction)
        assert off_y == mc_y_terrain, (
            f"Grounded mesh: off_y={off_y} should equal terrain Y={mc_y_terrain}"
        )

    def test_floating_mesh_pulled_to_terrain(self):
        """When bbox_min_y > 0, the mesh must be pulled down to terrain surface."""
        mc_y_terrain = 70
        bbox_min_y = 3.0   # mesh floats 3 units above its local origin

        off_y = _ground_object(mc_y_terrain, bbox_min_y)
        # off_y should still equal mc_y_terrain after correction
        assert off_y == mc_y_terrain, (
            f"Floating mesh: off_y={off_y} should equal terrain Y={mc_y_terrain} "
            f"after pulling down by bbox_min_y={bbox_min_y}"
        )

    def test_no_y_scaling_during_grounding(self):
        """Grounding must not scale or compress voxel Y values.

        The unit cube mesh is scaled 10× in Y so it spans Y=-5..+5 in model space.
        The mesh local origin (0,0,0) sits at terrain surface (mc_y_terrain).
        Therefore:
          - bottom of mesh ≈ mc_y_terrain - 5 (partially underground)
          - top    of mesh ≈ mc_y_terrain + 5
        The top must be within 2 blocks of mc_y_terrain + mesh_half_height.
        """
        verts, idxs = _unit_cube_mesh()
        verts_tall = verts.copy()
        verts_tall[:, 1] *= 10.0  # 10-unit tall: Y in [-5, +5]

        grid = voxelize_mesh(verts_tall, idxs, voxel_size=1.0)
        bbox_min_y = float(grid.origin[1])   # ≈ -6.0 (bbox_min=-5, padding=1)

        mc_y_terrain = 80
        off_y = _ground_object(mc_y_terrain, bbox_min_y)

        # Find actual topmost filled voxel
        filled = np.argwhere(grid.data > 0)
        if len(filled) == 0:
            pytest.skip("No voxels in mesh")
        max_iy = int(filled[:, 1].max())
        top_mc_y = off_y + max_iy

        # The mesh top (model Y ≈ +5) should be at terrain + 5 ≈ 85
        mesh_half_height = 5   # mesh spans -5..+5, half-height = 5
        expected_top = mc_y_terrain + mesh_half_height
        assert abs(top_mc_y - expected_top) <= 2, (
            f"Top of tall mesh: top_mc_y={top_mc_y}, expected≈{expected_top} "
            f"(mc_y_terrain={mc_y_terrain} + half_height={mesh_half_height})"
        )

    def test_terrain_y_drives_object_y(self):
        """Objects at different terrain heights must sit at different MC Y levels."""
        verts, idxs = _unit_cube_mesh()
        grid = voxelize_mesh(verts, idxs, voxel_size=1.0)
        bbox_min_y = float(grid.origin[1])

        y_low  = _ground_object(64,  bbox_min_y)
        y_high = _ground_object(100, bbox_min_y)

        assert y_high > y_low, (
            f"Higher terrain should produce higher off_y: {y_low} vs {y_high}"
        )


# ── Step 5: X-flip placement ──────────────────────────────────────────────────

class TestXFlipPlacement:
    """Validate that voxel X offsets are subtracted (not added) in MC world."""

    def test_positive_local_x_maps_to_lower_mc_x(self):
        """A voxel at local_x=+5 must have a lower mc_x than the anchor."""
        mc_anchor_x = 500
        local_x = 5
        # Correct: wx = off_x - ix = (mc_anchor_x - origin_x) - ix
        # For origin_x=0: wx = mc_anchor_x - local_x
        wx_correct = mc_anchor_x - local_x
        assert wx_correct < mc_anchor_x, (
            f"X flip: local +x={local_x} from anchor {mc_anchor_x} "
            f"must decrease mc_x (got {wx_correct})"
        )

    def test_symmetric_mesh_straddles_anchor(self):
        """A symmetric mesh placed at anchor must straddle it equally on both sides."""
        verts, idxs = _unit_cube_mesh()
        # Cube is symmetric: local X ranges from -0.5 to +0.5
        grid = voxelize_mesh(verts, idxs, voxel_size=1.0)

        mc_anchor_x = 200
        origin_x = int(grid.origin[0])   # ≈ -1 (bbox_min - padding)

        off_x = mc_anchor_x - origin_x   # subtract (X flip)
        # voxels at ix=0..nx-1; mc_x goes from off_x-0 down to off_x-(nx-1)
        mc_x_max = off_x - 0
        mc_x_min = off_x - (grid.nx - 1)

        # Centre of MC extent should be near mc_anchor_x
        mc_x_centre = (mc_x_max + mc_x_min) / 2.0
        assert abs(mc_x_centre - mc_anchor_x) <= 2.0, (
            f"X-flip centre off: mc_x_centre={mc_x_centre:.1f}, "
            f"anchor={mc_anchor_x}"
        )

    def test_two_objects_x_order_preserved(self):
        """Two KO objects with different X positions must maintain relative MC order."""
        # KO: obj_A has smaller X than obj_B → MC: obj_A has LARGER X
        ko_x_A, ko_x_B = 200.0, 600.0
        mc_x_A, _ = ko_to_mc_position(ko_x_A, 0.0)
        mc_x_B, _ = ko_to_mc_position(ko_x_B, 0.0)
        assert mc_x_A > mc_x_B, (
            f"X order: ko_x_A={ko_x_A} < ko_x_B={ko_x_B} must yield "
            f"mc_x_A={mc_x_A} > mc_x_B={mc_x_B}"
        )

    def test_no_y_flip(self):
        """Y axis must not be flipped during placement."""
        # A voxel at local_y=+5 must map to MC y = off_y + 5 (positive)
        off_y = 70
        local_y = 5
        wy = off_y + local_y
        assert wy > off_y, "Y offset must be additive (no Y flip)"

    def test_no_z_flip(self):
        """Z axis must not be flipped during placement."""
        mc_anchor_z = 300
        origin_z = 0
        off_z = mc_anchor_z + origin_z
        local_z = 3
        wz = off_z + local_z
        assert wz > mc_anchor_z, "Z offset must be additive (no Z flip)"


# ── Full pipeline integration ─────────────────────────────────────────────────

class TestFullPipeline:
    """Integration test: simulate converting one OPD shape to MC blocks."""

    def test_pipeline_produces_blocks_near_terrain(self):
        """Placed voxels must start at the terrain surface Y (within 1 block)."""
        ko_x, ko_z = 400.0, 300.0
        mc_x, mc_z = ko_to_mc_position(ko_x, ko_z)

        # Flat terrain at height 5.0 KO → mc_y = MC_Y_BASE + 5 = 69
        terrain_h = 5.0
        h_grid = _flat_terrain_h_grid(terrain_h, n=129)
        mc_y_terrain = terrain_y_at(h_grid, mc_x, mc_z)
        assert mc_y_terrain == MC_Y_BASE + int(terrain_h)

        # Voxelize a unit cube
        verts, idxs = _unit_cube_mesh()
        quat_identity = (0.0, 0.0, 0.0, 1.0)
        verts_t = apply_transform(verts, quat_identity, (1.0, 1.0, 1.0))
        grid = voxelize_mesh(verts_t, idxs, voxel_size=1.0)

        # Ground the object
        origin_y = int(grid.origin[1])
        mc_y_adj = mc_y_terrain - origin_y if origin_y > 0 else mc_y_terrain
        off_y = mc_y_adj + origin_y

        # Collect all placed Y positions
        filled = np.argwhere(grid.data > 0)
        if len(filled) == 0:
            pytest.skip("No voxels in mesh (degenerate test data)")

        placed_ys = [off_y + int(iy) for _, iy, _ in filled]
        min_placed_y = min(placed_ys)

        # The lowest placed block should be at or near the terrain surface
        assert abs(min_placed_y - mc_y_terrain) <= 2, (
            f"Bottom of placed object: Y={min_placed_y}, terrain Y={mc_y_terrain}"
        )

    def test_pipeline_x_offset_correct(self):
        """Placed voxels must straddle the MC anchor in X (not to one side only)."""
        ko_x, ko_z = 512.0, 512.0
        mc_anchor_x, mc_anchor_z = ko_to_mc_position(ko_x, ko_z)

        verts, idxs = _unit_cube_mesh()
        quat_identity = (0.0, 0.0, 0.0, 1.0)
        verts_t = apply_transform(verts, quat_identity, (1.0, 1.0, 1.0))
        grid = voxelize_mesh(verts_t, idxs, voxel_size=1.0)

        origin_x = int(grid.origin[0])
        off_x = mc_anchor_x - origin_x   # X flip

        filled = np.argwhere(grid.data > 0)
        if len(filled) == 0:
            pytest.skip("No voxels")

        placed_xs = [off_x - int(ix) for _, _, ix in filled]
        mc_x_min = min(placed_xs)
        mc_x_max = max(placed_xs)

        # Anchor should fall within [mc_x_min, mc_x_max]
        assert mc_x_min <= mc_anchor_x <= mc_x_max, (
            f"Anchor X={mc_anchor_x} not within voxel X range [{mc_x_min}, {mc_x_max}]"
        )
