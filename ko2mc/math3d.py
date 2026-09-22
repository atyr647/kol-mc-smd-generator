"""3D math utilities for KO→Minecraft coordinate conversion.

Coordinate system contract
--------------------------
  KO:  X-right, Y-up, Z-forward  (left-handed)
  MC:  X-east,  Y-up, Z-south    (right-handed, Z increases south)

Canonical KO→MC world transform
---------------------------------
The ONE authority for coordinate conversion.  Every module that needs to map
KO positions to MC positions MUST call these functions.  No other axis swaps,
flipud/rot90/fliplr, or sign flips are permitted anywhere else in the codebase.

  XZ:  mc_x = MAP_SIZE - ko_x   (mirrors the X axis)
       mc_z = ko_z              (Z unchanged)

  Y:   mc_y = MC_Y_BASE + ko_y * KO_HEIGHT_SCALE

  Yaw: mc_yaw = -ko_yaw         (X flip negates rotation around Y)

The terrain transform is empirically validated across 40 in-bounds landmarks:
np.rot90(gtd.heights, k=1) gives mean |error| 7.78 KO units vs 16.43 for flipud.
Root cause: the GTD binary is X-major (outer loop = KO_X, inner = KO_Z), but the
GTD parser stored heights[parser_x, parser_z] where parser's "z" index is the
outer (= KO_X) loop.  The stored array is therefore transposed relative to true
KO (tx, tz) indexing.  rot90 CCW corrects both the transpose and the X-mirror
in one operation: h_mc[i,j] = G[j, n-1-i], giving h_mc[n-1-ko_tx, ko_tz] =
G[ko_tz, ko_tx] = true height at KO tile (ko_tx, ko_tz). ✓

Height mapping
--------------
  KO height 0   → MC Y 64  (sea level)
  KO height 200 → MC Y 264

  With KO_HEIGHT_SCALE=1.0: 1 KO unit = 1 MC block (vertical).
  Horizontal: 1 KO unit = 1 MC block (4 KO units per GTD tile → 4 MC blocks).
"""

import math

import numpy as np

# ── Coordinate system constants ──────────────────────────────────────────────

KO_MAP_SIZE: int = 1024
"""Default KO world width/depth in KO units (256 tiles × 4 units/tile)."""

MC_Y_BASE: int = 64
"""MC block Y where KO Y=0 lands (sea level baseline)."""

KO_HEIGHT_SCALE: float = 1.0
"""KO height unit → MC blocks scale factor (1:1 vertical, matches horizontal)."""


# ── Canonical world transforms (THE ONLY PLACE transforms live) ──────────────

def ko_to_mc_position(ko_x: float, ko_z: float,
                      map_size: int = KO_MAP_SIZE) -> tuple[int, int]:
    """Convert KO world XZ to Minecraft block XZ.

    mc_x = map_size - ko_x   (X mirror so east increases as KO X decreases)
    mc_z = ko_z              (Z unchanged)

    This is the ONE authoritative XZ transform.  Do not replicate this
    arithmetic anywhere else.
    """
    return map_size - int(ko_x), int(ko_z)


def mc_to_ko_position(mc_x: int, mc_z: int,
                      map_size: int = KO_MAP_SIZE) -> tuple[float, float]:
    """Inverse of ko_to_mc_position.  Convert MC XZ back to KO world XZ.

    ko_x = map_size - mc_x
    ko_z = mc_z
    """
    return float(map_size - mc_x), float(mc_z)


def ko_to_mc_y(ko_y: float) -> int:
    """Convert KO world-space Y to Minecraft block Y coordinate.

    Uses math.floor(x + 0.5) — deterministic round-half-up — rather than
    int/floor (systematic downward bias) or Python round() (banker's rounding,
    where 0.5 ties round to even).  For positive terrain heights the three
    differ only on the exact n+0.5 boundary, but the downward bias of int/floor
    can sink objects by up to 1 block when their scaled KO height falls just
    below an integer.

    Max discretisation error: ±0.5 blocks (symmetric, not one-sided).
    """
    return math.floor(MC_Y_BASE + ko_y * KO_HEIGHT_SCALE + 0.5)


def mc_to_ko_y(mc_y: int) -> float:
    """Inverse of ko_to_mc_y.  Convert MC block Y back to KO world-space Y."""
    return (mc_y - MC_Y_BASE) / KO_HEIGHT_SCALE


# Keep legacy name as alias so existing callers don't break.
ko_height_to_mc = ko_to_mc_y


def ko_to_mc_yaw(ko_yaw: float) -> float:
    """Convert KO yaw (radians, Y-axis) to Minecraft yaw.

    Mirroring the X axis negates rotations around Y:
        mc_yaw = -ko_yaw
    """
    return -ko_yaw


def mc_to_ko_yaw(mc_yaw: float) -> float:
    """Inverse of ko_to_mc_yaw."""
    return -mc_yaw


def ko_pos_to_mc(ko_x: float, ko_y: float, ko_z: float,
                 map_size: int = KO_MAP_SIZE) -> tuple[int, int, int]:
    """Convert a full KO world position to Minecraft block coordinates (x, y, z).

    Delegates to the canonical per-axis functions.
    """
    mc_x, mc_z = ko_to_mc_position(ko_x, ko_z, map_size)
    mc_y = ko_to_mc_y(ko_y)
    return mc_x, mc_y, mc_z


# ── Terrain grid helper ────────────────────────────────────────────────────────

def build_mc_height_grid(gtd_heights: np.ndarray) -> np.ndarray:
    """Return the canonical MC-oriented height grid from a raw GTD heights array.

    Root cause: the GTD binary is X-major (outer binary loop = KO_X, inner = KO_Z),
    but the GTD parser variable naming treated outer as Z and inner as X, so the
    stored array has axes transposed: gtd_heights[ko_tz, ko_tx] = true height at
    KO tile (ko_tx, ko_tz).

    np.rot90(gtd_heights, k=1) corrects both the transpose and the required X-mirror
    (mc_tx = n-1-ko_tx) in one operation:
        h_mc[i, j] = gtd_heights[j, n-1-i]
        → h_mc[n-1-ko_tx, ko_tz] = gtd_heights[ko_tz, ko_tx] = height(ko_tx, ko_tz) ✓

    Validated against 40 in-bounds OPD landmarks: mean |error| 7.78 KO units
    (vs 16.43 for the previous flipud).

    Returns a float32 array of shape (n, n) where
        h_mc[mc_tx, mc_tz] == true_height(ko_tx, ko_tz)
    with mc_tx = (n-1) - ko_tx  (consistent with ko_to_mc_position X mirror).

    Use this result as the single terrain grid for both build_terrain() and
    terrain_y_at().  Never apply additional transforms to GTD heights elsewhere.
    """
    return np.rot90(gtd_heights, k=1).astype(np.float32)


def terrain_y_at(h_grid_mc: np.ndarray, mc_x: float, mc_z: float) -> int:
    """Sample terrain surface MC Y at (mc_x, mc_z) via bilinear interpolation.

    h_grid_mc: the canonical MC-oriented height grid from build_mc_height_grid().
    mc_x, mc_z: MC world block coordinates.

    Returns MC block Y of the terrain surface at that column.
    """
    n = h_grid_mc.shape[0]
    tx_f = mc_x / 4.0
    tz_f = mc_z / 4.0
    tx0 = max(0, min(int(tx_f), n - 2))
    tz0 = max(0, min(int(tz_f), n - 2))
    tx1, tz1 = tx0 + 1, tz0 + 1
    fx = tx_f - tx0
    fz = tz_f - tz0
    ko_h = (
        h_grid_mc[tx0, tz0] * (1.0 - fx) * (1.0 - fz)
        + h_grid_mc[tx1, tz0] * fx * (1.0 - fz)
        + h_grid_mc[tx0, tz1] * (1.0 - fx) * fz
        + h_grid_mc[tx1, tz1] * fx * fz
    )
    return ko_to_mc_y(float(ko_h))


# One terrain tile in MC space is 4 blocks wide.  A footprint whose
# half-extent reaches one tile boundary away from the centre spans two
# tiles and can land on a height step.  Use this threshold to decide
# whether multi-corner sampling is worth doing.
_MC_TILE_BLOCKS = 4


def terrain_y_at_footprint(
    h_grid_mc: np.ndarray,
    mc_x: float,
    mc_z: float,
    half_width: float = 0.0,
    half_depth: float = 0.0,
) -> int:
    """Return the lowest terrain Y under a rectangular footprint.

    Samples the four corners of the footprint and returns the minimum MC Y.
    This prevents large multi-tile structures from partially sinking into
    terrain when the slope under the footprint spans more than one GTD height
    step.

    When half_width and half_depth are both < one terrain-tile width
    (_MC_TILE_BLOCKS = 4 blocks) the footprint fits inside a single bilinear
    cell and a centre-point sample is returned directly — no extra sampling.

    Grounding rule (why minimum, not average or maximum)
    ----------------------------------------------------
    * maximum → structure sinks on the low side of a slope
    * average → structure clips terrain on both high corners
    * minimum → structure may appear to float by ≤1 block on high corners,
                but never sinks (partial burial)

    Floating by a block is invisible from the player's eye level; sinking
    (partial burial) breaks visual continuity.

    Invariant
    ---------
    The returned Y is always ≤ each of the four corner samples individually.
    It is NOT necessarily ≤ the centre-point sample (the centre may sit in a
    valley below all four corners).
    """
    if half_width < _MC_TILE_BLOCKS and half_depth < _MC_TILE_BLOCKS:
        # Footprint fits within one bilinear cell — single sample is accurate.
        return terrain_y_at(h_grid_mc, mc_x, mc_z)

    corners = [
        terrain_y_at(h_grid_mc, mc_x - half_width, mc_z - half_depth),
        terrain_y_at(h_grid_mc, mc_x + half_width, mc_z - half_depth),
        terrain_y_at(h_grid_mc, mc_x - half_width, mc_z + half_depth),
        terrain_y_at(h_grid_mc, mc_x + half_width, mc_z + half_depth),
    ]
    return min(corners)


# ── Rotation ──────────────────────────────────────────────────────────────────

def quat_to_matrix(q) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to a 3×3 rotation matrix (float32)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_to_euler(q) -> tuple:
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll  = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def quat_yaw(q) -> float:
    """Extract KO yaw (rotation around Y axis, radians) from a quaternion (x,y,z,w)."""
    _, _, yaw = quat_to_euler(q)
    return yaw


# ── Mesh transform ────────────────────────────────────────────────────────────

def apply_transform(
    vertices: np.ndarray,
    rotation_quat,
    scale,
) -> np.ndarray:
    """Apply rotation and non-uniform scale to a vertex array.

    vertices:      (N, 8) float32 — x,y,z,nx,ny,nz,u,v
    rotation_quat: (x, y, z, w) quaternion — KO local-space rotation from OPD
    scale:         (sx, sy, sz) scale factors

    Returns a new (N, 8) array. Positions are scaled then rotated;
    normals are rotated only (scale does not affect direction).

    NOTE: This transform operates entirely in model-local space and must be
    called BEFORE applying the world KO→MC transform.  The OPD rotation
    quaternion is a KO-space rotation; applying it here (in local space)
    is correct.  The world-space X mirror is then applied separately via
    ko_to_mc_position() when computing the MC anchor.
    """
    result = vertices.copy()
    R = quat_to_matrix(rotation_quat)
    S = np.asarray(scale, dtype=np.float32)

    result[:, :3] = (vertices[:, :3] * S) @ R.T
    result[:, 3:6] = vertices[:, 3:6] @ R.T

    return result


# ── Pivot / part-offset helper ────────────────────────────────────────────────

def transform_pivot(pivot_local, rotation_quat, scale) -> np.ndarray:
    """Return the world-space offset introduced by an OPD part pivot.

    In Knight Online each mesh part carries a *pivot* — the local-space offset
    from the shape's origin to the part's mesh anchor.  After the shape's
    rotation and scale are applied, this offset becomes:

        pivot_world = R × (S × pivot_local)

    Parameters
    ----------
    pivot_local : array-like (3,) — pivot in KO model-local space (x, y, z)
    rotation_quat : (x, y, z, w) quaternion — shape rotation from OPD
    scale         : (sx, sy, sz) — shape scale from OPD

    Returns
    -------
    np.ndarray (3,) float32 — pivot offset in KO world-local space (dx, dy, dz).
    Add this to the shape's KO world position to get the part's effective anchor.
    """
    piv = np.asarray(pivot_local, dtype=np.float32)
    S   = np.asarray(scale,       dtype=np.float32)
    R   = quat_to_matrix(rotation_quat)
    return R @ (S * piv)


def pivot_to_mc_offset(pivot_local, rotation_quat, scale) -> tuple[int, int, int]:
    """Convert an OPD part pivot to an MC block offset (dx, dy, dz).

    Applies the shape rotation/scale to the KO-local pivot and converts the
    result to MC coordinates.  The X component is negated because the canonical
    world transform mirrors X (mc_x = map_size − ko_x).

    Returns
    -------
    (mc_dx, mc_dy, mc_dz) integer block offsets to add to the MC shape anchor.
    """
    pw = transform_pivot(pivot_local, rotation_quat, scale)
    mc_dx = -int(round(float(pw[0])))   # X mirror
    mc_dy =  int(round(float(pw[1])))   # Y unchanged
    mc_dz =  int(round(float(pw[2])))   # Z unchanged
    return mc_dx, mc_dy, mc_dz


# ── AABB ──────────────────────────────────────────────────────────────────────

def compute_aabb(vertices: np.ndarray) -> tuple:
    """Compute axis-aligned bounding box from vertex array (N, 3+).

    Returns (min_xyz, max_xyz) as float32 numpy arrays.
    """
    if len(vertices) == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    pos = vertices[:, :3]
    return pos.min(axis=0).astype(np.float32), pos.max(axis=0).astype(np.float32)
