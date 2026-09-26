"""KO character/mob models: `.n3chr` + `.n3joint` + `.n3cpart` + `.n3cskins`.

Ported 1:1 from the validated TypeScript parsers in the `webmmoproj` repo
(src/ko/n3chr.ts, n3joint.ts, n3cpart.ts, n3skin.ts, n3imesh.ts, pose.ts,
mat4.ts, n3.ts) — those were checked byte-for-byte against a real 1298
client's mob_attila.n3chr. This is the character/skinned-mesh sibling of
ko_models.py's parse_n3pmesh (static progressive meshes): mobs and NPCs are
built from a skeleton (`.n3joint`) plus several skinned body parts
(`.n3cpart` + `.n3cskins`), not one static mesh.

Unlike static objects, a character has no single "at rest" geometry in the
file: the skin's vertices are stored in a bind pose, and the actual pose used
at runtime comes from evaluating the skeleton's baked animation timeline at
a frame. We only need frame 0 (a rest/idle pose) to voxelize a mob "statue" —
see `bind_world_matrices` for why frame 0 is the CORRECT bind pose (not the
joint's raw loaded transform: some skeletons' loaded rest pose differs from
frame 0, and the skin was authored against frame 0).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np


class Reader:
    """Little-endian binary cursor, mirroring webmmoproj's BinaryReader."""

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def i32(self) -> int:
        (v,) = struct.unpack_from("<i", self.data, self.pos)
        self.pos += 4
        return v

    def u32(self) -> int:
        (v,) = struct.unpack_from("<I", self.data, self.pos)
        self.pos += 4
        return v

    def f32(self) -> float:
        (v,) = struct.unpack_from("<f", self.data, self.pos)
        self.pos += 4
        return v

    def vec3(self):
        v = struct.unpack_from("<3f", self.data, self.pos)
        self.pos += 12
        return list(v)

    def quat(self):
        v = struct.unpack_from("<4f", self.data, self.pos)
        self.pos += 16
        return list(v)

    def f32_array(self, n: int) -> np.ndarray:
        v = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.pos)
        self.pos += 4 * n
        return v.astype(np.float32)

    def u16_array(self, n: int) -> np.ndarray:
        v = np.frombuffer(self.data, dtype="<u2", count=n, offset=self.pos)
        self.pos += 2 * n
        return v.astype(np.uint16)

    def fixed_string(self, n: int) -> str:
        raw = self.data[self.pos:self.pos + n]
        self.pos += n
        return raw.rstrip(b"\x00").decode("latin-1")

    def len_string(self) -> str:
        n = self.i32()
        if n <= 0:
            return ""
        return self.fixed_string(n)

    def skip(self, n: int) -> None:
        self.pos += n


# --- n3.ts: shared transform reader ----------------------------------------

def _read_empty_key_track(r: Reader, which: str) -> None:
    count = r.i32()
    if count != 0:
        raise ValueError(f"N3: keyframed {which} track (count={count}) unexpected on a static transform")


def read_transform(r: Reader):
    name = r.len_string()
    pos = r.vec3()
    rot = r.quat()
    scale = r.vec3()
    _read_empty_key_track(r, "position")
    _read_empty_key_track(r, "rotation")
    _read_empty_key_track(r, "scale")
    return name, pos, rot, scale


def read_transform_collision(r: Reader):
    name, pos, rot, scale = read_transform(r)
    collision_mesh_ref = r.len_string()
    climb_mesh_ref = r.len_string()
    return name, pos, rot, scale, collision_mesh_ref, climb_mesh_ref


# --- n3chr.ts: character manifest -------------------------------------------

@dataclass
class N3Character:
    name: str
    joint_ref: str
    part_refs: list
    plug_refs: list
    ani_ref: str
    fx_plug_ref: str


def parse_n3chr(data: bytes) -> N3Character:
    r = Reader(data)
    name, *_rest = read_transform_collision(r)
    joint_ref = r.len_string()
    part_refs = [r.len_string() for _ in range(r.i32())]
    plug_refs = [r.len_string() for _ in range(r.i32())]
    ani_ref = r.len_string()
    r.i32(); r.i32()  # jointPartStarts
    r.i32(); r.i32()  # jointPartEnds
    fx_plug_ref = r.len_string()
    return N3Character(name, joint_ref, part_refs, plug_refs, ani_ref, fx_plug_ref)


# --- n3joint.ts: skeleton ----------------------------------------------------

KEY_VECTOR3 = 0
KEY_QUATERNION = 1


@dataclass
class AnimKey:
    type: int
    sampling_rate: float
    keys: list  # list of [x,y,z] or [x,y,z,w]


EMPTY_KEY = AnimKey(-1, 0.0, [])


@dataclass
class Joint:
    name: str
    pos: list
    rot: list
    scale: list
    key_pos: AnimKey
    key_rot: AnimKey
    key_scale: AnimKey
    key_orient: AnimKey
    parent: int


@dataclass
class Skeleton:
    joints: list


def _read_anim_key(r: Reader) -> AnimKey:
    count = r.i32()
    if count <= 0:
        return AnimKey(EMPTY_KEY.type, EMPTY_KEY.sampling_rate, [])
    typ = r.u32()
    sampling_rate = r.f32()
    keys = []
    for _ in range(count):
        keys.append(r.quat() if typ == KEY_QUATERNION else r.vec3())
    return AnimKey(typ, sampling_rate, keys)


def _read_joint(r: Reader, parent: int, out: list) -> None:
    name = r.len_string()
    pos = r.vec3()
    rot = r.quat()
    scale = r.vec3()
    key_pos = _read_anim_key(r)
    key_rot = _read_anim_key(r)
    key_scale = _read_anim_key(r)
    key_orient = _read_anim_key(r)
    index = len(out)
    out.append(Joint(name, pos, rot, scale, key_pos, key_rot, key_scale, key_orient, parent))
    child_count = r.i32()
    for _ in range(child_count):
        _read_joint(r, index, out)


def parse_n3joint(data: bytes) -> Skeleton:
    r = Reader(data)
    joints: list = []
    _read_joint(r, -1, joints)
    return Skeleton(joints)


def sample_key_frame0(key: AnimKey, fallback):
    """Sample a key track at frame 0 -- always exactly keys[0] if any exist
    (see pose.ts's sampleKey: nIndex=floor(0)=0, fDelta=0 at frm=0), else the
    joint's raw loaded value."""
    if not key.keys:
        return fallback
    return key.keys[0]


# --- n3cpart.ts: body-part material/texture ---------------------------------

@dataclass
class N3CPart:
    name: str
    diffuse: tuple
    texture_ref: str
    skins_ref: str


_MATERIAL_BYTES = 92
_DIFFUSE_BYTES = 16


def parse_n3cpart(data: bytes) -> N3CPart:
    r = Reader(data)
    name = r.len_string()
    r.skip(4)  # m_dwReserved
    diffuse = tuple(r.vec3() + [r.f32()])  # rgba (vec3 helper reused for 3 floats + 1 more)
    r.skip(_MATERIAL_BYTES - _DIFFUSE_BYTES)
    texture_ref = r.len_string()
    skins_ref = r.len_string()
    return N3CPart(name, diffuse, texture_ref, skins_ref)


# --- n3imesh.ts: base indexed mesh -------------------------------------------

@dataclass
class N3IMesh:
    name: str
    face_count: int
    vertex_count: int
    uv_count: int
    positions: np.ndarray  # (V,3)
    normals: np.ndarray
    vertex_indices: np.ndarray  # (F*3,)
    uvs: np.ndarray  # (U,2)
    uv_indices: np.ndarray  # (F*3,)


def read_n3imesh(r: Reader) -> N3IMesh:
    name = r.len_string()
    face_count = r.i32()
    vertex_count = r.i32()
    uv_count = r.i32()

    positions = np.zeros((max(0, vertex_count), 3), dtype=np.float32)
    normals = np.zeros((max(0, vertex_count), 3), dtype=np.float32)
    vertex_indices = np.zeros(0, dtype=np.uint16)

    if face_count > 0 and vertex_count > 0:
        for i in range(vertex_count):
            positions[i] = r.vec3()
            normals[i] = r.vec3()
        vertex_indices = r.u16_array(face_count * 3)

    uvs = np.zeros((0, 2), dtype=np.float32)
    uv_indices = np.zeros(0, dtype=np.uint16)
    if uv_count > 0:
        uvs = r.f32_array(uv_count * 2).reshape(-1, 2)
        uv_indices = r.u16_array(face_count * 3)

    return N3IMesh(name, face_count, vertex_count, uv_count, positions, normals, vertex_indices, uvs, uv_indices)


# --- n3skin.ts: skinned geometry (.n3cskins), 4 LODs -------------------------

MAX_CHR_LOD = 4


@dataclass
class SkinVertex:
    origin: np.ndarray  # (3,)
    joints: list
    weights: list


@dataclass
class N3Skin:
    mesh: N3IMesh
    vertices: list  # list[SkinVertex]


@dataclass
class N3CPartSkins:
    name: str
    lods: list  # list[N3Skin], LOD 0 = highest detail


def _read_skin(r: Reader) -> N3Skin:
    mesh = read_n3imesh(r)
    vertices = []
    for _ in range(mesh.vertex_count):
        origin = np.array(r.vec3(), dtype=np.float32)
        n_affect = r.i32()
        r.skip(8)  # two dead pointers
        joints: list = []
        weights: list = []
        if n_affect > 1:
            joints = [r.i32() for _ in range(n_affect)]
            weights = [r.f32() for _ in range(n_affect)]
        elif n_affect == 1:
            joints = [r.i32()]
            weights = [1.0]
        vertices.append(SkinVertex(origin, joints, weights))
    return N3Skin(mesh, vertices)


def parse_n3skins(data: bytes) -> N3CPartSkins:
    r = Reader(data)
    name = r.len_string()
    lods = [_read_skin(r) for _ in range(MAX_CHR_LOD)]
    return N3CPartSkins(name, lods)


# --- mat4.ts: row-vector affine math (KO space) ------------------------------
# 4x4 matrices stored as flat length-16 float lists, row-major, row-vector
# convention (p' = p . M): matches webmmoproj's mat4.ts exactly so this port
# stays a direct translation, not a re-derivation.

def _from_quat(rot) -> np.ndarray:
    x, y, z, w = rot
    m = np.identity(4, dtype=np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m[0, 0] = 1 - 2 * (yy + zz)
    m[0, 1] = 2 * (xy + wz)
    m[0, 2] = 2 * (xz - wy)
    m[1, 0] = 2 * (xy - wz)
    m[1, 1] = 1 - 2 * (xx + zz)
    m[1, 2] = 2 * (yz + wx)
    m[2, 0] = 2 * (xz + wy)
    m[2, 1] = 2 * (yz - wx)
    m[2, 2] = 1 - 2 * (xx + yy)
    return m


def compose(rot, scale, pos) -> np.ndarray:
    """Joint local transform: fromQuat(rot) with columns 0-2 scaled, then
    translation set -- exactly mat4.ts's `compose` (used for joints)."""
    m = _from_quat(rot)
    if scale[0] != 1 or scale[1] != 1 or scale[2] != 1:
        m[:, 0] *= scale[0]
        m[:, 1] *= scale[1]
        m[:, 2] *= scale[2]
    m[3, 0], m[3, 1], m[3, 2] = pos
    return m


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a.b in row-vector convention (apply a then b) == numpy a @ b for
    row-major matrices used this way."""
    return a @ b


def transform_point(p, m: np.ndarray):
    x, y, z = p
    return np.array([
        x * m[0, 0] + y * m[1, 0] + z * m[2, 0] + m[3, 0],
        x * m[0, 1] + y * m[1, 1] + z * m[2, 1] + m[3, 1],
        x * m[0, 2] + y * m[1, 2] + z * m[2, 2] + m[3, 2],
    ])


def affine_inverse(m: np.ndarray) -> np.ndarray:
    r = m[:3, :3]
    t = m[3, :3]
    r_inv = np.linalg.inv(r) if abs(np.linalg.det(r)) > 1e-12 else np.identity(3)
    out = np.identity(4, dtype=np.float64)
    out[:3, :3] = r_inv
    out[3, :3] = -(t @ r_inv)
    return out


# --- pose.ts: FK + linear-blend skinning -------------------------------------

def _worlds_from(skel: Skeleton, locals_: list) -> list:
    world = [None] * len(skel.joints)
    for i, j in enumerate(skel.joints):
        world[i] = locals_[i] if j.parent < 0 else multiply(locals_[i], world[j.parent])
    return world


def bind_matrices(skel: Skeleton):
    """Bind-pose world matrices + inverses, evaluated at animation frame 0 --
    NOT the raw loaded joint TRS. KO's CN3Chr::Init ticks the skeleton to
    frame 0 before inverting; some skeletons' loaded rest pose differs from
    frame 0, and the skin's vertex origins were authored against frame 0, so
    using the raw transform puts joints away from the mesh and any later
    pose sweeps vertices into spikes. See pose.ts's `bindMatrices` docstring.
    """
    locals_ = []
    for j in skel.joints:
        rot = sample_key_frame0(j.key_rot, j.rot)
        pos = sample_key_frame0(j.key_pos, j.pos)
        scale = sample_key_frame0(j.key_scale, j.scale)
        locals_.append(compose(rot, scale, pos))
    world = _worlds_from(skel, locals_)
    inverse = [affine_inverse(w) for w in world]
    return world, inverse


def skin_positions(skin: N3Skin, inv_bind: list, world: list) -> np.ndarray:
    """Skin a part to world-space positions at the given pose (bind world for
    a rest-pose voxelization). KO's exact formula: posed = sum_b weight_b *
    (origin . invBind_b) . world_b."""
    v = skin.mesh.vertex_count
    out = np.zeros((v, 3), dtype=np.float64)
    for i in range(v):
        sv = skin.vertices[i]
        if not sv.joints:
            out[i] = sv.origin
            continue
        acc = np.zeros(3)
        for b, w in zip(sv.joints, sv.weights):
            if b < 0 or b >= len(inv_bind):
                continue
            t = transform_point(sv.origin, inv_bind[b])
            t = transform_point(t, world[b])
            acc += t * w
        out[i] = acc
    return out


# --- micro-voxel model builder ----------------------------------------------
# A mob's whole body is ~1-4m tall, so a real Minecraft block (1m) is far too
# coarse to be recognizable -- unlike buildings, a mob needs a MUCH finer grid
# (a handful of centimeters per voxel), so this is expressed as fractional
# offsets within a single block rather than one Minecraft block per voxel; the
# server side is expected to render each voxel as a small scaled/positioned
# block-display entity (Minecraft 1.19.4+), not a placed world block.

def load_character(chr_path: str, item_dir: str, chr_dir: str | None = None):
    """Load a `.n3chr` + its skeleton + every body part's skin+texture.
    Returns (character, skeleton, parts) where parts is a list of
    (N3Skin (best LOD), texture_rgba, N3CPart)."""
    import os
    chr_dir = chr_dir or os.path.dirname(chr_path)
    data = open(chr_path, "rb").read()
    ch = parse_n3chr(data)
    joint_name = ch.joint_ref.split("\\")[-1]
    skel = parse_n3joint(open(os.path.join(chr_dir, joint_name), "rb").read())

    from .ko_textures import read_n3_textures

    parts = []
    for part_ref in ch.part_refs:
        base = part_ref.split("\\")[-1].rsplit(".", 1)[0]
        cpart_path = os.path.join(item_dir, base + ".n3cpart")
        cskins_path = os.path.join(item_dir, base + ".n3cskins")
        if not (os.path.exists(cpart_path) and os.path.exists(cskins_path)):
            continue
        part = parse_n3cpart(open(cpart_path, "rb").read())
        skins = parse_n3skins(open(cskins_path, "rb").read())
        lod = next((l for l in skins.lods if l.mesh.vertex_count > 0 and l.mesh.face_count > 0), None)
        if lod is None:
            continue
        tex_name = part.texture_ref.split("\\")[-1]
        tex_path = os.path.join(item_dir, tex_name)
        if os.path.exists(tex_path):
            texs = read_n3_textures(open(tex_path, "rb").read())
            tex_rgba = texs[0].rgba if texs else np.full((4, 4, 4), 200, np.uint8)
        else:
            tex_rgba = np.full((4, 4, 4), 200, np.uint8)
        parts.append((lod, tex_rgba, part))
    return ch, skel, parts


def posed_triangle_soup(skel: Skeleton, parts: list):
    """Bind-pose (frame 0) world-space triangles + UVs + texture per part.
    Returns a list of (tris (F,3,3), uvs (F,3,2), tex_rgba) -- the same shape
    ko_models.py's sample_part / qa_render.py's ko_mesh already consume."""
    world, inv = bind_matrices(skel)
    out = []
    for lod, tex_rgba, _part in parts:
        posed = skin_positions(lod, inv, world)
        mesh = lod.mesh
        if mesh.face_count == 0 or mesh.uv_count == 0:
            continue
        vidx = mesh.vertex_indices.reshape(-1, 3).astype(np.int64)
        uvidx = mesh.uv_indices.reshape(-1, 3).astype(np.int64)
        tris = posed[vidx].astype(np.float32)
        uvs = mesh.uvs[uvidx].astype(np.float32)
        out.append((tris, uvs, tex_rgba))
    return out


def voxelize_character(parts, voxel_size: float = 0.08):
    """Surface-sample every part's posed triangles and bucket samples into a
    voxel grid (grid units = `voxel_size` metres each -- KO's own units, same
    ones the posed positions are already in). Overlapping parts (elbows,
    collars, ...) just average into the same cell, matching how the rest of
    this codebase treats ambiguous multi-surface samples.

    Returns {(vx, vy, vz): (r, g, b)} -- integer grid indices, average RGB.
    """
    from .ko_models import sample_part

    sums: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, int] = {}
    for tris, uvs, tex_rgba in parts:
        pts, _tri, tx, ty = sample_part(tris, uvs, tex_rgba, alpha_test=False, spacing=voxel_size * 0.6)
        if len(pts) == 0:
            continue
        colors = tex_rgba[ty, tx, :3].astype(np.float64)
        cells = np.floor(pts / voxel_size).astype(np.int64)
        for cell, color in zip(map(tuple, cells), colors):
            if cell in sums:
                sums[cell] += color
                counts[cell] += 1
            else:
                sums[cell] = color.copy()
                counts[cell] = 1
    return {cell: tuple((sums[cell] / counts[cell]).round().astype(int)) for cell in sums}


def voxels_to_blocks(voxels: dict) -> dict:
    """Map each voxel's average RGB to the nearest Minecraft block, using
    every block ko_models.py's BLOCK_COLORS knows (not just BUILD_BLOCKS --
    a mob's skin/fur/cloth needs wool/concrete/terracotta's much wider hue
    range, unlike a building's stone/wood palette)."""
    from .ko_models import BLOCK_COLORS, Palette

    if not hasattr(voxels_to_blocks, "_palette"):
        voxels_to_blocks._palette = Palette(BLOCK_COLORS)
        voxels_to_blocks._names = list(BLOCK_COLORS)
    palette = voxels_to_blocks._palette
    names = voxels_to_blocks._names
    if not voxels:
        return {}
    cells = list(voxels)
    rgb = np.array([voxels[c] for c in cells], dtype=np.float64)
    idx = palette.nearest(rgb)
    return {cell: names[i] for cell, i in zip(cells, idx)}
