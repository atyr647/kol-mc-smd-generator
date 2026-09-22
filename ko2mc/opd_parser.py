"""Parser for Knight Online .opd (Object Post Data) files.

OPD files contain collision meshes, shape/object definitions with
positions, rotations, scales, mesh parts, textures, and event info.
"""

import struct
from dataclasses import dataclass, field


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class ShapePart:
    name: str = ""
    pivot: "Vector3" = field(default_factory=lambda: Vector3(0.0, 0.0, 0.0))
    textures: list[str] = field(default_factory=list)


# Event type constants
OBJECT_BIND = 0
OBJECT_GATE = 1
OBJECT_GATE2 = 2
OBJECT_GATE_LEVER = 3
OBJECT_FLAG_LEVER = 4
OBJECT_WARP_GATE = 5
OBJECT_BARRICADE = 6
OBJECT_REMOVE_BIND = 7
OBJECT_ANVIL = 8
OBJECT_ARTIFACT = 9
OBJECT_NPC = 11

EVENT_TYPE_NAMES = {
    OBJECT_BIND: "Bind Point",
    OBJECT_GATE: "Gate (left-right)",
    OBJECT_GATE2: "Gate (up-down)",
    OBJECT_GATE_LEVER: "Gate Lever",
    OBJECT_FLAG_LEVER: "Flag",
    OBJECT_WARP_GATE: "Warp Gate",
    OBJECT_BARRICADE: "Barricade",
    OBJECT_REMOVE_BIND: "Resurrection Point",
    OBJECT_ANVIL: "Magic Anvil",
    OBJECT_ARTIFACT: "Artifact",
    OBJECT_NPC: "NPC",
}


@dataclass
class Shape:
    name: str = ""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))
    parts: list[ShapePart] = field(default_factory=list)
    belong: int = 0  # 0=all, 1=Karus, 2=Elmo
    event_id: int = 0
    event_type: int = 0
    npc_id: int = 0
    npc_status: int = 0

    @property
    def is_event_object(self) -> bool:
        return self.event_id > 0

    @property
    def event_type_name(self) -> str:
        return EVENT_TYPE_NAMES.get(self.event_type, "Unknown")


@dataclass
class OPDFile:
    map_name: str = ""
    map_width: float = 0.0
    map_length: float = 0.0
    collision_face_count: int = 0
    collision_vertices: list[Vector3] = field(default_factory=list)
    shapes: list[Shape] = field(default_factory=list)
    old_file_mode: bool = False


def _decrypt_string(enc_bytes: bytes) -> str:
    """Decrypt an OPD encrypted string."""
    cipher_key1 = 0x6081
    cipher_key2 = 0x1608
    volatile_key = 0x0816
    result = []

    for i in range(len(enc_bytes) - 1):
        raw_byte = enc_bytes[i] & 0xFF
        temp_key = (volatile_key & 0xFF00) >> 8
        decrypted_byte = temp_key ^ raw_byte
        volatile_key = ((raw_byte + volatile_key) * cipher_key1 + cipher_key2) & 0xFFFF
        result.append(decrypted_byte & 0xFF)

    return bytes(result).decode("ascii", errors="replace").rstrip("\x00")


def _read_string(fp) -> str:
    """Read a length-prefixed string."""
    (length,) = struct.unpack("<I", fp.read(4))
    if length == 0:
        return ""
    data = fp.read(length)
    return data.decode("ascii", errors="replace").rstrip("\x00")


def _read_vector3(fp) -> Vector3:
    x, y, z = struct.unpack("<fff", fp.read(12))
    return Vector3(x, y, z)


def _read_quaternion(fp) -> Quaternion:
    x, y, z, w = struct.unpack("<ffff", fp.read(16))
    return Quaternion(x, y, z, w)


def _read_shape(fp) -> Shape:
    """Parse a single shape/object from the OPD stream."""
    shape = Shape()

    # Name
    shape.name = _read_string(fp)

    # Transform
    shape.position = _read_vector3(fp)
    shape.rotation = _read_quaternion(fp)
    shape.scale = _read_vector3(fp)

    # Animation keys (3 sets: pos, rot, scale)
    for _ in range(3):
        (count,) = struct.unpack("<I", fp.read(4))
        if count > 0:
            (key_type,) = struct.unpack("<I", fp.read(4))
            fp.read(4)  # sampling rate
            if key_type == 0:  # KEY_VECTOR3
                fp.read(12 * count)
            elif key_type == 1:  # KEY_QUATERNION
                fp.read(16 * count)

    # Collision mesh filenames (2)
    for _ in range(2):
        (str_len,) = struct.unpack("<I", fp.read(4))
        if str_len > 0:
            fp.read(str_len)

    # Parts
    (part_count,) = struct.unpack("<i", fp.read(4))
    for _ in range(part_count):
        part = ShapePart()
        part.pivot = _read_vector3(fp)   # local attachment offset (KO model space)

        # Part name
        part.name = _read_string(fp)

        # Material: D3DMATERIAL8 (68 bytes: 4 D3DCOLORVALUE of 16 bytes + 1 float)
        # plus __Material extra fields (dwColorOp, dwColorArg1, dwColorArg2, nRenderFlags, dwSrcBlend, dwDestBlend)
        # D3DMATERIAL8 = 4*16 + 4 = 68, plus 6*4 = 24 => total 92
        # Actually: Diffuse(16) + Ambient(16) + Specular(16) + Emissive(16) + Power(4) = 68
        # + dwColorOp(4) + dwColorArg1(4) + dwColorArg2(4) + nRenderFlags(4) + dwSrcBlend(4) + dwDestBlend(4) = 24
        # Total = 92
        fp.read(92)  # __Material

        (tex_count,) = struct.unpack("<I", fp.read(4))
        fp.read(4)  # tex FPS

        for _ in range(tex_count):
            tex_name = _read_string(fp)
            if tex_name:
                part.textures.append(tex_name)

        shape.parts.append(part)

    # Event properties
    shape.belong, shape.event_id, shape.event_type, shape.npc_id, shape.npc_status = (
        struct.unpack("<iiiii", fp.read(20))
    )

    return shape


def parse_opd(filepath: str) -> OPDFile:
    """Parse an .opd file and return object/collision data."""
    opd = OPDFile()

    with open(filepath, "rb") as fp:
        # Read encrypted map name
        (count,) = struct.unpack("<i", fp.read(4))
        if count == 1:
            opd.old_file_mode = True
            (count,) = struct.unpack("<i", fp.read(4))

        name_bytes = fp.read(count)
        if opd.old_file_mode:
            opd.map_name = name_bytes.decode("ascii", errors="replace").rstrip("\x00")
        else:
            opd.map_name = _decrypt_string(name_bytes + b"\x00")

        # Unknown value (new format only)
        if not opd.old_file_mode:
            fp.read(4)

        # Collision data
        opd.map_width, opd.map_length = struct.unpack("<ff", fp.read(8))
        (opd.collision_face_count,) = struct.unpack("<i", fp.read(4))

        if opd.collision_face_count > 0:
            for _ in range(opd.collision_face_count * 3):
                opd.collision_vertices.append(_read_vector3(fp))

        # Cell data - we skip the detailed cell structure since we don't need
        # subcell collision indices for Minecraft conversion
        cells_x = int(opd.map_width / 16)
        cells_z = int(opd.map_length / 16)

        for _ in range(cells_z):
            for _ in range(cells_x):
                (exists,) = struct.unpack("<I", fp.read(4))
                if not exists:
                    continue
                # Read main cell
                (shape_count,) = struct.unpack("<i", fp.read(4))
                if shape_count > 0:
                    fp.read(shape_count * 2)  # WORD shape indices
                # Read 4x4 subcells
                for _sz in range(4):
                    for _sx in range(4):
                        (poly_count,) = struct.unpack("<i", fp.read(4))
                        if poly_count > 0:
                            fp.read(poly_count * 3 * 4)  # uint32 indices

        # Shape count
        (shape_count,) = struct.unpack("<I", fp.read(4))
        halfway = shape_count // 2

        for i in range(shape_count):
            # Object type
            (dw_type,) = struct.unpack("<I", fp.read(4))
            shape = _read_shape(fp)
            opd.shapes.append(shape)

            # Halfway string (new format quirk)
            if not opd.old_file_mode and halfway - 1 == i:
                (str_len,) = struct.unpack("<I", fp.read(4))
                fp.read(str_len)
            elif opd.old_file_mode and halfway - 1 == i:
                pass  # No extra data in old format

    event_objects = [s for s in opd.shapes if s.is_event_object]
    print(f"  OPD: map='{opd.map_name}', size={opd.map_width}x{opd.map_length}")
    print(f"  OPD: {len(opd.shapes)} shapes, {len(event_objects)} event objects")
    print(f"  OPD: {opd.collision_face_count} collision faces")

    return opd
