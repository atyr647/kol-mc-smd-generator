"""Minimal NBT (Named Binary Tag) reader/writer used for Minecraft files.

Values map to Python types like this when writing:
  Byte(v) / Short(v) / Int(v) / Long(v) / Float(v) / Double(v) wrappers,
  str -> String, dict -> Compound, List(tag_type, items) -> List,
  numpy int8 array -> Byte_Array, int32 array -> Int_Array, int64 array -> Long_Array.
Plain Python ints are written as Int, floats as Double, bools as Byte.

Reading returns plain Python values (numpy arrays for the array tags).
"""

import struct

import numpy as np

TAG_END, TAG_BYTE, TAG_SHORT, TAG_INT, TAG_LONG, TAG_FLOAT, TAG_DOUBLE = range(7)
TAG_BYTE_ARRAY, TAG_STRING, TAG_LIST, TAG_COMPOUND, TAG_INT_ARRAY, TAG_LONG_ARRAY = range(7, 13)


class _Scalar:
    tag = 0
    fmt = ""

    def __init__(self, value):
        self.value = value


class Byte(_Scalar):
    tag, fmt = TAG_BYTE, ">b"


class Short(_Scalar):
    tag, fmt = TAG_SHORT, ">h"


class Int(_Scalar):
    tag, fmt = TAG_INT, ">i"


class Long(_Scalar):
    tag, fmt = TAG_LONG, ">q"


class Float(_Scalar):
    tag, fmt = TAG_FLOAT, ">f"


class Double(_Scalar):
    tag, fmt = TAG_DOUBLE, ">d"


class List:
    def __init__(self, tag_type: int, items):
        self.tag_type = tag_type
        self.items = list(items)


def _tag_of(value) -> int:
    if isinstance(value, _Scalar):
        return value.tag
    if isinstance(value, bool):
        return TAG_BYTE
    if isinstance(value, int):
        return TAG_INT
    if isinstance(value, float):
        return TAG_DOUBLE
    if isinstance(value, str):
        return TAG_STRING
    if isinstance(value, dict):
        return TAG_COMPOUND
    if isinstance(value, List):
        return TAG_LIST
    if isinstance(value, np.ndarray):
        return {1: TAG_BYTE_ARRAY, 4: TAG_INT_ARRAY, 8: TAG_LONG_ARRAY}[value.dtype.itemsize]
    raise TypeError(f"Cannot encode {type(value)} as NBT")


def _write_payload(out: list, tag: int, value):
    if isinstance(value, _Scalar):
        out.append(struct.pack(value.fmt, value.value))
    elif tag == TAG_BYTE:
        out.append(struct.pack(">b", int(value)))
    elif tag == TAG_INT:
        out.append(struct.pack(">i", value))
    elif tag == TAG_DOUBLE:
        out.append(struct.pack(">d", value))
    elif tag == TAG_STRING:
        b = value.encode("utf-8")
        out.append(struct.pack(">H", len(b)))
        out.append(b)
    elif tag == TAG_COMPOUND:
        for k, v in value.items():
            t = _tag_of(v)
            kb = k.encode("utf-8")
            out.append(struct.pack(">bH", t, len(kb)))
            out.append(kb)
            _write_payload(out, t, v)
        out.append(b"\x00")
    elif tag == TAG_LIST:
        out.append(struct.pack(">bi", value.tag_type if value.items else TAG_END, len(value.items)))
        for item in value.items:
            _write_payload(out, value.tag_type, item)
    elif tag in (TAG_BYTE_ARRAY, TAG_INT_ARRAY, TAG_LONG_ARRAY):
        dt = {TAG_BYTE_ARRAY: ">i1", TAG_INT_ARRAY: ">i4", TAG_LONG_ARRAY: ">i8"}[tag]
        out.append(struct.pack(">i", len(value)))
        out.append(np.asarray(value).astype(dt, copy=False).tobytes())


def encode(root: dict, name: str = "") -> bytes:
    """Encode a root compound to (uncompressed) NBT bytes."""
    out = []
    nb = name.encode("utf-8")
    out.append(struct.pack(">bH", TAG_COMPOUND, len(nb)))
    out.append(nb)
    _write_payload(out, TAG_COMPOUND, root)
    return b"".join(out)


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def unpack(self, fmt):
        vals = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += struct.calcsize(fmt)
        return vals[0]

    def string(self):
        n = self.unpack(">H")
        s = self.data[self.pos:self.pos + n].decode("utf-8", errors="replace")
        self.pos += n
        return s

    def array(self, dt, size):
        n = self.unpack(">i")
        a = np.frombuffer(self.data, dtype=dt, count=n, offset=self.pos)
        self.pos += n * size
        return a

    def payload(self, tag):
        if tag == TAG_BYTE:
            return self.unpack(">b")
        if tag == TAG_SHORT:
            return self.unpack(">h")
        if tag == TAG_INT:
            return self.unpack(">i")
        if tag == TAG_LONG:
            return self.unpack(">q")
        if tag == TAG_FLOAT:
            return self.unpack(">f")
        if tag == TAG_DOUBLE:
            return self.unpack(">d")
        if tag == TAG_BYTE_ARRAY:
            return self.array(">i1", 1)
        if tag == TAG_STRING:
            return self.string()
        if tag == TAG_LIST:
            t = self.unpack(">b")
            n = self.unpack(">i")
            return [self.payload(t) for _ in range(n)]
        if tag == TAG_COMPOUND:
            d = {}
            while True:
                t = self.unpack(">b")
                if t == TAG_END:
                    return d
                k = self.string()
                d[k] = self.payload(t)
        if tag == TAG_INT_ARRAY:
            return self.array(">i4", 4)
        if tag == TAG_LONG_ARRAY:
            return self.array(">i8", 8)
        raise ValueError(f"Bad NBT tag type {tag} at {self.pos}")


def decode(data: bytes) -> dict:
    """Decode NBT bytes (uncompressed) and return the root compound."""
    r = _Reader(data)
    tag = r.unpack(">b")
    if tag != TAG_COMPOUND:
        raise ValueError("NBT root is not a compound")
    r.string()
    return r.payload(TAG_COMPOUND)
