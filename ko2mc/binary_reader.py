"""Reusable little-endian binary reader utility for KO asset parsers."""

import struct
from pathlib import Path

import numpy as np


class BinaryParseError(Exception):
    """Raised when binary data is malformed or a read overruns the buffer."""


class BinaryReader:
    """Bounds-checked little-endian binary reader.

    Accepts a file path, bytes buffer, or file-like object.
    """

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                self._buf = f.read()
        elif isinstance(source, (bytes, bytearray)):
            self._buf = bytes(source)
        elif hasattr(source, "read"):
            self._buf = source.read()
        else:
            raise BinaryParseError(f"Unsupported source type: {type(source)}")
        self._pos = 0

    # ── Position helpers ────────────────────────────────────────────────────

    def tell(self) -> int:
        return self._pos

    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def skip(self, n: int) -> None:
        if n < 0 or self._pos + n > len(self._buf):
            raise BinaryParseError(
                f"Skip overrun: skip {n} at offset {self._pos}, "
                f"have {self.remaining()} bytes remaining"
            )
        self._pos += n

    def expect_eof(self) -> None:
        if self._pos != len(self._buf):
            raise BinaryParseError(
                f"Expected EOF at offset {self._pos}, "
                f"{self.remaining()} bytes remaining"
            )

    # ── Raw read ─────────────────────────────────────────────────────────────

    def read_bytes(self, n: int) -> bytes:
        if n < 0 or self._pos + n > len(self._buf):
            raise BinaryParseError(
                f"Read overrun: need {n} bytes at offset {self._pos}, "
                f"have {self.remaining()} bytes remaining"
            )
        data = self._buf[self._pos : self._pos + n]
        self._pos += n
        return data

    # ── Scalars ──────────────────────────────────────────────────────────────

    def read_i8(self) -> int:
        return struct.unpack_from("<b", self.read_bytes(1))[0]

    def read_u8(self) -> int:
        return struct.unpack_from("<B", self.read_bytes(1))[0]

    def read_i16(self) -> int:
        return struct.unpack_from("<h", self.read_bytes(2))[0]

    def read_u16(self) -> int:
        return struct.unpack_from("<H", self.read_bytes(2))[0]

    def read_i32(self) -> int:
        return struct.unpack_from("<i", self.read_bytes(4))[0]

    def read_u32(self) -> int:
        return struct.unpack_from("<I", self.read_bytes(4))[0]

    def read_f32(self) -> float:
        return struct.unpack_from("<f", self.read_bytes(4))[0]

    # ── Arrays (numpy) ───────────────────────────────────────────────────────

    def read_f32_array(self, count: int) -> np.ndarray:
        data = self.read_bytes(count * 4)
        return np.frombuffer(data, dtype="<f4").copy()

    def read_u16_array(self, count: int) -> np.ndarray:
        data = self.read_bytes(count * 2)
        return np.frombuffer(data, dtype="<u2").copy()

    def read_i32_array(self, count: int) -> np.ndarray:
        data = self.read_bytes(count * 4)
        return np.frombuffer(data, dtype="<i4").copy()

    # ── Compound helpers ─────────────────────────────────────────────────────

    def read_bool_padded(self, n_bytes: int) -> bool:
        """Read a bool occupying n_bytes (MSVC struct alignment padding)."""
        val = struct.unpack_from("<B", self.read_bytes(1))[0]
        self.skip(n_bytes - 1)
        return bool(val)

    def read_len_string(self) -> str:
        """Read an int32-length-prefixed ASCII string."""
        length = self.read_i32()
        if length == 0:
            return ""
        if length < 0 or length > 65536:
            raise BinaryParseError(
                f"Implausible string length {length} at offset {self._pos}"
            )
        data = self.read_bytes(length)
        return data.decode("ascii", errors="replace").rstrip("\x00")

    def read_vec3(self) -> tuple:
        """Read three float32 values as (x, y, z)."""
        return struct.unpack_from("<fff", self.read_bytes(12))

    def read_quat(self) -> tuple:
        """Read four float32 values as (x, y, z, w)."""
        return struct.unpack_from("<ffff", self.read_bytes(16))

    def read_material(self) -> bytes:
        """Read a 92-byte D3D __Material block (returned as raw bytes)."""
        return self.read_bytes(92)
