"""Previews: see the KO map and the converted Minecraft world side by side.

Commands (run from the project folder):

  python -m ko2mc.preview ko gtd/moradon.gtd [opd/moradon.opd]
      Reference render of the original Knight Online map.

  python -m ko2mc.preview mc output/KO_Moradon
      Render the converted world, read back from the region files, the same
      way Minecraft would load it.

  python -m ko2mc.preview compare output/KO_Moradon
      Both of the above, lined up in the same coordinates, plus a
      side-by-side image and a 3D viewer with a split-screen mode.

Each command writes to a preview folder (default: <world>/preview or ./preview):
  *_map.png      top-down map (north is up, 1 pixel = 1 Minecraft block)
  *_3d.html      interactive 3D viewer, open it in a web browser
  compare_map.png  KO | Minecraft side by side (compare only)

Use --mc-jar or --download-textures to draw real Minecraft textures.
"""

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import zlib

import numpy as np

from . import materials
from .converter import CoordMap, _bilinear, classify_object, collision_triangles
from .gtd_parser import parse_gtd
from .mc_textures import CROSS, CUBE, LIQUID, build_appearance, find_client_jar
from .opd_parser import EVENT_TYPE_NAMES, parse_opd

VIEWER_TEMPLATE = os.path.join(os.path.dirname(__file__), "viewer.html")
MAX_VIEW_COLUMNS = 1536 * 1536   # larger worlds are cropped in the 3D viewer (use --area)

OBJECT_CATEGORIES = ["tree", "tree_spruce", "tree_palm", "bush", "grass", "flower", "sunflower",
                     "reed", "mushroom", "pumpkin", "rock", "lamp", "fire", "flag", "fence", "other"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _save_png(img: np.ndarray, path: str):
    with open(path, "wb") as f:
        f.write(_png_bytes(img))
    print(f"  wrote {path}")


def _hillshade(h: np.ndarray, spacing: float) -> np.ndarray:
    """Soft lighting from the north-west, h indexed [z(row, north up), x]."""
    gz, gx = np.gradient(h, spacing)   # rows go south as the index grows (Minecraft z)
    nx, ny, nz = -gx, np.ones_like(h), -gz
    light = np.array([-0.45, 0.8, -0.4])
    light /= np.linalg.norm(light)
    d = (nx * light[0] + ny * light[1] + nz * light[2]) / np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.clip(0.35 + 0.75 * d, 0.25, 1.15)


# ---------------------------------------------------------------------------
# Knight Online reference
# ---------------------------------------------------------------------------

class KOScene:
    """The original KO map in Minecraft block coordinates."""

    def __init__(self, gtd_path: str, opd_path: str | None, cm: CoordMap | None = None,
                 models_dir: str | None = None, textures_dir: str | None = None,
                 brightness: float = 1.3):
        print(f"Loading KO map {gtd_path}")
        self.models_dir = models_dir if models_dir and os.path.isdir(models_dir) else None
        self.textures_dir = textures_dir if textures_dir and os.path.isdir(textures_dir) else None
        self.brightness = brightness
        self.gtd = parse_gtd(gtd_path)
        self.opd = None
        if opd_path and os.path.exists(opd_path):
            try:
                self.opd = parse_opd(opd_path)
            except Exception as e:
                print(f"  Warning: could not read {opd_path}: {e}")
        self.cm = cm or CoordMap.for_map(self.gtd, 4)
        self.name = os.path.splitext(os.path.basename(gtd_path))[0]

    def objects(self):
        """(positions[n,3] in MC coords, sizes[n], categories[n], events list)."""
        if not self.opd:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros(0, np.uint8), []
        cm = self.cm
        pos, size, cat, events = [], [], [], []
        for s in self.opd.shapes:
            p = (float(cm.x(s.position.x)), float(cm.y(s.position.y)), float(cm.z(s.position.z)))
            if s.is_event_object:
                events.append({"name": s.name, "type": EVENT_TYPE_NAMES.get(s.event_type, "Event"),
                               "pos": p, "id": s.event_id})
                continue
            kind = classify_object(s.name)
            if kind is None and re.search(r"fx|smoke|fog|smog|alpha", s.name.lower()):
                continue
            pos.append(p)
            size.append(max(abs(s.scale.x), abs(s.scale.y), abs(s.scale.z), 0.2))
            cat.append(OBJECT_CATEGORIES.index(kind) if kind in OBJECT_CATEGORIES
                       else len(OBJECT_CATEGORIES) - 1)
        return (np.array(pos, np.float32).reshape(-1, 3), np.array(size, np.float32),
                np.array(cat, np.uint8), events)

    def top_down(self) -> np.ndarray:
        """Top-down map image, 1 pixel per Minecraft block, north up."""
        cm, gtd = self.cm, self.gtd
        size = cm.size_blocks
        s = cm.scale
        cols = np.arange(size) + 0.5
        FX, FZ = np.meshgrid(cols / s, (size - cols) / s)
        h = _bilinear(gtd.heights, FX, FZ)
        names, grid = materials.material_grid(gtd)
        n = gtd.heightmap_size
        mat = grid[np.clip(FX.astype(int), 0, n - 1), np.clip(FZ.astype(int), 0, n - 1)]
        colors = np.array([materials.MATERIALS[m].ko_color for m in names], np.float32)[mat]
        img = colors * _hillshade(h, 1.0 / cm.blocks_per_meter)[..., None]

        # Water: blend blue by depth
        level = np.full((size, size), -np.inf)
        for mesh in gtd.water:
            _raster_max(level, mesh.triangles(), cm, lambda t: t[:, :, 1])
        depth = level - h
        wet = depth > 0
        a = np.clip(0.55 + depth / 12.0, 0.55, 0.9)[..., None]
        img = np.where(wet[..., None], img * (1 - a) + np.array(materials.WATER_KO_COLOR) * a, img)

        # Buildings / walls from the collision mesh
        if self.opd:
            tris = collision_triangles(self.opd)
            if len(tris):
                cover = np.full((size, size), -np.inf)
                _raster_max(cover, tris, cm, lambda t: t[:, :, 1])
                above = cover > h + 0.5
                shade = np.clip(0.55 + (cover - h) / 30.0, 0.55, 1.0)
                img = np.where(above[..., None], np.array([196, 188, 176]) * shade[..., None], img)

        img = np.clip(img, 0, 255).astype(np.uint8)
        # Objects
        pos, sizes, cat, events = self.objects()
        col = {"tree": (34, 90, 34), "tree_spruce": (24, 70, 40), "tree_palm": (60, 110, 40),
               "bush": (60, 120, 50), "rock": (110, 110, 110), "lamp": (255, 220, 90),
               "fire": (255, 120, 30), "flag": (220, 40, 40), "flower": (230, 120, 200),
               "other": (150, 90, 170)}
        for (x, _, z), sz, c in zip(pos, sizes, cat):
            name = OBJECT_CATEGORIES[c]
            if name not in col:
                continue
            r = max(0, int(round((3.5 if name.startswith("tree") else 1.0) * sz * cm.blocks_per_meter)))
            r = min(r, 12)
            xi, zi = int(x), int(z)
            if 0 <= xi < img.shape[1] and 0 <= zi < img.shape[0]:
                img[max(zi - r, 0):zi + r + 1, max(xi - r, 0):xi + r + 1] = col[name]
        for ev in events:
            x, _, z = ev["pos"]
            xi, zi = int(x), int(z)
            r = max(2, int(3 * cm.blocks_per_meter))
            if not (0 <= xi < img.shape[1] and 0 <= zi < img.shape[0]):
                continue
            img[max(zi - r, 0):zi + r + 1, max(xi - r, 0):xi + r + 1] = (0, 230, 255)
        return img

    def viewer_data(self) -> dict:
        cm, gtd = self.cm, self.gtd
        n = gtd.heightmap_size
        names, grid = materials.material_grid(gtd)
        colors = np.array([materials.MATERIALS[m].ko_color for m in names], np.uint8)[grid]  # [x,z,3]
        # vertex (i, j) of the heightmap -> MC (i*s, size - j*s)
        ys = cm.y(gtd.heights).astype(np.float32)          # [x, z]
        water = [cm_tris(m.triangles(), cm) for m in gtd.water]
        water = np.concatenate(water) if water else np.zeros((0, 3, 3), np.float32)
        coll = cm_tris(collision_triangles(self.opd), cm) if self.opd else np.zeros((0, 3, 3), np.float32)
        pos, sizes, cat, events = self.objects()
        return {
            "name": self.name,
            "n": n,
            "step": cm.scale,
            "size": cm.size_blocks,
            "bpm": cm.blocks_per_meter,
            "heights": _b64(ys.T),                       # row-major [z_index][x_index]
            "colors": _b64(np.transpose(colors, (1, 0, 2))),
            "water": _b64(water.astype(np.float32)),
            "collision": _b64(coll.astype(np.float32)),
            "objPos": _b64(pos),
            "objSize": _b64(sizes * cm.blocks_per_meter),
            "objCat": _b64(cat),
            "categories": OBJECT_CATEGORIES,
            "events": events,
            "coords": cm.to_json(),
            "terrainTex": self._terrain_texture(),
            "models": self._models_data(),
        }

    def _terrain_texture(self, px: int = 16) -> str | None:
        """The KO ground (base + overlay textures, rotated like KO) baked into one JPEG."""
        if not self.textures_dir:
            return None
        from PIL import Image
        from .ko_ground import GroundBuilder
        from .ko_textures import TextureLibrary
        gb = GroundBuilder(self.gtd, TextureLibrary(self.textures_dir), self.brightness)
        if not gb.images:
            return None
        n = gb.t1.shape[0]
        names, grid = materials.material_grid(self.gtd)
        fallback = np.array([materials.MATERIALS[m].ko_color for m in names], np.uint8)
        inv, examples = gb.combos()
        small = []
        for tx, tz in examples:
            im = gb.tile_image(tx, tz)
            small.append(None if im is None else
                         np.asarray(Image.fromarray(im.astype(np.uint8)).resize((px, px), Image.BOX)))
        img = np.zeros((n * px, n * px, 3), np.uint8)
        for tx in range(n):
            for tz in range(n):
                tile = small[inv[tx, tz]]
                row = (n - 1 - tz) * px        # north up
                img[row:row + px, tx * px:(tx + 1) * px] = fallback[grid[tx, tz]] if tile is None else tile
        if img.shape[0] > 8192:
            img = np.asarray(Image.fromarray(img).resize((8192, 8192), Image.BOX))
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _models_data(self, max_tex: int = 128) -> dict | None:
        """Real KO object models: unique meshes + textures + one matrix per placed part."""
        if not self.models_dir or not self.opd:
            return None
        from PIL import Image
        from . import ko_models as km
        lib = km.ModelLibrary(self.models_dir)
        cm = self.cm
        # KO meters -> Minecraft coords as a 4x4 (column vectors)
        A = np.diag([cm.blocks_per_meter, cm.vertical_scale, -cm.blocks_per_meter, 1.0])
        A[1, 3] = cm.y_offset
        A[2, 3] = cm.map_size_m * cm.blocks_per_meter
        groups, tex_index, textures = {}, {}, []
        for shape in self.opd.shapes:
            if re.search(r"fx|smoke|fog|smog|collisioncube", shape.name.lower()):
                continue
            x, y, z, w = shape.rotation.x, shape.rotation.y, shape.rotation.z, shape.rotation.w
            R = np.eye(4)
            R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
            S = np.diag([shape.scale.x, shape.scale.y, shape.scale.z, 1.0])
            T = np.eye(4)
            T[:3, 3] = [shape.position.x, shape.position.y, shape.position.z]
            base = A @ T @ R @ S
            for part in shape.parts:
                mesh = lib.mesh(part.name)
                if mesh is None or len(mesh[1]) == 0 or part.dest_blend == 2:
                    continue
                tname = part.textures[0].lower() if part.textures else ""
                if tname not in tex_index:
                    rgba = lib.texture(tname) if tname else None
                    if rgba is None:
                        tex_index[tname] = -1
                    else:
                        im = Image.fromarray(rgba, "RGBA")
                        if im.width > max_tex or im.height > max_tex:
                            im.thumbnail((max_tex, max_tex), Image.BOX)
                        tex_index[tname] = len(textures)
                        textures.append("data:image/png;base64," + base64.b64encode(_png_bytes(np.asarray(im))).decode())
                P = np.eye(4)
                if part.pivot:
                    P[:3, 3] = [part.pivot.x, part.pivot.y, part.pivot.z]
                alpha = bool(part.render_flags & km.RF_ALPHABLENDING)
                key = (part.name.lower(), tname, alpha)
                groups.setdefault(key, []).append((base @ P).astype(np.float32).T.ravel())  # column-major
        meshes = []
        for (mname, tname, alpha), mats in groups.items():
            verts, idx = lib.mesh(mname)
            meshes.append({
                "v": _b64(np.ascontiguousarray(verts[:, :3])),
                "uv": _b64(np.ascontiguousarray(verts[:, 6:8])),
                "i": _b64(idx.astype(np.uint16 if len(verts) < 65536 else np.uint32)),
                "big": len(verts) >= 65536,
                "t": tex_index.get(tname, -1),
                "a": alpha,
                "m": _b64(np.concatenate(mats)),
            })
        print(f"  KO models for the viewer: {len(meshes)} meshes, {len(textures)} textures")
        return {"meshes": meshes, "textures": textures}


def cm_tris(tris: np.ndarray, cm: CoordMap) -> np.ndarray:
    out = np.empty_like(tris, dtype=np.float32)
    out[..., 0] = cm.x(tris[..., 0])
    out[..., 1] = cm.y(tris[..., 1])
    out[..., 2] = cm.z(tris[..., 2])
    return out


def _raster_max(grid: np.ndarray, tris_ko: np.ndarray, cm: CoordMap, value_fn):
    """Rasterize triangles onto grid[z, x], keeping the max interpolated KO value."""
    size = grid.shape[0]
    xs = cm.x(tris_ko[:, :, 0])
    zs = cm.z(tris_ko[:, :, 2])
    vs = value_fn(tris_ko)
    for (ax, bx, cx), (az, bz, cz), (va, vb, vc) in zip(xs, zs, vs):
        i0 = max(int(math.floor(min(ax, bx, cx))), 0)
        i1 = min(int(math.ceil(max(ax, bx, cx))), size - 1)
        j0 = max(int(math.floor(min(az, bz, cz))), 0)
        j1 = min(int(math.ceil(max(az, bz, cz))), size - 1)
        if i1 < i0 or j1 < j0:
            continue
        det = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
        if abs(det) < 1e-9:
            continue
        px, pz = np.meshgrid(np.arange(i0, i1 + 1) + 0.5, np.arange(j0, j1 + 1) + 0.5)
        w1 = ((bz - cz) * (px - cx) + (cx - bx) * (pz - cz)) / det
        w2 = ((cz - az) * (px - cx) + (ax - cx) * (pz - cz)) / det
        w3 = 1 - w1 - w2
        inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6)
        val = w1 * va + w2 * vb + w3 * vc
        region = grid[j0:j1 + 1, i0:i1 + 1]
        np.maximum(region, np.where(inside, val, -np.inf), out=region)


# ---------------------------------------------------------------------------
# Minecraft world
# ---------------------------------------------------------------------------

class MCScene:
    """A Minecraft world read back from its region files."""

    def __init__(self, world_dir: str, mc_jar: str | None = None, download: bool = False,
                 area: tuple[int, int, int, int] | None = None):
        from .mca_reader import WorldReader
        self.world_dir = world_dir
        self.reader = WorldReader(world_dir)
        info_path = os.path.join(world_dir, "ko2mc.json")
        self.info = json.load(open(info_path, encoding="utf-8")) if os.path.exists(info_path) else {}
        level = self.reader.level_info()
        self.spawn = (level.get("SpawnX", 0), level.get("SpawnY", 64), level.get("SpawnZ", 0))
        self.name = level.get("LevelName") or os.path.basename(os.path.normpath(world_dir))
        self.jar = find_client_jar(mc_jar, download)
        print(f"Reading Minecraft world {world_dir}")
        print(f"  Textures: {self.jar or 'built-in colours (use --mc-jar or --download-textures for real textures)'}")
        self._scan(area)

    def _scan(self, area):
        chunks = self.reader.chunk_positions()
        if not chunks:
            raise ValueError("World has no chunks")
        cxs = [c[0] for c in chunks]
        czs = [c[1] for c in chunks]
        x0, x1 = min(cxs) * 16, max(cxs) * 16 + 15
        z0, z1 = min(czs) * 16, max(czs) * 16 + 15
        if area:
            x0, z0, x1, z1 = max(x0, area[0]), max(z0, area[1]), min(x1, area[2]), min(z1, area[3])
        self.x0, self.z0 = x0, z0
        W, D = x1 - x0 + 1, z1 - z0 + 1
        self.W, self.D = W, D
        chunk_set = {c for c in chunks if c[0] * 16 + 15 >= x0 and c[0] * 16 <= x1
                     and c[1] * 16 + 15 >= z0 and c[1] * 16 <= z1}
        print(f"  {len(chunk_set)} chunks, area x {x0}..{x1}, z {z0}..{z1}")

        self.top_id = np.zeros((D, W), np.uint16)       # top-most block
        self.top_y = np.full((D, W), -64, np.int16)
        self.floor_y = np.full((D, W), -64, np.int16)   # first non-liquid below the top
        self.floor_id = np.zeros((D, W), np.uint16)
        self.run_bottom = np.full((D, W), -32768, np.int16)
        self.run_len = np.zeros((D, W), np.uint16)
        runs = {}

        cache = {}
        order = sorted(chunk_set, key=lambda c: (c[1], c[0]))
        lut_opaque = np.zeros(1, bool)

        def load(c):
            if c not in cache:
                cache[c] = self.reader.read_chunk(*c) if c in chunk_set else None
            return cache[c]

        count = 0
        for cx, cz in order:
            blocks = load((cx, cz))
            for old in [k for k in cache if k[1] < cz - 1]:
                del cache[old]
            if blocks is None:
                continue
            if len(lut_opaque) < len(self.reader.palette):
                lut_opaque = self._opaque_lut()
            P = np.zeros((386, 18, 18), np.uint16)
            P[1:-1, 1:-1, 1:-1] = blocks
            P[0] = 0xFFFF  # below the world counts as solid
            for (dx, dz), dst, src in (((-1, 0), (slice(1, -1), slice(1, -1), 0), (slice(None), slice(None), 15)),
                                       ((1, 0), (slice(1, -1), slice(1, -1), 17), (slice(None), slice(None), 0)),
                                       ((0, -1), (slice(1, -1), 0, slice(1, -1)), (slice(None), 15, slice(None))),
                                       ((0, 1), (slice(1, -1), 17, slice(1, -1)), (slice(None), 0, slice(None)))):
                nb = load((cx + dx, cz + dz))
                if nb is not None:
                    P[dst] = nb[src]
            if len(lut_opaque) < len(self.reader.palette):
                lut_opaque = self._opaque_lut()
            op = np.ones(P.shape, bool)
            mask = P != 0xFFFF
            op[mask] = lut_opaque[P[mask]]
            nonair = blocks != 0
            exposed = (~op[2:, 1:-1, 1:-1] | ~op[:-2, 1:-1, 1:-1] | ~op[1:-1, 2:, 1:-1]
                       | ~op[1:-1, :-2, 1:-1] | ~op[1:-1, 1:-1, 2:] | ~op[1:-1, 1:-1, :-2])
            visible = nonair & exposed

            has = nonair.any(axis=0)
            top = 383 - np.argmax(nonair[::-1], axis=0)
            vis_any = visible.any(axis=0)
            bottom = np.argmax(visible, axis=0)
            bottom = np.where(vis_any, bottom, top)

            # window of this chunk inside the scanned area
            gx0, gz0 = cx * 16, cz * 16
            xa, xb = max(gx0, x0), min(gx0 + 16, x1 + 1)
            za, zb = max(gz0, z0), min(gz0 + 16, z1 + 1)
            lx, lz = slice(xa - gx0, xb - gx0), slice(za - gz0, zb - gz0)
            gx, gz = slice(xa - x0, xb - x0), slice(za - z0, zb - z0)

            topc = top[lz, lx]
            hasc = has[lz, lx]
            zz, xx = np.meshgrid(np.arange(lz.start, lz.stop), np.arange(lx.start, lx.stop), indexing="ij")
            tid = blocks[topc, zz, xx]
            self.top_id[gz, gx] = np.where(hasc, tid, 0)
            self.top_y[gz, gx] = np.where(hasc, topc - 64, -64)
            # floor under liquids (for water depth on the map)
            liquid = self._liquid_lut()
            liq = liquid[blocks]
            below = (~liq) & nonair
            ys = np.arange(384)[:, None, None]
            below &= ys <= top[None]
            fl = np.where(below.any(axis=0), 383 - np.argmax(below[::-1], axis=0), top)
            flc = fl[lz, lx]
            self.floor_y[gz, gx] = flc - 64
            self.floor_id[gz, gx] = blocks[flc, zz, xx]

            bot = bottom[lz, lx]
            length = np.where(hasc, topc - bot + 1, 0)
            self.run_bottom[gz, gx] = np.where(hasc, bot - 64, -32768)
            self.run_len[gz, gx] = length
            ys = np.arange(384)[:, None, None]
            sub = blocks[:, lz, lx]
            in_run = (ys >= bot[None]) & (ys <= topc[None]) & hasc[None]
            # values ordered by (z, x, y) -> per column contiguous
            runs[(cx, cz)] = (za, zb, xa, xb, sub.transpose(1, 2, 0)[in_run.transpose(1, 2, 0)])
            count += 1
            if count % 256 == 0:
                print(f"    scanned {count}/{len(order)} chunks")

        # assemble runs in grid order (row-major z, x)
        lens = self.run_len.astype(np.int64).ravel()
        offsets = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=offsets[1:])
        values = np.zeros(int(offsets[-1]), np.uint16)
        for (za, zb, xa, xb, vals) in runs.values():
            pos = 0
            for z in range(za, zb):
                row = (z - z0) * W
                a = offsets[row + xa - x0]
                b = offsets[row + xb - x0]
                values[a:b] = vals[pos:pos + (b - a)]
                pos += b - a
        self.run_values = values
        self.palette = list(self.reader.palette)
        pack = self.info.get("resource_pack")
        if pack and not os.path.isabs(pack):
            pack = os.path.join(self.world_dir, pack)
        if not pack or not os.path.exists(pack):
            pack = os.path.join(self.world_dir, "resources.zip")
        self.look = build_appearance(self.palette, self.jar, [pack])

    def _opaque_lut(self):
        from .mc_textures import _kind_for, _split_state
        out = np.zeros(len(self.reader.palette), bool)
        for i, st in enumerate(self.reader.palette):
            if i == 0:
                continue
            name, _ = _split_state(st)
            out[i] = _kind_for(name, []) == CUBE and not name.endswith("air")
        return out

    def _liquid_lut(self):
        from .mc_textures import _kind_for, _split_state
        return np.array([_kind_for(_split_state(st)[0], []) == LIQUID for st in self.reader.palette], bool)

    def top_down(self) -> np.ndarray:
        """Minecraft map-item style render: block colour + height shading, north up."""
        colors = np.array([lk.color for lk in self.look.looks], np.float32)
        kinds = np.array([lk.kind for lk in self.look.looks])
        tid = self.top_id
        img = colors[tid].copy()
        h = self.top_y.astype(np.float32)
        north = np.vstack([h[:1], h[:-1]])
        shade = np.where(h > north, 1.0, np.where(h < north, 0.72, 0.86))
        wet = kinds[tid] == LIQUID
        depth = (self.top_y - self.floor_y).astype(np.float32)
        wshade = np.clip(1.0 - depth / 14.0, 0.62, 1.0)
        floor = colors[self.floor_id]
        water_col = colors[tid] * 0.8 + floor * 0.2
        img = np.where(wet[..., None], water_col * wshade[..., None], img * shade[..., None])
        img[self.top_y <= -64] = (0, 0, 0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def viewer_data(self) -> dict:
        W, D = self.W, self.D
        if W * D > MAX_VIEW_COLUMNS:
            raise ValueError("area too large")
        raw = b"".join([self.run_bottom.tobytes(), self.run_len.tobytes(), self.run_values.tobytes()])
        comp = zlib.compress(raw, 6)
        looks = [[lk.kind, lk.top, lk.side, lk.bottom] for lk in self.look.looks]
        atlas = self.look.atlas
        return {
            "name": self.name,
            "x0": self.x0, "z0": self.z0, "W": W, "D": D,
            "blob": base64.b64encode(comp).decode("ascii"),
            "nRuns": int(len(self.run_values)),
            "looks": looks,
            "palette": self.palette,
            "atlas": "data:image/png;base64," + base64.b64encode(_png_bytes(atlas)).decode("ascii"),
            "atlasW": int(atlas.shape[1]), "atlasH": int(atlas.shape[0]),
            "tilesPerRow": self.look.tiles_per_row,
            "textured": self.look.textured,
            "spawn": list(self.spawn),
        }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_viewer(path: str, title: str, ko: dict | None, mc: dict | None):
    with open(VIEWER_TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    data = json.dumps({"title": title, "ko": ko, "mc": mc}, separators=(",", ":"))
    html = html.replace("/*__TITLE__*/", title).replace("\"__DATA__\"", data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB) - open it in a web browser")


def _crop_area_for_view(scene: "MCScene") -> tuple | None:
    if scene.W * scene.D <= MAX_VIEW_COLUMNS:
        return None
    half = int(math.sqrt(MAX_VIEW_COLUMNS)) // 2
    sx, _, sz = scene.spawn
    return (sx - half, sz - half, sx + half - 1, sz + half - 1)


def preview_ko(gtd_path, opd_path=None, out_dir="preview", scale=4, html=True,
               models_dir=None, textures_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    gtd = parse_gtd(gtd_path, verbose=False)
    ko = KOScene(gtd_path, opd_path, CoordMap.for_map(gtd, scale), models_dir, textures_dir)
    _save_png(ko.top_down(), os.path.join(out_dir, f"ko_{ko.name}_map.png"))
    if html:
        write_viewer(os.path.join(out_dir, f"ko_{ko.name}_3d.html"), f"KO {ko.name}", ko.viewer_data(), None)
    return ko


def preview_mc(world_dir, out_dir=None, mc_jar=None, download=False, area=None, html=True):
    out_dir = out_dir or os.path.join(world_dir, "preview")
    os.makedirs(out_dir, exist_ok=True)
    mc = MCScene(world_dir, mc_jar, download)
    _save_png(mc.top_down(), os.path.join(out_dir, f"mc_{_safe(mc.name)}_map.png"))
    if html:
        view = mc if not (area or _crop_area_for_view(mc)) else \
            MCScene(world_dir, mc_jar, download, area or _crop_area_for_view(mc))
        if view is not mc and not area:
            print("  3D viewer shows the area around spawn (world is large); use --area to pick another")
        write_viewer(os.path.join(out_dir, f"mc_{_safe(mc.name)}_3d.html"), mc.name, None, view.viewer_data())
    return mc


def preview_compare(world_dir, out_dir=None, mc_jar=None, download=False, area=None,
                    gtd_path=None, opd_path=None):
    out_dir = out_dir or os.path.join(world_dir, "preview")
    os.makedirs(out_dir, exist_ok=True)
    info_path = os.path.join(world_dir, "ko2mc.json")
    info = json.load(open(info_path, encoding="utf-8")) if os.path.exists(info_path) else {}
    gtd_path = gtd_path or info.get("gtd")
    opd_path = opd_path or info.get("opd")
    if not gtd_path or not os.path.exists(gtd_path):
        raise SystemExit("Can't find the .gtd for this world; pass it with --gtd")
    c = info.get("coords")
    gtd = parse_gtd(gtd_path, verbose=False)
    cm = CoordMap(**c) if c else CoordMap.for_map(gtd, 4)
    ko = KOScene(gtd_path, opd_path, cm, info.get("ko_models"), info.get("ko_textures"),
                 info.get("pack_brightness", 1.3))
    mc = MCScene(world_dir, mc_jar, download)

    ko_img = ko.top_down()
    mc_img = mc.top_down()
    _save_png(ko_img, os.path.join(out_dir, f"ko_{ko.name}_map.png"))
    _save_png(mc_img, os.path.join(out_dir, f"mc_{_safe(mc.name)}_map.png"))
    # align the MC map with the KO map (KO map covers blocks 0..size-1)
    size = cm.size_blocks
    aligned = np.zeros((size, size, 3), np.uint8)
    xa, za = max(0, mc.x0), max(0, mc.z0)
    xb, zb = min(size, mc.x0 + mc.W), min(size, mc.z0 + mc.D)
    if xb > xa and zb > za:
        aligned[za:zb, xa:xb] = mc_img[za - mc.z0:zb - mc.z0, xa - mc.x0:xb - mc.x0]
    gap = np.full((size, max(8, size // 64), 3), 255, np.uint8)
    _save_png(_label(np.hstack([ko_img, gap, aligned]), ["Knight Online (reference)", "Minecraft (from world files)"],
                     [0, size + gap.shape[1]]), os.path.join(out_dir, "compare_map.png"))

    crop = area or _crop_area_for_view(mc)
    view = mc if not crop else MCScene(world_dir, mc_jar, download, crop)
    write_viewer(os.path.join(out_dir, "compare_3d.html"), f"{ko.name}: KO vs Minecraft",
                 ko.viewer_data(), view.viewer_data())
    return ko, mc


def _label(img: np.ndarray, texts, xs) -> np.ndarray:
    from PIL import Image, ImageDraw
    bar = 28
    out = np.full((img.shape[0] + bar, img.shape[1], 3), 255, np.uint8)
    out[bar:] = img
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    for t, x in zip(texts, xs):
        d.text((x + 6, 8), t, fill=(0, 0, 0))
    return np.array(im)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _area(text):
    try:
        v = [int(p) for p in text.split(",")]
        assert len(v) == 4
        return (min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3]))
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError("use x1,z1,x2,z2 (Minecraft block coordinates)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ko2mc.preview",
                                description="Render Knight Online maps and converted Minecraft worlds.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("ko", help="render the original KO map")
    k.add_argument("gtd")
    k.add_argument("opd", nargs="?")
    k.add_argument("-o", "--output", default="preview")
    k.add_argument("-s", "--scale", type=int, default=4, choices=[1, 2, 4])
    k.add_argument("--no-html", action="store_true")
    k.add_argument("--ko-models", default=None, help="KO Object folder (.n3pmesh/.dxt) for real models")
    k.add_argument("--ko-textures", default=None, help="KO DTex folder (.gtt) for real ground textures")

    for name, help_text in (("mc", "render a Minecraft world"), ("compare", "KO and Minecraft side by side")):
        m = sub.add_parser(name, help=help_text)
        m.add_argument("world", help="world folder (e.g. output/KO_Moradon)")
        m.add_argument("-o", "--output", default=None)
        m.add_argument("--mc-jar", default=None, help="Minecraft client .jar to take textures from")
        m.add_argument("--download-textures", action="store_true",
                       help="download the official Minecraft 1.20.4 client jar for textures")
        m.add_argument("--area", type=_area, default=None, help="x1,z1,x2,z2 region for the 3D viewer")
        if name == "mc":
            m.add_argument("--no-html", action="store_true")
        else:
            m.add_argument("--gtd", default=None)
            m.add_argument("--opd", default=None)

    a = p.parse_args(argv)
    if a.cmd == "ko":
        from .__main__ import find_opd
        preview_ko(a.gtd, a.opd or find_opd(a.gtd), a.output, a.scale, html=not a.no_html,
                   models_dir=a.ko_models, textures_dir=a.ko_textures)
    elif a.cmd == "mc":
        preview_mc(a.world, a.output, a.mc_jar, a.download_textures, a.area, html=not a.no_html)
    else:
        preview_compare(a.world, a.output, a.mc_jar, a.download_textures, a.area, a.gtd, a.opd)


if __name__ == "__main__":
    sys.exit(main())
