"""Isolated per-object QA renderer: KO model (with real texture) vs. what the converter
actually voxelized for it, side by side, plus numeric shape/colour metrics.

Usage: python -m ko2mc.qa_render gtd/moradon.gtd opd/moradon.opd --ko-models object --ko-textures dtex -o qa_out

For every distinct object "type" (grouped by name + mesh set) it:
  1. Re-runs the real object-building pipeline (ko_objects.build_objects) with a hook that
     records exactly which voxels each shape instance contributed, then looks up the final
     block at each of those voxels (same winner-takes-all result a real conversion would save).
  2. Extracts the KO model's real geometry + texture for one representative instance.
  3. Computes: bounding-box extent mismatch (shape) and a Lab colour-distance (texture/colour)
     between the KO surface's real average colour and the voxels' average colour.
  4. Renders both, face-culled and centred, into one three.js page (many objects share a single
     WebGL context via viewport/scissor, screenshotted once with headless Chromium) and crops
     each into its own comparison image.

Output: qa_out/report.csv (every object type, sorted worst-first) and qa_out/sheets/*.png.
"""

import argparse
import base64
import collections
import io
import json
import os
import re
import subprocess
import sys

import numpy as np

from .converter import CoordMap, TerrainModel, _key, _unkey, classify_object
from .ko_models import BLOCK_COLORS, ModelLibrary, _to_lab, sample_part
from . import ko_models as km
from .ko_objects import PLANT_KINDS, build_objects
from .ko_textures import TexturePack
from .mc_world import MinecraftWorld
from .opd_parser import parse_opd
from .gtd_parser import parse_gtd

FACES = [  # (normal, 4 corner offsets ccw) for a unit cube at origin
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)]),
    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, -1, 0), [(0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1)]),
    ((0, 0, 1), [(1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
]


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _png_b64(img: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img[..., :4] if img.shape[-1] >= 4 else
                    np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TextureRegistry:
    """Dedupes small PNGs into one list, returns an index per image."""

    def __init__(self, max_side=96):
        self.max_side = max_side
        self._index = {}
        self.images = []

    def add(self, rgba: np.ndarray) -> int:
        key = rgba.tobytes()
        if key in self._index:
            return self._index[key]
        from PIL import Image
        im = Image.fromarray(rgba[..., :4])
        if im.width > self.max_side or im.height > self.max_side:
            im = im.resize((self.max_side, self.max_side), Image.BOX)
        i = len(self.images)
        self.images.append(_png_b64(np.asarray(im)))
        self._index[key] = i
        return i


def voxel_mesh(cells: dict, registry: TextureRegistry):
    """cells: {(x,y,z): (image_rgba, half)} (half: None/'bottom'/'top') -> list of
    {"tex": idx, "v": base64 float32 positions, "uv": base64 float32 uvs, "n": vertex count}
    triangle batches, hidden faces culled."""
    occ = set(cells)
    batches = collections.defaultdict(lambda: ([], []))
    for (x, y, z), (img, half) in cells.items():
        y0, y1 = (0.0, 1.0) if not half else ((0.5, 1.0) if half == "top" else (0.0, 0.5))
        tex = registry.add(img)
        verts, uvs = batches[tex]
        for (nx, ny, nz), corners in FACES:
            if (x + nx, y + ny, z + nz) in occ:
                continue
            quad = []
            for cx, cy, cz in corners:
                yy = y0 if cy == 0 else y1
                quad.append((x + cx, y + yy, z + cz))
            for a, c, d in ((0, 1, 2), (0, 2, 3)):
                verts.extend(quad[a]); verts.extend(quad[c]); verts.extend(quad[d])
                uvs.extend([0, 0, 1, 0, 1, 1])
    out = []
    for t, (verts, uvs) in batches.items():
        v = np.asarray(verts, np.float32).reshape(-1, 3)
        out.append({"tex": t, "v": _b64(v), "uv": _b64(np.asarray(uvs, np.float32)), "n": len(v)})
    return out


def ko_mesh(parts, registry: TextureRegistry):
    """parts: [(mc_tris, uvs, tex_rgba, alpha)] -> list of {"tex", "v", "uv", "n"} (one per texture)."""
    batches = collections.defaultdict(lambda: ([], []))
    for mc, uvs, tex, alpha in parts:
        idx = registry.add(tex)
        verts, uvlist = batches[idx]
        verts.append(mc.reshape(-1, 3))
        uv = uvs.copy()
        uv[..., 1] = 1 - uv[..., 1]
        uvlist.append(uv.reshape(-1, 2))
    out = []
    for t, (verts, uvlist) in batches.items():
        v = np.concatenate(verts) if verts else np.zeros((0, 3), np.float32)
        uv = np.concatenate(uvlist) if uvlist else np.zeros((0, 2), np.float32)
        out.append({"tex": t, "v": _b64(v), "uv": _b64(uv), "n": len(v)})
    return out


def resolve_block(state, pack, solid_lookup, foliage_lookup):
    if state in solid_lookup:
        return solid_lookup[state], "solid"
    if state in foliage_lookup:
        return foliage_lookup[state], "leaf"
    base = state.split("[")[0].replace("minecraft:", "")
    props = state[state.index("[") + 1:-1] if "[" in state else ""
    if base.endswith("_stairs") and base[:-7] in pack.stairs:
        return pack.stairs[base[:-7]], "stairs"
    if base.endswith("_slab") and base[:-5] in pack.slabs:
        return pack.slabs[base[:-5]], "slab_top" if "type=top" in props else "slab_bottom"
    c = BLOCK_COLORS.get(base)
    if c:
        return np.tile(np.array(list(c) + [255], np.uint8), (4, 4, 1)), "flat"
    return np.tile(np.array([150, 150, 150, 255], np.uint8), (4, 4, 1)), "unknown"


def run(gtd_path, opd_path, ko_models, ko_textures, out_dir, scale=4, top_n=60,
        pack_resolution=32, pack_brightness=1.3):
    os.makedirs(out_dir, exist_ok=True)
    print("Parsing map + rebuilding objects (same pipeline as a real conversion)...")
    gtd = parse_gtd(gtd_path)
    opd = parse_opd(opd_path)
    cm = CoordMap.for_map(gtd, scale)
    world = MinecraftWorld(os.path.join(out_dir, "_dummy"), "QA")
    terrain = TerrainModel(gtd, cm, world, texture_pack=None, library=None)
    lib = ModelLibrary(ko_models)
    pack = TexturePack("QA", pack_resolution, pack_brightness)

    groups = collections.defaultdict(list)
    for i, shape in enumerate(opd.shapes):
        name = shape.name.lower()
        if not shape.parts or re.search(r"fx|smoke|fog|smog|collisioncube|alpha", name):
            continue
        if classify_object(shape.name) in PLANT_KINDS:
            continue
        sig = (name, tuple(sorted(p.name.lower() for p in shape.parts)))
        groups[sig].append(i)
    rep_of = {idxs[0]: sig for sig, idxs in groups.items()}

    shape_keys = {}

    def on_shape(i, keys):
        if i in rep_of:
            shape_keys[i] = keys

    build_objects(opd, terrain, world, cm, lib, pack, simple_plants=False, seed=0, on_shape=on_shape)

    world._flush_single()
    xs = np.concatenate([p[0] for p in world._pending])
    ys = np.concatenate([p[1] for p in world._pending])
    zs = np.concatenate([p[2] for p in world._pending])
    ids = np.concatenate([p[3] for p in world._pending])
    keyall = _key(xs, ys, zs)
    order = np.argsort(keyall, kind="stable")[::-1]
    uniq_keys, first_idx = np.unique(keyall[order], return_index=True)
    final_id_of_key = dict(zip(uniq_keys.tolist(), ids[order][first_idx].tolist()))
    solid_lookup = {pack._solid_slots[i][0]: img for i, img in enumerate(pack.solid)}
    foliage_lookup = {pack._foliage_slots[i][0]: img for i, img in enumerate(pack.foliage)}
    palette = world.palette

    def mean_lab(img):
        return _to_lab(img[..., :3].reshape(-1, 3).mean(0))

    print("Extracting geometry + metrics per object type...")
    results = []
    for i, sig in rep_of.items():
        shape = opd.shapes[i]
        parts_data, all_verts = [], []
        for tris, uvs, tex, part in lib.shape_parts(shape):
            if part.dest_blend == 2:
                continue
            mc = np.stack([cm.x(tris[..., 0]), cm.y(tris[..., 1]), cm.z(tris[..., 2])], -1)
            if tex is None:
                c = np.array([int(v * 255) for v in part.diffuse[:3]] + [255], np.uint8).clip(40, 230)
                tex = np.tile(c, (4, 4, 1))
            alpha = bool(part.render_flags & km.RF_ALPHABLENDING) or (tex[..., 3] < 128).mean() > 0.05
            parts_data.append((mc, uvs, tex, alpha))
            all_verts.append(mc.reshape(-1, 3))
        if not all_verts:
            continue
        allv = np.concatenate(all_verts)
        ko_min, ko_max = allv.min(0), allv.max(0)
        ko_ext = ko_max - ko_min

        lab_samples, n_samples = [], 0
        for mc, uvs, tex, alpha in parts_data:
            pts, tri, tx, ty = sample_part(mc, uvs, tex, alpha_test=alpha)
            if len(pts) == 0:
                continue
            rgb = tex[ty, tx, :3].astype(np.float32)
            lab_samples.append(_to_lab(rgb).sum(0))
            n_samples += len(pts)
        ko_lab = (sum(lab_samples) / n_samples) if n_samples else np.array([50., 0., 0.])

        keys = shape_keys.get(i, np.empty(0, np.int64))
        fids = np.array([final_id_of_key.get(int(k), 0) for k in keys])
        nz = fids != 0
        if nz.sum() == 0:
            continue
        vx, vy, vz = _unkey(keys[nz])
        vox_ext = np.array([vx.max() - vx.min() + 1, vy.max() - vy.min() + 1, vz.max() - vz.min() + 1], float)

        uniq_ids, counts = np.unique(fids[nz], return_counts=True)
        cells = {}
        vox_lab_num, vox_lab_den = np.zeros(3), 0
        for u, cnt in zip(uniq_ids, counts):
            img, kind = resolve_block(palette[u], pack, solid_lookup, foliage_lookup)
            half = "top" if kind == "slab_top" else ("bottom" if kind == "slab_bottom" else None)
            vox_lab_num = vox_lab_num + mean_lab(img) * cnt
            vox_lab_den += cnt
            m = fids[nz] == u
            for x, y, z in zip(vx[m].tolist(), vy[m].tolist(), vz[m].tolist()):
                cells[(x, y, z)] = (img, half)
        vox_lab = vox_lab_num / vox_lab_den
        dE = float(np.linalg.norm(ko_lab - vox_lab))
        ext_err = float(np.mean(np.abs(vox_ext - ko_ext) / np.maximum(ko_ext, 0.5)))

        results.append({
            "sig": sig, "shape_i": i, "count": len(groups[sig]), "voxels": int(nz.sum()),
            "ko_ext": ko_ext.tolist(), "vox_ext": vox_ext.tolist(), "ext_err": ext_err, "dE": dE,
            "parts_data": parts_data, "cells": cells, "anchor": ko_min.tolist(),
        })

    print(f"{len(results)} object types with placed voxels")
    return results, pack


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gtd"); ap.add_argument("opd")
    ap.add_argument("--ko-models", required=True); ap.add_argument("--ko-textures", default=None)
    ap.add_argument("-o", "--output", default="qa_out")
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    run(args.gtd, args.opd, args.ko_models, args.ko_textures, args.output, top_n=args.top)
