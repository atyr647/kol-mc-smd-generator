"""KO objects -> Minecraft blocks that show the real KO textures.

For every block a model covers we remember which KO texture it came from, where
in that texture (the texel at the block's centre) and how many texels one block
spans. That's enough to cut the matching piece out of the KO texture.

A resource pack can only add a limited number of block looks (see
custom_blocks.py), so similar-looking blocks are grouped (k-means on a small
3 x 3 colour grid per block) and every group gets the real KO piece of its most
typical block. Leaves get see-through leaf blocks with KO leaf textures, and
grass/flower objects become crossed plant sprites instead of blocks.
"""

import math
import re

import numpy as np

from . import ko_models as km
from .mc_world import MIN_Y

PLANT_KINDS = {"grass", "flower", "sunflower", "reed", "mushroom"}
OBJECT_BRIGHTNESS = 1.15      # KO lights objects a little brighter than stored
PIECE = 32                    # pixels per block texture


def _texel_scale(tris_mc, uvs, tex_shape):
    """Texels per block along the surface (median over the part's triangles)."""
    h, w = tex_shape[:2]
    duv = np.concatenate([uvs[:, 1] - uvs[:, 0], uvs[:, 2] - uvs[:, 0]])
    dpos = np.concatenate([tris_mc[:, 1] - tris_mc[:, 0], tris_mc[:, 2] - tris_mc[:, 0]])
    lp = np.linalg.norm(dpos, axis=1)
    lt = np.linalg.norm(duv * np.array([w, h]), axis=1)
    ok = lp > 0.05
    if not ok.any():
        return 16.0
    return float(np.clip(np.median(lt[ok] / lp[ok]), 2.0, max(w, h)))


def _sample_grid(tex, tx, ty, scale, n):
    """n x n texel colours around (tx, ty), spanning `scale` texels (wrapping)."""
    h, w = tex.shape[:2]
    offs = ((np.arange(n) + 0.5) / n - 0.5)
    ox = (tx[:, None, None] + offs[None, None, :] * scale[:, None, None])
    oy = (ty[:, None, None] + offs[None, :, None] * scale[:, None, None])
    ix = np.floor(ox).astype(np.int64) % w
    iy = np.floor(oy).astype(np.int64) % h
    return tex[iy, ix]                                     # (N, n, n, 4)


def _kmeans_fit_assign(feat, k, seed=0, fit_max=120_000):
    from .ko_ground import _kmeans
    uniq, inv, counts = np.unique(feat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    if len(uniq) <= k:
        return inv, uniq.astype(np.float32)
    rng = np.random.default_rng(seed)
    fit = rng.choice(len(uniq), size=min(fit_max, len(uniq)), replace=False,
                     p=counts / counts.sum()) if len(uniq) > fit_max else np.arange(len(uniq))
    x = uniq[fit].astype(np.float32)
    labels = _kmeans(x, counts[fit].astype(np.float64), k, seed, iters=15)
    k = int(labels.max()) + 1
    centres = np.stack([x[labels == g].mean(0) if (labels == g).any() else x[0] for g in range(k)])
    # assign every unique feature to its nearest centre
    u = uniq.astype(np.float32)
    lab = np.empty(len(u), np.int64)
    cc = (centres * centres).sum(1)
    for s in range(0, len(u), 16384):
        blk = u[s:s + 16384]
        lab[s:s + 16384] = (cc[None] - 2 * blk @ centres.T).argmin(1)
    return lab[inv], centres


def build_objects(opd, terrain, world, cm, library, pack, simple_plants=False, seed=0):
    """Place all objects as KO-textured blocks. Returns (blocks placed, ids of shapes built)."""
    from .converter import (FILL_BELOW_STEPS, MAX_Y, _close_diagonal_gaps, _key, _unkey,
                            classify_object)

    size = cm.size_blocks
    textures, tex_index = [], {}
    FULL, SLAB = 2, 1
    cols = {n: [] for n in ("key", "prio", "tex", "tx", "ty", "scale", "dist", "leaf", "walk")}
    plants = []            # (texture id, uv box, height in blocks, positions (n, 3))
    built, missing = set(), 0

    def tex_id(name, rgba):
        if name not in tex_index:
            tex_index[name] = len(textures)
            textures.append(rgba)
        return tex_index[name]

    def add(keys, prio, tid, tx, ty, scale, dist, leaf, walk):
        n = len(keys)
        if n == 0:
            return
        cols["key"].append(keys)
        cols["prio"].append(np.full(n, prio, np.int8))
        cols["tex"].append(np.full(n, tid, np.int32))
        cols["tx"].append(np.asarray(tx, np.float32))
        cols["ty"].append(np.asarray(ty, np.float32))
        cols["scale"].append(np.full(n, scale, np.float32))
        cols["dist"].append(np.asarray(dist, np.float32))
        cols["leaf"].append(np.full(n, leaf, bool))
        cols["walk"].append(np.full(n, walk, bool) if np.isscalar(walk) else walk)

    for i, shape in enumerate(opd.shapes):
        name = shape.name.lower()
        if re.search(r"fx|smoke|fog|smog|collisioncube|alpha", name) or not shape.parts:
            continue
        kind = classify_object(shape.name)
        if library.missing(shape):
            missing += 1
            continue
        parts = list(library.shape_parts(shape))
        if kind in PLANT_KINDS and not shape.is_event_object:
            if not simple_plants:
                _collect_plant(shape, parts, cm, terrain, tex_id, plants)
            built.add(i) if not simple_plants else None
            continue
        mirrored = shape.scale.x * shape.scale.y * shape.scale.z < 0
        for tris, uvs, tex, part in parts:
            if part.dest_blend == 2:
                continue
            mc = np.stack([cm.x(tris[..., 0]), cm.y(tris[..., 1]), cm.z(tris[..., 2])], -1)
            if tex is None:
                c = np.array([int(v * 255) for v in part.diffuse[:3]] + [255], np.uint8).clip(40, 230)
                tex = np.tile(c, (4, 4, 1))
                tname = f"#diffuse{tuple(c)}"
            else:
                tname = part.textures[0].lower()
            alpha = bool(part.render_flags & km.RF_ALPHABLENDING) or (tex[..., 3] < 128).mean() > 0.05
            tid = tex_id(tname, tex)
            scale = _texel_scale(mc, uvs, tex.shape)
            pts, tri, tx, ty = km.sample_part(mc, uvs, tex, alpha)
            if len(pts) == 0:
                continue
            nrm = -np.cross(mc[:, 1] - mc[:, 0], mc[:, 2] - mc[:, 0])
            if mirrored:
                nrm = -nrm
            ny = nrm[:, 1] / (np.linalg.norm(nrm, axis=1) + 1e-9)
            up = (ny > 0.7)[tri] & (not alpha)
            vx = np.floor(pts[:, 0]).astype(np.int64)
            vz = np.floor(pts[:, 2]).astype(np.int64)
            inside = (vx >= 0) & (vx < size) & (vz >= 0) & (vz < size)
            ground = np.full(len(pts), MAX_Y, np.int64)
            ground[inside] = terrain.top[vz[inside], vx[inside]]
            txf, tyf = tx.astype(np.float32), ty.astype(np.float32)

            wall = inside & ~up
            wy = np.floor(pts[:, 1]).astype(np.int64)
            wall &= wy >= ground
            if wall.any():
                d = np.linalg.norm(pts[wall] - (np.stack([vx[wall], wy[wall], vz[wall]], 1) + 0.5), axis=1)
                keys = _key(vx[wall], wy[wall], vz[wall])
                keys, src = _close_diagonal_gaps(keys, np.arange(len(keys)))
                dist = np.where(np.arange(len(keys)) < wall.sum(), d[src], 9.0)
                add(keys, FULL, tid, txf[wall][src], tyf[wall][src], scale, dist, alpha, False)

            top = inside & up
            if top.any():
                q = np.round(pts[top, 1] * 2) / 2
                half = (q % 1) != 0
                ty_ = np.where(half, np.floor(q), q - 1).astype(np.int64)
                x_, z_, g = vx[top], vz[top], ground[top]
                keep = ty_ > g
                d = np.hypot(pts[top, 0] - x_ - 0.5, pts[top, 2] - z_ - 0.5)
                for m, prio in ((keep & ~half, FULL), (keep & half, SLAB)):
                    add(_key(x_[m], ty_[m], z_[m]), prio, tid, txf[top][m], tyf[top][m], scale, d[m], False,
                        prio == FULL)
                low = keep & (ty_ - g <= FILL_BELOW_STEPS)
                if low.any():
                    idx = np.flatnonzero(low)
                    cols_ = np.stack([x_[idx], z_[idx], ty_[idx], g[idx]], 1)
                    _, first = np.unique(cols_[:, :3], axis=0, return_index=True)
                    idx, cols_ = idx[first], cols_[first]
                    depth = np.maximum(cols_[:, 2] - cols_[:, 3] - 1, 0)
                    rep = np.repeat(np.arange(len(cols_)), depth)
                    if len(rep):
                        off = np.arange(len(rep)) - np.repeat(np.cumsum(depth) - depth, depth)
                        fy = cols_[rep, 3] + 1 + off
                        add(_key(cols_[rep, 0], fy, cols_[rep, 1]), FULL, tid, txf[top][idx[rep]],
                            tyf[top][idx[rep]], scale, np.full(len(rep), 5.0), False, False)
        built.add(i)
        if len(built) % 2000 == 0:
            print(f"    {len(built)} objects built...")

    n_plants = _place_plants(plants, textures, terrain, world, cm, pack, seed)
    if not cols["key"]:
        return n_plants, built

    c = {k: np.concatenate(v) for k, v in cols.items()}
    # winner per block: full beats slab, then the sample closest to the block centre
    order = np.lexsort((c["dist"], -c["prio"], c["key"]))
    first = np.r_[True, c["key"][order][1:] != c["key"][order][:-1]]
    w = order[first]
    v = {k: a[w] for k, a in c.items()}
    walk_any = np.zeros(len(w), bool)
    # a block is walkable if any walkable top sample landed in it
    wk = np.unique(c["key"][c["walk"]])
    walk_any = np.isin(v["key"], wk) & (v["prio"] == FULL) & ~v["leaf"]

    # appearance: 3x3 colour grid of the KO texture piece each block shows
    feat = np.zeros((len(w), 3, 3, 4), np.uint8)
    for t in np.unique(v["tex"]):
        m = v["tex"] == t
        feat[m] = _sample_grid(textures[t], v["tx"][m], v["ty"][m], v["scale"][m], 3)
    leaf = v["leaf"]
    ids = np.zeros(len(w), np.int64)

    def piece(j):
        img = _sample_grid(textures[v["tex"][j]], v["tx"][j:j + 1], v["ty"][j:j + 1],
                           v["scale"][j:j + 1], PIECE)[0].astype(np.float32)
        img[..., :3] = np.clip(img[..., :3] * OBJECT_BRIGHTNESS, 0, 255)
        return img.astype(np.uint8)

    def representative(members, labels, centres, f):
        reps = {}
        for g in np.unique(labels):
            mm = members[labels == g]
            d = ((f[labels == g] - centres[g]) ** 2).sum(1)
            reps[int(g)] = int(mm[d.argmin()])
        return reps

    # solid blocks
    solid = np.flatnonzero(~leaf)
    if len(solid):
        f = feat[solid, :, :, :3].reshape(len(solid), -1).astype(np.float32)
        labels, centres = _kmeans_fit_assign(f.astype(np.uint8), pack.max_solid, seed)
        reps = representative(solid, labels, centres, f)
        state_of = {}
        for g, j in reps.items():
            img = piece(j)
            img[..., 3] = 255
            st = pack.add_solid(img)
            state_of[g] = world.block_id(st) if st else world.block_id("minecraft:stone")
        ids[solid] = [state_of[int(g)] for g in labels]
        # slabs and stairs: the most used step looks get their own stairs/slab type
        _steps(v, solid, labels, centres, reps, walk_any, ids, piece, terrain, world, pack, cm)
    # leaves
    lv = np.flatnonzero(leaf)
    if len(lv):
        f = feat[lv].reshape(len(lv), -1).astype(np.float32)
        labels, centres = _kmeans_fit_assign(f.astype(np.uint8), pack.max_foliage, seed)
        reps = representative(lv, labels, centres, f)
        state_of = {}
        for g, j in reps.items():
            st = pack.add_foliage(piece(j))
            state_of[g] = world.block_id(st) if st else world.block_id("minecraft:oak_leaves[persistent=true]")
        ids[lv] = [state_of[int(g)] for g in labels]

    x, y, z = _unkey(v["key"])
    world.set_blocks(x, y, z, ids.astype(np.uint16))
    if missing:
        print(f"  {missing} objects have model files missing; using simple stand-ins for them")
    print(f"  KO textures for objects: {len(pack.solid)} building, {len(pack.foliage)} foliage, "
          f"{len(pack.plants)} plant, {len(pack.stairs)} stairs, {len(pack.slabs)} slab looks")
    return len(w) + n_plants, built


def _steps(v, solid, labels, centres, reps, walk_any, ids, piece, terrain, world, pack, cm):
    """Slabs keep their half height; one-block rises on walkable surfaces become stairs."""
    from . import custom_blocks as cb
    from .converter import _add_stairs_facing
    types = cb.STAIR_SLAB_TYPES
    group = np.full(len(v["key"]), -1, np.int64)
    group[solid] = labels
    is_slab = v["prio"] == 1

    facing = _add_stairs_facing(v["key"], walk_any, ~v["leaf"] & ~is_slab, terrain, cm)

    def assign(mask, setter, state_fmt):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return
        g = group[idx]
        used, counts = np.unique(g, return_counts=True)
        chosen = used[np.argsort(-counts)][:len(types)]
        # every step look maps to the nearest chosen look
        d = ((centres[used][:, None] - centres[chosen][None]) ** 2).sum(-1)
        nearest = dict(zip(used.tolist(), chosen[d.argmin(1)].tolist()))
        type_of = {}
        for t, gg in zip(types, chosen):
            img = piece(reps[int(gg)])
            img[..., 3] = 255
            setter(t, img)
            type_of[int(gg)] = t
        for j, gg in zip(idx, g):
            t = type_of[nearest[int(gg)]]
            ids[j] = world.block_id(state_fmt(t, j))

    assign(is_slab & (group >= 0), pack.set_slab,
           lambda t, j: f"minecraft:{t}_slab[type=bottom,waterlogged=false]")
    stairs = np.array([k in facing for k in v["key"].tolist()]) & (group >= 0)
    assign(stairs, pack.set_stairs,
           lambda t, j: f"minecraft:{t}_stairs[facing={facing[int(v['key'][j])]},half=bottom,"
                        f"shape=straight,waterlogged=false]")
    if stairs.any():
        print(f"  {int(stairs.sum())} step edges turned into stairs")


# ---------------------------------------------------------------------------
# plants
# ---------------------------------------------------------------------------

def _collect_plant(shape, parts, cm, terrain, tex_id, plants):
    pts, best = [], None
    for tris, uvs, tex, part in parts:
        mc = np.stack([cm.x(tris[..., 0]), cm.y(tris[..., 1]), cm.z(tris[..., 2])], -1)
        pts.append(mc.reshape(-1, 3))
        if tex is None:
            continue
        area = np.linalg.norm(np.cross(mc[:, 1] - mc[:, 0], mc[:, 2] - mc[:, 0]), axis=1).sum()
        if best is None or area > best[0]:
            u = uvs.reshape(-1, 2)
            box = (float(u[:, 0].min()), float(u[:, 1].min()), float(u[:, 0].max()), float(u[:, 1].max()))
            best = (area, part.textures[0].lower(), tex, box)
    if best is None or not pts:
        return
    p = np.concatenate(pts)
    lo, hi = p.min(0), p.max(0)
    height = hi[1] - lo[1]
    tid = tex_id(best[1], best[2])
    ext = hi[[0, 2]] - lo[[0, 2]]
    if max(ext) <= 2.5:
        xs = np.array([(lo[0] + hi[0]) / 2])
        zs = np.array([(lo[2] + hi[2]) / 2])
    else:
        # patches of grass: scatter tufts over the footprint
        rng = np.random.default_rng(abs(int(shape.position.x * 131 + shape.position.z * 17)))
        gx, gz = np.meshgrid(np.arange(lo[0] + 1, hi[0], 2.0), np.arange(lo[2] + 1, hi[2], 2.0))
        keep = rng.random(gx.shape) < 0.45
        xs, zs = gx[keep] + rng.uniform(-0.7, 0.7, keep.sum()), gz[keep] + rng.uniform(-0.7, 0.7, keep.sum())
    base = math.floor(float(lo[1]))
    plants.append((tid, best[3], float(np.clip(height, 0.4, 2.0)), np.stack([xs, np.full(len(xs), base), zs], 1)))


def _place_plants(plants, textures, terrain, world, cm, pack, seed):
    if not plants:
        return 0
    size = cm.size_blocks
    # one look per (texture, uv box, height class)
    looks, look_of = [], []
    index = {}
    for tid, box, h, _ in plants:
        k = (tid, tuple(round(b, 2) for b in box), 2 if h > 1.2 else 1)
        if k not in index:
            index[k] = len(looks)
            looks.append((tid, box, h))
        look_of.append(index[k])
    imgs = []
    for tid, (u0, v0, u1, v1), h in looks:
        tex = textures[tid]
        th, tw = tex.shape[:2]
        xs = np.linspace(u0, u1, PIECE, endpoint=False) + (u1 - u0) / (2 * PIECE)
        ys = np.linspace(v0, v1, PIECE, endpoint=False) + (v1 - v0) / (2 * PIECE)
        ix = np.floor(np.mod(xs, 1.0) * tw).astype(np.int64) % tw
        iy = np.floor(np.mod(ys, 1.0) * th).astype(np.int64) % th
        img = tex[iy[:, None], ix[None, :]].astype(np.float32)
        img[..., :3] = np.clip(img[..., :3] * OBJECT_BRIGHTNESS, 0, 255)
        imgs.append(img.astype(np.uint8))
    # too many looks: merge by average colour
    group = np.arange(len(looks))
    if len(looks) > pack.max_plants:
        f = np.stack([im[im[..., 3] > 127][:, :3].mean(0) if (im[..., 3] > 127).any() else np.zeros(3)
                      for im in imgs]).astype(np.uint8)
        group, _ = _kmeans_fit_assign(f, pack.max_plants, seed)
    state = {}
    for g in np.unique(group):
        j = int(np.flatnonzero(group == g)[0])
        h = looks[j][2]
        st = pack.add_plant(imgs[j], int(round(min(h * 16, 32))))
        state[int(g)] = world.block_id(st) if st else world.block_id("minecraft:short_grass")
    xs, ys, zs, ids = [], [], [], []
    for (tid, box, h, pos), lk in zip(plants, look_of):
        x = np.floor(pos[:, 0]).astype(np.int64)
        z = np.floor(pos[:, 2]).astype(np.int64)
        ok = (x >= 0) & (x < size) & (z >= 0) & (z < size)
        x, z, yb = x[ok], z[ok], pos[ok, 1].astype(np.int64)
        top = terrain.top[z, x]
        dry = terrain.water_top[z, x] <= top
        y = np.maximum(top + 1, yb)
        x, y, z = x[dry], y[dry], z[dry]
        xs.append(x); ys.append(y); zs.append(z)
        ids.append(np.full(len(x), state[int(group[lk])], np.int64))
    x, y, z, i = (np.concatenate(a) for a in (xs, ys, zs, ids))
    world.set_blocks(x, y, z, i.astype(np.uint16))
    return len(x)
