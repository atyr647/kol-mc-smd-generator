"""KO objects -> Minecraft blocks that show the real KO textures.

For every block a model covers we remember which KO texture it came from, where
in that texture (the texel at the block's centre) and how many texels one block
spans. That's enough to cut the matching piece out of the KO texture.

Every KO texture is cut into block-sized pieces, so a wall shows its texture
continuously like in KO. A resource pack can only add about a thousand block
looks (see custom_blocks.py), so textures share those out and a texture with too
few looks merges its most similar pieces (never pieces of different textures). Leaves get see-through leaf blocks with KO leaf textures, and
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


FACINGS = ("north", "east", "south", "west")


def _facing(dx, dz):
    """Index into FACINGS of the horizontal direction (dx, dz) (Minecraft: north = -z)."""
    return np.where(np.abs(dx) > np.abs(dz), np.where(dx > 0, 1, 3), np.where(dz > 0, 2, 0))


def _texel_scale(tris_mc, uvs, tex_shape):
    """Texels per block across (u) and down (v) the texture, area-weighted median over the part."""
    h, w = tex_shape[:2]
    e1 = tris_mc[:, 1] - tris_mc[:, 0]
    e2 = tris_mc[:, 2] - tris_mc[:, 0]
    d1 = uvs[:, 1] - uvs[:, 0]
    d2 = uvs[:, 2] - uvs[:, 0]
    # metric of the triangle's plane; |grad u|^2 = du^T G^-1 du
    g11, g12, g22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    det = g11 * g22 - g12 * g12
    ok = det > 1e-6
    if not ok.any():
        return 16.0, 16.0
    out = []
    for k, size in ((0, w), (1, h)):
        a, b = d1[ok, k], d2[ok, k]
        grad2 = (g22[ok] * a * a - 2 * g12[ok] * a * b + g11[ok] * b * b) / det[ok]
        val = np.sqrt(np.maximum(grad2, 0)) * size
        wt = np.sqrt(det[ok])
        o = np.argsort(val)
        c = np.cumsum(wt[o])
        med = val[o][np.searchsorted(c, c[-1] / 2)]
        out.append(float(np.clip(med, 2.0, max(w, h))))
    return out[0], out[1]


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


def build_objects(opd, terrain, world, cm, library, pack, simple_plants=False, seed=0, on_shape=None):
    """Place all objects as KO-textured blocks. Returns (blocks placed, ids of shapes built).

    on_shape(i, keys): if given, called once per non-plant shape with the union of block
    keys (see converter._key) that shape contributed, before the global winner-takes-all
    step (so a neighbour may still have won a shared voxel). Used by qa_render.py."""
    from .converter import (FILL_BELOW_STEPS, MAX_Y, _close_diagonal_gaps, _key, _unkey,
                            classify_object)

    size = cm.size_blocks
    textures, tex_index = [], {}
    STAIR, FULL, SLAB = 3, 2, 1
    cols = {n: [] for n in ("key", "prio", "tex", "tx", "ty", "scale", "scale_y", "dist", "leaf", "walk",
                            "face")}
    plants = []            # (texture id, uv box, height in blocks, positions (n, 3))
    built, missing = set(), 0

    def tex_id(name, rgba):
        if name not in tex_index:
            tex_index[name] = len(textures)
            textures.append(rgba)
        return tex_index[name]

    cur_keys = []

    def add(keys, prio, tid, tx, ty, scale, dist, leaf, walk, face=-1):
        n = len(keys)
        if n == 0:
            return
        if on_shape is not None:
            cur_keys.append(keys)
        cols["key"].append(keys)
        cols["prio"].append(np.full(n, prio, np.int8))
        cols["tex"].append(np.full(n, tid, np.int32))
        cols["tx"].append(np.asarray(tx, np.float32))
        cols["ty"].append(np.asarray(ty, np.float32))
        cols["scale"].append(np.full(n, scale[0], np.float32))
        cols["scale_y"].append(np.full(n, scale[1], np.float32))
        cols["dist"].append(np.asarray(dist, np.float32))
        cols["leaf"].append(np.full(n, leaf, bool))
        cols["walk"].append(np.full(n, walk, bool) if np.isscalar(walk) else walk)
        cols["face"].append(np.full(n, face, np.int8) if np.isscalar(face) else face.astype(np.int8))

    for i, shape in enumerate(opd.shapes):
        cur_keys.clear()
        name = shape.name.lower()
        if re.search(r"fx|smoke|fog|smog|collisioncube|alpha", name) or not shape.parts:
            continue
        kind = classify_object(shape.name)
        if library.missing(shape):
            missing += 1
            continue
        parts = list(library.shape_parts(shape))
        if kind == "bush" and not simple_plants and _small_plant(parts, cm):
            kind = "grass"       # knee-high bushes: a plant sprite reads better than a lump of leaves
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
            a8 = tex[..., 3]
            # smoothly graduated alpha (many mid-range values) is a real blend effect (a
            # glowing crystal, a translucent aura) that Minecraft can't render see-through;
            # showing it solid, in its own colour, beats the alternative of a cutout test
            # dropping most of it and leaving next to nothing behind
            graduated = ((a8 > 20) & (a8 < 235)).mean() > 0.15
            alpha = (not graduated) and (bool(part.render_flags & km.RF_ALPHABLENDING)
                                         or (a8 < 128).mean() > 0.05)
            tid = tex_id(tname, tex)
            scale = _texel_scale(mc, uvs, tex.shape)
            pts, tri, tx, ty = km.sample_part(mc, uvs, tex, alpha)
            if len(pts) == 0:
                continue
            nrm = -np.cross(mc[:, 1] - mc[:, 0], mc[:, 2] - mc[:, 0])
            if mirrored:
                nrm = -nrm
            if part.render_flags & km.RF_DOUBLESIDED:
                nrm = np.where(nrm[:, 1:2] < 0, -nrm, nrm)     # one surface seen from both sides: face up
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
            # underside of a steep roof: same layer as the roof stairs above it
            under = ((ny < -0.7) & (ny > -0.9))[tri]
            wy = np.where(under, np.floor(pts[:, 1] - 0.5).astype(np.int64), wy)
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
                d = np.hypot(pts[top, 0] - x_ - 0.5, pts[top, 2] - z_ - 0.5)
                # steep roofs (26-45 degrees): stairs climbing the slope; gentler slopes keep
                # half/full steps
                steep = (ny < 0.9)[tri[top]]
                sy = np.floor(pts[top, 1] - 0.5).astype(np.int64)
                ty_ = np.where(steep, sy, ty_)
                # >= (not >): a flat decal painted right at ground level (a field line, a
                # carpet) should still show, not get silently dropped for tying the terrain
                keep = ty_ >= g
                uphill = _facing(-nrm[tri[top], 0], -nrm[tri[top], 2])
                for m, prio, face in ((keep & ~half & ~steep, FULL, -1), (keep & half & ~steep, SLAB, -1),
                                      (keep & steep, STAIR, uphill)):
                    add(_key(x_[m], ty_[m], z_[m]), prio, tid, txf[top][m], tyf[top][m], scale, d[m], False,
                        prio == FULL, face if np.isscalar(face) else face[m])
                # small perched decorations (signs, plaques, pots) aren't floors and don't
                # need a support pillar down to natural ground -- they usually rest on another
                # object's roof/ledge, which the ground reference here can't see
                small = max(x_.max() - x_.min(), z_.max() - z_.min()) < 4
                low = keep & (ty_ - g <= FILL_BELOW_STEPS) & (not small)
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
        if on_shape is not None:
            on_shape(i, np.unique(np.concatenate(cur_keys)) if cur_keys else np.empty(0, np.int64))
        if len(built) % 2000 == 0:
            print(f"    {len(built)} objects built...")

    n_plants = _place_plants(plants, textures, terrain, world, cm, pack, seed)
    if not cols["key"]:
        return n_plants, built

    c = {k: np.concatenate(v) for k, v in cols.items()}
    # winner per block: roof stairs beat full blocks beat slabs, then the sample closest to the centre
    order = np.lexsort((c["dist"], -c["prio"], c["key"]))
    first = np.r_[True, c["key"][order][1:] != c["key"][order][:-1]]
    w = order[first]
    v = {k: a[w] for k, a in c.items()}
    walk_any = np.zeros(len(w), bool)
    # a block is walkable if any walkable top sample landed in it
    wk = np.unique(c["key"][c["walk"]])
    walk_any = np.isin(v["key"], wk) & (v["prio"] == FULL) & ~v["leaf"]
    # a roof stair with a block right above it is inside the roof: make it a full block
    sx, sy, sz = _unkey(v["key"])
    covered = np.isin(_key(sx, sy + 1, sz), v["key"])
    v["prio"][(v["prio"] == STAIR) & covered] = FULL

    leaf = v["leaf"]
    ids = np.zeros(len(w), np.int64)

    # solid blocks
    solid = np.flatnonzero(~leaf)
    if len(solid):
        look, imgs, feats = _texture_looks(v, solid, textures, pack.max_solid, seed, alpha=False)
        state_of = []
        for img in imgs:
            st = pack.add_solid(img)
            state_of.append(world.block_id(st) if st else world.block_id("minecraft:stone"))
        ids[solid] = np.array(state_of)[look]
        # slabs and stairs: the most used step looks get their own stairs/slab type
        group = np.full(len(w), -1, np.int64)
        group[solid] = look
        _steps(v, group, feats, imgs, walk_any, ids, terrain, world, pack, cm)
    # leaves
    lv = np.flatnonzero(leaf)
    if len(lv):
        look, imgs, _ = _texture_looks(v, lv, textures, pack.max_foliage, seed, alpha=True)
        state_of = []
        for img in imgs:
            st = pack.add_foliage(img)
            state_of.append(world.block_id(st) if st else world.block_id("minecraft:oak_leaves[persistent=true]"))
        ids[lv] = np.array(state_of)[look]

    x, y, z = _unkey(v["key"])
    world.set_blocks(x, y, z, ids.astype(np.uint16))
    if missing:
        print(f"  {missing} objects have model files missing; using simple stand-ins for them")
    print(f"  KO textures for objects: {len(pack.solid)} building, {len(pack.foliage)} foliage, "
          f"{len(pack.plants)} plant, {len(pack.stairs)} stairs, {len(pack.slabs)} slab looks")
    return len(w) + n_plants, built


def _steps(v, group, feats, imgs, walk_any, ids, terrain, world, pack, cm):
    """Slabs keep their half height; one-block rises on walkable surfaces become stairs.

    group = look index of every block (-1 = none), feats/imgs = per look."""
    from . import custom_blocks as cb
    from .converter import _add_stairs_facing
    types = cb.STAIR_SLAB_TYPES
    is_slab = v["prio"] == 1

    facing = _add_stairs_facing(v["key"], walk_any, ~v["leaf"] & ~is_slab & (v["prio"] != 3), terrain, cm)

    def assign(mask, setter, state_fmt):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return
        g = group[idx]
        used, counts = np.unique(g, return_counts=True)
        chosen = used[np.argsort(-counts)][:len(types)]
        # every step look maps to the nearest chosen look
        d = ((feats[used][:, None] - feats[chosen][None]) ** 2).sum(-1)
        nearest = dict(zip(used.tolist(), chosen[d.argmin(1)].tolist()))
        type_of = {}
        for t, gg in zip(types, chosen):
            setter(t, imgs[int(gg)])
            type_of[int(gg)] = t
        for j, gg in zip(idx, g):
            t = type_of[nearest[int(gg)]]
            ids[j] = world.block_id(state_fmt(t, j))

    assign(is_slab & (group >= 0), pack.set_slab,
           lambda t, j: f"minecraft:{t}_slab[type=bottom,waterlogged=false]")
    roof = (v["prio"] == 3) & (v["face"] >= 0)
    for j in np.flatnonzero(roof):
        facing[int(v["key"][j])] = FACINGS[int(v["face"][j])]
    stairs = np.array([k in facing for k in v["key"].tolist()]) & (group >= 0)
    assign(stairs, pack.set_stairs,
           lambda t, j: f"minecraft:{t}_stairs[facing={facing[int(v['key'][j])]},half=bottom,"
                        f"shape=straight,waterlogged=false]")
    if stairs.any():
        print(f"  {int(stairs.sum())} step edges turned into stairs")


# ---------------------------------------------------------------------------
# block looks
# ---------------------------------------------------------------------------

MAX_CELLS = 32         # a texture is cut into at most 32 x 32 block-sized pieces


def _pow2_cells(size, scale):
    """How many block-sized pieces fit across a texture side (power of two, 1..MAX_CELLS)."""
    n = np.maximum(size / np.maximum(scale, 1e-3), 1.0)
    return np.clip(2 ** np.round(np.log2(n)), 1, MAX_CELLS).astype(np.int64)


def _cell_image(tex, x0, x1, y0, y1, alpha):
    """One block look: the tex[y0:y1, x0:x1] region, resampled to PIECE x PIECE."""
    h, w = tex.shape[:2]
    crop = tex[max(y0, 0):min(max(y1, y0 + 1), h), max(x0, 0):min(max(x1, x0 + 1), w)]
    ch, cw = crop.shape[:2]
    # area-average (or repeat) to PIECE x PIECE
    ys = (np.arange(PIECE + 1) * ch / PIECE).astype(np.int64)
    xs = (np.arange(PIECE + 1) * cw / PIECE).astype(np.int64)
    if ch >= PIECE and cw >= PIECE:
        c = np.cumsum(np.cumsum(np.pad(crop.astype(np.float64), ((1, 0), (1, 0), (0, 0))), 0), 1)
        tot = c[ys[1:, None], xs[None, 1:]] - c[ys[:-1, None], xs[None, 1:]] \
            - c[ys[1:, None], xs[None, :-1]] + c[ys[:-1, None], xs[None, :-1]]
        img = tot / ((ys[1:] - ys[:-1])[:, None, None] * (xs[1:] - xs[:-1])[None, :, None])
    else:
        img = crop[np.minimum(ys[:-1], ch - 1)[:, None], np.minimum(xs[:-1], cw - 1)[None, :]].astype(np.float64)
    img[..., :3] = np.clip(img[..., :3] * OBJECT_BRIGHTNESS, 0, 255)
    img = img.astype(np.uint8)
    if alpha:
        img[..., 3] = np.where(img[..., 3] >= 128, 255, 0)
    else:
        img[..., 3] = 255
    return img


def _texture_looks(v, members, textures, budget, seed, alpha):
    """Block looks for the blocks `members`.

    Every KO texture is cut into block-sized pieces (as big as one block on the
    model), and a block shows the piece its centre lies on, so neighbouring
    blocks continue the texture the way KO draws it. The pack only has `budget`
    looks, so textures share them out (more for textures on many blocks); a
    texture with fewer looks than pieces merges its similar pieces. Looks are
    never shared between textures, so every wall keeps its own material.

    Returns (look index per member, look images, look features)."""
    from .converter import _small
    from .ko_ground import _kmeans
    tex = v["tex"][members]
    scale, scale_y = v["scale"][members], v["scale_y"][members]
    units, unit_of = {}, np.empty(len(members), np.int64)
    unit_bbox = {}   # unit id -> (x0, x1, y0, y1) in texel space, from the samples that use it
    for t in np.unique(tex):
        m = np.flatnonzero(tex == t)
        h, w = textures[t].shape[:2]
        gx = _pow2_cells(w, scale[m])
        gy = _pow2_cells(h, scale_y[m])
        mtx = np.mod(v["tx"][members[m]], w)
        mty = np.mod(v["ty"][members[m]], h)
        cx = (mtx * gx // w).astype(np.int64)
        cy = (mty * gy // h).astype(np.int64)
        key = ((gx * 64 + gy) * 64 + cx) * 64 + cy
        uk, inv = np.unique(key, return_inverse=True)
        for k in uk.tolist():
            units[(int(t), k)] = len(units)
        u_ids = np.array([units[(int(t), k)] for k in uk.tolist()])[inv.ravel()]
        unit_of[m] = u_ids
        # crop each look from where its own samples actually sample the texture, not a
        # rigid grid slice -- a small prop (or one sharing a crowded sprite sheet) then
        # shows only its own material, never a neighbouring, unrelated piece of the sheet
        order = np.argsort(u_ids, kind="stable")
        u_sorted = u_ids[order]
        starts = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]])
        ends = np.r_[starts[1:], len(u_sorted)]
        pad = 3
        for s0, s1 in zip(starts, ends):
            idx = order[s0:s1]
            uid = int(u_sorted[s0])
            x0, x1 = int(mtx[idx].min()) - pad, int(mtx[idx].max()) + 1 + pad
            y0, y1 = int(mty[idx].min()) - pad, int(mty[idx].max()) + 1 + pad
            unit_bbox[uid] = (x0, x1, y0, y1)
    keys = list(units)
    count = np.bincount(unit_of, minlength=len(keys)).astype(np.float64)
    utex = np.array([t for t, _ in keys])
    imgs = []
    for u, (t, k) in enumerate(keys):
        x0, x1, y0, y1 = unit_bbox[u]
        imgs.append(_cell_image(textures[t], x0, x1, y0, y1, alpha))
    feat = np.stack([np.concatenate([_small(im), [im[..., 3].mean() / 4]]) for im in imgs])

    # share the budget: water-filling on sqrt(blocks) per texture, at least one look each
    tids = np.unique(utex)
    need = np.array([(utex == t).sum() for t in tids])
    weight = np.sqrt(np.array([count[utex == t].sum() for t in tids]))
    if need.sum() <= budget:
        alloc = need
    else:
        lo, hi = 0.0, float(budget)
        for _ in range(60):
            lam = (lo + hi) / 2
            alloc = np.minimum(need, np.maximum(1, np.floor(lam * weight / weight.sum()))).astype(np.int64)
            lo, hi = (lam, hi) if alloc.sum() <= budget else (lo, lam)
        alloc = np.minimum(need, np.maximum(1, np.floor(lo * weight / weight.sum()))).astype(np.int64)
        if alloc.sum() > budget:          # more textures than looks: the least used share one each
            order = np.argsort(-weight)
            alloc = np.zeros_like(need)
            alloc[order[:budget]] = 1
        # hand out what rounding left over, one look at a time to the most starved texture
        spare = budget - int(alloc.sum())
        while spare > 0 and (alloc < need).any():
            starved = np.where(alloc < need, weight / (alloc + 1), -1)
            alloc[int(starved.argmax())] += 1
            spare -= 1
    look_of_unit = np.full(len(keys), -1, np.int64)
    out_imgs, out_feat = [], []
    for t, k in zip(tids, alloc):
        u = np.flatnonzero(utex == t)
        if k <= 0:
            continue
        if len(u) <= k:
            lab = np.arange(len(u))
        else:
            lab = _kmeans(feat[u], count[u], int(k), seed)
        for g in np.unique(lab):
            mm = u[lab == g]
            centre = np.average(feat[mm], axis=0, weights=count[mm])
            rep = mm[((feat[mm] - centre) ** 2).sum(1).argmin()]
            look_of_unit[mm] = len(out_imgs)
            out_imgs.append(imgs[rep])
            out_feat.append(feat[rep])
    out_feat = np.stack(out_feat)
    # textures without a look of their own take the nearest look
    left = np.flatnonzero(look_of_unit < 0)
    if len(left):
        d = ((feat[left][:, None] - out_feat[None]) ** 2).sum(-1)
        look_of_unit[left] = d.argmin(1)
    print(f"    {len(tids)} KO textures, {len(keys)} block-sized pieces -> {len(out_imgs)} "
          f"{'foliage' if alpha else 'building'} looks")
    return look_of_unit[unit_of], out_imgs, out_feat


# ---------------------------------------------------------------------------
# plants
# ---------------------------------------------------------------------------

def _small_plant(parts, cm, max_height=2.0):
    ys = [tris[..., 1].ravel() for tris, _, _, _ in parts]
    if not ys:
        return False
    y = np.concatenate(ys)
    return (cm.y(y.max()) - cm.y(y.min())) <= max_height


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
