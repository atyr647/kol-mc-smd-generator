"""Rebuild the KO ground the way the KO client draws it, then fit it into a
Minecraft resource pack.

Every KO terrain tile (4 x 4 m) has a base texture and, on about a third of
the tiles, an overlay texture. Transition tiles fade to black and the overlay
fades the other way; KO adds the two together, which blends e.g. grass into
dirt. Each layer is also rotated/mirrored by a 5-bit "direction" code.

The direction codes aren't documented, so we work them out from the data: the
right transform for each code is the one that makes neighbouring tiles line up
without seams.

For Minecraft, each block shows its own piece of the tile (at true size a tile
is 4 x 4 blocks), so textures keep KO's scale. There are far more distinct
pieces than usable block states, so similar pieces are grouped (k-means) and
each group uses the real piece closest to its average.
"""

import numpy as np

TEX = 128  # KO terrain textures are 128 x 128

# Direction code -> transform (see dihedral), found with GroundBuilder.fit_directions on
# Moradon (and confirmed on Freezone): the choice that makes neighbouring tiles line up. 0 = as is, 1 = flip
# top/bottom, 2 = rotate 180, 3 = flip left/right. 4 and 5 are rare (~3% of tiles)
# and the fit is less certain about them.
DIR_TRANSFORMS = {0: 0, 1: 6, 2: 2, 3: 4, 4: 4, 5: 4, 6: 6}


def dihedral(img: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 rotations/mirrors of a square image (k in 0..7)."""
    out = np.rot90(img, k % 4)
    return out[:, ::-1] if k >= 4 else out


class GroundBuilder:
    def __init__(self, gtd, library, brightness: float = 1.3):
        self.gtd = gtd
        self.brightness = brightness
        n = gtd.heightmap_size - 1
        bits = gtd.texture_ids[:n, :n]
        self.t1 = gtd.tex1[:n, :n].astype(np.int32)
        self.t2 = gtd.tex2[:n, :n].astype(np.int32)
        self.d1 = ((bits >> 1) & 31).astype(np.int32)
        self.d2 = ((bits >> 6) & 31).astype(np.int32)
        # load every texture used
        from .ko_textures import tile_texture_key
        self.images = {}
        for idx in np.unique(np.concatenate([self.t1.ravel(), self.t2.ravel()])):
            key = tile_texture_key(gtd, int(idx))
            rgba = library.tile(*key) if key else None
            if rgba is not None and rgba.shape[0] == rgba.shape[1]:
                im = rgba[..., :3].astype(np.float32)
                if im.shape[0] != TEX:
                    from PIL import Image
                    im = np.asarray(Image.fromarray(rgba[..., :3]).resize((TEX, TEX)), np.float32)
                self.images[int(idx)] = im
        self.found = set(self.images)
        self.dir_map = {d: DIR_TRANSFORMS.get(d, 0) for d in range(32)}

    # ---- which transform does each direction code mean? -------------------

    def fit_directions(self):
        """Pick, for every direction code, the transform that minimises tile seams."""
        n = self.t1.shape[0]
        # edges of every texture under every transform: [tex][k] -> (4, TEX, 3) top,bottom,left,right
        edges = {}
        for idx, im in self.images.items():
            e = []
            for k in range(8):
                t = dihedral(im, k)
                e.append(np.stack([t[0], t[-1], t[:, 0], t[:, -1]]))
            edges[idx] = np.stack(e)
        zero = np.zeros((8, 4, TEX, 3), np.float32)
        ok = np.vectorize(lambda i: i in edges)(self.t1)
        used_dirs = sorted(set(np.unique(self.d1[ok])) | set(np.unique(self.d2[(self.t2 < 1023)])))

        def tile_edges(mapping):
            out = np.zeros((n, n, 4, TEX, 3), np.float32)
            for tx in range(n):
                for tz in range(n):
                    a = edges.get(int(self.t1[tx, tz]), zero)[mapping[int(self.d1[tx, tz])]]
                    if self.t2[tx, tz] < 1023:
                        a = a + edges.get(int(self.t2[tx, tz]), zero)[mapping[int(self.d2[tx, tz])]]
                    out[tx, tz] = a
            return np.minimum(out, 255)

        # Image rows run from the tile's north edge to its south edge, columns west to east
        # (the same layout the preview and pack use). North of tile (tx, tz) is (tx, tz+1).
        def cost(mapping):
            e = tile_edges(mapping)
            east = np.abs(e[:-1, :, 3] - e[1:, :, 2]).mean()     # right edge vs neighbour's left
            north = np.abs(e[:, :-1, 0] - e[:, 1:, 1]).mean()    # top edge vs northern neighbour's bottom
            return east + north

        mapping = dict(self.dir_map)
        best = cost(mapping)
        for _ in range(2):
            for d in used_dirs:
                for k in range(8):
                    trial = dict(mapping)
                    trial[d] = k
                    c = cost(trial)
                    if c < best - 1e-6:
                        best, mapping = c, trial
        self.dir_map = mapping
        return {d: mapping[d] for d in used_dirs}, best

    # ---- composite tiles ---------------------------------------------------

    def tile_image(self, tx: int, tz: int) -> np.ndarray | None:
        a = self.images.get(int(self.t1[tx, tz]))
        if a is None:
            return None
        img = dihedral(a, self.dir_map[int(self.d1[tx, tz])])
        b = self.images.get(int(self.t2[tx, tz])) if self.t2[tx, tz] < 1023 else None
        if b is not None:
            img = img + dihedral(b, self.dir_map[int(self.d2[tx, tz])])
        return np.clip(img * self.brightness, 0, 255)

    def combos(self):
        """Unique (t1, d1, t2, d2) per tile -> (combo id grid [x, z], list of example tiles)."""
        key = (((self.t1.astype(np.int64) * 32 + self.d1) * 1024 + self.t2) * 32 + self.d2)
        uk, inv = np.unique(key.ravel(), return_inverse=True)
        inv = inv.reshape(key.shape)
        first = {}
        for (tx, tz), c in np.ndenumerate(inv):
            first.setdefault(int(c), (tx, tz))
        return inv, [first[c] for c in range(len(uk))]

    # ---- block textures ----------------------------------------------------

    def block_textures(self, scale: int, max_textures: int, seed: int = 0):
        """Group the per-block texture pieces.

        Returns (group [x_tile, z_tile, sub_x, sub_z] -> group id or -1, list of group images).
        sub_x runs west->east, sub_z north->south inside a tile.
        """
        inv, examples = self.combos()
        px = TEX // scale
        pieces, owners = [], []
        for c, (tx, tz) in enumerate(examples):
            img = self.tile_image(tx, tz)
            if img is None:
                continue
            for sz in range(scale):
                for sx in range(scale):
                    pieces.append(img[sz * px:(sz + 1) * px, sx * px:(sx + 1) * px])
                    owners.append((c, sx, sz))
        if not pieces:
            return None, []
        pieces = np.stack(pieces)                                   # (P, px, px, 3)
        # weight = how many blocks use each piece
        counts = np.bincount(inv.ravel(), minlength=len(examples))
        w = np.array([counts[c] for c, _, _ in owners], np.float64)
        f = 4 if px >= 4 else px
        feat = pieces.reshape(len(pieces), f, px // f, f, px // f, 3).mean((2, 4)).reshape(len(pieces), -1)
        k = min(max_textures, len(pieces))
        labels = _kmeans(feat, w, k, seed)
        # representative = real piece closest to the group's weighted mean
        reps = []
        for g in range(k):
            m = np.flatnonzero(labels == g)
            if len(m) == 0:
                reps.append(None)
                continue
            centre = np.average(feat[m], axis=0, weights=w[m])
            best = m[np.argmin(((feat[m] - centre) ** 2).sum(1))]
            reps.append(pieces[best].astype(np.uint8))
        group = np.full(inv.shape + (scale, scale), -1, np.int32)
        piece_group = {}
        for (c, sx, sz), g in zip(owners, labels):
            piece_group[(c, sx, sz)] = g
        lut = np.full((len(examples), scale, scale), -1, np.int32)
        for (c, sx, sz), g in piece_group.items():
            lut[c, sx, sz] = g
        group = lut[inv]                                            # [x, z, sx, sz]
        return group, reps


def _kmeans(x: np.ndarray, w: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    """Weighted k-means; returns a label per row."""
    rng = np.random.default_rng(seed)
    x = x.astype(np.float32)
    if k >= len(x):
        return np.arange(len(x))
    # k-means++ style init (sampled)
    centres = [x[rng.choice(len(x), p=w / w.sum())]]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 * w
        centres.append(x[rng.choice(len(x), p=p / p.sum())])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(1))
    c = np.stack(centres)
    xx = (x * x).sum(1)[:, None]
    for _ in range(iters):
        labels = np.empty(len(x), np.int64)
        for s in range(0, len(x), 8192):
            blk = x[s:s + 8192]
            d = xx[s:s + 8192] - 2 * blk @ c.T + (c * c).sum(1)[None]
            labels[s:s + 8192] = d.argmin(1)
        for g in range(k):
            m = labels == g
            if m.any():
                c[g] = np.average(x[m], axis=0, weights=w[m])
    return labels
