"""
Windowed dataset over the tensors cached by `tensors.py` (S4 / S6 plumbing).

One training example is one **query column**:

    x : (W, 4)   gamete counts (or frequencies) for the pairs (i, i+d), one row
                 per circular offset d, with the self-pair (d = 0) at the centre
    y : 0/1      "a real breakpoint sits immediately to the left of column i"

The cached target has length `M_tx + 1` (boundaries `0..M_tx`).  The genome is
circular, so boundary `M_tx` and boundary `0` are the same point; the trailing
boundary is therefore folded into column 0.  Boundary labels then have length
`M_tx` and line up exactly with the query columns.

Nothing here imports a training framework — it is plain numpy over a
memory-mapped array, so it is cheap to verify on its own (see
`tests/test_data.py`) and can feed any framework.

Input mode
----------
`mode="freq"` (default) sends counts / n_sequences, so every dataset is on the
same scale regardless of how many sequences it has.  `mode="count"` sends raw
counts (Dr. Martin asked for the counts; this is open question Q5).
"""
from __future__ import annotations
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_candidate_base = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASE = os.environ.get(
    "GT_BASE_DIR",
    _candidate_base if os.path.exists(os.path.join(_candidate_base, "WINDOWS_RDP5_PACKAGE"))
    else os.path.abspath(os.path.join(HERE, ".."))
)
CACHE = os.environ.get("GT_CACHE_DIR", os.path.join(HERE, "cache"))
MANIFEST = os.environ.get(
    "GT_MANIFEST",
    os.path.join(BASE, "WINDOWS_RDP5_PACKAGE", "batch_manifest_v2.json"),
)


def all_configs(cache=None, manifest_path=None):
    cache = cache or CACHE
    manifest_path = manifest_path or os.environ.get("GT_MANIFEST", MANIFEST)
    if manifest_path and os.path.exists(manifest_path):
        entries = json.load(open(manifest_path))
        # Filter to configs that are present in cache if cache is a custom dir
        if cache != os.path.join(HERE, "cache") and os.path.isdir(cache):
            return [e["config_name"] for e in entries
                    if os.path.exists(os.path.join(cache, e["config_name"], "meta.json"))]
        return [e["config_name"] for e in entries]
    if os.path.isdir(cache):
        return sorted(
            d for d in os.listdir(cache)
            if os.path.exists(os.path.join(cache, d, "meta.json"))
        )
    return []


def load_meta(config, cache=None):
    cache = cache or CACHE
    with open(os.path.join(cache, config, "meta.json")) as fh:
        return json.load(fh)


def fold_circular(y):
    """(M+1,) boundary labels -> (M,), folding boundary M onto boundary 0."""
    y = np.asarray(y)
    out = y[:-1].copy()
    out[0] = 1 if (y[0] or y[-1]) else 0
    return out


def _window_from_haplo(H, i, offsets_arr):
    """Compute (W, 4) gamete counts for query column i from haplo H (M, S) uint8."""
    M = H.shape[0]
    j = (int(i) + offsets_arr) % M
    b_i = H[int(i)]          # (S,)
    b_j = H[j]               # (W, S)
    code = (b_i[None, :] << 1) | b_j   # (W, S) in 0..3
    W = len(offsets_arr)
    out = np.empty((W, 4), dtype=np.float32)
    for g in range(4):
        out[:, g] = (code == g).sum(axis=1)
    return out


class WindowDataset:
    """One dataset, one example per transformed column."""

    def __init__(self, config, cache=None, mode="freq", lazy_mmap=False):
        assert mode in ("freq", "count")
        self.config = config
        self.cache = cache or CACHE
        self.mode = mode
        d = os.path.join(self.cache, config)
        self.dir = d
        with open(os.path.join(d, "meta.json")) as fh:
            self.meta = json.load(fh)
        y_raw = np.load(os.path.join(d, "target.npy"))
        self.y = fold_circular(y_raw)
        self.W = self.meta["W"]
        self.M = self.meta["M_tx"]
        self.n_sequences = self.meta["n_sequences"]
        self.offsets = self.meta["offsets"]
        self._offsets_arr = np.asarray(self.offsets, dtype=np.int64)
        self.centre = self.offsets.index(0)
        embed_path = os.path.join(d, "embed.npy")
        self._has_embed = os.path.exists(embed_path)
        self._lazy_mmap = lazy_mmap
        self.F = None
        self._H_cache = None
        if self._has_embed and not lazy_mmap:
            # memory-mapped: the 50-dataset panel is ~0.6 GB
            self.F = np.load(embed_path, mmap_mode="r")
            assert self.F.shape == (self.M, self.W, 4), self.F.shape
        assert self.y.shape == (self.M,), self.y.shape

    def __len__(self):
        return self.M

    def _get_haplo(self):
        if self._H_cache is None:
            self._H_cache = np.load(os.path.join(self.dir, "haplo.npy"))
        return self._H_cache

    def x(self, i):
        if self.F is not None:
            row = np.asarray(self.F[i], dtype=np.float32)
        elif self._has_embed:
            arr = np.load(os.path.join(self.dir, "embed.npy"), mmap_mode="r")
            row = np.asarray(arr[i], dtype=np.float32).copy()
            del arr
        else:
            row = _window_from_haplo(self._get_haplo(), i, self._offsets_arr)
        if self.mode == "freq":
            row = row / float(self.n_sequences)
        return row

    def __getitem__(self, i):
        return self.x(i), float(self.y[i])

    def batch(self, indices):
        """numpy batch -> (x (B, W, 4) float32, y (B,) float32)."""
        xs = np.stack([self.x(i) for i in indices]).astype(np.float32)
        ys = np.asarray([self.y[i] for i in indices], dtype=np.float32)
        return xs, ys

    def positives(self):
        return int(self.y.sum())

    # ---------------------------------------------- level-2 artefacts (T10.1)
    def haplo(self, mmap=True):
        """(M_tx, S) uint8 per-sequence states in transformed coordinates."""
        return np.load(os.path.join(self.cache, self.config, "haplo.npy"),
                       mmap_mode="r" if mmap else None)

    def seq_event(self):
        """(S, E) uint8, 1 where sequence s carries event e (events.json order)."""
        return np.load(os.path.join(self.cache, self.config, "seq_event.npy"))

    def events(self):
        return json.load(open(os.path.join(self.cache, self.config,
                                           "events.json")))

    def coords(self):
        return np.load(os.path.join(self.cache, self.config, "coords.npz"))


class Panel:
    """A concatenation of WindowDatasets, addressed by a flat global index."""

    def __init__(self, configs, cache=None, mode="freq"):
        lazy = len(configs) > 128
        self.parts = [
            WindowDataset(c, cache=cache, mode=mode, lazy_mmap=lazy)
            for c in configs
        ]
        self.bounds = []
        self._ends = np.empty(len(self.parts), dtype=np.int64)
        start = 0
        for idx, p in enumerate(self.parts):
            end = start + len(p)
            self.bounds.append((start, end))
            self._ends[idx] = end
            start = end
        self.n = start
        self.W = max((p.W for p in self.parts), default=314)

    def __len__(self):
        return self.n

    def owner(self, i):
        if i < 0 or i >= self.n:
            raise IndexError(i)
        k = int(np.searchsorted(self._ends, i, side="right"))
        a = self.bounds[k][0]
        return k, int(i - a)

    def __getitem__(self, i):
        k, j = self.owner(i)
        x, y = self.parts[k][j]
        if x.shape[0] < self.W:
            # Circularly wrap-pad if an ultra-sparse alignment had M_tx < W
            reps = (self.W // max(1, x.shape[0])) + 2
            tiled = np.tile(x, (reps, 1))
            pad_left = (self.W - x.shape[0]) // 2
            x = tiled[pad_left:pad_left + self.W]
        return x, y

    def positives(self):
        return sum(p.positives() for p in self.parts)

    def batch(self, indices):
        """numpy batch -> (x (B, W, 4) float32, y (B,) float32)."""
        xs = np.stack([self[i][0] for i in indices]).astype(np.float32)
        ys = np.asarray([self[i][1] for i in indices], dtype=np.float32)
        return xs, ys


def splits(seed=0, frac=(0.70, 0.15, 0.15), cache=None, manifest_path=None):
    """Deterministic, stratified train/val/test split over datasets.

    Split by *dataset*, so no alignment contributes to more than one split.
    Stratified on whether the dataset contains any breakpoint at all: 17 of the 50
    alignments have zero events, and an unstratified shuffle can land most of the
    event-bearing datasets in train, which makes the held-out base rate a lottery
    (and the metric meaningless).
    """
    cache = cache or CACHE
    configs = all_configs(cache=cache, manifest_path=manifest_path)
    rng = np.random.default_rng(seed)
    strata = {}
    for c in configs:
        tpath = os.path.join(cache, c, "target.npy")
        if not os.path.exists(tpath):
            continue
        has = int(np.load(tpath).sum() > 0)
        strata.setdefault(has, []).append(c)
    out = {"train": [], "val": [], "test": []}
    for has in sorted(strata):
        items = [str(x) for x in rng.permutation(strata[has])]
        n = len(items)
        n_tr = int(round(frac[0] * n))
        n_va = int(round(frac[1] * n))
        if n >= 3:
            if n_va == 0:
                n_va = 1
            if n_tr + n_va >= n:
                n_tr = max(1, n - n_va - 1)
        out["train"] += items[:n_tr]
        out["val"] += items[n_tr:n_tr + n_va]
        out["test"] += items[n_tr + n_va:]
    for k in out:
        out[k] = sorted(out[k])
    return out


def main():
    import os as _os
    if not _os.path.isdir(CACHE):
        print("no cache yet - run: python3 tensors.py --all")
        return 1
    d = WindowDataset("d013_L4584_s34_r005")
    print(f"dataset   : {d.config}")
    print(f"columns   : {d.M}  window {d.W}  sequences {d.n_sequences}")
    print(f"centre    : offset index {d.centre} (offset 0)")
    print(f"positives : {d.positives()} (folded target)")
    x, y = d[0]
    print(f"example 0 : x{x.shape} {x.dtype} sum={x.sum():.3f}  y={y}")
    idx = np.flatnonzero(d.y)[:5]
    xs, ys = d.batch(idx)
    print(f"batch     : x{xs.shape} {xs.dtype}  y{ys.shape} sum={ys.sum():.0f}")
    sp = splits()
    print(f"splits    : train {len(sp['train'])}  val {len(sp['val'])}  "
          f"test {len(sp['test'])} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
