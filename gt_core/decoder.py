"""
Minimal-event decoder  --  S10 / T10.4, first cut.

Turn a dataset's 4-gamete pair observations into a *minimal* set of breakpoints,
then score it against the true breakpoints at the **event** level.  This is the
crudest honest version of what Dr. Martin asked for: "reconcile all those
instances where you have all 4 gamete combinations for a pair of alignment
coordinates into a cohesive recombination hypothesis that invokes a minimal number
of recombination events".

Method
------
1. A 4-gamete pair (i, j) is evidence for exactly one breakpoint **anywhere in the
   open interval (i, j)**: a single crossover between i and j is enough to make
   that pair show all four gametes.  So each observed 4-gamete pair votes for
   every boundary it straddles.
2. `support(b)` = number of 4-gamete pairs straddling boundary `b`.  This is the
   classical 4-gamete plot, computed in O(#pairs + M) with two `bincount`s and a
   cumulative sum.
3. *Greedy minimal explanation.*  Take the boundary with the most support, record
   it, delete every pair it explains, and repeat.  Stop when the best remaining
   support falls below `min_support` (so homoplasy is not chased indefinitely) or
   when `max_events` is reached.  This is set cover over the straddling
   constraint: "minimal" in the greedy sense, which is the point -- it is
   measurable, not trusted.
4. Map boundaries back to original coordinates and score against the truth.

Distance decay (`--pair-weight exp|inv`) is available because plain interval
containment is too permissive: a pair spanning 150 columns votes as hard for a
boundary in the middle as for one at its edge.  Decaying the vote with span is the
cheapest correction and is what the classical 4-gamete plots do in effect.

Honest limitation, stated rather than hidden: the greedy pass explains *pairs*,
not *haplotypes*.  It never checks that one sequence's haplotype actually switches
at the breakpoint, which is the T10.2/T10.3 step and the reason the first cut sits
close to the random control (see ACTION_ITEMS.md §H, fourth finding).

Run:  python3 decoder.py                       # sweep min_support over the panel
      python3 decoder.py --datasets d013_L4584_s34_r005
      python3 decoder.py --tol 20 --min-support 10
"""
from __future__ import annotations
import os
import csv
import json
import argparse
import numpy as np

try:
    from . import data as D
    from . import pipeline as P
except (ImportError, ValueError):
    import data as D
    import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


# --------------------------------------------------------------- pair extraction
def gamete_pairs(mask, offsets):
    """4-gamete pairs as (Li, Rj) in *doubled* linear coordinates.

    `mask[i, oi]` is True where the pair (i, i+offset[oi]) shows all four gametes.
    Only positive offsets are used (each unordered pair counted once).  The column
    space is doubled (`i`, `i + M`) so circular pairs need no special case: a pair
    that wraps simply lands in the second copy.
    """
    offs = np.asarray(offsets)
    sel = np.flatnonzero(offs > 0)
    rows, cols = np.nonzero(np.asarray(mask)[:, sel])
    o = offs[sel][cols]
    return rows.astype(np.int64), (rows + o).astype(np.int64)


def pair_weights(d, mode="none", decay=20.0):
    """Weight a 4-gamete pair by its span `d`.

    Interval containment is too permissive: a pair spanning 150 columns votes as
    strongly for a boundary in the middle as for one at its edge.  Weighting by
    distance (a pair seen 1 column apart is far stronger evidence than one seen
    150 apart) is the cheapest fix, and the classical 4-gamete plots all decay
    with distance in effect.
    """
    if mode == "none":
        return np.ones_like(d, dtype=np.float64)
    if mode == "inv":
        return 1.0 / d
    if mode == "exp":
        return np.exp(-(d - 1.0) / float(decay))
    raise ValueError(mode)


def _profile(Li, Rj, n2, w=None):
    """f[k] = weighted support of boundary k+1 (doubled coordinates)."""
    if w is None:
        starts = np.bincount(Li, minlength=n2)
        ends = np.bincount(Rj, minlength=n2)
    else:
        starts = np.bincount(Li, weights=w, minlength=n2)
        ends = np.bincount(Rj, weights=w, minlength=n2)
    return np.cumsum(starts - ends)


def decode_trace(mask, offsets, M, max_events=64, max_span=None,
                 pair_weight="none", decay=20.0):
    """Greedy trace: ordered list of (boundary, support) picks.

    Support is non-increasing along the trace (each pick removes straddling
    pairs, which can only lower later supports), so `min_support` is just a
    prefix cut: decode once, read off every threshold.
    """
    Li, Rj = gamete_pairs(mask, offsets)
    if max_span is not None:
        keep = (Rj - Li) <= max_span
        Li, Rj = Li[keep], Rj[keep]
    if len(Li) == 0:
        return []
    w = pair_weights(Rj - Li, mode=pair_weight, decay=decay)
    alive = np.ones(len(Li), dtype=bool)
    n2 = 2 * M
    trace = []
    while alive.any() and len(trace) < max_events:
        f = _profile(Li[alive], Rj[alive], n2, w[alive])
        fold = f[:M - 1] + f[M:2 * M - 1]       # boundaries 1 .. M-1
        best = int(np.argmax(fold)) + 1
        best_sup = float(f[best - 1])
        if float(f[M - 1]) > best_sup:          # boundary M (== boundary 0)
            best, best_sup = M, float(f[M - 1])
        trace.append((best, best_sup))
        explained = (((Li < best) & (best <= Rj)) |
                     ((Li < best + M) & (best + M <= Rj)))
        alive &= ~explained
    return trace


def decode(mask, offsets, M, min_support=5, max_events=64, max_span=None,
           pair_weight="none", decay=20.0):
    """Greedy minimal-event decoder.  Boundary `b` is left of transformed column
    `b` (so `b = M` is the circular wrap, the same point as boundary 0)."""
    return [b for b, sup in
            decode_trace(mask, offsets, M, max_events, max_span,
                         pair_weight, decay)
            if sup >= min_support]


# ------------------------------------------------------------------- evaluation
def match(pred, gt, tol):
    """One-to-one greedy nearest matching of sorted position lists -> tp, fp, fn."""
    used = set()
    tp = 0
    for p in pred:
        best_i, best_d = None, None
        for i, g in enumerate(gt):
            if i in used:
                continue
            d = abs(p - g)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_d <= tol:
            used.add(best_i)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def truth_for(config, cache=D.CACHE):
    """(true original breakpoints, L, maps) for one cached dataset."""
    d = os.path.join(cache, config)
    ev = json.load(open(os.path.join(d, "events.json")))
    gt = sorted({b for e in ev for b in e["breakpoints_orig"]})
    coords = np.load(os.path.join(d, "coords.npz"))
    L = int(json.load(open(os.path.join(d, "meta.json")))["genome_length"])
    return gt, L, {"boundary": coords["boundary"], "tx_to_orig": coords["tx_to_orig"]}


def decode_dataset(config, min_support=5, max_events=64, max_span=None,
                   pair_weight="none", decay=20.0, cache=D.CACHE):
    """-> (predicted original coords, true original coords)."""
    meta = json.load(open(os.path.join(cache, config, "meta.json")))
    F = np.load(os.path.join(cache, config, "embed.npy"), mmap_mode="r")
    mask = (np.asarray(F) > 0).all(axis=-1)
    chosen = decode(mask, meta["offsets"], meta["M_tx"], min_support=min_support,
                    max_events=max_events, max_span=max_span,
                    pair_weight=pair_weight, decay=decay)
    gt, L, maps = truth_for(config, cache)
    pred = sorted({P.unshrink(P.normalise_boundary(b, maps), maps, L)
                   for b in chosen})
    return pred, gt


def sweep(configs, min_supports, tol, max_events, max_span, pair_weight="none",
          decay=20.0, cache=D.CACHE):
    rows = []
    for c in configs:
        meta = json.load(open(os.path.join(cache, c, "meta.json")))
        F = np.load(os.path.join(cache, c, "embed.npy"), mmap_mode="r")
        mask = (np.asarray(F) > 0).all(axis=-1)
        trace = decode_trace(mask, meta["offsets"], meta["M_tx"], max_events,
                             max_span, pair_weight, decay)
        del F, mask
        gt, L, maps = truth_for(c, cache)
        for ms in min_supports:
            chosen = [b for b, sup in trace if sup >= ms]
            pred = sorted({P.unshrink(P.normalise_boundary(b, maps), maps, L)
                           for b in chosen})
            tp, fp, fn = match(pred, gt, tol)
            rows.append({"config": c, "min_support": ms, "n_gt": len(gt),
                         "n_pred": len(pred), "tp": tp, "fp": fp, "fn": fn})
    return rows


def _print_table(rows, ms, label):
    print(f"\n=== {label} ===")
    print(f"  {'min_sup':>7} {'TP':>7} {'FP':>7} {'FN':>7} {'n_pred':>8} "
          f"{'prec':>7} {'rec':>7} {'F1':>7}")
    best = None
    for m in ms:
        rr = [r for r in rows if r["min_support"] == m]
        tp = sum(r["tp"] for r in rr)
        fp = sum(r["fp"] for r in rr)
        fn = sum(r["fn"] for r in rr)
        n_pred = sum(r["n_pred"] for r in rr)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  {m:>7} {tp:>7} {fp:>7} {fn:>7} {n_pred:>8} "
              f"{prec:7.3f} {rec:7.3f} {f1:7.3f}")
        if best is None or f1 > best[1]:
            best = (m, f1, prec, rec)
    return best


def random_control(configs, n_pred_by_cfg, tol, reps, seed=0, cache=D.CACHE):
    """Random-boundary control at matched prediction counts.

    Same number of predicted boundaries per dataset as the decoder produced, but
    placed uniformly at random.  This is the honest floor for an event-level F1 at
    tolerance `tol` -- the tolerance means a random hit is not rare, so precision
    must be compared against *this*, not against `n_gt / M`.
    """
    rng = np.random.default_rng(seed)
    tp = fp = fn = 0
    for c in configs:
        n_pred = n_pred_by_cfg.get(c, 0)
        gt, L, maps = truth_for(c, cache)
        M = json.load(open(os.path.join(cache, c, "meta.json")))["M_tx"]
        for _ in range(reps):
            if n_pred and M:
                b = rng.integers(1, M + 1, size=n_pred)
                pred = sorted({P.unshrink(P.normalise_boundary(int(x), maps),
                                         maps, L) for x in b})
            else:
                pred = []
            t, f, n = match(pred, gt, tol)
            tp += t
            fp += f
            fn += n
    return tp, fp, fn


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--tol", type=int, default=10)
    ap.add_argument("--min-support", type=int, default=None)
    ap.add_argument("--min-supports", nargs="*", type=int,
                    default=[2, 5, 10, 20, 40, 80])
    ap.add_argument("--max-events", type=int, default=64)
    ap.add_argument("--max-span", type=int, default=None)
    ap.add_argument("--pair-weight", choices=["none", "inv", "exp"], default="none",
                    help="distance decay applied to each 4-gamete pair's vote")
    ap.add_argument("--decay", type=float, default=20.0,
                    help="decay length (columns) for --pair-weight exp")
    ap.add_argument("--reps", type=int, default=10,
                    help="random-control repeats per dataset")
    ap.add_argument("--out", default=os.path.join(OUT, "decoder_results.csv"))
    args = ap.parse_args()

    configs = args.datasets or D.all_configs()
    ms = [args.min_support] if args.min_support is not None else args.min_supports

    print(f"decoder: {len(configs)} datasets, tol=+/-{args.tol} bp, "
          f"max_events={args.max_events}, max_span={args.max_span}, "
          f"pair_weight={args.pair_weight}"
          + (f"(decay={args.decay})" if args.pair_weight == "exp" else ""))
    rows = sweep(configs, ms, args.tol, args.max_events, args.max_span,
                 args.pair_weight, args.decay)

    _print_table(rows, ms, f"ALL {len(configs)} datasets")
    with_events = [r["config"] for r in rows if r["n_gt"] > 0]
    ev_set = set(with_events)
    _print_table([r for r in rows if r["config"] in ev_set], ms,
                 f"event-bearing only ({len(ev_set)} datasets)")

    # random control at matched prediction counts
    print(f"\n=== random-boundary control ({args.reps} reps/dataset) ===")
    print(f"  {'min_sup':>7} {'prec':>7} {'rec':>7} {'F1':>7}")
    for m in ms:
        n_pred_by_cfg = {r["config"]: r["n_pred"] for r in rows
                         if r["min_support"] == m}
        tp, fp, fn = random_control(with_events, n_pred_by_cfg, args.tol,
                                    args.reps)
        prec, rec, f1 = _prf(tp, fp, fn)
        print(f"  {m:>7} {prec:7.3f} {rec:7.3f} {f1:7.3f}")

    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["config", "min_support", "n_gt",
                                            "n_pred", "tp", "fp", "fn"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
