"""
Haplotype-based recombination localisation  --  S10 / T10.3, second cut.

The first cut (`decoder.py`) explained **4-gamete pairs** by interval containment
and landed only ~1.3-1.4x above a random control.  This module attacks the actual
weakness: it uses the **haplotypes**, and it asks the question a recombination
event really poses --

    is there a sequence whose closest relative on the left of this boundary is a
    different sequence from its closest relative on the right?

A single crossover at boundary `b` makes exactly one sequence `r` a mosaic: its
left segment matches parent `L`, its right segment matches parent `R`, with `L != R`.
Non-recombinant sequences have the *same* closest relative on both sides (`L == R`)
and are excluded by construction -- which is why the score does not simply track
overall similarity.  This is the haplotype-based statement the pair model was
missing, and it produces level 2 for free (the contributing sequences are the
recombinants).

Score
-----
For boundary `b` (1..M) and sequence `r`:

    leftsim[x, r]  = fraction of columns i <  b with H[i, x] == H[i, r]
    rightsim[y, r] = fraction of columns i >= b with H[i, y] == H[i, r]
    L(r) = argmax_{x != r} leftsim[x, r],   LB(r) = that maximum
    R(r) = argmax_{y != r} rightsim[y, r],  RB(r) = that maximum

`r` shows a switch at `b` iff

    L(r) != R(r)   and   LB(r) >= thrL   and   RB(r) >= thrR
    and  leftsim[R(r), r]  <= LB(r) - margin     (right parent is worse on the left)
    and  rightsim[L(r), r] <= RB(r) - margin     (left parent is worse on the right)

The two margin conditions are what stop two *identical* haplotypes from being read
as different parents: if `L` and `R` were the same haplotype, both differences would
be 0 and the switch is rejected.  `score(b)` is the number of such sequences.

Oracle
------
`--mode oracle` ranks the true breakpoints inside the score profile itself.  If the
true breakpoints are not near the top of *any* score, no decoder built on that score
can localise them, and the ceiling is the data, not the model.  This is the most
decision-relevant number in the file, and it is cheap.

Run:  python3 haldec.py --mode oracle                     # headroom of each score
      python3 haldec.py --mode decode --thr 0.9          # event-level P/R/F1
      python3 haldec.py --mode decode --datasets d013_L4584_s34_r005
"""
from __future__ import annotations
import os
import csv
import json
import argparse
import warnings
import numpy as np

try:
    from . import data as D
    from . import pipeline as P
    from . import decoder as DEC
except (ImportError, ValueError):
    import data as D
    import pipeline as P
    import decoder as DEC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


# ------------------------------------------------------------------- score: pairs
def pair_support_score(config, cache=D.CACHE):
    """Unweighted 4-gamete straddle support per boundary, length M+1 (index 0 unused)."""
    meta = json.load(open(os.path.join(cache, config, "meta.json")))
    F = np.load(os.path.join(cache, config, "embed.npy"), mmap_mode="r")
    mask = (np.asarray(F) > 0).all(axis=-1)          # (M, W)
    M = meta["M_tx"]
    Li, Rj = DEC.gamete_pairs(mask, meta["offsets"])
    f = DEC._profile(Li, Rj, 2 * M)                  # f[k] = support of boundary k+1
    score = np.zeros(M + 1, dtype=np.float64)
    score[1:M] = f[:M - 1] + f[M:2 * M - 1]
    score[M] = f[M - 1]
    return score, meta


# ----------------------------------------------------------- score: haplotype switch
def segment_switch(H, thrL=0.9, thrR=0.9, margin=0.02, min_seg=3):
    """Per-boundary recombination-switch score, using the haplotypes.

    `H` is `(M, S)` with entries in {0, 1}.  Returns `(score, switch)` where `score`
    has length `M + 1` (index 0 unused) and `switch[b]` is a boolean `(S,)` mask of
    the sequences that show a parent switch at boundary `b` -- i.e. level 2 directly.

    Cost is O(M * S^2) with a small constant: one `(S, S)` equality per column plus a
    handful of `(S, S)` reductions per boundary.
    """
    M, S = H.shape
    H = H.astype(np.int8)
    both1 = (H.T.astype(np.int32) @ H.astype(np.int32))       # (S, S) count of 1-1
    ones = H.sum(axis=0).astype(np.int32)                     # (S,)
    both0 = M - ones[:, None] - ones[None, :] + both1         # count of 0-0
    total = both1 + both0                                     # (S, S) matches over all M

    C = np.zeros((S, S), dtype=np.int32)                      # matches among i < b
    rows = np.arange(S)
    score = np.zeros(M + 1, dtype=np.float64)
    switch = np.zeros((M + 1, S), dtype=bool)
    for b in range(1, M + 1):
        col = H[b - 1]
        C += (col[:, None] == col[None, :])
        if b < min_seg or (M - b) < min_seg:
            continue
        leftsim = C / b
        rightsim = (total - C) / (M - b)
        np.fill_diagonal(leftsim, -1.0)
        np.fill_diagonal(rightsim, -1.0)
        pL = np.argmax(leftsim, axis=0)
        pR = np.argmax(rightsim, axis=0)
        LB = leftsim[pL, rows]
        RB = rightsim[pR, rows]
        cond = ((LB >= thrL) & (RB >= thrR) & (pL != pR) &
                (leftsim[pR, rows] <= LB - margin) &
                (rightsim[pL, rows] <= RB - margin))
        switch[b] = cond
        score[b] = float(cond.sum())
    return score, switch


def flank_kl_score(H, L_w=250, min_seg=50, beta=20.0, power=0.5):
    """Symmetric variance-stabilised local flanking KL score, length M + 1.

    Why this improves on the global prefix/suffix `segment_score`:
      1. The global prefix `[0, b)` vs suffix `[b, M)` has binomial sampling
         variance proportional to `1/b + 1/(M-b)`, creating a 3-4.4x U-shaped
         baseline distortion near alignment ends and preventing `min_seg` from
         being lowered below 200 without inflating false positives.
      2. On double-crossover events `[b1, b2]`, a global suffix `[b1, M)`
         dilutes the recombinant tract `[b1, b2]` with the non-recombinant
         tail `[b2, M)`.

    At each boundary `b in [min_seg, M - min_seg]`, this compares the symmetric
    local flanking windows `[b - w, b)` and `[b, b + w)` where
    `w = min(L_w, b, M - b)`, scaled by the variance-stabilising factor
    `(w / L_w)**power`.  Uses O(M * S^2) rolling {-1, +1} outer products with
    O(S^2) memory.
    """
    M, S = H.shape
    Z = 2 * H.astype(np.int16) - 1
    score = np.zeros(M + 1, dtype=np.float64)
    diag = np.arange(S)
    w_prev = min(L_w, min_seg, M - min_seg)
    if w_prev < min_seg:
        return score
    b0 = min_seg
    L_blk = Z[b0 - w_prev : b0].astype(np.int32)
    R_blk = Z[b0 : b0 + w_prev].astype(np.int32)
    dotL = L_blk.T @ L_blk
    dotR = R_blk.T @ R_blk
    l_start, l_end = b0 - w_prev, b0
    r_start, r_end = b0, b0 + w_prev
    for b in range(min_seg, M - min_seg + 1):
        w = min(L_w, b, M - b)
        nl_s, nl_e = b - w, b
        nr_s, nr_e = b, b + w
        while l_start < nl_s:
            v = Z[l_start]
            dotL -= np.outer(v, v)
            l_start += 1
        while l_start > nl_s:
            l_start -= 1
            v = Z[l_start]
            dotL += np.outer(v, v)
        while l_end < nl_e:
            v = Z[l_end]
            dotL += np.outer(v, v)
            l_end += 1
        while r_start < nr_s:
            v = Z[r_start]
            dotR -= np.outer(v, v)
            r_start += 1
        while r_end < nr_e:
            v = Z[r_end]
            dotR += np.outer(v, v)
            r_end += 1
        while r_end > nr_e:
            r_end -= 1
            v = Z[r_end]
            dotR -= np.outer(v, v)
        leftsim = (dotL + w) / (2.0 * w)
        rightsim = (dotR + w) / (2.0 * w)
        leftsim[diag, diag] = -1.0
        rightsim[diag, diag] = -1.0
        qL = np.exp(beta * (leftsim - leftsim.max(axis=0, keepdims=True)))
        qL /= qL.sum(axis=0, keepdims=True)
        qR = np.exp(beta * (rightsim - rightsim.max(axis=0, keepdims=True)))
        qR /= qR.sum(axis=0, keepdims=True)
        logL = np.log(np.maximum(qL, 1e-300))
        logR = np.log(np.maximum(qR, 1e-300))
        kl = 0.5 * ((qL * (logL - logR)).sum(axis=0)
                    + (qR * (logR - logL)).sum(axis=0))
        scale = (w / float(L_w)) ** power if power != 0.0 else 1.0
        score[b] = float(kl.sum()) * scale
    return score


def segment_score(H, kind="hard", thrL=0.9, thrR=0.9, margin=0.02, beta=20.0,
                  min_seg=3, flank_window=250, power=0.5):
    """Per-boundary localisation score, in four flavours.

    All share the same per-boundary machinery (closest relative left vs
    right) but aggregate differently, to separate "the hard nearest-neighbour
    choice is the problem" from "the evidence is the problem":

    - ``hard``     : number of sequences with a *hard* parent switch (the count used
                     by `segment_switch`; requires thr, margin and pL != pR).
    - ``gain``     : sum of the positive *asymmetry* of the mosaic, i.e.
                     ``max(0, LB - leftsim[pR]) + max(0, RB - rightsim[pL])``.  No
                     threshold or argmax-equality test -- it uses the magnitude.
    - ``soft``     : ``sum_r (1 - sum_x qL[x] qR[x])`` where ``qL``/``qR`` are
                     softmax(beta * similarity) parent distributions on each side.
                     This **marginalises over parents** instead of committing to the
                     single nearest sampled relative (Dr. Martin's unsampled-parent
                     point).
    - ``kl``       : symmetric KL divergence between global prefix/suffix `qL` and `qR`.
    - ``flank_kl`` : symmetric variance-stabilised local flanking KL divergence over
                     `[b - w, b)` vs `[b, b + w)`, `w = min(flank_window, b, M - b)`.
    """
    if kind == "flank_kl":
        return flank_kl_score(H, L_w=flank_window, min_seg=min_seg, beta=beta,
                              power=power)
    M, S = H.shape
    H8 = H.astype(np.int8)
    both1 = (H8.T.astype(np.int32) @ H8.astype(np.int32))
    ones = H8.sum(axis=0).astype(np.int32)
    total = both1 + (M - ones[:, None] - ones[None, :] + both1)
    C = np.zeros((S, S), dtype=np.int32)
    rows = np.arange(S)
    score = np.zeros(M + 1, dtype=np.float64)
    for b in range(1, M + 1):
        col = H8[b - 1]
        C += (col[:, None] == col[None, :])
        if b < min_seg or (M - b) < min_seg:
            continue
        leftsim = C / b
        rightsim = (total - C) / (M - b)
        np.fill_diagonal(leftsim, -1.0)
        np.fill_diagonal(rightsim, -1.0)
        if kind in ("soft", "kl"):
            qL = np.exp(beta * leftsim - beta * leftsim.max(axis=0, keepdims=True))
            qL /= qL.sum(axis=0, keepdims=True)
            qR = np.exp(beta * rightsim - beta * rightsim.max(axis=0, keepdims=True))
            qR /= qR.sum(axis=0, keepdims=True)
            if kind == "soft":
                score[b] = float((1.0 - (qL * qR).sum(axis=0)).sum())
            else:
                logL = np.log(np.maximum(qL, 1e-300))
                logR = np.log(np.maximum(qR, 1e-300))
                kl = 0.5 * ((qL * (logL - logR)).sum(axis=0)
                            + (qR * (logR - logL)).sum(axis=0))
                score[b] = float(kl.sum())
            continue
        pL = np.argmax(leftsim, axis=0)
        pR = np.argmax(rightsim, axis=0)
        LB = leftsim[pL, rows]
        RB = rightsim[pR, rows]
        if kind == "gain":
            g = (np.maximum(0.0, LB - leftsim[pR, rows]) +
                 np.maximum(0.0, RB - rightsim[pL, rows]))
            score[b] = float(g.sum())
            continue
        cond = ((LB >= thrL) & (RB >= thrR) & (pL != pR) &
                (leftsim[pR, rows] <= LB - margin) &
                (rightsim[pL, rows] <= RB - margin))
        score[b] = float(cond.sum())
    return score


def sequence_switch_matrix(H, beta=20.0, min_seg=100, kind="overlap"):
    """Per-boundary, per-sequence switch evidence -> `(M, S)` float array.

    Row `b-1` is boundary `b` (left of column `b`); column `r` is sequence `r`.
    Rows too close to either end to have a legal segment (`b < min_seg` or
    `M - b < min_seg`) are **NaN**, so aggregators can ignore them without a
    separate mask.

    `kind` controls how "different" the left- and right-parent distributions are,
    each a softmax over the *other* sampled sequences:

    * `overlap` -- `1 - sum_x qL(x) qR(x)` (the original soft score; a
      Bhattacharyya-style overlap, in `[0, 1]`).
    * `kl` -- symmetric KL `0.5 (KL(qL||qR) + KL(qR||qL))`, which is sharper when
      the two distributions disagree only slightly but consistently.
    """
    M, S = H.shape
    H8 = H.astype(np.int8)
    both1 = (H8.T.astype(np.int32) @ H8.astype(np.int32))
    ones = H8.sum(axis=0).astype(np.int32)
    total = both1 + (M - ones[:, None] - ones[None, :] + both1)
    per_b = np.full((M, S), np.nan, dtype=np.float64)
    C = np.zeros((S, S), dtype=np.int32)
    diag = np.arange(S)
    bufc, bufb = [], []

    def flush():
        if not bufb:
            return
        Cv = np.stack(bufc)
        bids = np.asarray(bufb, dtype=np.float64)
        bL = bids[:, None, None]
        bR = (M - bids)[:, None, None]
        leftsim = Cv / bL
        rightsim = (total[None, :, :] - Cv) / bR
        leftsim[:, diag, diag] = -1.0
        rightsim[:, diag, diag] = -1.0
        qL = np.exp(beta * (leftsim - leftsim.max(axis=1, keepdims=True)))
        qL /= qL.sum(axis=1, keepdims=True)
        qR = np.exp(beta * (rightsim - rightsim.max(axis=1, keepdims=True)))
        qR /= qR.sum(axis=1, keepdims=True)
        if kind == "overlap":
            val = 1.0 - (qL * qR).sum(axis=1)
        elif kind == "kl":
            logL = np.log(np.maximum(qL, 1e-300))
            logR = np.log(np.maximum(qR, 1e-300))
            val = 0.5 * ((qL * (logL - logR)).sum(axis=1)
                         + (qR * (logR - logL)).sum(axis=1))
        else:
            raise ValueError(kind)
        per_b[(np.asarray(bufb) - 1)] = val
        bufc.clear()
        bufb.clear()

    for b in range(1, M + 1):
        col = H8[b - 1]
        C += (col[:, None] == col[None, :])
        if b < min_seg or (M - b) < min_seg:
            continue
        bufc.append(C.copy())
        bufb.append(b)
        if len(bufb) >= 128:
            flush()
    flush()
    return per_b


def aggregate_sequence_score(run, agg="max", top=3):
    """Reduce an `(M, S)` switch matrix to `(S,)` per-sequence scores.

    `max`/`top`: raw evidence.  `zmax`/`ztop`: standardise each boundary **across
    sequences** first (so a boundary where *everyone* looks switched does not help),
    then aggregate.  `mean`: average over all legal boundaries.
    """
    if agg in ("zmax", "ztop"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.nanmean(run, axis=1, keepdims=True)
            sd = np.nanstd(run, axis=1, keepdims=True)
        run = (run - mu) / np.where(sd > 1e-12, sd, 1.0)
    with np.errstate(all="ignore"):
        if agg in ("max", "zmax"):
            return np.nanmax(run, axis=0)
        if agg in ("top", "ztop"):
            k = min(top, run.shape[0])
            part = np.sort(np.nan_to_num(run, nan=-np.inf), axis=0)[-k:]
            return part.mean(axis=0)
        if agg == "mean":
            return np.nanmean(run, axis=0)
    raise ValueError(agg)


def sequence_switch_score(H, beta=20.0, min_seg=100, agg="max", top=3,
                          kind="overlap", return_matrix=False):
    """Q1: per-sequence evidence that a sequence is *recombinant at all*.

    The soft switch score elsewhere is summed over sequences; here we keep the
    per-sequence term at every boundary (`sequence_switch_matrix`) and aggregate it
    over boundaries.  A truly recombinant sequence is a mosaic, so at *its* own
    breakpoint it should show a high switch probability; a non-recombinant should
    show none anywhere.

    Returns `(S,)` scores, aligned to the haplotype rows (sequence order).  See
    `aggregate_sequence_score` for the aggregations; `agg="zmax"` removes the
    per-boundary baseline that every sequence shares.
    """
    run = sequence_switch_matrix(H, beta=beta, min_seg=min_seg, kind=kind)
    if return_matrix:
        return run
    return aggregate_sequence_score(run, agg=agg, top=top)


def switch_detail(H, b, thrL=0.9, thrR=0.9, margin=0.02, min_seg=3, top_parents=3):
    """Level-2 and level-3 output for one predicted breakpoint `b`.

    Returns a list, one entry per sequence `r` that shows a parent switch at `b`,
    each with the recombinant's index and a **ranked** list of parental-like
    candidates (nearest sampled relatives on the left and on the right), because
    the true parents are almost never sampled (Dr. Martin's point).  Recomputed
    directly from the haplotypes, so it is cheap to call only at chosen peaks.
    """
    M, S = H.shape
    if b < min_seg or (M - b) < min_seg:
        return []
    Lf = H[:b].astype(np.float64)
    Rf = H[b:].astype(np.float64)
    nl = M - b
    leftsim = (Lf.T @ Lf + (1.0 - Lf).T @ (1.0 - Lf)) / b
    rightsim = (Rf.T @ Rf + (1.0 - Rf).T @ (1.0 - Rf)) / (M - b)
    np.fill_diagonal(leftsim, -1.0)
    np.fill_diagonal(rightsim, -1.0)
    rows = np.arange(S)
    orderL = np.argsort(-leftsim, axis=0)         # (S, S): rows per candidate seq
    orderR = np.argsort(-rightsim, axis=0)
    pL = orderL[0]
    pR = orderR[0]
    LB = leftsim[pL, rows]
    RB = rightsim[pR, rows]
    out = []
    for r in range(S):
        if not (LB[r] >= thrL and RB[r] >= thrR and pL[r] != pR[r]
                and leftsim[pR[r], r] <= LB[r] - margin
                and rightsim[pL[r], r] <= RB[r] - margin):
            continue
        left_rank = [{"seq": int(x), "sim": float(leftsim[x, r])}
                     for x in orderL[:top_parents, r]]
        right_rank = [{"seq": int(x), "sim": float(rightsim[x, r])}
                      for x in orderR[:top_parents, r]]
        out.append({"recombinant": int(r),
                    "left_sim": float(LB[r]), "right_sim": float(RB[r]),
                    "parent_left": int(pL[r]), "parent_right": int(pR[r]),
                    "left_candidates": left_rank, "right_candidates": right_rank})
    return out


def soft_switch_detail(H, b, beta=20.0, min_seg=100, kind="kl",
                       top_members=20, member_frac=0.5, top_parents=3):
    """Level-2/3 membership at boundary `b` from the **soft** switch evidence.

    Unlike `switch_detail` (which needs a hard nearest-neighbour switch with
    thresholds and margins), this ranks every sequence by its own switch
    evidence at `b` and keeps those within `member_frac` of the best, up to
    `top_members`.  It is the membership that matches the KL score used to pick
    the peak, so the report and the detector agree.  Parental candidates are still
    hard nearest sampled relatives (ranked), because exact parents are unsampled.
    """
    M, S = H.shape
    if b < min_seg or (M - b) < min_seg:
        return []
    Lf = H[:b].astype(np.float64)
    Rf = H[b:].astype(np.float64)
    leftsim = (Lf.T @ Lf + (1.0 - Lf).T @ (1.0 - Lf)) / b
    rightsim = (Rf.T @ Rf + (1.0 - Rf).T @ (1.0 - Rf)) / (M - b)
    np.fill_diagonal(leftsim, -1.0)
    np.fill_diagonal(rightsim, -1.0)
    qL = np.exp(beta * (leftsim - leftsim.max(axis=0, keepdims=True)))
    qL /= qL.sum(axis=0, keepdims=True)
    qR = np.exp(beta * (rightsim - rightsim.max(axis=0, keepdims=True)))
    qR /= qR.sum(axis=0, keepdims=True)
    logL = np.log(np.maximum(qL, 1e-300))
    logR = np.log(np.maximum(qR, 1e-300))
    if kind == "kl":
        ev = 0.5 * ((qL * (logL - logR)).sum(axis=0)
                    + (qR * (logR - logL)).sum(axis=0))
    else:
        ev = 1.0 - (qL * qR).sum(axis=0)
    order = np.argsort(-ev)
    orderL = np.argsort(-leftsim, axis=0)
    orderR = np.argsort(-rightsim, axis=0)
    thr = member_frac * float(ev[order[0]]) if ev[order[0]] > 0 else 0.0
    out = []
    for r in order:
        if ev[r] < thr or len(out) >= top_members or ev[r] <= 0:
            break
        pl, pr = int(orderL[0, r]), int(orderR[0, r])
        out.append({
            "recombinant": int(r),
            "switch": float(ev[r]),
            "parent_left": pl, "parent_right": pr,
            "left_sim": float(leftsim[pl, r]),
            "right_sim": float(rightsim[pr, r]),
            "left_candidates": [{"seq": int(x), "sim": float(leftsim[x, r])}
                                for x in orderL[:top_parents, r]],
            "right_candidates": [{"seq": int(x), "sim": float(rightsim[x, r])}
                                 for x in orderR[:top_parents, r]],
        })
    return out


def flank_soft_switch_detail(H, b, L_w=250, beta=20.0, min_seg=3, kind="kl",
                             top_members=20, member_frac=0.5, top_parents=3):
    """Level-2/3 membership at boundary `b` using symmetric local flanking windows.

    Matches `flank_kl_score`: compares `[b - w, b)` vs `[b, b + w)` where
    `w = min(L_w, b, M - b)`.  Avoids diluting double-crossover tracts `[b1, b2]`
    with the non-recombinant tail `[b2, M)`.
    """
    M, S = H.shape
    eff_min = max(3, min_seg)
    if b < eff_min or (M - b) < eff_min:
        return []
    w = min(L_w, b, M - b)
    Lf = H[b - w : b].astype(np.float64)
    Rf = H[b : b + w].astype(np.float64)
    leftsim = (Lf.T @ Lf + (1.0 - Lf).T @ (1.0 - Lf)) / w
    rightsim = (Rf.T @ Rf + (1.0 - Rf).T @ (1.0 - Rf)) / w
    np.fill_diagonal(leftsim, -1.0)
    np.fill_diagonal(rightsim, -1.0)
    qL = np.exp(beta * (leftsim - leftsim.max(axis=0, keepdims=True)))
    qL /= qL.sum(axis=0, keepdims=True)
    qR = np.exp(beta * (rightsim - rightsim.max(axis=0, keepdims=True)))
    qR /= qR.sum(axis=0, keepdims=True)
    logL = np.log(np.maximum(qL, 1e-300))
    logR = np.log(np.maximum(qR, 1e-300))
    if kind == "kl":
        ev = 0.5 * ((qL * (logL - logR)).sum(axis=0)
                    + (qR * (logR - logL)).sum(axis=0))
    else:
        ev = 1.0 - (qL * qR).sum(axis=0)
    order = np.argsort(-ev)
    orderL = np.argsort(-leftsim, axis=0)
    orderR = np.argsort(-rightsim, axis=0)
    thr = member_frac * float(ev[order[0]]) if ev[order[0]] > 0 else 0.0
    out = []
    for r in order:
        if ev[r] < thr or len(out) >= top_members or ev[r] <= 0:
            break
        pl, pr = int(orderL[0, r]), int(orderR[0, r])
        out.append({
            "recombinant": int(r),
            "switch": float(ev[r]),
            "parent_left": pl, "parent_right": pr,
            "left_sim": float(leftsim[pl, r]),
            "right_sim": float(rightsim[pr, r]),
            "left_candidates": [{"seq": int(x), "sim": float(leftsim[x, r])}
                                for x in orderL[:top_parents, r]],
            "right_candidates": [{"seq": int(x), "sim": float(rightsim[x, r])}
                                 for x in orderR[:top_parents, r]],
        })
    return out


# ------------------------------------------------------------------- peak picking
def boundary_zscore_flank(H, L_w=250, beta=20.0, min_seg=50, power=1.0,
                          rng=None, n_perm=3, origin=None):
    """Calibrated boundary profile for `flank_kl_score` using window-stratified permutation null.

    Why window-stratified null standardization:
      1. Effective flanking half-windows truncate near sequence boundaries:
         w(b) = min(L_w, b, M - b). As w shrinks near boundaries, pairwise binomial
         sampling variance increases as 1/w, inflating raw flanking KL scores.
      2. Setting power=1.0 variance-stabilises the score by (w / L_w)**1.0, cancelling
         the 1/w divergence across window scales.
      3. Scalar pooling across all boundaries b in [min_seg, M - min_seg] creates a
         pooling fallacy: because >85% of boundaries are in the flat interior (w = L_w),
         the pooled mean and std are dominated by the interior, shifting expected edge
         null z-scores to +3.90 standard deviations and triggering spurious edge calls.
      4. Standardizing using stratified median and robust scale profiles within windows
         of effective window width w(b) eliminates edge false positives while preserving
         high sensitivity (E[z] = 0, Var(z) = 1 across all window depths).
      5. Grouping transformed columns by original alignment site (`origin`) preserves
         multi-allelic duplicated column pairings under permutation null.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M, _S = H.shape
    score = flank_kl_score(H, L_w=L_w, min_seg=min_seg, beta=beta, power=power)
    if M <= 2 * min_seg:
        return np.zeros_like(score), score, np.zeros_like(score)

    if origin is not None and len(origin) == M:
        orig_dict = {}
        unique_origs = []
        for i, o in enumerate(origin):
            if o not in orig_dict:
                unique_origs.append(o)
                orig_dict[o] = []
            orig_dict[o].append(i)
        groups = [orig_dict[o] for o in unique_origs]
    else:
        # Group adjacent duplicate / near-duplicate columns from multi-allelic site binarisation
        groups = []
        cur_group = [0]
        for i in range(1, M):
            if np.count_nonzero(H[i] != H[i - 1]) <= 1:
                cur_group.append(i)
            else:
                groups.append(cur_group)
                cur_group = [i]
        groups.append(cur_group)

    def _permute_H(rng):
        perm_g = rng.permutation(len(groups))
        idx = []
        for g in perm_g:
            idx.extend(groups[g])
        return H[idx]

    null_matrix = np.zeros((n_perm, M + 1), dtype=np.float64)
    for p in range(n_perm):
        Hp = _permute_H(rng)
        null_matrix[p] = flank_kl_score(Hp, L_w=L_w, min_seg=min_seg, beta=beta, power=power)

    null_mean_prof = null_matrix.mean(axis=0)

    # Continuous Parametric Variance Model with Interior Pooling & Scale Floor
    w_arr = np.array([min(L_w, b, M - b) for b in range(M + 1)])
    legal_indices = np.arange(min_seg, M - min_seg + 1)
    legal_w = w_arr[legal_indices]

    mask_int = (legal_w == L_w)
    if np.any(mask_int):
        idx_int = legal_indices[mask_int]
        vals_int = null_matrix[:, idx_int].flatten()
        med_ref = float(np.median(vals_int))
        mad_ref = float(np.median(np.abs(vals_int - med_ref)))
        std_ref = float(np.std(vals_int))
        scale_ref = 0.5 * (std_ref + 1.4826 * mad_ref) if mad_ref > 1e-6 else max(std_ref, 1e-6)
        w_ref = float(L_w)
    else:
        w_max = float(np.max(legal_w))
        mask_ref = (legal_w >= 0.80 * w_max)
        idx_ref = legal_indices[mask_ref]
        vals_ref = null_matrix[:, idx_ref].flatten()
        med_ref = float(np.median(vals_ref))
        mad_ref = float(np.median(np.abs(vals_ref - med_ref)))
        std_ref = float(np.std(vals_ref))
        scale_ref = 0.5 * (std_ref + 1.4826 * mad_ref) if mad_ref > 1e-6 else max(std_ref, 1e-6)
        w_ref = w_max

    u = np.clip(w_arr / max(w_ref, 1.0), 1e-4, 1.0)
    gamma = 0.12
    mu_w = med_ref * (u ** gamma)
    scale_w = np.maximum(scale_ref * (u ** gamma), 0.70 * scale_ref)

    z = np.zeros_like(score)
    z[legal_indices] = (score[legal_indices] - mu_w[legal_indices]) / np.maximum(scale_w[legal_indices], 1e-6)

    return z, score, null_mean_prof

def boundary_null(H, beta=20.0, min_seg=200, kind="kl", rng=None, n_perm=2):
    """Permutation null for the boundary score profile.

    Each sequence is circularly shifted by an independent random offset.  This
    keeps every sequence's own mosaic and its composition, but scatters the
    breakpoints so they no longer line up across sequences -- i.e. it removes the
    *shared* recombination signal while preserving the marginal statistics.  The
    mean profile over `n_perm` shifts is the null.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M, S = H.shape
    acc = None
    for _ in range(n_perm):
        Hs = np.empty_like(H)
        for s in range(S):
            Hs[:, s] = np.roll(H[:, s], int(rng.integers(M)))
        sc = segment_score(Hs, kind, beta=beta, min_seg=min_seg)
        acc = sc if acc is None else acc + sc
    return acc / n_perm


def boundary_zscore(H, beta=20.0, min_seg=200, kind="kl", rng=None, n_perm=2):
    """Calibrated boundary profile: `(z, score, null)`, `z = (score - null)/sd`.

    Peaks are then `z` peaks, so a threshold is in units of the null's own noise
    and a predicted event count is comparable across datasets of different size.
    """
    score = segment_score(H, kind, beta=beta, min_seg=min_seg)
    null = boundary_null(H, beta=beta, min_seg=min_seg, kind=kind, rng=rng,
                         n_perm=n_perm)
    mu, sd = float(null.mean()), float(null.std())
    z = (score - mu) / max(sd, 1e-12)
    return z, score, null


def boundary_null_rotate(H, beta=20.0, min_seg=200, kind="kl", rng=None, n_perm=32):
    """**Phylogeny-preserving** null: global circular rotations (not per-sequence).

    Why this and not `boundary_null`.  `boundary_null` rolls every sequence by an
    *independent* offset.  That does keep each sequence's own mosaic, but it also
    breaks the one thing the statistic is built on: at column `t` sequence `s`
    then shows genomic position ``t - k_s`` while sequence ``s'`` shows
    ``t - k_{s'}``, so the two are compared at *different* loci.  The
    left-/right-half similarity matrices collapse towards the marginal rate, `qL`
    and `qR` both flatten towards uniform, and KL -> ~0.  Every boundary scores
    below the real data (measured: null mean 5.9 vs observed 23.5) and the
    p-values pile up bimodally -- the null was degenerate.

    A **single shared** offset ``roll(H, k, 0)`` moves every sequence together:

    * the column vector ``H[t, :]`` is carried around intact, so the grouping of
      sequences -- the tree -- is preserved **exactly** at every column;
    * what changes is only the *origin* of the prefix splits.  Boundary ``b`` in
      the rotated frame covers the circular block ``[-k, -k + b)``, so as ``k``
      varies we sample all circular blocks of length ``b``.

    Hence: the phylogeny is untouched, and the *position* a breakpoint is tested
    at is randomised -- which is precisely Dr. Martin's open Q6 ("randomise the
    circular origin"), and the alignment is already treated as circular
    everywhere else (the doubled coordinates of §8).

    Returns the full ``(n_perm, M + 1)`` stack of profiles rather than their mean:
    the caller needs the *per-position* spread to standardise, and averaging
    would dilute the occasional peak that a rotation lands on a position.

    Identity (``k = 0``) is excluded -- it would just be the observed profile.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M, _S = H.shape
    out = np.empty((int(n_perm), M + 1), dtype=np.float64)
    for i in range(int(n_perm)):
        k = int(rng.integers(1, max(M, 2)))
        out[i] = segment_score(np.roll(H, k, axis=0), kind, beta=beta,
                               min_seg=min_seg)
    return out


def pick_peaks(score, min_score, radius=10, max_peaks=200):
    """Greedy peaks with non-maximum suppression of `radius` boundaries."""
    order = np.argsort(-score)
    taken = []
    for b in order:
        if b == 0 or score[b] < min_score:
            break
        if all(abs(b - t) > radius for t in taken):
            taken.append(int(b))
        if len(taken) >= max_peaks:
            break
    return sorted(taken)


# ---------------------------------------------------------------------- oracle
def percentiles(score, idxs):
    """Mid-rank percentiles of several boundary indices at once (vectorised).

    Ties matter here: the switch score takes small integer values and many
    boundaries share the same value, so a naive `<=` count inflates every
    percentile (including the null).  Mid-rank is the fair version:
    `(n_less + 0.5 * n_equal) / n`.
    """
    s = np.sort(score[1:])
    n = len(s)
    if n == 0:
        return np.zeros(len(idxs))
    v = np.asarray(score)[np.asarray(idxs)]
    less = np.searchsorted(s, v, side="left")
    right = np.searchsorted(s, v, side="right")
    return (less + 0.5 * (right - less)) / n


def gt_boundaries_tx(config, cache=D.CACHE):
    ev = json.load(open(os.path.join(cache, config, "events.json")))
    return sorted({int(b) for e in ev for b in e.get("boundaries_tx", [])})


def oracle(config, score, label, gt_tx, rng):
    """Rank the true breakpoints inside `score` (the ceiling test).

    Reports the mean mid-rank percentile of the true boundaries, the hit rate of
    simply taking the top-`n_gt` boundaries, and the same two numbers for random
    boundaries (the null).  `lift_*` is true minus null.  If the lift is ~0, no
    decoder built on this score can localise these breakpoints.
    """
    M = len(score) - 1
    gt = [b for b in gt_tx if 1 <= b <= M]
    k = len(gt)
    p_true = percentiles(score, gt) if gt else np.array([0.0])
    rnd = rng.integers(1, M + 1, size=max(k, 1) * 50)
    p_null = percentiles(score, rnd)
    top = set((np.argsort(-score[1:])[:k] + 1).tolist()) if k else set()
    hit = sum(1 for b in gt if b in top) / k if k else 0.0
    null_hit = (sum(1 for b in rnd if b in top) / len(rnd)) if len(rnd) else 0.0
    mpt, mpn = float(np.mean(p_true)), float(np.mean(p_null))
    return {"config": config, "score": label, "n_gt": k, "M_tx": M,
            "mean_pct_true": mpt, "mean_pct_null": mpn, "lift_pct": mpt - mpn,
            "topk_hit_true": hit, "topk_hit_null": null_hit,
            "lift_topk": hit - null_hit}


# ------------------------------------------------------------------- evaluation
def decode_scores(config, score, min_score, radius, max_peaks, cache=D.CACHE):
    """Peaks -> original coordinates (via the exact coordinate maps)."""
    _, _, maps = DEC.truth_for(config, cache)
    L = json.load(open(os.path.join(cache, config, "meta.json")))["genome_length"]
    peaks = pick_peaks(score, min_score, radius, max_peaks)
    return sorted({P.unshrink(P.normalise_boundary(b, maps), maps, L) for b in peaks}), peaks


def event_level(configs, score_fn, min_score, radius, max_peaks, tol, reps,
                cache=D.CACHE):
    rows = []
    n_pred_cfg = {}
    for c in configs:
        score = score_fn(c)
        pred, peaks = decode_scores(c, score, min_score, radius, max_peaks, cache)
        gt, _, _ = DEC.truth_for(c, cache)
        tp, fp, fn = DEC.match(pred, gt, tol)
        rows.append({"config": c, "n_gt": len(gt), "n_pred": len(pred),
                     "tp": tp, "fp": fp, "fn": fn})
        n_pred_cfg[c] = len(pred)
    return rows, n_pred_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "decode"], default="oracle")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--thr", type=float, default=0.9, help="thrL = thrR")
    ap.add_argument("--thrs", nargs="*", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--margins", nargs="*", type=float, default=None)
    ap.add_argument("--min-seg", type=int, default=3)
    ap.add_argument("--min-segs", nargs="*", type=int, default=None)
    ap.add_argument("--kinds", nargs="*", default=None,
                    help="oracle score families: hard gain soft")
    ap.add_argument("--betas", nargs="*", type=float, default=None,
                    help="softmax sharpness for the soft (marginalised) score")
    ap.add_argument("--scores", nargs="*", default=None,
                    help="decode score families: pair hard gain soft")
    ap.add_argument("--beta", type=float, default=20.0,
                    help="softmax sharpness for the soft score in decode mode")
    ap.add_argument("--min-score", type=float, default=1.0)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--max-peaks", type=int, default=200)
    ap.add_argument("--tol", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configs = args.datasets or D.all_configs()
    rng = np.random.default_rng(args.seed)
    S = lambda c: segment_switch(np.load(os.path.join(D.CACHE, c, "haplo.npy")),
                                 args.thr, args.thr, args.margin,
                                 args.min_seg)[0]
    PS = lambda c: pair_support_score(c)[0]

    if args.mode == "oracle":
        thrs = args.thrs or [args.thr]
        min_segs = args.min_segs or [args.min_seg]
        margins = args.margins or [args.margin]
        kinds = args.kinds or ["hard"]
        betas = args.betas or [20.0]
        specs = [("pair_support",)]
        for kind in kinds:
            if kind == "hard":
                for ms in min_segs:
                    for thr in thrs:
                        for mg in margins:
                            specs.append((f"hard_thr{thr}_seg{ms}_m{mg}",
                                          kind, ms, thr, mg, None))
            elif kind == "gain":
                for ms in min_segs:
                    specs.append((f"gain_seg{ms}", kind, ms, None, None, None))
            elif kind == "soft":
                for ms in min_segs:
                    for beta in betas:
                        specs.append((f"soft_beta{beta:g}_seg{ms}",
                                      kind, ms, None, None, beta))
            elif kind == "kl":
                for ms in min_segs:
                    for beta in betas:
                        specs.append((f"kl_beta{beta:g}_seg{ms}",
                                      kind, ms, None, None, beta))
            elif kind == "flank_kl":
                for ms in min_segs:
                    for beta in betas:
                        specs.append((f"flank_kl_beta{beta:g}_seg{ms}",
                                      kind, ms, None, None, beta))
        rows = []
        for c in configs:
            gt_tx = gt_boundaries_tx(c)
            if not gt_tx:
                continue
            H = np.load(os.path.join(D.CACHE, c, "haplo.npy"))
            rows.append(oracle(c, PS(c), "pair_support", gt_tx, rng))
            for label, *rest in specs[1:]:
                kind, ms, thr, mg, beta = rest
                if kind == "hard":
                    sc = segment_score(H, "hard", thr, thr, mg, min_seg=ms)
                elif kind == "gain":
                    sc = segment_score(H, "gain", min_seg=ms)
                else:
                    sc = segment_score(H, kind, beta=beta, min_seg=ms)
                rows.append(oracle(c, sc, label, gt_tx, rng))
            del H
            print(".", end="", flush=True)
        print()
        labels = [lab for lab, *_ in specs]
        print(f"\n{'score':>28} {'n_gt':>5} {'pct_true':>9} {'pct_null':>9} "
              f"{'lift_pct':>9} {'topk_pool':>10} {'topk_null':>10} {'lift':>9}")
        for label in labels:
            sub = [r for r in rows if r["score"] == label]
            if not sub:
                continue
            w = np.array([r["n_gt"] for r in sub], dtype=float)
            hit = np.array([r["topk_hit_true"] for r in sub], dtype=float)
            null = np.array([r["n_gt"] / r["M_tx"] for r in sub], dtype=float)
            pooled_true = float((hit * w).sum() / w.sum())
            pooled_null = float((null * w).sum() / w.sum())
            print(f"{label:>28} {int(w.sum()):>5} "
                  f"{np.mean([r['mean_pct_true'] for r in sub]):9.3f} "
                  f"{np.mean([r['mean_pct_null'] for r in sub]):9.3f} "
                  f"{np.mean([r['lift_pct'] for r in sub]):9.3f} "
                  f"{pooled_true:10.4f} {pooled_null:10.4f} "
                  f"{pooled_true - pooled_null:10.4f}")
        out = args.out or os.path.join(OUT, "haldec_oracle.csv")
        os.makedirs(OUT, exist_ok=True)
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                wr.writeheader(); wr.writerows(rows)
        print(f"wrote {out}")
        return 0

    # decode mode ------------------------------------------------------------
    scores = args.scores or ["pair", "hard", "soft", "kl"]
    _hcache = {}

    def _hap(c):
        if c not in _hcache:
            _hcache[c] = np.load(os.path.join(D.CACHE, c, "haplo.npy"))
        return _hcache[c]

    def make_fn(kind):
        ms = args.min_seg
        if kind == "pair":
            return "pair_support", PS
        if kind == "hard":
            return (f"switch_hard(thr={args.thr},seg={ms})",
                    lambda c: segment_score(_hap(c), "hard", args.thr, args.thr,
                                            args.margin, min_seg=ms))
        if kind == "gain":
            return (f"switch_gain(seg={ms})",
                    lambda c: segment_score(_hap(c), "gain", min_seg=ms))
        if kind == "soft":
            return (f"switch_soft(beta={args.beta},seg={ms})",
                    lambda c: segment_score(_hap(c), "soft", beta=args.beta,
                                            min_seg=ms))
        if kind == "kl":
            return (f"switch_kl(beta={args.beta},seg={ms})",
                    lambda c: segment_score(_hap(c), "kl", beta=args.beta,
                                            min_seg=ms))
        if kind == "flank_kl":
            return (f"flank_kl(beta={args.beta},seg={ms})",
                    lambda c: flank_kl_score(_hap(c), L_w=250, min_seg=ms,
                                             beta=args.beta, power=0.5))
        raise SystemExit(f"unknown score family: {kind}")

    for kind in scores:
        label, fn = make_fn(kind)
        rows, n_pred_cfg = event_level(configs, fn, args.min_score, args.radius,
                                       args.max_peaks, args.tol, args.reps)
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn_ = sum(r["fn"] for r in rows)
        prec, rec, f1 = DEC._prf(tp, fp, fn_)
        rtp, rfp, rfn = DEC.random_control(configs, n_pred_cfg, args.tol, args.reps)
        _, _, rf1 = DEC._prf(rtp, rfp, rfn)
        print(f"{label:>28}  tol={args.tol:>2}  TP={tp:>5} FP={fp:>6} FN={fn_:>5} "
              f"n_pred={sum(n_pred_cfg.values()):>6}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}  "
              f"(rand F1 {rf1:.3f}, {f1 / max(rf1, 1e-9):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
