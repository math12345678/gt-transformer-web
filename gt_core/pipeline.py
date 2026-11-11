"""
Reference implementation of the SantaSim-alignment -> binary column-pair embedding
pipeline for the ground-truth recombination-log transformer.

Stages
------
1. read alignment
2. filter columns: parsimony-informative sites only
       (>=2 distinct nucleotide states, every state present >=2 times)
3. binarise columns:
       - the state(s) that are NOT the two most common become '-' (null)
       - most common state -> '1', second most common -> '0'
       then apply a null policy:
         'drop'      : discard any column containing a null
         'zero'      : turn nulls into '0'
         'duplicate' : replace the column by TWO binary columns,
                       one with nulls->1 and one with nulls->0
4. coordinate bookkeeping: transformed column index <-> original alignment column
5. circular moving-window pairwise column embedding
       pair feature = counts of the four gametes (00,01,10,11) over the sequences
6. remap SantaSim ground-truth breakpoints into transformed coordinates

No indels exist in this panel, so an alignment column index equals the genome
coordinate; the bookkeeping is written explicitly so the pipeline keeps working
if an indel-enabled panel is used later.
"""
from __future__ import annotations
import os
import numpy as np

_ACGT = {b"A", b"C", b"G", b"T"}


# ---------------------------------------------------------------- 1. alignment
def read_fasta(path):
    names, seqs, cur = [], [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                names.append(line[1:])
                seqs.append([])
                cur = seqs[-1]
            else:
                cur.append(line.upper())
    return names, ["".join(s) for s in seqs]


# ------------------------------------------------- 2. informative-site filter
def informative_columns(seqs):
    """0-based column indices that are parsimony-informative:
    >=2 distinct nucleotide states and every present state occurs >=2 times."""
    n, L = len(seqs), len(seqs[0])
    mat = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(n, L)
    kept = []
    for c in range(L):
        vals, counts = np.unique(mat[:, c], return_counts=True)
        if not set(vals.tolist()) <= _ACGT:
            continue
        if len(vals) < 2 or counts.min() < 2:
            continue
        kept.append(c)
    return kept


# ---------------------------------------------------------------- 3. binarise
NULL_MODES = ("drop", "zero", "duplicate")


def binarise(seqs, kept, null_mode="duplicate"):
    """Return (cols, origin, kind).

    cols[k]   : list of '0'/'1' states, one per sequence (transformed column k)
    origin[k] : original alignment column that transformed column k came from
    kind[k]   : 'info' (no nulls present) or 'dup' (a null-filled duplicate)
    """
    if null_mode not in NULL_MODES:
        raise ValueError(null_mode)
    cols, origin, kind = [], [], []
    for c in kept:
        counts = {}
        for s in seqs:
            counts[s[c]] = counts.get(s[c], 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        major, minor = ordered[0][0], ordered[1][0]
        nulls = {k for k, _ in ordered[2:]}
        raw = ["-" if s[c] in nulls else ("1" if s[c] == major else "0")
               for s in seqs]
        has_null = "-" in raw
        if null_mode == "drop":
            if not has_null:
                cols.append(raw); origin.append(c); kind.append("info")
        elif null_mode == "zero":
            cols.append(["0" if x == "-" else x for x in raw])
            origin.append(c); kind.append("info")
        else:  # duplicate
            if has_null:
                cols.append(["1" if x == "-" else x for x in raw])
                origin.append(c); kind.append("dup")
                cols.append(["0" if x == "-" else x for x in raw])
                origin.append(c); kind.append("dup")
            else:
                cols.append(raw); origin.append(c); kind.append("info")
    return cols, origin, kind


# ------------------------------------------------- 4. coordinate bookkeeping
def build_maps(origin, n_original):
    """origin[k] = original column of transformed column k.

    Returns:
      tx_to_orig : list, transformed col -> original col
      orig_to_tx : dict, original col -> [transformed cols] (filtered -> [])
      boundary   : array length n_original+1; boundary[p] = number of
                   transformed columns whose original column is < p.  This is an
                   insertion point in [0, M_tx] and is the robust way to carry a
                   breakpoint coordinate across filtering/duplication.
    """
    prefix = np.zeros(n_original + 1, dtype=np.int64)
    for c in origin:
        prefix[c + 1] += 1
    boundary = np.cumsum(prefix)
    orig_to_tx = {}
    for k, c in enumerate(origin):
        orig_to_tx.setdefault(c, []).append(k)
    return {"tx_to_orig": list(origin),
            "orig_to_tx": orig_to_tx,
            "boundary": boundary}


# ------------------------------------------------------- 5. pairwise embedding
def _offsets(W):
    """Circular offsets with the self-pair (offset 0) as close to the centre as
    possible.  For odd W the self-pair sits exactly at the centre."""
    if W % 2 == 1:
        h = W // 2
        return list(range(-h, h + 1))
    return list(range(-(W // 2), W // 2))            # self at index W//2


def circular_pair_embedding(binary_cols, window=None):
    """binary_cols: list of M columns, each a list of '0'/'1' per sequence.

    Returns F (M, W, 4) and offsets (length W).
    F[i, pos, g] = number of sequences whose (column i, column i+offset) pair
    equals gamete g, where gamete g encodes (state_i, state_j):
        g = 2*state_i + state_j   ->  0=00, 1=01, 2=10, 3=11
    The window wraps circularly; W = M unless a smaller `window` is given.
    """
    n_seq = len(binary_cols[0])
    n_col = len(binary_cols)
    B = np.array([[int(binary_cols[c][s]) for c in range(n_col)]
                  for s in range(n_seq)], dtype=np.int8)   # (S, M)
    S, M = B.shape
    W = M if (window is None or window > M) else window
    offs = _offsets(W)
    idx = np.arange(M)
    F = np.zeros((M, len(offs), 4), dtype=np.int32)
    for pos, d in enumerate(offs):
        j = (idx + d) % M
        code = 2 * B[:, idx] + B[:, j]                # (S, M) in 0..3
        for g in range(4):
            F[:, pos, g] = (code == g).sum(axis=0)
    return F, offs


def four_gamete_mask(F):
    """Boolean (M, W): True where all four gametes are present for that pair."""
    return (F > 0).all(axis=2)


# ------------------------------------------- 6. ground-truth coordinate remap
def remap_breakpoints(breakpoints, boundary_map):
    """Map original 0-based breakpoint positions to transformed insertion points."""
    b = boundary_map
    n = len(b) - 1
    out = []
    for p in breakpoints:
        p = int(p)
        p = 0 if p < 0 else (n if p > n else p)
        out.append(int(b[p]))
    return out


def unshrink(boundary_index, maps, n_original):
    """Inverse of the insertion-point map.

    Given a transformed-coordinate boundary index (0..M_tx), return the original
    alignment coordinate of the first transformed column at or after it.

    Caveat: under the 'duplicate' null policy two adjacent transformed columns
    share one original column, so a boundary *between* them has no exact original
    counterpart.  The exact round-trip statement is therefore
        shrink(unshrink(b)) == normalise(b)
    where `normalise` collapses b to the first boundary of its origin-group
    (see `normalise_boundary`).  For the 'drop' and 'zero' policies the origin
    columns are unique and `normalise` is the identity.
    """
    b = int(boundary_index)
    tx_to_orig = maps["tx_to_orig"]
    if b < 0:
        b = 0
    if b >= len(tx_to_orig):
        return int(n_original)
    return int(tx_to_orig[b])


def normalise_boundary(boundary_index, maps):
    """Collapse a boundary index to the first boundary of its origin-group.

    Boundaries strictly inside a duplicated pair are degenerate (they do not
    correspond to a distinct original coordinate) and decode to the same position
    as the start of the group.  This is the idempotent 'unshrink-then-shrink'
    projection used when turning predictions back into original coordinates.
    """
    b = int(boundary_index)
    tx_to_orig = maps["tx_to_orig"]
    if b <= 0:
        return 0
    if b >= len(tx_to_orig):
        return len(tx_to_orig)
    c = tx_to_orig[b]
    # walk back over any preceding columns sharing this original column
    while b > 0 and tx_to_orig[b - 1] == c:
        b -= 1
    return b


def shrink(orig_pos, maps):
    """Forward insertion-point map: original coordinate -> transformed boundary."""
    b = maps["boundary"]
    n = len(b) - 1
    p = int(orig_pos)
    p = 0 if p < 0 else (n if p > n else p)
    return int(b[p])


def genome_length_from_name(config_name):
    """d013_L4584_s34_r005 -> 4584."""
    try:
        return int(config_name.split("_L")[1].split("_")[0])
    except (IndexError, ValueError):
        return None


def strip_anchor(breakpoints, genome_length):
    """Split a logged breakpoint set into (real breakpoints, anchor).

    SantaSim's sampler appends a SYNTHETIC anchor to single-crossover events
    (AlignmentSampler.java:711-721): when the event has exactly one homologous
    breakpoint it adds 0 if that point lies in the first half of the genome, or
    `genome_length` if it lies in the second half.  The anchor marks the side the
    recombined segment reaches; it is *not* a recombination boundary, carries no
    alignment signal, and must not become a training label.

    Real (drawn) breakpoints lie in [1, min_length-4] and are multiples of 3, so
    an anchor can be recognised exactly: it equals 0 or the genome length.
    Returns (real_breakpoints, "start" | "end" | None).
    """
    if 0 in breakpoints:
        return [b for b in breakpoints if b != 0], "start"
    if genome_length is not None and genome_length in breakpoints:
        return [b for b in breakpoints if b != genome_length], "end"
    return list(breakpoints), None


def gt_events(config_name, gt_dir, genome_length=None):
    """Full parse of a recombination_events file.

    Returns a list of dicts:
      event_id, generation,
      logged_breakpoints : exactly as written by SantaSim (may include an anchor)
      real_breakpoints   : anchors removed, i.e. the actual crossover points
      anchor             : 'start' | 'end' | None
      event_type         : 'single' (1 real breakpoint) | 'double' (2)
    """
    path = os.path.join(gt_dir, f"recombination_events_{config_name}.txt")
    events = []
    if not os.path.exists(path):
        return events
    if genome_length is None:
        genome_length = genome_length_from_name(config_name)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "EventNum" in line:
                continue
            parts = line.split("*")
            if len(parts) < 3:
                continue
            logged = [int(x) for x in parts[1].strip("[]").split(",")
                      if x.strip().lstrip("-").isdigit()]
            real, anchor = strip_anchor(logged, genome_length)
            events.append({"event_id": int(parts[0]),
                           "generation": int(parts[2]),
                           "logged_breakpoints": logged,
                           "real_breakpoints": real,
                           "anchor": anchor,
                           "event_type": "single" if anchor else "double"})
    return events


def gt_breakpoints(config_name, gt_dir, real=True):
    """List of breakpoint lists.  `real=True` drops the synthetic anchors."""
    key = "real_breakpoints" if real else "logged_breakpoints"
    return [e[key] for e in gt_events(config_name, gt_dir)]


def gt_sequence_events(config_name, gt_dir):
    """Parse sequence_events_map: {sequence_index (1-based) -> [event_id, ...]}."""
    path = os.path.join(gt_dir, f"sequence_events_map_{config_name}.txt")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Sequence"):
                continue
            seq, rest = line.split("*", 1)
            ids = [int(x) for x in rest.strip("[]").split(",") if x.strip()]
            out[int(seq)] = ids
    return out


# --------------------------------------------------------------------- helper
def prepare(path, null_mode="duplicate", window=None):
    names, seqs = read_fasta(path)
    kept = informative_columns(seqs)
    cols, origin, kind = binarise(seqs, kept, null_mode=null_mode)
    maps = build_maps(origin, len(seqs[0]))
    out = {"names": names, "seqs": seqs, "kept": kept, "cols": cols,
           "origin": origin, "kind": kind, "maps": maps}
    if window is not None:
        F, offs = circular_pair_embedding(cols, window=window)
        out["F"], out["offsets"] = F, offs
    return out
