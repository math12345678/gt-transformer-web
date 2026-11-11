"""
Training scaffold for the window transformer (S6 / M3).

    python3 train.py --smoke                      # overfit one dataset (fast)
    python3 train.py --smoke --dataset d003_L3074_s25_r005 --steps 500
    python3 train.py --train                      # train on the dataset split

Positives are ~1 in 476 boundaries, so each training batch is drawn with a fixed
positive fraction (`--pos-frac`, default 0.5) and the loss up-weights positives.
Evaluation is *not* oversampled - it runs on all positives plus a random sample
of real columns - so the reported AUPRC is honest.  Read AUPRC against the base
rate printed alongside it, not against 0.5.
"""
from __future__ import annotations
import os
import pickle
import argparse
import numpy as np

import jax
import jax.numpy as jnp
import optax

try:
    from . import data as D
    from . import model as M
except (ImportError, ValueError):
    import data as D
    import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")


# ------------------------------------------------------------------ metrics
def auprc(scores, labels):
    """Average precision (area under the precision-recall curve)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(float(y.sum()), 1e-12)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def best_f1(scores, labels):
    """Best achievable F1, its threshold, and the positive/negative mean logits."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    k = int(np.argmax(f1))
    above, below = s[order][:k + 1], s[order][k + 1:]
    return (float(f1[k]), float(s[order][k]),
            float(above.mean()) if len(above) else float("nan"),
            float(below.mean()) if len(below) else float("nan"))


# ------------------------------------------------------------------ batching
def sample_batch(ds, rng, batch_size, pos_frac, pos_idx=None, neg_idx=None):
    if pos_idx is None:
        pos_idx = np.flatnonzero(ds.y)
        neg_idx = np.flatnonzero(1 - ds.y)
    n_pos = int(round(batch_size * pos_frac))
    n_neg = batch_size - n_pos
    if len(pos_idx) == 0:
        pick = rng.choice(neg_idx, batch_size, replace=False)
    else:
        p = rng.choice(pos_idx, n_pos, replace=len(pos_idx) < n_pos)
        n = rng.choice(neg_idx, n_neg, replace=len(neg_idx) < n_neg)
        pick = np.concatenate([p, n])
    rng.shuffle(pick)
    return ds.batch(pick)


def make_step(cfg, lr):
    opt = optax.adamw(lr)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(M.loss_fn)(params, x, y, cfg)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step, opt.init


def make_score(cfg):
    @jax.jit
    def scores(params, x):
        return M.forward(params, x, cfg)

    return scores


def eval_scores(params, ds, score_fn, rng, n_neg=40000, chunk=512):
    """All positives + a random sample of negatives -> (logits, labels).

    Scored in chunks. The full eval set is ~40k windows of length 314; a single
    forward pass materialises (40k, 314, 4) and runs 40k x 314^2 attention, which
    OOMs / gets SIGKILLed. AUPRC is rank-based, so chunking changes nothing but
    memory and speed.
    """
    y = ds.y if hasattr(ds, "y") else np.concatenate([p.y for p in ds.parts])
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(1 - y)
    if len(neg) > n_neg:
        neg = rng.choice(neg, n_neg, replace=False)
    idx = np.concatenate([pos, neg])
    scores = np.empty(len(idx), dtype=np.float64)
    labels = np.empty(len(idx), dtype=np.float64)
    for s in range(0, len(idx), chunk):
        piece = idx[s:s + chunk]
        x, yy = ds.batch(piece)
        scores[s:s + chunk] = np.asarray(score_fn(params, x))
        labels[s:s + chunk] = np.asarray(yy)
    return scores, labels


# ------------------------------------------------------------------ modes
def smoke(args):
    ds = D.WindowDataset(args.dataset, mode=args.mode)
    cfg = M.Config(W=ds.W, centre=ds.centre, d_model=args.d_model,
                   n_layers=args.layers, d_ff=args.d_model * 2)
    pos_idx = np.flatnonzero(ds.y)
    neg_idx = np.flatnonzero(1 - ds.y)
    print(f"dataset {ds.config}: {ds.M} columns, window {ds.W}, "
          f"{len(pos_idx)} positives (base rate {len(pos_idx) / ds.M:.4f})")
    print(f"model    {M.n_params(M.init_params(jax.random.PRNGKey(0), cfg)):,} "
          f"parameters  d_model={cfg.d_model} layers={cfg.n_layers}")

    rng = np.random.default_rng(args.seed)
    params = M.init_params(jax.random.PRNGKey(args.seed), cfg)
    step, opt_init = make_step(cfg, args.lr)
    opt_state = opt_init(params)
    score = make_score(cfg)

    for it in range(1, args.steps + 1):
        x, y = sample_batch(ds, rng, args.batch, args.pos_frac, pos_idx, neg_idx)
        params, opt_state, loss = step(params, opt_state,
                                       jnp.asarray(x), jnp.asarray(y))
        if it == 1 or it % max(1, args.steps // 10) == 0:
            print(f"  step {it:>5}  loss {float(loss):.4f}")

    sc, yy = eval_scores(params, ds, score, rng)
    ap = auprc(sc, yy)
    f1, thr, mu_pos, mu_neg = best_f1(sc, yy)
    base = float(yy.mean())
    print("\noverfit sanity:")
    print(f"  AUPRC                {ap:.4f}   (base rate {base:.4f})")
    print(f"  best F1              {f1:.4f} at threshold {thr:.3f}")
    print(f"  mean logit pos / neg {mu_pos:.3f} / {mu_neg:.3f}")
    ok = ap > max(0.10, 3 * base)
    print(f"  verdict              {'LEARNING' if ok else 'NOT LEARNING'}")
    return 0 if ok else 1


def train(args):
    sp = D.splits(seed=args.seed, cache=args.cache, manifest_path=args.manifest)
    print(f"splits: train {len(sp['train'])}  val {len(sp['val'])}  "
          f"test {len(sp['test'])} datasets")
    tr = D.Panel(sp["train"], cache=args.cache, mode=args.mode)
    va = D.Panel(sp["val"], cache=args.cache, mode=args.mode)
    y = np.concatenate([p.y for p in tr.parts])
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(1 - y)
    print(f"train columns {len(tr):,}  positives {len(pos_idx):,} "
          f"(base rate {len(pos_idx) / len(tr):.5f})")

    cfg = M.Config(W=tr.W, centre=tr.W // 2,
                   d_model=args.d_model, n_layers=args.layers,
                   d_ff=args.d_model * 2, pos_weight=args.pos_weight)
    params = M.init_params(jax.random.PRNGKey(args.seed), cfg)
    print(f"model {M.n_params(params):,} parameters")
    step, opt_init = make_step(cfg, args.lr)
    opt_state = opt_init(params)
    score = make_score(cfg)
    rng = np.random.default_rng(args.seed)

    ckpt_dir = args.ckpt or CKPT
    os.makedirs(ckpt_dir, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        losses = []
        for _ in range(args.steps_per_epoch):
            x, yb = sample_batch(tr, rng, args.batch, args.pos_frac,
                                 pos_idx, neg_idx)
            params, opt_state, loss = step(params, opt_state,
                                           jnp.asarray(x), jnp.asarray(yb))
            losses.append(float(loss))
        sc, yv = eval_scores(params, va, score, rng)
        ap = auprc(sc, yv)
        f1, thr, _, _ = best_f1(sc, yv)
        print(f"epoch {epoch:>3}  loss {np.mean(losses):.4f}  "
              f"val AUPRC {ap:.4f}  best F1 {f1:.4f}  (base {yv.mean():.4f})",
              flush=True)
        if ap > best:
            best = ap
            with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as fh:
                pickle.dump({"params": params, "cfg": cfg._asdict(),
                             "val_auprc": ap, "epoch": epoch}, fh)
    print(f"best val AUPRC {best:.4f} -> {os.path.join(ckpt_dir, 'params.pkl')}")
    if sp["test"]:
        te = D.Panel(sp["test"], cache=args.cache, mode=args.mode)
        with open(os.path.join(ckpt_dir, "params.pkl"), "rb") as fh:
            best_params = pickle.load(fh)["params"]
        sc_te, y_te = eval_scores(best_params, te, score, rng)
        ap_te = auprc(sc_te, y_te)
        f1_te, thr_te, _, _ = best_f1(sc_te, y_te)
        print(f"held-out test ({len(sp['test'])} datasets): "
              f"test AUPRC {ap_te:.4f}  best F1 {f1_te:.4f}  (base {y_te.mean():.4f})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    ap.add_argument("--dataset", default="d013_L4584_s34_r005")
    ap.add_argument("--cache", default=None, help="Cache directory")
    ap.add_argument("--manifest", default=None, help="Path to batch_manifest_v2.json")
    ap.add_argument("--ckpt", default=None, help="Checkpoint output directory")
    ap.add_argument("--mode", default="freq", choices=["freq", "count"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-frac", type=float, default=0.5,
                    help="positive fraction per batch; keep --pos-weight at 1 "
                         "while this is balanced")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.train:
        return train(args)
    return smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
