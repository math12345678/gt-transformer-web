"""
The window transformer (S6 / M3 scaffold).

One training example is a query column.  Input is the circular pair-window
`(W, 4)` of gamete counts (self-pair at the centre); output is a single logit for
the boundary immediately to the left of that column.

This is deliberately small and explicit - a hand-rolled pre-norm encoder in pure
JAX, no Flax.  PyTorch is the obvious first choice for this kind of model, but
there is no torch wheel for the Python 3.15 interpreter in this environment
(`pip` finds nothing), so the scaffold is written against JAX + optax, which are
available.  The parameter tree is plain nested dicts/lists of arrays, so porting
to `torch.nn.Module` is mechanical if that is preferred.
"""
from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Config(NamedTuple):
    W: int = 314
    centre: int = 157          # index of offset 0 in the window
    n_in: int = 4              # gamete counts
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    d_ff: int = 256
    # BCE positive-class weight.  With the default balanced batches
    # (--pos-frac 0.5) this must stay at 1.0; it is only useful when sampling
    # naturally, where the base rate is ~1 in 476.
    pos_weight: float = 1.0


# ------------------------------------------------------------------ init
def _linear(key, din, dout, scale=None):
    scale = scale if scale is not None else 1.0 / (din ** 0.5)
    return {"w": jax.random.normal(key, (din, dout)) * scale,
            "b": jnp.zeros((dout,))}


def init_params(key, cfg):
    n_keys = 3 + 4 * cfg.n_layers
    keys = jax.random.split(key, n_keys)
    i = 0
    p = {}
    p["embed"] = _linear(keys[i], cfg.n_in, cfg.d_model)
    i += 1
    p["pos"] = jax.random.normal(keys[i], (cfg.W, cfg.d_model)) * 0.02
    i += 1
    layers = []
    for _ in range(cfg.n_layers):
        qkv = _linear(keys[i], cfg.d_model, 3 * cfg.d_model)
        i += 1
        attn_out = _linear(keys[i], cfg.d_model, cfg.d_model)
        i += 1
        ff1 = _linear(keys[i], cfg.d_model, cfg.d_ff)
        i += 1
        ff2 = _linear(keys[i], cfg.d_ff, cfg.d_model)
        i += 1
        layers.append({
            "ln1": {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))},
            "qkv": qkv,
            "attn_out": attn_out,
            "ln2": {"w": jnp.ones((cfg.d_model,)), "b": jnp.zeros((cfg.d_model,))},
            "ff1": ff1,
            "ff2": ff2,
        })
    p["layers"] = layers
    p["head"] = _linear(keys[i], cfg.d_model, 1)
    i += 1
    assert i <= n_keys, (i, n_keys)
    return p


# ------------------------------------------------------------------ forward
def _ln(x, p, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * p["w"] + p["b"]


def _lin(x, p):
    return x @ p["w"] + p["b"]


def _block(x, lp, cfg):
    h = _ln(x, lp["ln1"])
    B, W, D = h.shape
    H, dh = cfg.n_heads, D // cfg.n_heads
    q, k, v = jnp.split(_lin(h, lp["qkv"]), 3, axis=-1)

    def heads(t):
        return t.reshape(B, W, H, dh).transpose(0, 2, 1, 3)   # (B, H, W, dh)

    q, k, v = heads(q), heads(k), heads(v)
    att = jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(dh), -1)
    o = jnp.einsum("bhqk,bhkd->bhqd", att, v).transpose(0, 2, 1, 3)
    x = x + _lin(o.reshape(B, W, D), lp["attn_out"])
    h = _ln(x, lp["ln2"])
    h = jax.nn.gelu(_lin(h, lp["ff1"]))
    return x + _lin(h, lp["ff2"])


def forward(params, x, cfg):
    """x: (B, W, n_in) -> logits (B,) for the boundary left of each query column."""
    h = _lin(x, params["embed"]) + params["pos"][None, :, :]
    for lp in params["layers"]:
        h = _block(h, lp, cfg)
    return _lin(h[:, cfg.centre, :], params["head"]).squeeze(-1)


def bce_logits(logits, y):
    """Numerically stable BCE-with-logits, elementwise."""
    return jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def loss_fn(params, x, y, cfg):
    logits = forward(params, x, cfg)
    w = jnp.where(y > 0.5, cfg.pos_weight, 1.0)
    return (bce_logits(logits, y) * w).sum() / w.sum()


def n_params(params):
    return int(sum(a.size for a in jax.tree_util.tree_leaves(params)))
