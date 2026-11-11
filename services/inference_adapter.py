"""
Live GT-Transformer Inference Engine Adapter.

Connects the FastAPI web application directly to the real `gt-transformer`
pipeline, JAX Window Transformer checkpoints, local flanking KL localiser
(`flank_kl`), permutation-null calibrator, and 3-level event consolidation:
  - `pipeline.py`        : informative_columns, binarise, build_maps, unshrink, normalise_boundary
  - `model.py`           : Config, forward, n_params
  - `train.py`           : make_score
  - `haldec.py`          : boundary_zscore_flank, boundary_zscore, flank_soft_switch_detail,
                           soft_switch_detail, pick_peaks
  - `report.py`          : consolidate_peaks
  - `event_eval_orig.py` : load_truth_orig
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Wire parent gt-transformer infrastructure directory onto sys.path
GT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = GT_ROOT.parents[1]
if str(GT_ROOT) not in sys.path:
    sys.path.insert(0, str(GT_ROOT))

import haldec as H  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402
import report as REP  # noqa: E402
import train as T  # noqa: E402

from schemas import (  # noqa: E402
    AlignmentMetadata,
    ConsolidatedEvent,
    GenomeTrackProfile,
    InferenceConfig,
    InferenceResponse,
    ParentalCandidate,
    RawPeakCall,
    RecombinantCall,
)
from services.diagnostics import evaluate_predictions_against_ground_truth  # noqa: E402
from services.fasta_utils import (  # noqa: E402
    compute_alignment_base_stats,
    parse_fasta_text,
    parse_ground_truth_payload,
)
from services.timing import TimingHistoryStore, TimingRecorder  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint Registry (Real Saved Pickle Checkpoints)
# ---------------------------------------------------------------------------
CHECKPOINT_REGISTRY: dict[str, dict[str, Any]] = {
    "ckpt_N10000_extreme": {
        "id": "ckpt_N10000_extreme",
        "path": GT_ROOT / "ckpt_N10000_extreme" / "params.pkl",
        "label": "ckpt_N10000_extreme — Latest Multi-Rate (7k In-Range + Extreme)",
        "description": (
            "Latest checkpoint trained jointly on 7,000 in-range alignments and extreme "
            "mutation/recombination arms (0.2x–5.0x rates). 0.0000 AUPRC on extreme clonal controls."
        ),
        "recommended": True,
    },
    "ckpt_N10000": {
        "id": "ckpt_N10000",
        "path": GT_ROOT / "ckpt_N10000" / "params.pkl",
        "label": "ckpt_N10000 — 10k In-Range Scaling Model (val AUPRC 0.572)",
        "description": (
            "Trained on 7,000 in-range SantaSim alignments (10,000 corpus; 1,500 held-out test, "
            "test AUPRC 0.366, best F1 0.435)."
        ),
        "recommended": False,
    },
    "ckpt_N8000": {
        "id": "ckpt_N8000",
        "path": GT_ROOT / "ckpt_N8000" / "params.pkl",
        "label": "ckpt_N8000 — 8k Scaling Checkpoint (val AUPRC 0.539)",
        "description": "Trained on 5,600 in-range alignments from the 8k scaling tier.",
        "recommended": False,
    },
    "ckpt_N4000": {
        "id": "ckpt_N4000",
        "path": GT_ROOT / "ckpt_N4000" / "params.pkl",
        "label": "ckpt_N4000 — 4k Scaling Checkpoint (val AUPRC 0.362)",
        "description": "Intermediate scaling checkpoint trained on 2,800 alignments.",
        "recommended": False,
    },
    "ckpt_N2000": {
        "id": "ckpt_N2000",
        "path": GT_ROOT / "ckpt_N2000" / "params.pkl",
        "label": "ckpt_N2000 — 2k Scaling Checkpoint (val AUPRC 0.195)",
        "description": "Early scaling checkpoint trained on 1,400 alignments.",
        "recommended": False,
    },
    "ckpt_N1000": {
        "id": "ckpt_N1000",
        "path": GT_ROOT / "ckpt_N1000" / "params.pkl",
        "label": "ckpt_N1000 — 1k Scaling Checkpoint (val AUPRC 0.101)",
        "description": "1,000-alignment baseline scaling tier (700 training alignments).",
        "recommended": False,
    },
    "ckpt_4x_panel": {
        "id": "ckpt_4x_panel",
        "path": GT_ROOT / "ckpt" / "model_4x_data.pkl",
        "label": "ckpt_4x_panel — 50-Dataset Panel (4x Augmented Regime)",
        "description": "Original 2,000-step Window Transformer trained on the 35-alignment panel.",
        "recommended": False,
    },
}


# ---------------------------------------------------------------------------
# Preset Benchmark Datasets (Backed Directly by Real FASTA & Cache Files)
# ---------------------------------------------------------------------------
PRESET_REGISTRY: dict[str, dict[str, Any]] = {
    "d005_L3536_s96_r001": {
        "id": "d005_L3536_s96_r001",
        "label": "d005_L3536_s96_r001 — Benchmark Diagnostic (S=96, L=3,536 bp, 13 GT bps)",
        "category": "Benchmark Panel (Annotated by Dr. Martin)",
        "description": (
            "96 sequences × 3,536 bp SantaSim alignment with 10 events (13 real breakpoints). "
            "Tests edge events (78 bp, 3399 bp) and internal Event 4 (2523 bp)."
        ),
        "fasta_path": WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d005_L3536_s96_r001.fas",
        "cache_dir": GT_ROOT / "cache",
        "is_control": False,
    },
    "d048_L4577_s79_r001": {
        "id": "d048_L4577_s79_r001",
        "label": "d048_L4577_s79_r001 — Held-Out Benchmark Test (S=79, L=4,577 bp, 9 GT bps)",
        "category": "Benchmark Panel (Held-Out Test Split)",
        "description": "Unseen held-out benchmark alignment (79 sequences × 4,577 bp, r=0.001, 6 events / 9 real bps).",
        "fasta_path": WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d048_L4577_s79_r001.fas",
        "cache_dir": GT_ROOT / "cache",
        "is_control": False,
    },
    "d008_L8383_s79_r001": {
        "id": "d008_L8383_s79_r001",
        "label": "d008_L8383_s79_r001 — Held-Out Benchmark Test (S=79, L=8,383 bp, 8 GT bps)",
        "category": "Benchmark Panel (Held-Out Test Split)",
        "description": "Unseen held-out benchmark alignment (79 sequences × 8,383 bp, r=0.001, 6 events / 8 real bps).",
        "fasta_path": WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d008_L8383_s79_r001.fas",
        "cache_dir": GT_ROOT / "cache",
        "is_control": False,
    },
    "d047_L15068_s27_ctrl": {
        "id": "d047_L15068_s27_ctrl",
        "label": "d047_L15068_s27_ctrl — Held-Out Clonal Control (S=27, L=15,068 bp, 0 GT)",
        "category": "Benchmark Panel (Held-Out Negative Control)",
        "description": "Strictly clonal held-out benchmark control alignment with 0 recombination events (specificity test).",
        "fasta_path": WORKSPACE_ROOT / "WINDOWS_RDP5_PACKAGE" / "01_ALIGNMENTS" / "d047_L15068_s27_ctrl.fas",
        "cache_dir": GT_ROOT / "cache",
        "is_control": True,
    },
    "d00003_L2487_s28_r005": {
        "id": "d00003_L2487_s28_r005",
        "label": "d00003_L2487_s28_r005 — 10k Corpus Held-Out Test (S=28, L=2,487 bp, r=0.005)",
        "category": "10,000-Alignment Corpus (Held-Out Test Split)",
        "description": "Unseen high-recombination test alignment from the 10k simulation corpus (94 events).",
        "fasta_path": GT_ROOT / "datasets_sims" / "d00003_L2487_s28_r005.fas",
        "cache_dir": GT_ROOT / "cache_sims",
        "is_control": False,
    },
    "d00011_L9812_s43_r0005": {
        "id": "d00011_L9812_s43_r0005",
        "label": "d00011_L9812_s43_r0005 — 10k Corpus Sparse Test (S=43, L=9,812 bp, 2 GT bps)",
        "category": "10,000-Alignment Corpus (Held-Out Test Split)",
        "description": "Unseen sparse-recombination test alignment from the 10k corpus (2 events / 2 real breakpoints).",
        "fasta_path": GT_ROOT / "datasets_sims" / "d00011_L9812_s43_r0005.fas",
        "cache_dir": GT_ROOT / "cache_sims",
        "is_control": False,
    },
    "d00005_L7268_s23_ctrl": {
        "id": "d00005_L7268_s23_ctrl",
        "label": "d00005_L7268_s23_ctrl — 10k Corpus Clonal Control (S=23, L=7,268 bp, 0 GT)",
        "category": "10,000-Alignment Corpus (Held-Out Negative Control)",
        "description": "Unseen clonal negative control from the 10k corpus (0 true events; tests false-positive rate).",
        "fasta_path": GT_ROOT / "datasets_sims" / "d00005_L7268_s23_ctrl.fas",
        "cache_dir": GT_ROOT / "cache_sims",
        "is_control": True,
    },
    "ext_0076_L4413_s95_high_mu_high_r_mu25_r025": {
        "id": "ext_0076_L4413_s95_high_mu_high_r_mu25_r025",
        "label": "ext_0076 — Extreme 5x Mu + 5x Recomb (S=95, L=4,413 bp, mu=2.5e-3, r=0.25)",
        "category": "Out-of-Range Extreme Suite (500% Rate Arm)",
        "description": "Extreme out-of-range alignment at 5x mutation and 5x recombination rates.",
        "fasta_path": GT_ROOT / "datasets_extreme" / "ext_0076_L4413_s95_high_mu_high_r_mu25_r025.fas",
        "cache_dir": GT_ROOT / "cache_extreme",
        "is_control": False,
    },
    "ext_0001_L6809_s57_low_mu_low_r_mu01_r0001": {
        "id": "ext_0001_L6809_s57_low_mu_low_r_mu01_r0001",
        "label": "ext_0001 — Extreme 0.2x Mu + 0.2x Recomb (S=57, L=6,809 bp, mu=1e-4, r=0.001)",
        "category": "Out-of-Range Extreme Suite (20% Rate Arm)",
        "description": "Extreme low-diversity, low-recombination alignment (20% rates).",
        "fasta_path": GT_ROOT / "datasets_extreme" / "ext_0001_L6809_s57_low_mu_low_r_mu01_r0001.fas",
        "cache_dir": GT_ROOT / "cache_extreme",
        "is_control": False,
    },
    "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl": {
        "id": "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl",
        "label": "ext_0101 — Extreme Low-Mu Clonal Control (S=27, L=3,291 bp, 0 GT)",
        "category": "Out-of-Range Extreme Suite (Negative Control)",
        "description": "Extreme low-mutation clonal control (r=0.0, 0 true breakpoints).",
        "fasta_path": GT_ROOT / "datasets_extreme" / "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl.fas",
        "cache_dir": GT_ROOT / "cache_extreme",
        "is_control": True,
    },
    "sarbecovirus_aligned": {
        "id": "sarbecovirus_aligned",
        "label": "sarbecovirus_aligned — Empirical Coronavirus Alignment (S=19, L=30,619 bp)",
        "category": "Empirical Viral Alignment (Unlabelled)",
        "description": "19 empirical aligned Sarbecovirus genomes (30,619 bp) with gaps/ambiguities.",
        "fasta_path": WORKSPACE_ROOT / "sarbecovirus_aligned.fasta",
        "cache_dir": None,
        "is_control": False,
    },
}


class BaseInferenceEngine(ABC):
    """Abstract interface for recombination inference engines."""

    @abstractmethod
    def list_presets(self) -> list[dict[str, Any]]:
        """Return available preset benchmark datasets."""

    @abstractmethod
    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return available trained checkpoints on disk."""

    @abstractmethod
    def run_inference(
        self,
        config: InferenceConfig,
        preset_id: Optional[str] = None,
        fasta_text: Optional[str] = None,
        fasta_filename: Optional[str] = None,
        gt_text: Optional[str] = None,
        gt_filename: Optional[str] = None,
    ) -> InferenceResponse:
        """Execute inference and return real timings + 3-level report + diagnostics."""


class GTTransformerInferenceEngine(BaseInferenceEngine):
    """Live GT-Transformer inference engine wired directly to model & pipeline infrastructure."""

    CHUNK_SIZE = 512

    def __init__(self, timing_store: TimingHistoryStore) -> None:
        self.timing_store = timing_store
        self._ckpt_cache: dict[str, tuple[Any, M.Config, int]] = {}
        self._score_fn_cache: dict[M.Config, Any] = {}
        self._warmed_configs: set[M.Config] = set()
        self._preset_meta_cache: Optional[list[dict[str, Any]]] = None
        self._alignment_cache: dict[str, dict[str, Any]] = {}
        self._alignment_cache_order: list[str] = []

    def get_alignment_window(
        self, run_id: str, start_bp: int = 0, width: int = 120
    ) -> dict[str, Any]:
        """Return actual nucleotide alignment slice & informative site coordinates for the RDP5 viewer."""
        entry = self._alignment_cache.get(run_id)
        if entry is None:
            raise ValueError(f"Alignment session '{run_id}' not found or expired; re-run inference.")
        L = int(entry["L"])
        width = max(20, min(int(width), 300))
        start = max(0, min(int(start_bp), max(0, L - 1)))
        end = min(L, start + width)
        if end - start < width and L > width:
            start = max(0, L - width)
            end = L

        kept_set: set[int] = entry["kept_set"]
        info_cols = [c for c in range(start, end) if c in kept_set]
        names: list[str] = entry["names"]
        seqs: list[str] = entry["seqs"]
        seq_rows = [
            {
                "seq_idx": i,
                "name": names[i],
                "slice": seqs[i][start:end],
            }
            for i in range(len(seqs))
        ]
        return {
            "run_id": run_id,
            "start_bp": start,
            "end_bp": end,
            "genome_length_bp": L,
            "informative_cols": info_cols,
            "sequences": seq_rows,
        }

    def list_checkpoints(self) -> list[dict[str, Any]]:
        out = []
        for ckpt_id, meta in CHECKPOINT_REGISTRY.items():
            p = Path(meta["path"])
            if p.exists():
                out.append(
                    {
                        "id": ckpt_id,
                        "label": meta["label"],
                        "description": meta["description"],
                        "recommended": meta["recommended"],
                        "path": str(p),
                    }
                )
        return out

    def list_presets(self) -> list[dict[str, Any]]:
        if self._preset_meta_cache is not None:
            return self._preset_meta_cache

        out = []
        for pid, pmeta in PRESET_REGISTRY.items():
            fpath = Path(pmeta["fasta_path"])
            if not fpath.exists():
                continue
            cdir = pmeta.get("cache_dir")
            n_seq = None
            length = None
            n_info = None
            n_gt = 0
            if cdir is not None and (Path(cdir) / pid / "meta.json").exists():
                try:
                    m = json.loads((Path(cdir) / pid / "meta.json").read_text(encoding="utf-8"))
                    n_seq = m.get("n_sequences")
                    length = m.get("genome_length")
                    n_info = m.get("informative_columns")
                    evs = json.loads((Path(cdir) / pid / "events.json").read_text(encoding="utf-8"))
                    n_gt = len({int(b) for e in evs for b in e.get("breakpoints_orig", [])})
                except Exception:
                    pass
            if n_seq is None:
                try:
                    names, seqs = P.read_fasta(str(fpath))
                    n_seq = len(names)
                    length = max(len(s) for s in seqs)
                except Exception:
                    n_seq, length = 0, 0
            out.append(
                {
                    "id": pid,
                    "label": pmeta["label"],
                    "category": pmeta["category"],
                    "description": pmeta["description"],
                    "n_sequences": n_seq,
                    "genome_length_bp": length,
                    "n_informative_cols": n_info,
                    "n_ground_truth_breakpoints": n_gt,
                    "is_control": bool(pmeta.get("is_control", False)),
                }
            )
        self._preset_meta_cache = out
        return out

    def warmup(self, checkpoint_id: str = "ckpt_N10000_extreme") -> None:
        """Load the default checkpoint and pre-compile the static-shape JAX score function."""
        if checkpoint_id not in CHECKPOINT_REGISTRY or not Path(CHECKPOINT_REGISTRY[checkpoint_id]["path"]).exists():
            avail = self.list_checkpoints()
            if not avail:
                return
            checkpoint_id = avail[0]["id"]
        params, cfg, _ = self._load_checkpoint(checkpoint_id)
        self._get_score_fn(params, cfg)

    def _load_checkpoint(self, checkpoint_id: str) -> tuple[Any, M.Config, int]:
        if checkpoint_id in self._ckpt_cache:
            return self._ckpt_cache[checkpoint_id]

        if checkpoint_id not in CHECKPOINT_REGISTRY:
            raise ValueError(f"Unknown checkpoint '{checkpoint_id}'.")
        ckpt_path = Path(CHECKPOINT_REGISTRY[checkpoint_id]["path"])
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint file not found on disk: {ckpt_path}")

        with open(ckpt_path, "rb") as fh:
            payload = pickle.load(fh)
        params = payload["params"]
        cfg_dict = payload.get("cfg", {})
        cfg = M.Config(**cfg_dict) if cfg_dict else M.Config()
        n_params = int(M.n_params(params))
        self._ckpt_cache[checkpoint_id] = (params, cfg, n_params)
        return params, cfg, n_params

    def _get_score_fn(self, params: Any, cfg: M.Config) -> Any:
        if cfg not in self._score_fn_cache:
            self._score_fn_cache[cfg] = T.make_score(cfg)
        score_fn = self._score_fn_cache[cfg]
        if cfg not in self._warmed_configs:
            dummy = np.zeros((self.CHUNK_SIZE, cfg.W, cfg.n_in), dtype=np.float32)
            out = score_fn(params, dummy)
            out.block_until_ready()
            self._warmed_configs.add(cfg)
        return score_fn

    def _load_preset_ground_truth(self, preset_id: str, cache_dir: Optional[Path]) -> list[dict[str, Any]]:
        if cache_dir is None:
            return []
        ds_dir = Path(cache_dir) / preset_id
        events_path = ds_dir / "events.json"
        if not events_path.exists():
            return []

        events = json.loads(events_path.read_text(encoding="utf-8"))
        seq_event_path = ds_dir / "seq_event.npy"
        seq_ev = np.load(seq_event_path) if seq_event_path.exists() else None

        gt_items: list[dict[str, Any]] = []
        for idx, ev in enumerate(events):
            carriers: list[int] = []
            if seq_ev is not None and idx < seq_ev.shape[1]:
                carriers = [int(s) for s in np.flatnonzero(seq_ev[:, idx])]
            bps = ev.get("breakpoints_orig", [])
            txs = ev.get("boundaries_tx", [])
            for b_i, bp in enumerate(bps):
                gt_items.append(
                    {
                        "gt_bp_orig": int(bp),
                        "gt_boundary_tx": int(txs[b_i]) if b_i < len(txs) else None,
                        "gt_event_id": ev.get("event_id", idx + 1),
                        "event_type": str(ev.get("event_type", "unknown")),
                        "generation": ev.get("generation"),
                        "carrier_seqs": carriers,
                    }
                )
        return sorted(gt_items, key=lambda x: x["gt_bp_orig"])

    @staticmethod
    def _select_informative_and_binarise(
        seqs: list[str],
        null_mode: str,
        strict_min: int = 30,
        max_non_acgt_frac: float = 0.25,
    ) -> tuple[list[int], list[list[str]], list[int], str]:
        """Select parsimony-informative columns and binarise for 4-gamete embedding.

        Uses exact `pipeline.informative_columns` + `pipeline.binarise` whenever at
        least `strict_min` strict ACGT informative columns exist (preserving bit-for-bit
        parity with `pipeline.py`). For real-world FASTAs with scattered gaps or
        ambiguities (`-`, `N`, IUPAC) that leave fewer than `strict_min` gap-free sites,
        falls back to gap-tolerant ACGT informative columns (`<= max_non_acgt_frac`
        non-ACGT) and treats gaps/ambiguities as missing (`'-'`) under `null_mode`.
        """
        kept_strict = P.informative_columns(seqs)
        if len(kept_strict) >= strict_min:
            cols, origin, _kind = P.binarise(seqs, kept_strict, null_mode=null_mode)
            return kept_strict, cols, origin, "strict_acgt"

        acgt = ("A", "C", "G", "T")
        n_seq = len(seqs)
        genome_len = len(seqs[0])
        gap_kept: list[int] = []
        col_major_minor: list[tuple[str, str]] = []

        for c in range(genome_len):
            counts = {b: 0 for b in acgt}
            non_acgt = 0
            for s in seqs:
                ch = s[c]
                if ch in counts:
                    counts[ch] += 1
                else:
                    non_acgt += 1
            if non_acgt / max(n_seq, 1) > max_non_acgt_frac:
                continue
            ordered = sorted(
                ((b, cnt) for b, cnt in counts.items() if cnt > 0),
                key=lambda kv: (-kv[1], kv[0]),
            )
            if len(ordered) < 2 or ordered[1][1] < 2:
                continue
            gap_kept.append(c)
            col_major_minor.append((ordered[0][0], ordered[1][0]))

        if len(gap_kept) <= len(kept_strict):
            cols, origin, _kind = P.binarise(seqs, kept_strict, null_mode=null_mode)
            return kept_strict, cols, origin, "strict_acgt"

        if null_mode not in P.NULL_MODES:
            raise ValueError(null_mode)

        cols: list[list[str]] = []
        origin: list[int] = []
        for c, (major, minor) in zip(gap_kept, col_major_minor):
            raw = [
                "1" if s[c] == major else ("0" if s[c] == minor else "-")
                for s in seqs
            ]
            has_null = "-" in raw
            if null_mode == "drop":
                if not has_null:
                    cols.append(raw)
                    origin.append(c)
            elif null_mode == "zero":
                cols.append(["0" if x == "-" else x for x in raw])
                origin.append(c)
            else:  # duplicate
                if has_null:
                    cols.append(["1" if x == "-" else x for x in raw])
                    origin.append(c)
                    cols.append(["0" if x == "-" else x for x in raw])
                    origin.append(c)
                else:
                    cols.append(raw)
                    origin.append(c)
        return gap_kept, cols, origin, "gap_tolerant"

    @staticmethod
    def _build_circular_windows_freq(haplo: np.ndarray, window_size: int) -> np.ndarray:
        """Compute (M_tx, window_size, 4) normalized gamete frequency windows.

        Matches `pipeline.circular_pair_embedding(..., window=window_size)` / `S`,
        and handles `M_tx < window_size` cleanly via circular index wrapping.
        """
        M_tx, S = haplo.shape
        B = haplo.T.astype(np.int8)  # (S, M_tx)
        offs = P._offsets(window_size)
        idx = np.arange(M_tx)
        X = np.empty((M_tx, len(offs), 4), dtype=np.float32)
        inv_s = np.float32(1.0 / max(S, 1))
        for pos, d in enumerate(offs):
            j = (idx + d) % M_tx
            code = (B[:, idx] << 1) | B[:, j]  # (S, M_tx) in 0..3
            for g in range(4):
                X[:, pos, g] = (code == g).sum(axis=0, dtype=np.float32) * inv_s
        return X

    def _score_transformer_static(self, params: Any, cfg: M.Config, score_fn: Any, X: np.ndarray) -> np.ndarray:
        """Score every transformed column using static-shape chunked JAX forward passes."""
        M_tx = X.shape[0]
        scores = np.empty(M_tx, dtype=np.float32)
        buf = np.zeros((self.CHUNK_SIZE, cfg.W, cfg.n_in), dtype=np.float32)
        for s in range(0, M_tx, self.CHUNK_SIZE):
            e = min(s + self.CHUNK_SIZE, M_tx)
            n = e - s
            if n == self.CHUNK_SIZE:
                out = score_fn(params, X[s:e])
            else:
                buf[:n] = X[s:e]
                if n < self.CHUNK_SIZE:
                    buf[n:] = 0.0
                out = score_fn(params, buf)
            out.block_until_ready()
            scores[s:e] = np.asarray(out)[:n]
        return scores

    def run_inference(
        self,
        config: InferenceConfig,
        preset_id: Optional[str] = None,
        fasta_text: Optional[str] = None,
        fasta_filename: Optional[str] = None,
        gt_text: Optional[str] = None,
        gt_filename: Optional[str] = None,
    ) -> InferenceResponse:
        # Load checkpoint upfront (cached after first load)
        params, cfg, n_model_params = self._load_checkpoint(config.checkpoint)
        score_fn = self._get_score_fn(params, cfg)

        # Resolve alignment input (uploaded FASTA vs preset FASTA file)
        is_known_control = False
        if fasta_text and fasta_text.strip():
            ds_name = (fasta_filename or "uploaded_alignment.fasta").rsplit(".", 1)[0]
            source_type = "upload"
            raw_fasta_str = fasta_text
            preset_cache_dir = None
        else:
            pid = preset_id if (preset_id in PRESET_REGISTRY) else "d005_L3536_s96_r001"
            pmeta = PRESET_REGISTRY[pid]
            fpath = Path(pmeta["fasta_path"])
            if not fpath.exists():
                raise ValueError(f"Preset alignment file not found on disk: {fpath}")
            ds_name = pid
            source_type = "preset"
            raw_fasta_str = fpath.read_text(encoding="utf-8", errors="replace")
            preset_cache_dir = pmeta.get("cache_dir")
            is_known_control = bool(pmeta.get("is_control", False))

        # We start an outer timing recorder once dimensions are known; measure stage 1 directly
        import time
        t_s1_start = time.perf_counter()
        names, seqs = parse_fasta_text(raw_fasta_str)
        S = len(seqs)
        L = len(seqs[0])
        if S < 4:
            raise ValueError(f"Alignment has only {S} sequences; at least 4 sequences are required.")

        t_s1_ms = (time.perf_counter() - t_s1_start) * 1000.0
        gc_pct, ambig_pct = compute_alignment_base_stats(seqs)

        # Stage 2: Informative-site selection (strict ACGT first, gap-tolerant fallback if <30),
        # binarisation, coordinate maps & haplotype matrix
        t_s2_start = time.perf_counter()
        kept, cols, origin, info_mode = self._select_informative_and_binarise(seqs, config.null_policy)
        if len(kept) < 10:
            raise ValueError(
                f"Alignment '{ds_name}' has only {len(kept)} parsimony-informative sites "
                "(>= 10 required after filtering heavily gapped columns and singletons)."
            )
        maps = P.build_maps(origin, L)
        M_tx = len(cols)
        if M_tx < 10:
            raise ValueError(
                f"Alignment '{ds_name}' has only {M_tx} columns after '{config.null_policy}' binarisation."
            )
        haplo = np.array([[1 if v == "1" else 0 for v in col] for col in cols], dtype=np.uint8)
        t_s2_ms = (time.perf_counter() - t_s2_start) * 1000.0

        recorder = TimingRecorder(
            dataset_name=ds_name,
            checkpoint=config.checkpoint,
            scoring_mode=config.scoring_mode,
            n_sequences=S,
            genome_length_bp=L,
            n_informative_cols=len(kept),
            n_transformed_cols=M_tx,
            window_size=cfg.W,
        )

        # Record already-measured Stages 1 & 2 cleanly via TimingRecorder
        info_desc = (
            "parsimony-informative ACGT sites"
            if info_mode == "strict_acgt"
            else "parsimony-informative sites (gap-tolerant fallback)"
        )
        recorder._stages_raw.append(
            {
                "stage_id": "parse_filter",
                "stage_name": "1. FASTA Parse & Alignment Validation",
                "duration_ms": round(t_s1_ms, 2),
                "items_processed": L,
                "items_unit": "bp",
                "throughput": round(L / max(t_s1_ms / 1000.0, 1e-6), 1),
                "throughput_unit": "bp/s",
                "notes": f"{S} sequences × {L:,} bp parsed (GC={gc_pct}%, non-ACGT={ambig_pct}%)",
            }
        )
        recorder._stages_raw.append(
            {
                "stage_id": "binarise_coords",
                "stage_name": "2. Informative Filter, Binarisation & Coordinate Maps",
                "duration_ms": round(t_s2_ms, 2),
                "items_processed": len(kept),
                "items_unit": "cols",
                "throughput": round(len(kept) / max(t_s2_ms / 1000.0, 1e-6), 1),
                "throughput_unit": "cols/s",
                "notes": f"{len(kept):,} {info_desc} → {M_tx:,} cols ({config.null_policy})",
            }
        )

        # Ground-truth resolution (uploaded GT overrides preset GT)
        if gt_text and gt_text.strip():
            gt_items = parse_ground_truth_payload(gt_text, gt_filename or "", L)
            has_known_gt = True
        elif preset_cache_dir is not None:
            gt_items = self._load_preset_ground_truth(ds_name, Path(preset_cache_dir))
            has_known_gt = is_known_control or len(gt_items) > 0
        else:
            gt_items = []
            has_known_gt = False

        # Stage 3: Circular Moving-Window Pairwise 4-Gamete Embedding (W=314)
        need_tf = config.scoring_mode in ("hybrid_flank", "standalone_tf")
        with recorder.measure_stage(
            stage_id="pairwise_embed",
            stage_name="3. Circular Window 4-Gamete Pairwise Embedding",
            items_processed=M_tx if need_tf else 0,
            items_unit="windows",
            throughput_unit="win/s",
            notes=(
                f"Circular gamete frequency tensor ({M_tx:,}, {cfg.W}, 4)"
                if need_tf
                else "Skipped (KL-only scoring mode)"
            ),
        ):
            X_embed = self._build_circular_windows_freq(haplo, cfg.W) if need_tf else None

        # Stage 4: Window Transformer Forward Pass (JAX JIT Encoder)
        with recorder.measure_stage(
            stage_id="transformer_fwd",
            stage_name="4. Window Transformer Forward Pass",
            items_processed=M_tx if need_tf else 0,
            items_unit="windows",
            throughput_unit="win/s",
            notes=(
                f"{config.checkpoint} ({n_model_params:,} params, d={cfg.d_model}, {cfg.n_layers} layers)"
                if need_tf
                else f"Skipped (scoring_mode={config.scoring_mode})"
            ),
        ):
            if need_tf and X_embed is not None:
                tf_raw = self._score_transformer_static(params, cfg, score_fn, X_embed)
                tf_logits_b = np.concatenate([tf_raw, [tf_raw[0]]]).astype(np.float64)
            else:
                tf_logits_b = np.zeros(M_tx + 1, dtype=np.float64)
            tf_prob_b = 1.0 / (1.0 + np.exp(-np.clip(tf_logits_b, -10.0, 10.0)))

        # Adaptive bounds for very sparse empirical alignments (e.g. M_tx < 250)
        eff_min_seg = min(config.min_seg, max(8, M_tx // 6))
        eff_L_w = min(config.flank_window, max(16, M_tx // 3))
        if config.scoring_mode == "global_kl":
            eff_min_seg_global = min(max(config.min_seg, 200), max(10, M_tx // 4))
        else:
            eff_min_seg_global = eff_min_seg

        # Stage 5: Local Flanking KL / Global KL & Permutation Null Calibration
        rng = np.random.default_rng(0)
        need_kl = config.scoring_mode in ("hybrid_flank", "flank_kl", "global_kl")
        with recorder.measure_stage(
            stage_id="flank_kl_null",
            stage_name="5. Flanking KL & Permutation Null Calibration",
            items_processed=(M_tx + 1) if need_kl else 0,
            items_unit="boundaries",
            throughput_unit="bound/s",
            notes=(
                f"Legacy global KL (min_seg={eff_min_seg_global}, beta={config.beta:g})"
                if config.scoring_mode == "global_kl"
                else (
                    f"Local flank_kl (Lw={eff_L_w}, min_seg={eff_min_seg}, beta={config.beta:g}, 2x col-perm null)"
                    if need_kl
                    else "Skipped global null scan (standalone_tf mode)"
                )
            ),
        ):
            if config.scoring_mode == "global_kl":
                z_kl, _raw_kl, _null_kl = H.boundary_zscore(
                    haplo, beta=config.beta, min_seg=eff_min_seg_global, kind="kl", rng=rng, n_perm=2
                )
            elif config.scoring_mode in ("hybrid_flank", "flank_kl"):
                z_kl, _raw_kl, _null_kl = H.boundary_zscore_flank(
                    haplo,
                    L_w=eff_L_w,
                    beta=config.beta,
                    min_seg=eff_min_seg,
                    power=0.5,
                    rng=rng,
                    n_perm=2,
                )
            else:
                z_kl = np.zeros(M_tx + 1, dtype=np.float64)

        # Stage 6: Peak Picking, Consolidation & 3-Level Carrier/Parent Attribution
        with recorder.measure_stage(
            stage_id="consolidate_3level",
            stage_name="6. Peak Consolidation & 3-Level Attribution",
            items_processed=0,
            items_unit="peaks",
            throughput_unit="peaks/s",
            notes="",
        ) as s6_meta:
            events, raw_peaks, track_profile = self._extract_and_consolidate_events(
                config=config,
                haplo=haplo,
                maps=maps,
                kept=kept,
                L=L,
                seq_names=names,
                z_kl=z_kl,
                tf_logits_b=tf_logits_b,
                tf_prob_b=tf_prob_b,
                eff_L_w=eff_L_w,
                eff_min_seg=eff_min_seg,
                eff_min_seg_global=eff_min_seg_global,
            )
            s6_meta["items_processed"] = max(1, len(raw_peaks))
            s6_meta["notes"] = (
                f"{len(raw_peaks)} raw peak(s) → {len(events)} consolidated event(s) "
                f"(merge_radius={config.merge_radius_bp} bp)"
            )

        timings = recorder.finalize()
        self.timing_store.record(timings)

        # Store alignment sequences in bounded LRU cache for RDP5 nucleotide alignment viewer
        self._alignment_cache[timings.run_id] = {
            "names": names,
            "seqs": seqs,
            "kept_set": set(int(c) for c in kept),
            "L": L,
        }
        self._alignment_cache_order.append(timings.run_id)
        while len(self._alignment_cache_order) > 8:
            old_id = self._alignment_cache_order.pop(0)
            self._alignment_cache.pop(old_id, None)

        diagnostics = evaluate_predictions_against_ground_truth(
            consolidated_events=events,
            gt_raw_items=gt_items,
            track_profile=track_profile,
            genome_length_bp=L,
            active_tolerance_bp=config.eval_tolerance_bp,
            is_known_control=is_known_control or (has_known_gt and len(gt_items) == 0),
        )

        return InferenceResponse(
            engine_mode="live",
            model_param_count=n_model_params,
            config=config,
            alignment=AlignmentMetadata(
                dataset_name=ds_name,
                source_type=source_type,  # type: ignore[arg-type]
                n_sequences=S,
                genome_length_bp=L,
                n_informative_cols=len(kept),
                n_transformed_cols=M_tx,
                window_size=cfg.W,
                null_policy=config.null_policy,
                sequence_names=names,
                gc_content_pct=gc_pct,
                ambiguity_pct=ambig_pct,
            ),
            timings=timings,
            consolidated_events=events,
            raw_peaks=raw_peaks,
            diagnostics=diagnostics,
            track_profile=track_profile,
        )

    def _extract_and_consolidate_events(
        self,
        config: InferenceConfig,
        haplo: np.ndarray,
        maps: dict[str, Any],
        kept: list[int],
        L: int,
        seq_names: list[str],
        z_kl: np.ndarray,
        tf_logits_b: np.ndarray,
        tf_prob_b: np.ndarray,
        eff_L_w: int,
        eff_min_seg: int,
        eff_min_seg_global: int,
    ) -> tuple[list[ConsolidatedEvent], list[RawPeakCall], GenomeTrackProfile]:
        """Pick peaks, run Level-2/3 switch attribution, consolidate via report.consolidate_peaks."""
        M_tx, S = haplo.shape
        hybrid_profile = np.maximum(z_kl, 0.0) + np.where(
            tf_logits_b > 0.5, 7.0 * (tf_logits_b - 0.5), 0.0
        )

        # Pick candidate peaks depending on scoring mode
        peak_sources: dict[int, str] = {}
        if config.scoring_mode == "flank_kl":
            kl_peaks = H.pick_peaks(z_kl, config.z_thresh, radius=20, max_peaks=64)
            for b in kl_peaks:
                peak_sources[int(b)] = "flank_kl"
            ordered_peaks = kl_peaks
            pick_by = "prod"
        elif config.scoring_mode == "global_kl":
            kl_peaks = H.pick_peaks(z_kl, config.z_thresh, radius=20, max_peaks=64)
            for b in kl_peaks:
                peak_sources[int(b)] = "global_kl"
            ordered_peaks = kl_peaks
            pick_by = "score"
        elif config.scoring_mode == "standalone_tf":
            tf_peaks = H.pick_peaks(tf_logits_b, config.tf_logit_thresh, radius=20, max_peaks=64)
            for b in tf_peaks:
                peak_sources[int(b)] = "transformer"
            ordered_peaks = tf_peaks
            pick_by = "score"
        else:
            # hybrid_flank: combine calibrated flank_kl peaks + non-overlapping confirmed TF peaks
            kl_peaks = H.pick_peaks(z_kl, config.z_thresh, radius=20, max_peaks=64)
            tf_peaks = H.pick_peaks(tf_logits_b, config.tf_logit_thresh, radius=20, max_peaks=64)
            for b in kl_peaks:
                peak_sources[int(b)] = "flank_kl"
            merged_peaks = list(kl_peaks)
            for b_tf in tf_peaks:
                if all(abs(int(b_tf) - int(b_k)) > 20 for b_k in merged_peaks):
                    if len(merged_peaks) < 64:
                        merged_peaks.append(int(b_tf))
                        peak_sources[int(b_tf)] = "transformer"
                else:
                    # Mark nearby KL peak as TF-corroborated
                    near_k = min(kl_peaks, key=lambda k: abs(int(k) - int(b_tf)))
                    peak_sources[int(near_k)] = "hybrid"
            ordered_peaks = sorted(merged_peaks)
            pick_by = "prod"

        raw_for_cons: list[dict[str, Any]] = []
        raw_peak_meta: dict[int, dict[str, Any]] = {}

        for b in ordered_peaks:
            b_int = int(b)
            src = peak_sources.get(b_int, config.scoring_mode)
            if config.scoring_mode == "global_kl":
                detail = H.soft_switch_detail(
                    haplo,
                    b_int,
                    beta=config.beta,
                    min_seg=eff_min_seg_global,
                    kind="kl",
                    top_members=20,
                    member_frac=0.5,
                    top_parents=5,
                )
                min_sw = 0.0
            else:
                detail = H.flank_soft_switch_detail(
                    haplo,
                    b_int,
                    L_w=eff_L_w,
                    beta=config.beta,
                    min_seg=eff_min_seg,
                    kind="kl",
                    top_members=20,
                    member_frac=0.5,
                    top_parents=5,
                )
                min_sw = 0.5 if src == "transformer" else 1.0

            if min_sw > 0.0 and (not detail or detail[0]["switch"] < min_sw):
                continue
            if not detail:
                continue

            b_orig = int(P.unshrink(P.normalise_boundary(b_int, maps), maps, L))
            if config.scoring_mode == "standalone_tf":
                sc_val = float(tf_logits_b[b_int])
            elif config.scoring_mode == "hybrid_flank":
                sc_val = float(hybrid_profile[b_int]) if src == "transformer" else float(z_kl[b_int])
            else:
                sc_val = float(z_kl[b_int])

            raw_for_cons.append(
                {
                    "boundary_tx": b_int,
                    "score": sc_val,
                    "breakpoint_orig": b_orig,
                    "n_switch": len(detail),
                    "recombinants": detail,
                }
            )
            raw_peak_meta[b_int] = {
                "source": src,
                "z_kl": float(z_kl[b_int]),
                "tf_logit": float(tf_logits_b[b_int]),
                "tf_prob": float(tf_prob_b[b_int]),
                "hybrid": float(hybrid_profile[b_int]),
            }

        cons_list = REP.consolidate_peaks(
            raw_for_cons,
            merge_radius_bp=config.merge_radius_bp,
            pick_by=pick_by,
        )

        # Map each raw peak's orig coordinate to its consolidated event_id
        orig_to_event_id: dict[int, int] = {}
        for ev in cons_list:
            for bp_c in ev.get("constituent_breakpoints_orig", []):
                orig_to_event_id[int(bp_c)] = int(ev["event_id"])

        def sname(idx: int) -> str:
            return seq_names[idx] if 0 <= idx < len(seq_names) else f"Seq_{idx}"

        consolidated_events: list[ConsolidatedEvent] = []
        for ev in cons_list:
            b_tx = int(ev.get("boundary_tx", 0))
            p_meta = raw_peak_meta.get(
                b_tx,
                {
                    "z_kl": float(z_kl[b_tx]) if 0 <= b_tx < len(z_kl) else 0.0,
                    "tf_logit": float(tf_logits_b[b_tx]) if 0 <= b_tx < len(tf_logits_b) else 0.0,
                    "tf_prob": float(tf_prob_b[b_tx]) if 0 <= b_tx < len(tf_prob_b) else 0.5,
                    "hybrid": float(hybrid_profile[b_tx]) if 0 <= b_tx < len(hybrid_profile) else 0.0,
                },
            )
            carrier_set = set(int(s) for s in ev.get("all_carrier_seqs", []))
            recombs_out: list[RecombinantCall] = []
            for r in ev.get("recombinants", []):
                r_idx = int(r["recombinant"])
                pL = int(r["parent_left"])
                pR = int(r["parent_right"])
                l_cands = r.get("left_candidates", [])
                r_cands = r.get("right_candidates", [])
                ext_L = next((int(c["seq"]) for c in l_cands if int(c["seq"]) not in carrier_set), pL)
                ext_R = next((int(c["seq"]) for c in r_cands if int(c["seq"]) not in carrier_set), pR)

                recombs_out.append(
                    RecombinantCall(
                        seq_idx=r_idx,
                        seq_name=sname(r_idx),
                        p_switch=round(float(r.get("switch", 0.0)), 4),
                        is_recombinant=True,
                        parent_left=pL,
                        parent_left_name=sname(pL),
                        parent_right=pR,
                        parent_right_name=sname(pR),
                        ext_parent_left=ext_L,
                        ext_parent_left_name=sname(ext_L),
                        ext_parent_right=ext_R,
                        ext_parent_right_name=sname(ext_R),
                        left_sim=round(float(r.get("left_sim", 0.0)), 4),
                        right_sim=round(float(r.get("right_sim", 0.0)), 4),
                        left_candidates=[
                            ParentalCandidate(
                                seq_idx=int(c["seq"]),
                                seq_name=sname(int(c["seq"])),
                                similarity=round(float(c["sim"]), 4),
                                is_external_to_event=(int(c["seq"]) not in carrier_set),
                            )
                            for c in l_cands
                        ],
                        right_candidates=[
                            ParentalCandidate(
                                seq_idx=int(c["seq"]),
                                seq_name=sname(int(c["seq"])),
                                similarity=round(float(c["sim"]), 4),
                                is_external_to_event=(int(c["seq"]) not in carrier_set),
                            )
                            for c in r_cands
                        ],
                    )
                )

            consolidated_events.append(
                ConsolidatedEvent(
                    event_id=int(ev["event_id"]),
                    most_plausible_bp_orig=int(ev["most_plausible_breakpoint_orig"]),
                    span_orig_start=int(ev["breakpoint_span_orig"][0]),
                    span_orig_end=int(ev["breakpoint_span_orig"][1]),
                    n_peaks_consolidated=int(ev["n_peaks_consolidated"]),
                    constituent_bps_orig=[int(x) for x in ev.get("constituent_breakpoints_orig", [])],
                    score_z=round(float(p_meta["z_kl"]), 4),
                    tf_logit=round(float(p_meta["tf_logit"]), 4),
                    tf_prob=round(float(p_meta["tf_prob"]), 4),
                    hybrid_score=round(float(p_meta["hybrid"]), 4),
                    boundary_tx=b_tx,
                    n_switch_recombinants=len(recombs_out),
                    recombinant_sequences=[r.seq_idx for r in recombs_out],
                    recombinants=recombs_out,
                )
            )

        raw_peaks_out: list[RawPeakCall] = []
        for rp in raw_for_cons:
            b_tx = int(rp["boundary_tx"])
            bp_orig = int(rp["breakpoint_orig"])
            p_meta = raw_peak_meta[b_tx]
            raw_peaks_out.append(
                RawPeakCall(
                    boundary_tx=b_tx,
                    breakpoint_orig=bp_orig,
                    score_z=round(float(p_meta["z_kl"]), 4),
                    tf_logit=round(float(p_meta["tf_logit"]), 4),
                    tf_prob=round(float(p_meta["tf_prob"]), 4),
                    n_switch=int(rp["n_switch"]),
                    source=str(p_meta["source"]),
                    consolidated_event_id=orig_to_event_id.get(bp_orig),
                )
            )

        # Build binned continuous genome profile in original coordinates for plotting
        track_profile = self._build_track_profile(
            maps=maps,
            kept=kept,
            L=L,
            z_kl=z_kl,
            tf_logits_b=tf_logits_b,
            tf_prob_b=tf_prob_b,
            hybrid_profile=hybrid_profile,
            threshold_z=config.z_thresh,
            threshold_tf_logit=config.tf_logit_thresh,
        )
        return consolidated_events, raw_peaks_out, track_profile

    @staticmethod
    def _build_track_profile(
        maps: dict[str, Any],
        kept: list[int],
        L: int,
        z_kl: np.ndarray,
        tf_logits_b: np.ndarray,
        tf_prob_b: np.ndarray,
        hybrid_profile: np.ndarray,
        threshold_z: float,
        threshold_tf_logit: float,
        max_bins: int = 400,
    ) -> GenomeTrackProfile:
        """Downsample boundary score arrays onto original bp bins taking local maxima per bin."""
        tx2o = np.asarray(maps["tx_to_orig"], dtype=np.int64)
        M_tx = len(tx2o)
        b_orig = np.empty(M_tx + 1, dtype=np.int64)
        b_orig[:M_tx] = tx2o
        b_orig[M_tx] = L
        kept_arr = np.asarray(kept, dtype=np.int64)

        n_bins = min(max_bins, max(120, L // 10))
        edges = np.linspace(0, L, n_bins + 1)
        positions: list[int] = []
        z_bins: list[float] = []
        tf_prob_bins: list[float] = []
        tf_logit_bins: list[float] = []
        hyb_bins: list[float] = []
        dens_bins: list[float] = []

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mid = int(round(0.5 * (lo + hi)))
            positions.append(mid)
            bin_span = max(hi - lo, 1.0)
            info_cnt = int(np.sum((kept_arr >= lo) & (kept_arr < hi))) if len(kept_arr) else 0
            dens_bins.append(round(min(1.0, info_cnt / bin_span), 3))

            mask = (b_orig >= lo) & (b_orig < hi) if i < n_bins - 1 else (b_orig >= lo) & (b_orig <= hi)
            if not np.any(mask):
                idx = int(np.argmin(np.abs(b_orig - mid)))
                z_bins.append(round(float(z_kl[idx]), 3))
                tf_prob_bins.append(round(float(tf_prob_b[idx]), 4))
                tf_logit_bins.append(round(float(tf_logits_b[idx]), 3))
                hyb_bins.append(round(float(hybrid_profile[idx]), 3))
            else:
                z_bins.append(round(float(np.max(z_kl[mask])), 3))
                tf_prob_bins.append(round(float(np.max(tf_prob_b[mask])), 4))
                tf_logit_bins.append(round(float(np.max(tf_logits_b[mask])), 3))
                hyb_bins.append(round(float(np.max(hybrid_profile[mask])), 3))

        return GenomeTrackProfile(
            positions_orig_bp=positions,
            flank_kl_z=z_bins,
            tf_prob=tf_prob_bins,
            tf_logit=tf_logit_bins,
            hybrid_score=hyb_bins,
            informative_density=dens_bins,
            threshold_z=threshold_z,
            threshold_tf_logit=threshold_tf_logit,
        )
