"""
Stage-by-stage wall-clock inference timing instrumentation and persistent log store.

Tracks wall-clock duration (via time.perf_counter), percentage of total runtime,
and domain throughput (bp/s, columns/s, windows/s) across each stage of the
live GT-Transformer inference pipeline:
    1. FASTA Parsing & Informative-Site Filtering
    2. Binarisation, Coordinate Map Construction & Haplotype Matrix
    3. Circular Moving-Window Pairwise 4-Gamete Embedding (W=314)
    4. Window Transformer Forward Pass (JAX JIT Compiled Encoder)
    5. Local Flanking KL (flank_kl) & Permutation Null Calibration
    6. Connected-Component Peak Consolidation & 3-Level Carrier/Parent Attribution
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Iterator, Optional

from schemas import InferenceTimingReport, StageTiming


class TimingRecorder:
    """Context-managed stage timer for recording live inference pipeline telemetry."""

    def __init__(
        self,
        dataset_name: str,
        checkpoint: str,
        scoring_mode: str,
        n_sequences: int,
        genome_length_bp: int,
        n_informative_cols: int,
        n_transformed_cols: int,
        window_size: int,
    ) -> None:
        self.run_id = f"run_{uuid.uuid4().hex[:8]}"
        self.timestamp_iso = datetime.now(timezone.utc).isoformat()
        self.dataset_name = dataset_name
        self.checkpoint = checkpoint
        self.scoring_mode = scoring_mode
        self.n_sequences = n_sequences
        self.genome_length_bp = genome_length_bp
        self.n_informative_cols = n_informative_cols
        self.n_transformed_cols = n_transformed_cols
        self.window_size = window_size
        self._stages_raw: list[dict] = []
        self._t0_wall = time.perf_counter()

    @contextlib.contextmanager
    def measure_stage(
        self,
        stage_id: str,
        stage_name: str,
        items_processed: Optional[int] = None,
        items_unit: Optional[str] = None,
        throughput_unit: Optional[str] = None,
        notes: str = "",
    ) -> Iterator[dict]:
        """Measure a pipeline stage using high-resolution time.perf_counter()."""
        meta_box: dict = {"notes": notes, "items_processed": items_processed}
        t_start = time.perf_counter()
        try:
            yield meta_box
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            n_items = meta_box.get("items_processed", items_processed)
            throughput = None
            if n_items is not None and n_items > 0 and elapsed_ms > 0:
                throughput = round((n_items / (elapsed_ms / 1000.0)), 1)
            self._stages_raw.append(
                {
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "duration_ms": round(elapsed_ms, 2),
                    "items_processed": n_items,
                    "items_unit": items_unit,
                    "throughput": throughput,
                    "throughput_unit": throughput_unit,
                    "notes": str(meta_box.get("notes", notes)),
                }
            )

    def finalize(self) -> InferenceTimingReport:
        """Build the finalized InferenceTimingReport with stage percentages."""
        total_ms = sum(s["duration_ms"] for s in self._stages_raw)
        if total_ms <= 0:
            total_ms = max((time.perf_counter() - self._t0_wall) * 1000.0, 0.01)

        stages: list[StageTiming] = []
        for s in self._stages_raw:
            pct = (s["duration_ms"] / total_ms * 100.0) if total_ms > 0 else 0.0
            stages.append(
                StageTiming(
                    stage_id=s["stage_id"],
                    stage_name=s["stage_name"],
                    duration_ms=round(s["duration_ms"], 2),
                    pct_of_total=round(pct, 1),
                    items_processed=s["items_processed"],
                    items_unit=s["items_unit"],
                    throughput=s["throughput"],
                    throughput_unit=s["throughput_unit"],
                    notes=s["notes"],
                )
            )

        total_sec = max(total_ms / 1000.0, 1e-6)
        bp_per_sec = round(self.genome_length_bp / total_sec, 1)
        cols_per_sec = round(self.n_transformed_cols / total_sec, 1)

        return InferenceTimingReport(
            run_id=self.run_id,
            timestamp_iso=self.timestamp_iso,
            dataset_name=self.dataset_name,
            checkpoint=self.checkpoint,
            scoring_mode=self.scoring_mode,
            is_live=True,
            n_sequences=self.n_sequences,
            genome_length_bp=self.genome_length_bp,
            n_informative_cols=self.n_informative_cols,
            n_transformed_cols=self.n_transformed_cols,
            window_size=self.window_size,
            total_ms=round(total_ms, 2),
            throughput_bp_per_sec=bp_per_sec,
            throughput_cols_per_sec=cols_per_sec,
            stages=stages,
        )


class TimingHistoryStore:
    """Persists and serves historical live inference timing logs."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_path = log_path or os.path.join(here, "timing_history.json")
        self._records: list[InferenceTimingReport] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.log_path):
            self._records = []
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            # Only retain real live runs (discard legacy mock scaffold entries)
            live_items = [
                item for item in raw
                if item.get("is_live", False) and not item.get("is_mock_scaffold", False)
            ]
            self._records = [InferenceTimingReport(**item) for item in live_items]
        except Exception:
            self._records = []

    def _save(self) -> None:
        try:
            with open(self.log_path, "w", encoding="utf-8") as fh:
                json.dump([r.model_dump() for r in self._records], fh, indent=2)
        except Exception:
            pass

    def record(self, report: InferenceTimingReport) -> None:
        self._records.insert(0, report)
        self._records = self._records[:150]
        self._save()

    def list_history(self) -> list[InferenceTimingReport]:
        return list(self._records)

    def export_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "run_id",
                "timestamp_iso",
                "dataset_name",
                "checkpoint",
                "scoring_mode",
                "is_live",
                "n_sequences",
                "genome_length_bp",
                "n_informative_cols",
                "n_transformed_cols",
                "window_size",
                "total_ms",
                "throughput_bp_per_sec",
                "throughput_cols_per_sec",
                "stage_1_parse_filter_ms",
                "stage_2_binarise_coords_ms",
                "stage_3_pairwise_embed_ms",
                "stage_4_transformer_fwd_ms",
                "stage_5_flank_kl_null_ms",
                "stage_6_consolidate_3level_ms",
            ]
        )
        for rec in self._records:
            by_id = {s.stage_id: s.duration_ms for s in rec.stages}
            writer.writerow(
                [
                    rec.run_id,
                    rec.timestamp_iso,
                    rec.dataset_name,
                    rec.checkpoint,
                    rec.scoring_mode,
                    rec.is_live,
                    rec.n_sequences,
                    rec.genome_length_bp,
                    rec.n_informative_cols,
                    rec.n_transformed_cols,
                    rec.window_size,
                    rec.total_ms,
                    rec.throughput_bp_per_sec,
                    rec.throughput_cols_per_sec,
                    by_id.get("parse_filter", 0.0),
                    by_id.get("binarise_coords", 0.0),
                    by_id.get("pairwise_embed", 0.0),
                    by_id.get("transformer_fwd", 0.0),
                    by_id.get("flank_kl_null", 0.0),
                    by_id.get("consolidate_3level", 0.0),
                ]
            )
        return buf.getvalue()
