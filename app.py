#!/usr/bin/env python3
"""
FastAPI Web Application Entrypoint for GT-Transformer Recombination Diagnostic Interface.

Runs live inference using the trained JAX Window Transformer checkpoints,
variance-stabilised local flanking KL (`flank_kl`), permutation-null calibration,
and connected-component event consolidation (`report.py`).
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from schemas import FeedbackAnnotation, InferenceConfig, InferenceResponse
from services.benchmarks import load_benchmark_summaries
from services.fasta_utils import (
    AlignmentValidationError,
    BinaryPayloadError,
    EmptyPayloadError,
)
from services.inference_adapter import GTTransformerInferenceEngine
from services.timing import TimingHistoryStore

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
TEMPLATES_DIR = HERE / "templates"
ANNOTATIONS_PATH = HERE / "reviewer_annotations.json"

timing_store = TimingHistoryStore(log_path=str(HERE / "timing_history.json"))
engine = GTTransformerInferenceEngine(timing_store=timing_store)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pre-compile JAX forward graph for static (512, 314, 4) chunks on startup
    engine.warmup("ckpt_N10000_extreme")
    yield


app = FastAPI(
    title="gt-transformer | Recombination Inference & Diagnostic Interface",
    description=(
        "Web interface for running live gt-transformer inference, recording wall-clock timings, "
        "inspecting 3-level outputs, and diagnosing model successes (TP) vs failures (FP/FN)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(AlignmentValidationError)
async def alignment_validation_handler(request: Request, exc: AlignmentValidationError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    detail = str(exc)
    status_code = (
        422
        if any(
            k in detail
            for k in (
                "identical lengths",
                "at least 4 sequences",
                "polymorphic sites",
                "gap characters",
            )
        )
        else 400
    )
    return JSONResponse(status_code=status_code, content={"detail": detail})

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


AVAILABLE_SCORING_MODES = [
    {
        "id": "hybrid_flank",
        "label": "Hybrid (flank_kl + Window Transformer) — Default",
        "description": "Combines null-calibrated local flanking KL z-score (Lw=250) with Window Transformer boundary logits.",
    },
    {
        "id": "standalone_tf",
        "label": "Standalone Window Transformer",
        "description": "Trained 4-layer pre-norm encoder over circular (W=314, 4) pairwise 4-gamete windows + carrier switch verification.",
    },
    {
        "id": "flank_kl",
        "label": "Local Flanking KL (flank_kl only)",
        "description": "Symmetric flanking window [b-w, b) vs [b, b+w) soft-parent KL localiser calibrated against column-permutation null.",
    },
    {
        "id": "global_kl",
        "label": "Legacy Global Segment KL (kl, min_seg=200)",
        "description": "Original prefix [0, b) vs suffix [b, M) KL scorer (retained for side-by-side comparison).",
    },
]


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    """Serve the diagnostic interface HTML page."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="Frontend template index.html not found.")
    return FileResponse(str(index_file))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Return server status, live engine state, and available checkpoints."""
    return {
        "status": "ok",
        "engine_mode": "live",
        "infrastructure_imported": True,
        "checkpoints": engine.list_checkpoints(),
        "scoring_modes": AVAILABLE_SCORING_MODES,
    }


@app.get("/api/presets")
async def list_presets() -> dict[str, Any]:
    """List verified benchmark, 10k simulation, extreme, and empirical alignments on disk."""
    return {
        "presets": engine.list_presets(),
        "checkpoints": engine.list_checkpoints(),
        "scoring_modes": AVAILABLE_SCORING_MODES,
    }


@app.get("/api/benchmarks")
async def get_benchmarks() -> dict[str, Any]:
    """Return verified 10,000-alignment scaling and out-of-range specificity summaries."""
    return load_benchmark_summaries()


@app.get("/api/alignment/window")
async def get_alignment_window(
    run_id: str,
    start: int = 0,
    width: int = 110,
) -> dict[str, Any]:
    """Return a real nucleotide window slice and informative-site flags for the alignment viewer."""
    try:
        return engine.get_alignment_window(run_id=run_id, start_bp=start, width=width)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/infer", response_model=InferenceResponse)
async def run_inference_endpoint(
    request: Request,
    preset_id: Optional[str] = Form(default=None),
    checkpoint: str = Form(default="ckpt_N10000_extreme"),
    scoring_mode: str = Form(default="hybrid_flank"),
    null_policy: str = Form(default="duplicate"),
    flank_window: int = Form(default=250),
    min_seg: int = Form(default=50),
    beta: float = Form(default=20.0),
    z_thresh: float = Form(default=6.0),
    tf_logit_thresh: float = Form(default=0.8),
    merge_radius_bp: int = Form(default=75),
    eval_tolerance_bp: int = Form(default=30),
    pasted_fasta: Optional[str] = Form(default=None),
    pasted_gt: Optional[str] = Form(default=None),
    fasta_file: Optional[UploadFile] = File(default=None),
    gt_file: Optional[UploadFile] = File(default=None),
) -> InferenceResponse:
    """Execute live inference from multipart form upload or JSON body."""
    content_type = request.headers.get("content-type", "")
    fasta_text: Optional[str] = None
    fasta_filename: Optional[str] = None
    gt_text: Optional[str] = None
    gt_filename: Optional[str] = None
    has_upload = False

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        preset_id = body.get("preset_id", preset_id)
        checkpoint = body.get("checkpoint", checkpoint)
        scoring_mode = body.get("scoring_mode", scoring_mode)
        null_policy = body.get("null_policy", null_policy)
        flank_window = int(body.get("flank_window", flank_window))
        min_seg = int(body.get("min_seg", min_seg))
        beta = float(body.get("beta", beta))
        z_thresh = float(body.get("z_thresh", z_thresh))
        tf_logit_thresh = float(body.get("tf_logit_thresh", tf_logit_thresh))
        merge_radius_bp = int(body.get("merge_radius_bp", merge_radius_bp))
        eval_tolerance_bp = int(body.get("eval_tolerance_bp", eval_tolerance_bp))
        fasta_text = body.get("fasta_text") or body.get("pasted_fasta")
        if "fasta_text" in body or "pasted_fasta" in body:
            has_upload = True
        fasta_filename = body.get("fasta_filename")
        gt_text = body.get("gt_text") or body.get("pasted_gt")
        gt_filename = body.get("gt_filename")
    else:
        if fasta_file is not None and fasta_file.filename:
            has_upload = True
            raw_bytes = await fasta_file.read()
            if len(raw_bytes.strip()) == 0:
                raise HTTPException(status_code=400, detail="Empty alignment payload")
            if b"\x00" in raw_bytes:
                raise HTTPException(status_code=400, detail="Binary or non-text alignment content detected")
            try:
                fasta_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Binary or non-text alignment content detected")
            fasta_filename = fasta_file.filename
        elif pasted_fasta is not None:
            has_upload = True
            fasta_text = pasted_fasta
            fasta_filename = "pasted_alignment.fasta"

        if gt_file is not None and gt_file.filename:
            gt_bytes = await gt_file.read()
            gt_text = gt_bytes.decode("utf-8", errors="replace")
            gt_filename = gt_file.filename
        elif pasted_gt and pasted_gt.strip():
            gt_text = pasted_gt
            gt_filename = "pasted_ground_truth.txt"

    if has_upload and (fasta_text is None or not fasta_text.strip()):
        raise HTTPException(status_code=400, detail="Empty alignment payload")
    if not has_upload and not preset_id:
        raise HTTPException(status_code=400, detail="Empty alignment payload")
    if fasta_text and "\x00" in fasta_text:
        raise HTTPException(status_code=400, detail="Binary or non-text alignment content detected")

    try:
        cfg = InferenceConfig(
            checkpoint=checkpoint,
            scoring_mode=scoring_mode,  # type: ignore[arg-type]
            null_policy=null_policy,  # type: ignore[arg-type]
            flank_window=flank_window,
            min_seg=min_seg,
            beta=beta,
            z_thresh=z_thresh,
            tf_logit_thresh=tf_logit_thresh,
            merge_radius_bp=merge_radius_bp,
            eval_tolerance_bp=eval_tolerance_bp,
        )
        return engine.run_inference(
            config=cfg,
            preset_id=preset_id,
            fasta_text=fasta_text,
            fasta_filename=fasta_filename,
            gt_text=gt_text,
            gt_filename=gt_filename,
        )
    except HTTPException:
        raise
    except AlignmentValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = (
            422
            if any(
                k in detail
                for k in (
                    "identical lengths",
                    "at least 4 sequences",
                    "polymorphic sites",
                    "gap characters",
                )
            )
            else 400
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        import logging

        logging.getLogger("uvicorn.error").exception("Unhandled inference exception")
        raise HTTPException(status_code=500, detail=f"Internal inference error: {str(exc)}") from exc


@app.get("/api/timings")
async def get_timing_history() -> dict[str, Any]:
    """Return recorded live inference timing runs."""
    records = timing_store.list_history()
    return {
        "count": len(records),
        "records": [r.model_dump() for r in records],
    }


@app.get("/api/timings/export.csv")
async def export_timings_csv() -> PlainTextResponse:
    """Download all recorded inference timings as a CSV file."""
    csv_text = timing_store.export_csv()
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="gt_transformer_inference_timings.csv"'},
    )


def _load_annotations() -> list[dict[str, Any]]:
    if ANNOTATIONS_PATH.exists():
        try:
            return json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


@app.get("/api/annotations")
async def list_annotations() -> dict[str, Any]:
    """Return saved reviewer diagnostic feedback annotations."""
    items = _load_annotations()
    return {"count": len(items), "annotations": items}


@app.post("/api/annotations")
async def save_annotation(annotation: FeedbackAnnotation) -> dict[str, Any]:
    """Save a reviewer feedback note on a predicted or missed recombination event."""
    items = _load_annotations()
    entry = annotation.model_dump()
    entry["created_at_iso"] = datetime.now(timezone.utc).isoformat()
    items.insert(0, entry)
    ANNOTATIONS_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return {"status": "saved", "annotation": entry, "total_count": len(items)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8050"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
