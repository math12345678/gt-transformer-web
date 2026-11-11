"""
Data contracts and Pydantic schemas for the GT-Transformer Recombination Web Application.

Defines request/response schemas for alignment uploads, model configuration,
stage-by-stage wall-clock inference timing telemetry, the three-level
recombination event report (breakpoints -> recombinant carriers -> parental
candidates), tolerant TP/FP/FN diagnostics, and 10k/extreme validation metrics.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Configuration parameters for running a live GT-Transformer inference job."""
    checkpoint: str = Field(
        default="ckpt_N10000_extreme",
        description="Trained model checkpoint identifier.",
    )
    scoring_mode: Literal["hybrid_flank", "standalone_tf", "flank_kl", "global_kl"] = Field(
        default="hybrid_flank",
        description="Boundary scoring method: Hybrid (Flank-KL + Transformer), Standalone TF, Local Flank-KL, or Global KL.",
    )
    null_policy: Literal["duplicate", "zero", "drop"] = Field(
        default="duplicate",
        description="Policy for third/fourth nucleotide states at parsimony-informative columns.",
    )
    flank_window: int = Field(
        default=250,
        ge=15,
        le=2000,
        description="Flanking window size L_w (in transformed columns) for local flanking KL.",
    )
    min_seg: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Minimum segment length near alignment boundaries.",
    )
    beta: float = Field(
        default=20.0,
        ge=1.0,
        le=100.0,
        description="Softmax inverse-temperature beta for marginalising over parental candidates.",
    )
    z_thresh: float = Field(
        default=6.0,
        ge=0.0,
        le=25.0,
        description="Calibrated z-score peak threshold against column-permutation null (for KL modes).",
    )
    tf_logit_thresh: float = Field(
        default=0.8,
        ge=-5.0,
        le=10.0,
        description="Transformer logit threshold (logit >= 0.8 corresponds to P(bp) >= 0.69) for TF peak calling.",
    )
    merge_radius_bp: int = Field(
        default=75,
        ge=1,
        le=500,
        description="Consolidation radius in original alignment base pairs for adjacent peaks sharing carriers.",
    )
    eval_tolerance_bp: int = Field(
        default=30,
        ge=0,
        le=200,
        description="Primary bp tolerance window (±k bp) for TP/FP/FN ground-truth evaluation.",
    )


class StageTiming(BaseModel):
    """Timing telemetry for a single stage of the inference pipeline."""
    stage_id: str
    stage_name: str
    duration_ms: float
    pct_of_total: float
    items_processed: Optional[int] = None
    items_unit: Optional[str] = None
    throughput: Optional[float] = None
    throughput_unit: Optional[str] = None
    notes: str = ""


class InferenceTimingReport(BaseModel):
    """End-to-end and stage-by-stage wall-clock timing breakdown for an inference run."""
    run_id: str
    timestamp_iso: str
    dataset_name: str
    checkpoint: str
    scoring_mode: str
    is_live: bool = True
    n_sequences: int
    genome_length_bp: int
    n_informative_cols: int
    n_transformed_cols: int
    window_size: int
    total_ms: float
    throughput_bp_per_sec: float
    throughput_cols_per_sec: float
    stages: list[StageTiming]


class ParentalCandidate(BaseModel):
    """Level 3: Ranked parental-like sequence candidate."""
    seq_idx: int
    seq_name: str
    similarity: float
    is_external_to_event: bool = True


class RecombinantCall(BaseModel):
    """Level 2 & Level 3: Recombinant carrier sequence and its left/right parental candidates."""
    seq_idx: int
    seq_name: str
    p_switch: float
    is_recombinant: bool = True
    parent_left: int
    parent_left_name: str
    parent_right: int
    parent_right_name: str
    ext_parent_left: int
    ext_parent_left_name: str
    ext_parent_right: int
    ext_parent_right_name: str
    left_sim: float
    right_sim: float
    left_candidates: list[ParentalCandidate] = Field(default_factory=list)
    right_candidates: list[ParentalCandidate] = Field(default_factory=list)


class ConsolidatedEvent(BaseModel):
    """Level 1: Consolidated breakpoint event with Level 2/3 carrier & parent details."""
    event_id: int
    most_plausible_bp_orig: int
    span_orig_start: int
    span_orig_end: int
    n_peaks_consolidated: int
    constituent_bps_orig: list[int] = Field(default_factory=list)
    score_z: float
    tf_logit: float = 0.0
    tf_prob: float
    hybrid_score: float
    boundary_tx: int
    n_switch_recombinants: int
    recombinant_sequences: list[int] = Field(default_factory=list)
    recombinants: list[RecombinantCall] = Field(default_factory=list)
    # Signal attribution & diagnostic honesty (PROJECT.md contract)
    signal_attribution: str = "hybrid_flank"
    attribution_note: Optional[str] = None
    # Diagnostic classification when ground truth is present
    match_status: Literal["TP", "FP", "UNLABELLED"] = "UNLABELLED"
    matched_gt_bp: Optional[int] = None
    matched_gt_event_id: Optional[int | str] = None
    bp_error: Optional[int] = None
    diagnostic_explanation: str = ""


class RawPeakCall(BaseModel):
    """Un-consolidated localiser peak in original and transformed coordinates."""
    boundary_tx: int
    breakpoint_orig: int
    score_z: float
    tf_logit: float = 0.0
    tf_prob: float
    n_switch: int
    source: str = "flank_kl"
    consolidated_event_id: Optional[int] = None


class GroundTruthBreakpointItem(BaseModel):
    """A single real ground-truth breakpoint and whether the model detected it."""
    gt_bp_orig: int
    gt_boundary_tx: Optional[int] = None
    gt_event_id: Optional[int | str] = None
    event_type: str = "unknown"
    generation: Optional[int] = None
    carrier_seqs: list[int] = Field(default_factory=list)
    detected: bool = False
    matched_pred_event_id: Optional[int] = None
    matched_pred_bp_orig: Optional[int] = None
    bp_error: Optional[int] = None
    nearest_subthreshold_peak_bp: Optional[int] = None
    nearest_subthreshold_score_z: Optional[float] = None
    failure_hypothesis: str = ""


class ToleranceMetrics(BaseModel):
    """Evaluation metrics at a specific ±k bp tolerance."""
    tolerance_bp: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    control_f1: float
    lift: float
    median_error_bp: Optional[float] = None


class DiagnosticBreakdown(BaseModel):
    """Where the model succeeds (TP) and where it fails (FP / FN)."""
    has_ground_truth: bool = False
    active_tolerance_bp: int = 30
    n_ground_truth_breakpoints: int = 0
    n_predicted_events: int = 0
    tolerance_metrics: list[ToleranceMetrics] = Field(default_factory=list)
    ground_truth_items: list[GroundTruthBreakpointItem] = Field(default_factory=list)
    summary_notes: list[str] = Field(default_factory=list)


class GenomeTrackProfile(BaseModel):
    """Sampled coordinate profile along the genome for interactive plotting."""
    positions_orig_bp: list[int]
    flank_kl_z: list[float]
    tf_prob: list[float]
    tf_logit: list[float] = Field(default_factory=list)
    hybrid_score: list[float]
    informative_density: list[float] = Field(default_factory=list)
    threshold_z: float
    threshold_tf_logit: float = 0.8


class AlignmentMetadata(BaseModel):
    """Summary metadata for the uploaded or selected alignment."""
    dataset_name: str
    source_type: Literal["preset", "upload"]
    n_sequences: int
    genome_length_bp: int
    n_informative_cols: int
    n_transformed_cols: int
    window_size: int
    null_policy: str
    sequence_names: list[str]
    gc_content_pct: float
    ambiguity_pct: float
    is_rna: bool = False


class InferenceResponse(BaseModel):
    """Complete response returned to the web client for an inference run."""
    status: str = "ok"
    warnings: list[str] = Field(default_factory=list)
    engine_mode: str = "live"
    model_param_count: int = 0
    config: InferenceConfig
    alignment: AlignmentMetadata
    timings: InferenceTimingReport
    consolidated_events: list[ConsolidatedEvent]
    raw_peaks: list[RawPeakCall]
    diagnostics: DiagnosticBreakdown
    track_profile: GenomeTrackProfile


class FeedbackAnnotation(BaseModel):
    """Reviewer annotation from Dr. Martin on a specific predicted or missed event."""
    dataset_name: str
    run_id: str
    target_type: Literal["predicted_event", "missed_gt_breakpoint", "general"]
    target_id: str
    verdict: Literal[
        "confirmed_true_event",
        "tract_tail_echo",
        "homoplasy_false_positive",
        "boundary_dead_zone_miss",
        "parent_assignment_issue",
        "other",
    ]
    reviewer_name: str = "Darren Martin"
    comment: str = ""
    created_at_iso: Optional[str] = None
