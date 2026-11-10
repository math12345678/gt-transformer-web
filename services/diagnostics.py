"""
Tolerant (±k bp) breakpoint evaluation and TP/FP/FN diagnostic classifier.

Directly supports Dr. Darren Martin's request:
    "I'll be able to give you much better feedback if I can see where the model
     is failing and where it is succeeding."

Matches predicted consolidated events against ground-truth breakpoints in original
alignment coordinates across tolerances k in {0, 3, 10, 30} bp (plus any custom k),
computes Precision, Recall (Power), F1, matched random-boundary control F1, Lift,
and Median Localisation Error, and annotates each event/breakpoint with clear
diagnostic explanations.
"""
from __future__ import annotations

import math
import random
from typing import Any

from schemas import (
    ConsolidatedEvent,
    DiagnosticBreakdown,
    GenomeTrackProfile,
    GroundTruthBreakpointItem,
    ToleranceMetrics,
)


def match_tolerant_greedy(
    preds_bp: list[int],
    truth_bp: list[int],
    tolerance_bp: int,
) -> tuple[int, int, int, float, float, float, list[int], dict[int, int], dict[int, int]]:
    """Greedy closest-first 1-to-1 bipartite matching within ±tolerance_bp.

    Returns:
      (tp, fp, fn, precision, recall, f1, errors_bp, pred_idx_to_gt_idx, gt_idx_to_pred_idx)
    """
    if not preds_bp and not truth_bp:
        return 0, 0, 0, 1.0, 1.0, 1.0, [], {}, {}
    if not truth_bp:
        return 0, len(preds_bp), 0, 0.0, 0.0, 0.0, [], {}, {}
    if not preds_bp:
        return 0, 0, len(truth_bp), 0.0, 0.0, 0.0, [], {}, {}

    candidate_pairs: list[tuple[int, int, int]] = []
    for p_idx, p_val in enumerate(preds_bp):
        for g_idx, g_val in enumerate(truth_bp):
            dist = abs(int(p_val) - int(g_val))
            if dist <= tolerance_bp:
                candidate_pairs.append((dist, p_idx, g_idx))

    candidate_pairs.sort(key=lambda t: (t[0], t[1], t[2]))
    matched_p: dict[int, int] = {}
    matched_g: dict[int, int] = {}
    errors: list[int] = []

    for dist, p_idx, g_idx in candidate_pairs:
        if p_idx not in matched_p and g_idx not in matched_g:
            matched_p[p_idx] = g_idx
            matched_g[g_idx] = p_idx
            errors.append(dist)

    tp = len(matched_p)
    fp = len(preds_bp) - tp
    fn = len(truth_bp) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1, errors, matched_p, matched_g


def _median(vals: list[int | float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def evaluate_predictions_against_ground_truth(
    consolidated_events: list[ConsolidatedEvent],
    gt_raw_items: list[dict[str, Any]],
    track_profile: GenomeTrackProfile,
    genome_length_bp: int,
    active_tolerance_bp: int = 30,
    is_known_control: bool = False,
) -> DiagnosticBreakdown:
    """Annotate consolidated_events in-place with TP/FP diagnostics and return DiagnosticBreakdown."""
    if not gt_raw_items and is_known_control:
        for ev in consolidated_events:
            ev.match_status = "FP"
            ev.matched_gt_bp = None
            ev.matched_gt_event_id = None
            ev.diagnostic_explanation = (
                f"FALSE POSITIVE ON CLONAL CONTROL: Dataset has 0 true recombination events, "
                f"but model called {ev.most_plausible_bp_orig:,} bp (z={ev.score_z:.2f}, TF_prob={ev.tf_prob:.2f})."
            )
        n_fp = len(consolidated_events)
        spec_note = (
            "Clonal negative control (0 true breakpoints): 0 False Positives — 100% clean specificity."
            if n_fp == 0
            else f"Clonal negative control (0 true breakpoints): {n_fp} False Positive call(s) above threshold."
        )
        return DiagnosticBreakdown(
            has_ground_truth=True,
            active_tolerance_bp=active_tolerance_bp,
            n_ground_truth_breakpoints=0,
            n_predicted_events=n_fp,
            tolerance_metrics=[
                ToleranceMetrics(
                    tolerance_bp=k,
                    tp=0,
                    fp=n_fp,
                    fn=0,
                    precision=1.0 if n_fp == 0 else 0.0,
                    recall=1.0,
                    f1=1.0 if n_fp == 0 else 0.0,
                    control_f1=0.0,
                    lift=0.0,
                    median_error_bp=None,
                )
                for k in sorted({0, 3, 10, 30, int(active_tolerance_bp)})
            ],
            ground_truth_items=[],
            summary_notes=[spec_note],
        )

    if not gt_raw_items:
        for ev in consolidated_events:
            ev.match_status = "UNLABELLED"
            ev.diagnostic_explanation = (
                "No ground-truth event log provided. Upload a SantaSim events.json or breakpoint CSV "
                "to classify as True Positive (TP) or False Positive (FP)."
            )
        return DiagnosticBreakdown(
            has_ground_truth=False,
            active_tolerance_bp=active_tolerance_bp,
            n_ground_truth_breakpoints=0,
            n_predicted_events=len(consolidated_events),
            tolerance_metrics=[],
            ground_truth_items=[],
            summary_notes=[
                "Running in unlabelled discovery mode (no ground-truth breakpoint log attached).",
                f"Detected {len(consolidated_events)} consolidated recombination events across {genome_length_bp:,} bp.",
            ],
        )

    # Deduplicate ground-truth breakpoints by coordinate while preserving metadata
    by_bp: dict[int, dict[str, Any]] = {}
    for item in gt_raw_items:
        bp = int(item["gt_bp_orig"])
        if bp not in by_bp:
            by_bp[bp] = dict(item)
        else:
            existing_id = str(by_bp[bp].get("gt_event_id", ""))
            new_id = str(item.get("gt_event_id", ""))
            if new_id and new_id not in existing_id.split("/"):
                by_bp[bp]["gt_event_id"] = f"{existing_id}/{new_id}" if existing_id else new_id

    gt_sorted = [by_bp[k] for k in sorted(by_bp.keys())]
    truth_bps = [int(g["gt_bp_orig"]) for g in gt_sorted]
    pred_bps = [int(ev.most_plausible_bp_orig) for ev in consolidated_events]

    tolerances = sorted({0, 3, 10, 30, int(active_tolerance_bp)})
    rng = random.Random(42)
    min_gt = min(truth_bps) if truth_bps else 1
    max_gt = max(truth_bps) if truth_bps else max(genome_length_bp - 1, 2)
    span_lo = max(1, min_gt - 50)
    span_hi = min(genome_length_bp - 1, max_gt + 50)

    tol_metrics: list[ToleranceMetrics] = []
    for k in tolerances:
        tp, fp, fn, prec, rec, f1, errs, _, _ = match_tolerant_greedy(pred_bps, truth_bps, k)
        # Matched random-boundary control (40 trials)
        ctrl_f1s: list[float] = []
        if pred_bps and truth_bps and span_hi > span_lo:
            for _ in range(40):
                rand_preds = [rng.randint(span_lo, span_hi) for _ in range(len(pred_bps))]
                _, _, _, _, _, cf1, _, _, _ = match_tolerant_greedy(rand_preds, truth_bps, k)
                ctrl_f1s.append(cf1)
        ctrl_f1 = sum(ctrl_f1s) / len(ctrl_f1s) if ctrl_f1s else 0.0
        lift = f1 - ctrl_f1
        med_err = _median(errs)
        tol_metrics.append(
            ToleranceMetrics(
                tolerance_bp=k,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                control_f1=round(ctrl_f1, 4),
                lift=round(lift, 4),
                median_error_bp=round(med_err, 1) if med_err is not None else None,
            )
        )

    # Perform matching at active_tolerance_bp to label each event and GT item
    tp, fp, fn, prec, rec, f1, errs, matched_p, matched_g = match_tolerant_greedy(
        pred_bps, truth_bps, active_tolerance_bp
    )

    # Annotate predicted ConsolidatedEvents
    for p_idx, ev in enumerate(consolidated_events):
        if p_idx in matched_p:
            g_idx = matched_p[p_idx]
            gt_obj = gt_sorted[g_idx]
            gt_bp = int(gt_obj["gt_bp_orig"])
            err = abs(ev.most_plausible_bp_orig - gt_bp)
            ev.match_status = "TP"
            ev.matched_gt_bp = gt_bp
            ev.matched_gt_event_id = gt_obj.get("gt_event_id")
            ev.bp_error = err
            top_carrier = (
                f"Seq {ev.recombinants[0].seq_idx} (p_switch={ev.recombinants[0].p_switch:.2f})"
                if ev.recombinants
                else "none"
            )
            ev.diagnostic_explanation = (
                f"SUCCEEDED (TP @ ±{active_tolerance_bp} bp): Predicted at {ev.most_plausible_bp_orig:,} bp, "
                f"matching true breakpoint {gt_bp:,} bp (Event #{ev.matched_gt_event_id}) with Δ = {err} bp error. "
                f"Consolidated {ev.n_peaks_consolidated} peak(s) across [{ev.span_orig_start:,}–{ev.span_orig_end:,} bp]; "
                f"lead carrier: {top_carrier}."
            )
        else:
            ev.match_status = "FP"
            ev.matched_gt_bp = None
            ev.matched_gt_event_id = None
            # Find nearest true breakpoint even if outside tolerance to diagnose why FP occurred
            if truth_bps:
                nearest_gt = min(truth_bps, key=lambda g: abs(g - ev.most_plausible_bp_orig))
                dist_nearest = abs(ev.most_plausible_bp_orig - nearest_gt)
            else:
                nearest_gt = None
                dist_nearest = 999999

            if nearest_gt is not None and dist_nearest <= 110:
                ev.bp_error = dist_nearest
                ev.diagnostic_explanation = (
                    f"NEAR-MISS / TRACT ECHO (FP @ ±{active_tolerance_bp} bp): Predicted at "
                    f"{ev.most_plausible_bp_orig:,} bp, which lies {dist_nearest} bp from true breakpoint "
                    f"{nearest_gt:,} bp (outside ±{active_tolerance_bp} bp window). Likely caused by "
                    f"informative-site sparsity or multi-carrier tract shoulder drift."
                )
            else:
                ev.bp_error = None
                ev.diagnostic_explanation = (
                    f"SPURIOUS CALL (FP @ ±{active_tolerance_bp} bp): Predicted at "
                    f"{ev.most_plausible_bp_orig:,} bp (z={ev.score_z:.2f}, TF={ev.tf_prob:.2f}) with no "
                    f"ground-truth breakpoint nearby (nearest GT is {dist_nearest:,} bp away). "
                    f"Typical cause: homoplasy cluster or nested ancestral 4-gamete soup."
                )

    # Annotate GroundTruthBreakpointItems
    gt_items_out: list[GroundTruthBreakpointItem] = []
    positions = track_profile.positions_orig_bp
    z_vals = track_profile.flank_kl_z

    for g_idx, gt_obj in enumerate(gt_sorted):
        gt_bp = int(gt_obj["gt_bp_orig"])
        # Find local score peak near gt_bp in track profile
        near_pos = None
        near_z = None
        if positions and z_vals:
            best_i = min(range(len(positions)), key=lambda i: abs(positions[i] - gt_bp))
            window_indices = [
                i for i in range(len(positions)) if abs(positions[i] - gt_bp) <= max(active_tolerance_bp, 40)
            ] or [best_i]
            peak_i = max(window_indices, key=lambda i: z_vals[i])
            near_pos = int(positions[peak_i])
            near_z = round(float(z_vals[peak_i]), 2)

        if g_idx in matched_g:
            p_idx = matched_g[g_idx]
            ev = consolidated_events[p_idx]
            err = abs(ev.most_plausible_bp_orig - gt_bp)
            gt_items_out.append(
                GroundTruthBreakpointItem(
                    gt_bp_orig=gt_bp,
                    gt_boundary_tx=gt_obj.get("gt_boundary_tx"),
                    gt_event_id=gt_obj.get("gt_event_id"),
                    event_type=str(gt_obj.get("event_type", "unknown")),
                    generation=gt_obj.get("generation"),
                    carrier_seqs=gt_obj.get("carrier_seqs", []),
                    detected=True,
                    matched_pred_event_id=ev.event_id,
                    matched_pred_bp_orig=ev.most_plausible_bp_orig,
                    bp_error=err,
                    nearest_subthreshold_peak_bp=near_pos,
                    nearest_subthreshold_score_z=near_z,
                    failure_hypothesis=(
                        f"Detected by Consolidated Event #{ev.event_id} at {ev.most_plausible_bp_orig:,} bp "
                        f"(Δ = {err} bp, z = {ev.score_z:.2f})."
                    ),
                )
            )
        else:
            # Diagnose why this true breakpoint was missed (FN)
            if gt_bp < 100 or gt_bp > genome_length_bp - 100:
                reason = (
                    f"Boundary-proximity attenuation: {gt_bp:,} bp sits within 100 bp of alignment edge "
                    f"(short flanking window reduces KL statistical power)."
                )
            elif near_z is not None and near_z >= track_profile.threshold_z * 0.72:
                reason = (
                    f"Sub-threshold local peak: local score reached z = {near_z:.2f} near {near_pos:,} bp "
                    f"(just below threshold z = {track_profile.threshold_z:.1f}) or was merged into an "
                    f"adjacent stronger crossover peak."
                )
            else:
                reason = (
                    f"Signal erasure / un-homoplasy: low local switch contrast (max local z = "
                    f"{near_z if near_z is not None else 0.0:.2f}). Subsequent overprinting crossovers or "
                    f"unsampled parental lineage divergence masked the 4-gamete boundary."
                )

            gt_items_out.append(
                GroundTruthBreakpointItem(
                    gt_bp_orig=gt_bp,
                    gt_boundary_tx=gt_obj.get("gt_boundary_tx"),
                    gt_event_id=gt_obj.get("gt_event_id"),
                    event_type=str(gt_obj.get("event_type", "unknown")),
                    generation=gt_obj.get("generation"),
                    carrier_seqs=gt_obj.get("carrier_seqs", []),
                    detected=False,
                    matched_pred_event_id=None,
                    matched_pred_bp_orig=None,
                    bp_error=None,
                    nearest_subthreshold_peak_bp=near_pos,
                    nearest_subthreshold_score_z=near_z,
                    failure_hypothesis=reason,
                )
            )

    active_m = next((m for m in tol_metrics if m.tolerance_bp == active_tolerance_bp), tol_metrics[-1])
    med_str = f"{active_m.median_error_bp:.1f} bp" if active_m.median_error_bp is not None else "N/A"
    summary_notes = [
        (
            f"At ±{active_tolerance_bp} bp tolerance: {active_m.tp}/{ len(truth_bps) } true breakpoints detected "
            f"(Recall = {active_m.recall * 100:.1f}%), with {active_m.tp}/{ len(pred_bps) } predictions matching truth "
            f"(Precision = {active_m.precision * 100:.1f}%, F1 = {active_m.f1:.3f}, Lift = {active_m.lift:+.3f} over matched random control)."
        ),
        f"Median localisation error on detected events: {med_str}.",
    ]

    return DiagnosticBreakdown(
        has_ground_truth=True,
        active_tolerance_bp=active_tolerance_bp,
        n_ground_truth_breakpoints=len(truth_bps),
        n_predicted_events=len(consolidated_events),
        tolerance_metrics=tol_metrics,
        ground_truth_items=gt_items_out,
        summary_notes=summary_notes,
    )
