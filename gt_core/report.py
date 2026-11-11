"""
End-to-end event report  --  S10 / T10.2, T10.2b, T10.4.

This turns the decoders into the **three-level output** Dr. Martin asked for, in
**original coordinates**:

    level 1  breakpoints                     (where the crossover is)
    level 2  recombinant sequences           (which sampled sequence carries it)
    level 3  parental-like sequences         (ranked closest sampled relatives)

Level 1 uses the **soft, parent-marginalised** haplotype score in `haldec.py`
(`--score kl` by default, the best localiser found: oracle lift +0.148 vs +0.090
for the 4-gamete pair support).  Peaks are **calibrated**: by default they must
exceed a z-score (`--z-thresh 3`) against the sequence-shift permutation null, so
the event count adapts to each dataset instead of filling a cap.  Level 2/3 come
from `haldec.soft_switch_detail`: every sequence is ranked by its own switch
evidence at the boundary, and the parental candidates are the ranked hard nearest
sampled relatives -- a **ranking**, not an identity, because the true parents are
essentially never sampled.

Honesty note.  The oracle in `haldec.py` measures the ceiling, and it is low (the
best score is ~2.4x a matched random-boundary control at the event level), so most
of these predicted events are expected false positives.  The report is the correct
*interface* and is meant to be read with that ceiling in mind -- which is exactly
what a real recombination detector (RDP5) also has to contend with.

Run:
    python3 report.py                       # all datasets with >=1 event
    python3 report.py --all                 # every dataset
    python3 report.py --datasets d013_L4584_s34_r005
    python3 report.py --score kl --beta 20 --min-seg 200 --radius 20
    python3 report.py --score hard          # the old hard-switch membership
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
    from . import decoder as DEC
    from . import haldec as H
except (ImportError, ValueError):
    import data as D
    import pipeline as P
    import decoder as DEC
    import haldec as H


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "events_report")


def load(dataset, cache=D.CACHE):
    meta = json.load(open(os.path.join(cache, dataset, "meta.json")))
    Hap = np.load(os.path.join(cache, dataset, "haplo.npy"))
    return meta, Hap


def consolidate_peaks(raw_events, merge_radius_bp=75, pick_by="score"):
    """Consolidate adjacent/nearby peaks sharing recombinant evidence into single events.
    
    Directly addresses Dr. Martin's 2026-10-29 review of the diagnostic bundle:
    listing dozens of distinct boundaries for the same biological recombination event
    is unparsimonious. This groups raw localiser peaks into cohesive events via
    connected components over:
      1. Coordinate distance in original alignment coordinates <= merge_radius_bp (default 75 bp)
      2. AND (shared primary recombinant carrier OR shared top-3 carriers).
      
    For each consolidated event, it selects the single most plausible breakpoint
    (the constituent peak with maximum score or `score * primary_switch`).
    """
    if not raw_events:
        return []
        
    p_list = []
    for p in raw_events:
        recombs = p.get("recombinants", [])
        recombs_sorted = sorted(recombs, key=lambda r: -r.get("switch", 0.0))
        cset = set(r["recombinant"] for r in recombs)
        primary = recombs_sorted[0]["recombinant"] if recombs_sorted else None
        max_sw = float(recombs_sorted[0].get("switch", 0.0)) if recombs_sorted else 0.0
        top3 = set(r["recombinant"] for r in recombs_sorted[:3])
        p_list.append({
            "boundary_tx": p.get("boundary_tx", 0),
            "breakpoint_orig": p["breakpoint_orig"],
            "score": p["score"],
            "max_sw": max_sw,
            "carrier_set": cset,
            "primary": primary,
            "top3": top3,
            "recombinants": recombs
        })
        
    N = len(p_list)
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            bp_dist = abs(p_list[i]["breakpoint_orig"] - p_list[j]["breakpoint_orig"])
            if bp_dist <= merge_radius_bp:
                if (p_list[i]["primary"] == p_list[j]["primary"] or 
                    (p_list[i]["top3"] & p_list[j]["top3"])):
                    adj[i].append(j)
                    adj[j].append(i)
                    
    visited = [False] * N
    consolidated = []
    for i in range(N):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        queue.append(nbr)
                        
            comp_peaks = [p_list[idx] for idx in comp]
            if pick_by == "prod":
                best_peak = max(comp_peaks, key=lambda x: x["score"] * x["max_sw"])
            elif pick_by == "sw":
                best_peak = max(comp_peaks, key=lambda x: (x["max_sw"], x["score"]))
            else:
                best_peak = max(comp_peaks, key=lambda x: x["score"])
            all_origs = [p["breakpoint_orig"] for p in comp_peaks]
            all_carriers = set.union(*[p["carrier_set"] for p in comp_peaks]) if comp_peaks else set()
            
            consolidated.append({
                "most_plausible_breakpoint_orig": best_peak["breakpoint_orig"],
                "boundary_tx": best_peak["boundary_tx"],
                "breakpoint_span_orig": [min(all_origs), max(all_origs)],
                "n_peaks_consolidated": len(comp_peaks),
                "score": best_peak["score"],
                "recombinants": best_peak["recombinants"],
                "all_carrier_seqs": sorted(all_carriers),
                "constituent_breakpoints_orig": sorted(all_origs)
            })
            
    consolidated = sorted(consolidated, key=lambda ev: ev["most_plausible_breakpoint_orig"])
    for idx, ev in enumerate(consolidated, 1):
        ev["event_id"] = idx
    return consolidated


def events_for(dataset, score_kind="kl", beta=20.0, thr=0.9, margin=0.02,
               min_seg=200, min_score=0.0, radius=20, max_peaks=64,
               member_frac=0.5, top_members=20, top_parents=3,
               z_thresh=None, n_perm=2, merge_radius_bp=75, flank_window=250,
               min_switch=1.0, rng=None, cache=D.CACHE):
    meta, Hap = load(dataset, cache)
    M = meta["M_tx"]
    L = int(meta["genome_length"])
    if score_kind == "flank_kl":
        eff_min_seg = 50 if min_seg == 200 else min_seg
        if z_thresh is not None:
            eff_z = 6.0 if z_thresh == 3.0 else z_thresh
            score, _raw, _null = H.boundary_zscore_flank(
                Hap, L_w=flank_window, beta=beta, min_seg=eff_min_seg,
                power=0.5, rng=rng, n_perm=n_perm
            )
            min_score = eff_z
        else:
            score = H.flank_kl_score(Hap, L_w=flank_window, min_seg=eff_min_seg,
                                     beta=beta, power=0.5)
    elif z_thresh is not None and score_kind == "kl":
        # calibrated peak threshold: events are z-scores against the
        # sequence-shift permutation null, so the count adapts per dataset.
        score, _raw, _null = H.boundary_zscore(Hap, beta=beta, min_seg=min_seg,
                                               kind=score_kind, rng=rng,
                                               n_perm=n_perm)
        min_score = z_thresh
        eff_min_seg = min_seg
    else:
        score = H.segment_score(Hap, score_kind, thrL=thr, thrR=thr, margin=margin,
                                beta=beta, min_seg=min_seg)
        eff_min_seg = min_seg
    peaks = H.pick_peaks(score, min_score, radius, max_peaks)
    _, _, maps = DEC.truth_for(dataset, cache)
    out = []
    for b in peaks:
        b_orig = P.unshrink(P.normalise_boundary(b, maps), maps, L)
        if score_kind == "flank_kl":
            detail = H.flank_soft_switch_detail(
                Hap, b, L_w=flank_window, beta=beta, min_seg=eff_min_seg,
                kind="kl", top_members=top_members, member_frac=member_frac,
                top_parents=top_parents
            )
            if min_switch > 0.0 and (not detail or detail[0]["switch"] < min_switch):
                continue
        elif score_kind in ("kl", "soft"):
            detail = H.soft_switch_detail(Hap, b, beta=beta, min_seg=min_seg,
                                          kind=score_kind, top_members=top_members,
                                          member_frac=member_frac,
                                          top_parents=top_parents)
        else:
            detail = H.switch_detail(Hap, b, thr, thr, margin, min_seg)
        out.append({
            "boundary_tx": int(b),
            "score": float(score[b]),
            "breakpoint_orig": int(b_orig),
            "n_switch": len(detail),
            "recombinants": detail,
        })
        
    pick_by = "prod" if score_kind == "flank_kl" else "score"
    consolidated = consolidate_peaks(out, merge_radius_bp=merge_radius_bp,
                                     pick_by=pick_by)
    
    return {
        "dataset": dataset,
        "genome_length": L,
        "n_sequences": int(Hap.shape[1]),
        "window": min(int(meta.get("M_tx", Hap.shape[0])), 314),
        "parameters": {"score": score_kind, "beta": beta, "thr": thr,
                       "margin": margin, "min_seg": eff_min_seg,
                       "min_score": min_score, "z_thresh": z_thresh,
                       "n_perm": n_perm, "radius": radius,
                       "max_peaks": max_peaks, "member_frac": member_frac,
                       "top_members": top_members,
                       "merge_radius_bp": merge_radius_bp},
        "n_events": len(out),
        "events": out,
        "n_consolidated_events": len(consolidated),
        "consolidated_events": consolidated,
    }


def write_dataset_csvs(rep, out_dir):
    """Write both consolidated and raw CSV reports for a dataset."""
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Consolidated breakpoints CSV (Q2 consolidated)
    cons_bp_path = os.path.join(out_dir, "breakpoints_original_alignment_kl_consolidated.csv")
    with open(cons_bp_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["event_id", "most_plausible_bp_orig", "span_orig_start", "span_orig_end",
                     "n_peaks_consolidated", "score_z", "boundary_tx", "n_switch_recombinants",
                     "recombinant_sequences"])
        for ev in rep.get("consolidated_events", []):
            recombs = [r["recombinant"] for r in ev.get("recombinants", [])]
            wr.writerow([
                ev["event_id"],
                ev["most_plausible_breakpoint_orig"],
                ev["breakpoint_span_orig"][0],
                ev["breakpoint_span_orig"][1],
                ev["n_peaks_consolidated"],
                f"{ev['score']:.4f}",
                ev.get("boundary_tx", ""),
                len(recombs),
                ";".join(str(s) for s in recombs)
            ])
            
    # 2. Consolidated sequence calls CSV (Q1 concise)
    cons_seq_path = os.path.join(out_dir, "sequence_calls_kl_consolidated.csv")
    with open(cons_seq_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["event_id", "most_plausible_bp_orig", "span_orig_start", "span_orig_end",
                     "seq_idx", "p_switch", "is_recombinant", "parent_left", "parent_right",
                     "ext_parent_left", "ext_parent_right", "left_sim", "right_sim"])
        for ev in rep.get("consolidated_events", []):
            carrier_set = set(ev.get("all_carrier_seqs", []))
            for r in ev.get("recombinants", []):
                ext_left = next((c["seq"] for c in r.get("left_candidates", []) if c["seq"] not in carrier_set), r["parent_left"])
                ext_right = next((c["seq"] for c in r.get("right_candidates", []) if c["seq"] not in carrier_set), r["parent_right"])
                wr.writerow([
                    ev["event_id"],
                    ev["most_plausible_breakpoint_orig"],
                    ev["breakpoint_span_orig"][0],
                    ev["breakpoint_span_orig"][1],
                    r["recombinant"],
                    f"{r['switch']:.4f}",
                    True,
                    r["parent_left"],
                    r["parent_right"],
                    ext_left,
                    ext_right,
                    f"{r['left_sim']:.4f}",
                    f"{r['right_sim']:.4f}"
                ])
                
    # 3. Consolidated sequence calls CSV (Q1 detailed with candidates)
    cons_det_path = os.path.join(out_dir, "sequence_calls_kl_detailed_consolidated.csv")
    with open(cons_det_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["event_id", "most_plausible_bp_orig", "span_orig_start", "span_orig_end",
                     "n_peaks_consolidated", "recombinant_seq", "p_switch", "is_recombinant",
                     "parent_left", "parent_right", "ext_parent_left", "ext_parent_right",
                     "left_sim", "left_cands_json", "right_sim", "right_cands_json"])
        for ev in rep.get("consolidated_events", []):
            carrier_set = set(ev.get("all_carrier_seqs", []))
            for r in ev.get("recombinants", []):
                ext_left = next((c["seq"] for c in r.get("left_candidates", []) if c["seq"] not in carrier_set), r["parent_left"])
                ext_right = next((c["seq"] for c in r.get("right_candidates", []) if c["seq"] not in carrier_set), r["parent_right"])
                wr.writerow([
                    ev["event_id"],
                    ev["most_plausible_breakpoint_orig"],
                    ev["breakpoint_span_orig"][0],
                    ev["breakpoint_span_orig"][1],
                    ev["n_peaks_consolidated"],
                    r["recombinant"],
                    f"{r['switch']:.4f}",
                    True,
                    r["parent_left"],
                    r["parent_right"],
                    ext_left,
                    ext_right,
                    f"{r['left_sim']:.4f}",
                    json.dumps(r.get("left_candidates", [])),
                    f"{r['right_sim']:.4f}",
                    json.dumps(r.get("right_candidates", []))
                ])

    # 4. Raw breakpoints CSV
    raw_bp_path = os.path.join(out_dir, "breakpoints_original_alignment_kl.csv")
    with open(raw_bp_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["boundary_tx", "score_z", "breakpoint_orig_alignment", "n_switch_recombinants"])
        for ev in rep.get("events", []):
            wr.writerow([
                ev["boundary_tx"],
                f"{ev['score']:.4f}",
                ev["breakpoint_orig"],
                ev["n_switch"]
            ])

    # 5. Raw sequence calls CSV
    raw_seq_path = os.path.join(out_dir, "sequence_calls_kl.csv")
    with open(raw_seq_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["breakpoint_orig_alignment", "seq_idx", "p_switch", "is_recombinant",
                     "parent_left", "parent_right", "left_sim", "right_sim"])
        for ev in rep.get("events", []):
            for r in ev.get("recombinants", []):
                wr.writerow([
                    ev["breakpoint_orig"],
                    r["recombinant"],
                    f"{r['switch']:.4f}",
                    True,
                    r["parent_left"],
                    r["parent_right"],
                    f"{r['left_sim']:.4f}",
                    f"{r['right_sim']:.4f}"
                ])

    # 6. Raw detailed sequence calls CSV
    raw_det_path = os.path.join(out_dir, "sequence_calls_kl_detailed.csv")
    with open(raw_det_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["breakpoint_orig_alignment", "recombinant_seq", "p_switch", "is_recombinant",
                     "parent_left", "left_sim", "left_cands_json", "parent_right", "right_sim",
                     "right_cands_json"])
        for ev in rep.get("events", []):
            for r in ev.get("recombinants", []):
                wr.writerow([
                    ev["breakpoint_orig"],
                    r["recombinant"],
                    f"{r['switch']:.4f}",
                    True,
                    r["parent_left"],
                    f"{r['left_sim']:.4f}",
                    json.dumps(r.get("left_candidates", [])),
                    r["parent_right"],
                    f"{r['right_sim']:.4f}",
                    json.dumps(r.get("right_candidates", []))
                ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--all", action="store_true",
                    help="include datasets with no events (default: skip them)")
    ap.add_argument("--score", choices=["kl", "flank_kl", "soft", "hard"], default="kl")
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--min-seg", type=int, default=200)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--z-thresh", type=float, default=3.0,
                    help="calibrated peak threshold on the KL z-score; "
                         "None/0 falls back to raw --min-score")
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--max-peaks", type=int, default=64)
    ap.add_argument("--member-frac", type=float, default=0.5)
    ap.add_argument("--top-members", type=int, default=20)
    ap.add_argument("--merge-radius-bp", type=int, default=75,
                    help="Radius in original alignment bp to consolidate adjacent peaks sharing carriers (default: 75)")
    ap.add_argument("--update-consolidated", action="store_true",
                    help="Read existing JSON reports in --out and update consolidation + CSVs without recomputing raw nulls")
    ap.add_argument("--csv", action="store_true",
                    help="Write per-dataset CSV reports (both consolidated and raw)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if args.datasets:
        configs = list(args.datasets)
    else:
        configs = D.all_configs()
        if not args.all:
            configs = [c for c in configs if H.gt_boundaries_tx(c)]

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for c in configs:
        json_path = os.path.join(args.out, c + ".json")
        if args.update_consolidated and os.path.exists(json_path):
            with open(json_path) as fh:
                rep = json.load(fh)
            rep["consolidated_events"] = consolidate_peaks(rep.get("events", []),
                                                           merge_radius_bp=args.merge_radius_bp)
            rep["n_consolidated_events"] = len(rep["consolidated_events"])
            rep.setdefault("parameters", {})["merge_radius_bp"] = args.merge_radius_bp
        else:
            rep = events_for(c, args.score, args.beta, args.thr, args.margin,
                             args.min_seg, args.min_score, args.radius,
                             args.max_peaks, args.member_frac, args.top_members,
                             z_thresh=args.z_thresh or None, n_perm=args.n_perm,
                             merge_radius_bp=args.merge_radius_bp,
                             rng=rng)
        with open(json_path, "w") as fh:
            json.dump(rep, fh, indent=1)
            
        if args.csv or args.datasets:
            ds_dir = os.path.join(args.out, c)
            write_dataset_csvs(rep, ds_dir)
            
        n_rows = sum(e["n_switch"] for e in rep["events"])
        n_cons_rows = sum(len(e["recombinants"]) for e in rep["consolidated_events"])
        rows.append({"dataset": c,
                     "n_events": rep["n_events"],
                     "n_consolidated_events": rep["n_consolidated_events"],
                     "n_recombinant_rows": n_rows,
                     "n_consolidated_rows": n_cons_rows,
                     "genome_length": rep["genome_length"],
                     "n_sequences": rep["n_sequences"]})
        print(f"  {c:<28} raw {rep['n_events']:>3} -> cons {rep['n_consolidated_events']:>2} events  "
              f"recomb rows {n_rows:>4} -> {n_cons_rows:>3}", flush=True)

    sum_path = os.path.join(args.out, "summary.csv")
    with open(sum_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["dataset", "n_events", "n_consolidated_events",
                                            "n_recombinant_rows", "n_consolidated_rows",
                                            "genome_length", "n_sequences"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n{len(rows)} datasets -> {args.out}/  (per-dataset JSON + {sum_path})")
    print(f"totals: {sum(r['n_events'] for r in rows)} raw peaks -> {sum(r['n_consolidated_events'] for r in rows)} consolidated events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
