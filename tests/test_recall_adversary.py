"""
Adversarial Stress Test Suite: True Recombinant Recall & Sensitivity
Challenger M1_2

Batteries:
1. Planted Edge Breakpoints: bp = 80, 150, L-150, L-80 (tolerance +-30 bp).
2. Short Tracts: lengths 300 bp, 400 bp, 500 bp (both breakpoints +-30 bp).
3. Sibling Carriers & Orphan-Donor alignments.
4. Window-Stratification Signal Attenuation & Suppression analysis.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np

GT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = GT_ROOT / "web"
if str(GT_ROOT) not in sys.path:
    sys.path.insert(0, str(GT_ROOT))
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

import pipeline as P
import haldec as H
import app as web_app
from schemas import InferenceConfig

web_app.engine.warmup("ckpt_N10000_extreme")
cfg_hybrid = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank", min_seg=20)
cfg_flank = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="flank_kl", min_seg=20)

BASES = np.array(["A", "C", "G", "T"])

def make_panel(seed: int, L: int = 3000):
    rng = np.random.default_rng(seed)
    root = rng.choice(BASES, size=L)
    
    def mutate(seq, rate):
        s = seq.copy()
        mask = rng.random(L) < rate
        s[mask] = rng.choice(BASES, size=int(mask.sum()))
        return s

    clade_A = mutate(root, 0.03)
    clade_B = mutate(root, 0.03)
    clade_C = mutate(root, 0.08)

    seqs, names = [], []
    for i in range(3):
        seqs.append("".join(mutate(clade_A, 0.005)))
        names.append(f"A_{i}")
    for i in range(3):
        seqs.append("".join(mutate(clade_B, 0.005)))
        names.append(f"B_{i}")
    for i in range(3):
        seqs.append("".join(mutate(clade_C, 0.005)))
        names.append(f"C_{i}")
    return rng, root, clade_A, clade_B, clade_C, seqs, names, mutate


# ============================================================================
# Battery 1: Planted Edge Breakpoints
# ============================================================================
def run_battery_1():
    print("\n" + "=" * 70)
    print("BATTERY 1: PLANTED EDGE BREAKPOINTS (bp = 80, 150, L-150, L-80)")
    print("=" * 70)

    seeds = [42, 100, 2026, 777, 999]
    lengths = [3000, 5000]
    results = []

    for L in lengths:
        for seed in seeds:
            rng, root, clade_A, clade_B, clade_C, seqs, names, mutate = make_panel(seed, L)
            test_bps = [
                ("bp=80", 80, lambda A, B: np.concatenate([A[:80], B[80:]])),
                ("bp=150", 150, lambda A, B: np.concatenate([A[:150], B[150:]])),
                (f"bp=L-150 ({L-150})", L - 150, lambda A, B: np.concatenate([A[:L-150], B[L-150:]])),
                (f"bp=L-80 ({L-80})", L - 80, lambda A, B: np.concatenate([A[:L-80], B[L-80:]])),
            ]

            for label, gt_bp, plant_fn in test_bps:
                r_seq = plant_fn(clade_A, clade_B)
                cur_seqs = list(seqs) + ["".join(r_seq)]
                cur_names = list(names) + ["Recombinant"]
                fasta = "\n".join(f">{cur_names[i]}\n{cur_seqs[i]}" for i in range(len(cur_names)))

                res = web_app.engine.run_inference(
                    cfg_hybrid,
                    fasta_text=fasta,
                    fasta_filename=f"edge_s{seed}_L{L}_{label}.fas",
                    gt_text=str(gt_bp),
                )
                preds = [ev.most_plausible_bp_orig for ev in res.consolidated_events]
                hits = [p for p in preds if abs(p - gt_bp) <= 30]
                best_err = min([abs(p - gt_bp) for p in preds]) if preds else None
                rec_carriers = [ev.recombinant_sequences for ev in res.consolidated_events]

                # Check informative column count in edge window
                kept = P.informative_columns(cur_seqs)
                cols, origin, _ = P.binarise(cur_seqs, kept, null_mode="duplicate")
                maps = P.build_maps(origin, L)
                b_tx = maps["boundary"][gt_bp]
                M_tx = len(cols)
                edge_cols = b_tx if gt_bp < L // 2 else (M_tx - b_tx)

                status = "PASS" if hits else "FAIL"
                results.append({
                    "L": L, "seed": seed, "label": label, "gt_bp": gt_bp,
                    "status": status, "preds": preds, "hits": hits, "best_err": best_err,
                    "edge_cols": edge_cols, "M_tx": M_tx,
                })
                print(f"[{status}] L={L} s={seed} {label:18s} GT={gt_bp:4d} | edge_cols={edge_cols:2d}/{M_tx:3d} | preds={preds} | hits={hits}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_total = len(results)
    print(f"\nBattery 1 Summary: {n_pass}/{n_total} PASSED ({n_pass/n_total*100:.1f}%)")
    return results


# ============================================================================
# Battery 2: Short Tracts (>= 300 bp)
# ============================================================================
def run_battery_2():
    print("\n" + "=" * 70)
    print("BATTERY 2: SHORT TRACT RECOMBINATION (>= 300 bp)")
    print("=" * 70)

    seeds = [42, 100, 2026]
    tract_lengths = [300, 400, 500]
    results = []
    L = 3000

    for tract_len in tract_lengths:
        for seed in seeds:
            rng, root, clade_A, clade_B, clade_C, seqs, names, mutate = make_panel(seed, L)
            bp1 = 1000
            bp2 = 1000 + tract_len
            r_seq = clade_A.copy()
            r_seq[bp1:bp2] = clade_B[bp1:bp2]
            cur_seqs = list(seqs) + ["".join(r_seq)]
            cur_names = list(names) + ["Recombinant_tract"]
            fasta = "\n".join(f">{cur_names[i]}\n{cur_seqs[i]}" for i in range(len(cur_names)))

            res = web_app.engine.run_inference(
                cfg_hybrid,
                fasta_text=fasta,
                fasta_filename=f"tract_{tract_len}_s{seed}.fas",
                gt_text=f"{bp1}, {bp2}",
            )
            preds = [ev.most_plausible_bp_orig for ev in res.consolidated_events]
            hit1 = [p for p in preds if abs(p - bp1) <= 30]
            hit2 = [p for p in preds if abs(p - bp2) <= 30]

            # Detailed check on soft switch evidence at true boundaries
            kept = P.informative_columns(cur_seqs)
            cols, origin, _ = P.binarise(cur_seqs, kept, null_mode="duplicate")
            maps = P.build_maps(origin, L)
            haplo = np.array([[int(ch) for ch in col] for col in cols], dtype=np.int8)
            M_tx = len(cols)
            eff_min_seg = min(20, max(6, int(M_tx * 0.015)))
            eff_L_w = min(250, max(16, M_tx // 3))
            z_kl, _, _ = H.boundary_zscore_flank(haplo, L_w=eff_L_w, beta=20.0, min_seg=eff_min_seg, power=1.0)
            
            b1_tx = maps["boundary"][bp1]
            b2_tx = maps["boundary"][bp2]
            dt1 = H.flank_soft_switch_detail(haplo, b1_tx, L_w=eff_L_w, beta=20.0, min_seg=eff_min_seg)
            dt2 = H.flank_soft_switch_detail(haplo, b2_tx, L_w=eff_L_w, beta=20.0, min_seg=eff_min_seg)
            sw1 = dt1[0]["switch"] if dt1 else 0.0
            sw2 = dt2[0]["switch"] if dt2 else 0.0

            both_hit = bool(hit1 and hit2)
            status = "PASS" if both_hit else "FAIL"
            results.append({
                "tract_len": tract_len, "seed": seed, "gt": [bp1, bp2],
                "status": status, "preds": preds, "hit1": hit1, "hit2": hit2,
                "sw1": sw1, "sw2": sw2, "z1": float(z_kl[b1_tx]), "z2": float(z_kl[b2_tx]),
            })
            print(f"[{status}] Tract {tract_len:3d} bp (s={seed}) GT=[{bp1}, {bp2}] | preds={preds} | bp1_hit={hit1} (sw={sw1:.2f}, z={z_kl[b1_tx]:.1f}) | bp2_hit={hit2} (sw={sw2:.2f}, z={z_kl[b2_tx]:.1f})")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_total = len(results)
    print(f"\nBattery 2 Summary: {n_pass}/{n_total} PASSED ({n_pass/n_total*100:.1f}%)")
    return results


# ============================================================================
# Battery 3: Sibling Carriers and Orphan-Donor Alignments
# ============================================================================
def run_battery_3():
    print("\n" + "=" * 70)
    print("BATTERY 3: SIBLING CARRIERS AND ORPHAN-DONOR ALIGNMENTS")
    print("=" * 70)

    seeds = [42, 100, 2026, 777]
    L = 3000
    bp1, bp2 = 1000, 1600
    results = []

    for seed in seeds:
        rng, root, clade_A, clade_B, clade_C, seqs, names, mutate = make_panel(seed, L)

        # Sub-case 3A: Sibling carriers
        r_sib1 = clade_A.copy()
        r_sib1[bp1:bp2] = clade_B[bp1:bp2]
        r_sib2 = mutate(r_sib1, 0.003)
        cur_seqs = list(seqs) + ["".join(r_sib1), "".join(r_sib2)]
        cur_names = list(names) + ["Sib1", "Sib2"]
        fasta = "\n".join(f">{cur_names[i]}\n{cur_seqs[i]}" for i in range(len(cur_names)))

        res = web_app.engine.run_inference(
            cfg_hybrid,
            fasta_text=fasta,
            fasta_filename=f"sibs_s{seed}.fas",
            gt_text=f"{bp1}, {bp2}",
        )
        preds = [ev.most_plausible_bp_orig for ev in res.consolidated_events]
        h1 = [p for p in preds if abs(p - bp1) <= 30]
        h2 = [p for p in preds if abs(p - bp2) <= 30]
        # Check if siblings 9 and 10 are recognized as carriers
        carriers_found = any((9 in ev.recombinant_sequences and 10 in ev.recombinant_sequences) for ev in res.consolidated_events)
        status_sibs = "PASS" if (h1 and h2 and carriers_found) else "FAIL"
        results.append({
            "type": "sibling", "seed": seed, "status": status_sibs,
            "preds": preds, "h1": h1, "h2": h2, "shared_carrier": carriers_found,
            "events": [(ev.most_plausible_bp_orig, ev.recombinant_sequences, round(ev.score_z, 1)) for ev in res.consolidated_events],
        })
        print(f"[{status_sibs}] Siblings (s={seed}) GT=[{bp1}, {bp2}] | preds={preds} | h1={h1} | h2={h2} | shared_carrier={carriers_found}")

        # Sub-case 3B: Orphan donor (donor clade D not present in panel)
        clade_D = mutate(root, 0.15)
        r_orphan = clade_A.copy()
        r_orphan[bp1:bp2] = clade_D[bp1:bp2]
        cur_seqs_o = list(seqs) + ["".join(r_orphan)]
        cur_names_o = list(names) + ["OrphanRec"]
        fasta_o = "\n".join(f">{cur_names_o[i]}\n{cur_seqs_o[i]}" for i in range(len(cur_names_o)))

        res_o = web_app.engine.run_inference(
            cfg_hybrid,
            fasta_text=fasta_o,
            fasta_filename=f"orphan_s{seed}.fas",
            gt_text=f"{bp1}, {bp2}",
        )
        preds_o = [ev.most_plausible_bp_orig for ev in res_o.consolidated_events]
        h1_o = [p for p in preds_o if abs(p - bp1) <= 30]
        h2_o = [p for p in preds_o if abs(p - bp2) <= 30]
        rec9_found = any(9 in ev.recombinant_sequences for ev in res_o.consolidated_events)
        status_orph = "PASS" if (h1_o and h2_o and rec9_found) else "FAIL"
        results.append({
            "type": "orphan", "seed": seed, "status": status_orph,
            "preds": preds_o, "h1": h1_o, "h2": h2_o, "rec9": rec9_found,
            "events": [(ev.most_plausible_bp_orig, ev.recombinant_sequences, round(ev.score_z, 1)) for ev in res_o.consolidated_events],
        })
        print(f"[{status_orph}] Orphan donor (s={seed}) GT=[{bp1}, {bp2}] | preds={preds_o} | h1={h1_o} | h2={h2_o} | carrier9={rec9_found}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_total = len(results)
    print(f"\nBattery 3 Summary: {n_pass}/{n_total} PASSED ({n_pass/n_total*100:.1f}%)")
    return results


# ============================================================================
# Battery 4: Window-Stratification Signal Attenuation Analysis
# ============================================================================
def run_battery_4():
    print("\n" + "=" * 70)
    print("BATTERY 4: WINDOW-STRATIFICATION SIGNAL ATTENUATION")
    print("=" * 70)

    # Check whether window stratification artificially suppresses signal at edge vs interior
    seed = 100
    L = 3000
    rng, root, clade_A, clade_B, clade_C, seqs, names, mutate = make_panel(seed, L)

    # Plant single crossover at various positions from edge to interior
    positions = [80, 150, 300, 600, 1000, 1500, 2000, 2400, 2700, 2850, 2920]
    z_scores = []
    raw_scores = []
    top_switches = []

    for bp in positions:
        r = np.concatenate([clade_A[:bp], clade_B[bp:]])
        cur_seqs = list(seqs) + ["".join(r)]
        kept = P.informative_columns(cur_seqs)
        cols, origin, _ = P.binarise(cur_seqs, kept, null_mode="duplicate")
        maps = P.build_maps(origin, L)
        haplo = np.array([[int(ch) for ch in col] for col in cols], dtype=np.int8)
        M_tx = len(cols)
        eff_min_seg = min(20, max(6, int(M_tx * 0.015)))
        eff_L_w = min(250, max(16, M_tx // 3))
        b_tx = maps["boundary"][bp]
        w_eff = min(eff_L_w, b_tx, M_tx - b_tx)

        z_kl, raw_score, _ = H.boundary_zscore_flank(haplo, L_w=eff_L_w, beta=20.0, min_seg=eff_min_seg, power=1.0)
        dt = H.flank_soft_switch_detail(haplo, b_tx, L_w=eff_L_w, beta=20.0, min_seg=eff_min_seg)
        top_sw = dt[0]["switch"] if dt else 0.0

        z_val = float(z_kl[b_tx]) if 0 <= b_tx < len(z_kl) else 0.0
        raw_val = float(raw_score[b_tx]) if 0 <= b_tx < len(raw_score) else 0.0
        z_scores.append(z_val)
        raw_scores.append(raw_val)
        top_switches.append(top_sw)

        print(f"bp={bp:4d} | b_tx={b_tx:3d}/{M_tx:3d} | w_eff={w_eff:2d} | z_kl={z_val:6.2f} | raw_kl={raw_val:7.2f} | top_sw={top_sw:5.2f}")

    return positions, z_scores, raw_scores, top_switches


if __name__ == "__main__":
    t0 = time.time()
    b1_res = run_battery_1()
    b2_res = run_battery_2()
    b3_res = run_battery_3()
    b4_res = run_battery_4()
    print(f"\nTotal test suite runtime: {time.time() - t0:.2f} s")
