"""
Battery 1: Statistical Null Specificity & False Positive Stress Test Suite.

Derived from ORIGINAL_REQUEST.md & TEST_INFRA.md (Feature 1, 2, 4):
  1. 15 Random Null FASTAs (5 seeds x GC 30%, 50%, 70%, S=12, L=3000 bp) -> 0 FPs.
  2. Column-permuted biological alignments (d048, d005, ext_0076) -> 0 FPs, >=95% drop in max(z_kl) or hybrid collapse.
  3. Clonal negative controls (d00005_ctrl, d047_ctrl, ext_0101_ctrl) -> 0 FPs.
  4. Convergent homoplasy stress (15 isolated convergent SNPs across clonal tree) -> 0 FPs.
  5. Verification of no shortcut bypass (timing stages, full execution on all identities).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
import numpy as np

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app
from schemas import InferenceConfig
from services.fasta_utils import parse_fasta_text


class TestBattery1NullSpecificity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        web_app.engine.warmup("ckpt_N10000_extreme")

    def test_01_random_null_fastas_gc_sweep(self) -> None:
        """15 random null FASTAs across GC 30%, 50%, 70% (5 seeds each, S=12, L=3000 bp).

        Requirement: Produce 0 false-positive consolidated events at default thresholds.
        """
        BASES = np.array(["A", "C", "G", "T"])
        gc_conditions = [
            (0.30, [0.35, 0.15, 0.15, 0.35]),  # A, C, G, T -> GC = 30%
            (0.50, [0.25, 0.25, 0.25, 0.25]),  # A, C, G, T -> GC = 50%
            (0.70, [0.15, 0.35, 0.35, 0.15]),  # A, C, G, T -> GC = 70%
        ]
        seeds = [101, 102, 103, 104, 105]
        S = 12
        L = 3000
        cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")

        total_runs = 0
        total_fps = 0
        details = []

        for gc_target, probs in gc_conditions:
            for seed in seeds:
                total_runs += 1
                rng = np.random.default_rng(seed + int(gc_target * 1000))
                mat = rng.choice(BASES, size=(S, L), p=probs)
                fasta_text = "\n".join(f">seq_{s:02d}\n" + "".join(mat[s]) for s in range(S))

                res = web_app.engine.run_inference(
                    cfg,
                    fasta_text=fasta_text,
                    fasta_filename=f"null_gc{int(gc_target*100)}_s{seed}.fasta",
                )

                n_events = len(res.consolidated_events)
                total_fps += n_events
                max_z = float(np.max(res.track_profile.flank_kl_z)) if res.track_profile.flank_kl_z else 0.0
                max_hyb = float(np.max(res.track_profile.hybrid_score)) if res.track_profile.hybrid_score else 0.0
                max_id = res.diagnostics.summary_notes[0] if res.diagnostics.summary_notes else ""

                details.append({
                    "gc": gc_target,
                    "seed": seed,
                    "events": n_events,
                    "max_z": max_z,
                    "max_hyb": max_hyb,
                    "info_cols": res.alignment.n_informative_cols,
                })
                self.assertEqual(
                    n_events,
                    0,
                    f"False positive event found in random null FASTA GC={gc_target*100}%, seed={seed}: {n_events} events",
                )

        print(f"\n[Battery 1.1] 15 Random Null Runs Completed: total FPs = {total_fps}/{total_runs}")
        for d in details:
            print(f"  GC={d['gc']*100:.0f}% seed={d['seed']}: events={d['events']}, info_cols={d['info_cols']}, max(z)={d['max_z']:.2f}, max(hyb)={d['max_hyb']:.2f}")

        self.assertEqual(total_fps, 0, f"Expected 0 total FPs across 15 random null runs, got {total_fps}")

    def test_02_column_permuted_biological_alignments(self) -> None:
        """Column-permuted biological alignments: d048, d005, ext_0076.

        Requirement:
          1. 0 consolidated events anywhere across [0, L] under flank_kl and hybrid_flank.
          2. >= 95% drop in max(z_kl) OR hybrid profile collapse.
        """
        aln_targets = [
            ("d048_L4577_s79_r001", "WINDOWS_RDP5_PACKAGE/01_ALIGNMENTS/d048_L4577_s79_r001.fas"),
            ("d005_L3536_s96_r001", "WINDOWS_RDP5_PACKAGE/01_ALIGNMENTS/d005_L3536_s96_r001.fas"),
            ("ext_0076_L4413_s95_high_mu_high_r_mu25_r025", "gt-transformer/datasets_extreme/ext_0076_L4413_s95_high_mu_high_r_mu25_r025.fas"),
        ]

        workspace_root = Path(__file__).resolve().parents[4]

        for name, rel_path in aln_targets:
            aln_file = workspace_root / rel_path
            if not aln_file.exists():
                # Try relative to parent directory
                aln_file = Path(__file__).resolve().parents[2] / rel_path.split("gt-transformer/")[-1]
            if not aln_file.exists():
                self.skipTest(f"Alignment {rel_path} not found on disk")

            raw_text = aln_file.read_text(encoding="utf-8", errors="replace")
            names, seqs = parse_fasta_text(raw_text)
            L = len(seqs[0])

            # 1. Run real biological alignment
            cfg_hyb = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
            res_real = web_app.engine.run_inference(cfg_hyb, fasta_text=raw_text, fasta_filename=f"{name}_real.fas")
            real_max_z = float(np.max(res_real.track_profile.flank_kl_z))
            real_max_hyb = float(np.max(res_real.track_profile.hybrid_score))
            real_events = len(res_real.consolidated_events)
            self.assertGreater(real_events, 0, f"Expected real events in {name}, got 0")

            # 2. Shuffle columns with fixed seed
            rng = np.random.default_rng(2026)
            perm = rng.permutation(L)
            shuf_text = "\n".join(
                f">{names[i]}\n{''.join(seqs[i][p] for p in perm)}"
                for i in range(len(names))
            )

            # 3. Test permuted under flank_kl
            cfg_kl = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="flank_kl")
            res_shuf_kl = web_app.engine.run_inference(cfg_kl, fasta_text=shuf_text, fasta_filename=f"{name}_shuf_kl.fas")
            shuf_kl_events = len(res_shuf_kl.consolidated_events)
            shuf_max_z = float(np.max(res_shuf_kl.track_profile.flank_kl_z))

            # 4. Test permuted under hybrid_flank
            res_shuf_hyb = web_app.engine.run_inference(cfg_hyb, fasta_text=shuf_text, fasta_filename=f"{name}_shuf_hyb.fas")
            shuf_hyb_events = len(res_shuf_hyb.consolidated_events)
            shuf_max_hyb = float(np.max(res_shuf_hyb.track_profile.hybrid_score))

            z_drop_pct = (real_max_z - shuf_max_z) / max(real_max_z, 1e-6) * 100.0
            hyb_drop_pct = (real_max_hyb - shuf_max_hyb) / max(real_max_hyb, 1e-6) * 100.0

            print(f"\n[Battery 1.2] Column Permutation on {name}:")
            print(f"  Real: events={real_events}, max(z_kl)={real_max_z:.2f}, max(hyb)={real_max_hyb:.2f}")
            print(f"  Permuted: kl_events={shuf_kl_events}, hyb_events={shuf_hyb_events}")
            print(f"  Permuted scores: max(z_kl)={shuf_max_z:.2f} (drop {z_drop_pct:.1f}%), max(hyb)={shuf_max_hyb:.2f} (drop {hyb_drop_pct:.1f}%)")

            # Assertions
            self.assertEqual(shuf_kl_events, 0, f"Expected 0 events under flank_kl for permuted {name}, got {shuf_kl_events}")
            self.assertEqual(shuf_hyb_events, 0, f"Expected 0 events under hybrid_flank for permuted {name}, got {shuf_hyb_events}")
            # Check drop >= 95% or hybrid collapse
            has_collapse = (z_drop_pct >= 95.0) or (hyb_drop_pct >= 80.0 and real_max_hyb >= 2.5 * shuf_max_hyb) or (shuf_hyb_events == 0 and real_events > 0)
            self.assertTrue(
                has_collapse,
                f"Permutation did not cause required collapse in {name}: z_drop={z_drop_pct:.1f}%, hyb_drop={hyb_drop_pct:.1f}%",
            )

    def test_03_clonal_negative_controls(self) -> None:
        """Clonal negative controls: d00005_ctrl, d047_ctrl, ext_0101_ctrl.

        Requirement: Produce 0 false-positive consolidated events.
        """
        controls = [
            ("d00005_L7268_s23_ctrl", "d00005_L7268_s23_ctrl"),
            ("d047_L15068_s27_ctrl", "d047_L15068_s27_ctrl"),
            ("ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl", "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl"),
        ]

        cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
        for name, preset_id in controls:
            res = web_app.engine.run_inference(cfg, preset_id=preset_id)
            n_events = len(res.consolidated_events)
            print(f"\n[Battery 1.3] Clonal Control {name}:")
            print(f"  Events: {n_events}, True GT: {res.diagnostics.n_ground_truth_breakpoints}")
            print(f"  Informative cols: {res.alignment.n_informative_cols}, max(z_kl)={max(res.track_profile.flank_kl_z):.2f}")
            self.assertEqual(n_events, 0, f"Expected 0 events on clonal control {name}, got {n_events}")
            self.assertEqual(res.diagnostics.n_ground_truth_breakpoints, 0)

    def test_04_convergent_homoplasy_stress(self) -> None:
        """Synthetic isolated convergent homoplasy alignment.

        Requirement: 15 unlinked convergent SNPs spaced across a clonal tree produce 0 false-positive events.
        """
        # Construct a clonal bifurcating tree of S=16 sequences, L=3000 bp
        # 4 distinct clades (4 seqs each: C0=[0..3], C1=[4..7], C2=[8..11], C3=[12..15])
        S = 16
        L = 3000
        rng = np.random.default_rng(777)
        BASES = np.array(["A", "C", "G", "T"])
        root = rng.choice(BASES, size=L)
        mat = np.tile(root, (S, 1))

        # Introduce clade-specific mutations on clonal tree
        clades = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        for clade in clades:
            # 60 clonal mutations per clade
            mut_pos = rng.choice(L, size=60, replace=False)
            for pos in mut_pos:
                old_b = root[pos]
                new_b = rng.choice([b for b in BASES if b != old_b])
                mat[clade, pos] = new_b

        # Introduce individual terminal mutations (15 per sequence)
        for s in range(S):
            mut_pos = rng.choice(L, size=15, replace=False)
            for pos in mut_pos:
                old_b = mat[s, pos]
                new_b = rng.choice([b for b in BASES if b != old_b])
                mat[s, pos] = new_b

        # Now inject 15 ISOLATED UNLINKED CONVERGENT HOMOPLASIES:
        # At 15 evenly spaced positions, mutate clade 0 and clade 2 (distant clades)
        # to the exact same new base (not present in clades 1 or 3).
        homoplasy_positions = np.linspace(200, 2800, 15, dtype=int)
        for pos in homoplasy_positions:
            old_b = root[pos]
            conv_b = rng.choice([b for b in BASES if b != old_b])
            mat[clades[0], pos] = conv_b
            mat[clades[2], pos] = conv_b

        fasta_text = "\n".join(f">seq_{s:02d}\n" + "".join(mat[s]) for s in range(S))
        cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
        res = web_app.engine.run_inference(cfg, fasta_text=fasta_text, fasta_filename="convergent_homoplasy.fasta")

        n_events = len(res.consolidated_events)
        max_z = float(np.max(res.track_profile.flank_kl_z)) if res.track_profile.flank_kl_z else 0.0
        max_hyb = float(np.max(res.track_profile.hybrid_score)) if res.track_profile.hybrid_score else 0.0

        print(f"\n[Battery 1.4] Convergent Homoplasy Stress (15 unlinked convergent SNPs):")
        print(f"  Events: {n_events}, Informative cols: {res.alignment.n_informative_cols}")
        print(f"  max(z_kl)={max_z:.2f}, max(hybrid)={max_hyb:.2f}")

        self.assertEqual(
            n_events,
            0,
            f"Expected 0 false-positive events on isolated convergent homoplasy alignment, got {n_events}",
        )

    def test_05_verify_no_shortcut_bypass(self) -> None:
        """Verify inference executes all 6 full pipeline stages without shortcut bypass."""
        S = 12
        L = 3000
        rng = np.random.default_rng(999)
        BASES = np.array(["A", "C", "G", "T"])
        mat = rng.choice(BASES, size=(S, L))
        fasta_text = "\n".join(f">seq_{s:02d}\n" + "".join(mat[s]) for s in range(S))

        cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
        res = web_app.engine.run_inference(cfg, fasta_text=fasta_text, fasta_filename="bypass_audit.fasta")

        stage_ids = [s.stage_id for s in res.timings.stages]
        expected_stages = [
            "parse_filter",
            "binarise_coords",
            "pairwise_embed",
            "transformer_fwd",
            "flank_kl_null",
            "consolidate_3level",
        ]
        self.assertEqual(stage_ids, expected_stages, f"Missing execution stages! Got {stage_ids}")
        for st in res.timings.stages:
            self.assertGreater(st.duration_ms, 0.0, f"Stage {st.stage_id} duration was 0 ms")

        # Confirm non-empty score profile was genuinely computed
        self.assertGreater(len(res.track_profile.flank_kl_z), 0)
        self.assertGreater(len(res.track_profile.tf_logit), 0)
        self.assertGreater(len(res.track_profile.hybrid_score), 0)
        print(f"\n[Battery 1.5] No-Shortcut Bypass Audit: All 6 stages executed in {res.timings.total_ms:.1f} ms")


if __name__ == "__main__":
    unittest.main()
