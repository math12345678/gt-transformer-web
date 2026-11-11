"""
Integration and causal-perturbation verification tests for the live gt-transformer web app.

Validates:
  1. Live engine health and checkpoint availability.
  2. Real end-to-end FASTA inference + `/api/alignment/window` nucleotide slicing.
  3. Column-permutation causal test: randomly shuffling alignment columns destroys
     spatial recombination linkage while preserving base frequencies, causing
     flank_kl and Transformer predictions to drop sharply.
  4. Clonal negative-control specificity (zero-recombination alignments).
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app  # noqa: E402
from schemas import InferenceConfig  # noqa: E402
from services.fasta_utils import parse_fasta_text  # noqa: E402


class TestLiveWebWorkbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        web_app.engine.warmup("ckpt_N10000_extreme")

    def test_01_live_engine_and_checkpoints(self) -> None:
        ckpts = web_app.engine.list_checkpoints()
        self.assertGreaterEqual(len(ckpts), 1)
        presets = web_app.engine.list_presets()
        self.assertGreaterEqual(len(presets), 1)

    def test_02_alignment_window_and_permutation_collapse(self) -> None:
        """Real alignment vs column-shuffled alignment proves predictions come from real linkage."""
        local_aln = Path(__file__).resolve().parents[1] / "sample_data" / "d048_L4577_s79_r001.fas"
        aln_path = local_aln if local_aln.exists() else (
            Path(__file__).resolve().parents[4]
            / "WINDOWS_RDP5_PACKAGE"
            / "01_ALIGNMENTS"
            / "d048_L4577_s79_r001.fas"
        )
        if not aln_path.exists():
            self.skipTest("Benchmark alignment d048_L4577_s79_r001.fas not present")

        raw_text = aln_path.read_text(encoding="utf-8")
        cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
        res_real = web_app.engine.run_inference(cfg, preset_id="d048_L4577_s79_r001")
        self.assertEqual(res_real.engine_mode, "live")
        self.assertGreater(res_real.model_param_count, 500000)

        # Verify /api/alignment/window serves real nucleotides from this run
        win = web_app.engine.get_alignment_window(res_real.timings.run_id, start_bp=100, width=80)
        self.assertEqual(len(win["sequences"]), res_real.alignment.n_sequences)
        self.assertEqual(len(win["sequences"][0]["slice"]), 80)

        # Shuffle columns identically across all sequences (preserves per-column & overall freqs, destroys linkage)
        names, seqs = parse_fasta_text(raw_text)
        L = len(seqs[0])
        perm = list(range(L))
        random.Random(7).shuffle(perm)
        shuffled_fasta = "\n".join(
            f">{names[i]}\n{''.join(seqs[i][p] for p in perm)}"
            for i in range(len(names))
        )
        res_shuf = web_app.engine.run_inference(
            cfg,
            fasta_text=shuffled_fasta,
            fasta_filename="d048_shuffled.fas",
        )
        max_z_real = max(res_real.track_profile.flank_kl_z)
        max_z_shuf = max(res_shuf.track_profile.flank_kl_z)
        self.assertGreater(max_z_real, max_z_shuf)
        max_hyb_real = max(res_real.track_profile.hybrid_score)
        max_hyb_shuf = max(res_shuf.track_profile.hybrid_score)
        self.assertGreater(max_hyb_real, max_hyb_shuf * 1.5)
        self.assertGreater(len(res_real.consolidated_events), len(res_shuf.consolidated_events))
        # Verify pure flank_kl produces zero interior false positives once spatial linkage is destroyed
        res_shuf_kl = web_app.engine.run_inference(
            InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="flank_kl"),
            fasta_text=shuffled_fasta,
            fasta_filename="d048_shuffled.fas",
        )
        interior_shuf_kl = [
            ev for ev in res_shuf_kl.consolidated_events
            if 200 <= ev.most_plausible_bp_orig <= (res_shuf_kl.alignment.genome_length_bp - 200)
        ]
        self.assertEqual(len(interior_shuf_kl), 0)

    def test_03_clonal_negative_control_specificity(self) -> None:
        """Unseen 10k clonal negative control yields 0 consolidated events."""
        local_ctrl = Path(__file__).resolve().parents[1] / "sample_data" / "d00005_L7268_s23_ctrl.fas"
        ctrl_path = local_ctrl if local_ctrl.exists() else (
            Path(__file__).resolve().parents[2]
            / "datasets_sims"
            / "d00005_L7268_s23_ctrl.fas"
        )
        if not ctrl_path.exists():
            self.skipTest("10k control dataset d00005_L7268_s23_ctrl.fas not present")
        res_ctrl = web_app.engine.run_inference(
            InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank"),
            preset_id="d00005_L7268_s23_ctrl",
        )
        self.assertTrue(res_ctrl.diagnostics.has_ground_truth)
        self.assertEqual(res_ctrl.diagnostics.n_ground_truth_breakpoints, 0)
        self.assertEqual(len(res_ctrl.consolidated_events), 0)

    def test_04_gapped_and_rna_fasta_upload(self) -> None:
        """Verify RNA U->T normalization and gap-tolerant fallback on real/gapped FASTAs."""
        # Part A: Synthetic gapped RNA FASTA (every column has >=1 gap + U so strict ACGT = 0)
        rng = random.Random(123)
        L = 2400
        a_base = [rng.choice("ACGU") for _ in range(L)]
        b_base = [rng.choice("ACGU") for _ in range(L)]

        def make_seq(tmpl: list[str], n_mut: int, gap_idx: int) -> str:
            arr = list(tmpl)
            for pos in rng.sample(range(L), n_mut):
                arr[pos] = rng.choice([x for x in "ACGU" if x != arr[pos]])
            arr[gap_idx::6] = ["-"] * len(arr[gap_idx::6])
            return "".join(arr)

        a0 = make_seq(a_base, 25, 0)
        a1 = make_seq(a_base, 25, 1)
        a2 = make_seq(a_base, 25, 2)
        b0 = make_seq(b_base, 25, 3)
        b1 = make_seq(b_base, 25, 4)
        b2 = make_seq(b_base, 25, 5)
        recomb = a0[:800] + b0[800:1600] + a0[1600:]
        rna_gapped_fasta = "\n".join(
            [
                f">A0\n{a0}",
                f">A1\n{a1}",
                f">A2\n{a2}",
                f">B0\n{b0}",
                f">B1\n{b1}",
                f">B2\n{b2}",
                f">Recomb_RNA_Gapped\n{recomb}",
            ]
        )
        res_rna = web_app.engine.run_inference(
            InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank"),
            fasta_text=rna_gapped_fasta,
            fasta_filename="rna_gapped.fasta",
            gt_text="800, 1600",
        )
        self.assertGreater(res_rna.alignment.n_informative_cols, 100)
        self.assertIn("gap-tolerant fallback", res_rna.timings.stages[1].notes)
        self.assertGreaterEqual(len(res_rna.consolidated_events), 2)
        m30 = next(m for m in res_rna.diagnostics.tolerance_metrics if m.tolerance_bp == 30)
        self.assertEqual(m30.tp, 2)

        # Part B: Real OpenRDP Neisseria alignment (4 seqs, 787 bp, 62 informative sites)
        neis_path = (
            Path(__file__).resolve().parents[4]
            / "tools"
            / "openrdp"
            / "tests"
            / "test_neisseria.fasta"
        )
        if neis_path.exists():
            res_neis = web_app.engine.run_inference(
                InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank"),
                fasta_text=neis_path.read_text(encoding="utf-8", errors="replace"),
                fasta_filename="test_neisseria.fasta",
            )
            self.assertEqual(res_neis.alignment.n_sequences, 4)
            self.assertEqual(res_neis.alignment.genome_length_bp, 787)
            self.assertEqual(res_neis.alignment.n_informative_cols, 62)

        # Part C: Sarbecovirus 5-sequence alignment has only 9 parsimony-informative sites (55 singletons)
        # and must fail gracefully with an informative ValueError rather than faking results.
        sarb_path = Path(__file__).resolve().parents[4] / "sarbecovirus_5_aligned.fasta"
        if sarb_path.exists():
            with self.assertRaises(ValueError) as ctx:
                web_app.engine.run_inference(
                    InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank"),
                    fasta_text=sarb_path.read_text(encoding="utf-8", errors="replace"),
                    fasta_filename="sarbecovirus_5_aligned.fasta",
                )
            self.assertIn("only 9 parsimony-informative sites", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
