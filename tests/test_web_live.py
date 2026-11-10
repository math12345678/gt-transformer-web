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
        aln_path = (
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
        self.assertGreater(max_z_real, max_z_shuf * 1.5)
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
        ctrl_path = (
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


if __name__ == "__main__":
    unittest.main()
