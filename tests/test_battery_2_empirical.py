"""
Battery 2: Empirical Alignment Support & Structured HTTP 400/422 Validation Suite.

Validates:
  1. Empty payload rejection (HTTP 400: "Empty alignment payload")
  2. Binary / non-text payload rejection (HTTP 400: "Binary or non-text alignment content detected")
  3. Unaligned headers / unequal sequence lengths (HTTP 422: "Alignment sequences must have identical lengths")
  4. Panel size < 4 sequences on CRF_07_test.fasta & synthetics (HTTP 422: "Alignment requires at least 4 sequences for recombination detection")
  5. Insufficient informative sites on sarbecovirus_5_aligned.fasta & synthetics (HTTP 422: "Insufficient informative polymorphic sites (found X, required >= 10)")
  6. All-gap sequence rejection (HTTP 422: "Sequence 'X' contains only gap characters")
  7. Empirical HTTP 200 execution on test_neisseria.fasta, sarbecovirus_aligned.fasta, RNA (U->T), and gapped FASTAs
  8. JSON error shape invariant: always {"detail": "..."} and NEVER unhandled HTTP 500 tracebacks.
"""
from __future__ import annotations

import io
import random
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app
from services.fasta_utils import (
    AlignmentValidationError,
    AllGapSequenceError,
    BinaryPayloadError,
    EmptyPayloadError,
    InsufficientInformativeSitesError,
    InsufficientSequencesError,
    UnequalLengthError,
    count_parsimony_informative_sites,
    parse_and_validate_fasta,
)

SAMPLE_DIR = WEB_DIR / "sample_data"


class TestBattery2EmpiricalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(web_app.app)
        web_app.engine.warmup("ckpt_N10000_extreme")

    # --- HTTP 400: Empty Payloads ---
    def test_01_empty_payload_json(self) -> None:
        res = self.client.post("/api/infer", json={"fasta_text": ""})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

    def test_02_empty_payload_whitespace(self) -> None:
        res = self.client.post("/api/infer", json={"pasted_fasta": "   \n\t  \n"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

    def test_03_empty_payload_no_body(self) -> None:
        res = self.client.post("/api/infer", json={})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

    def test_04_empty_payload_file_upload(self) -> None:
        files = {"fasta_file": ("empty.fasta", io.BytesIO(b""), "text/plain")}
        res = self.client.post("/api/infer", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Empty alignment payload"})

    # --- HTTP 400: Binary / Non-Text Payloads ---
    def test_05_binary_payload_null_byte_file(self) -> None:
        content = b">seq1\nACGT\x00ACGT\n>seq2\nACGTACGT\n>seq3\nACGTACGT\n>seq4\nACGTACGT\n"
        files = {"fasta_file": ("binary.fasta", io.BytesIO(content), "application/octet-stream")}
        res = self.client.post("/api/infer", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Binary or non-text alignment content detected"})

    def test_06_binary_payload_raw_bytes(self) -> None:
        content = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"  # gzip magic header
        files = {"fasta_file": ("archive.tar.gz", io.BytesIO(content), "application/octet-stream")}
        res = self.client.post("/api/infer", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Binary or non-text alignment content detected"})

    def test_07_binary_payload_json_null_byte(self) -> None:
        res = self.client.post("/api/infer", json={"fasta_text": ">s1\nACGT\x00\n>s2\nACGT\n>s3\nACGT\n>s4\nACGT"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Binary or non-text alignment content detected"})

    # --- HTTP 422: Unaligned Headers / Unequal Sequence Lengths ---
    def test_08_unequal_sequence_lengths(self) -> None:
        payload = ">s1\nACGTACGTAC\n>s2\nACGTACGT\n>s3\nACGTACGTAC\n>s4\nACGTACGTAC\n"
        res = self.client.post("/api/infer", json={"fasta_text": payload})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Alignment sequences must have identical lengths"})

    # --- HTTP 422: Panel Size < 4 Sequences ---
    def test_09_panel_size_under_4_crf07(self) -> None:
        crf_path = SAMPLE_DIR / "CRF_07_test.fasta"
        raw = crf_path.read_text(encoding="utf-8")
        res = self.client.post("/api/infer", json={"fasta_text": raw, "fasta_filename": "CRF_07_test.fasta"})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Alignment requires at least 4 sequences for recombination detection"})

    def test_10_panel_size_under_4_synthetic(self) -> None:
        payload = ">s1\nACGTACGTAC\n>s2\nACGTACGTAC\n>s3\nACGTACGTAC\n"
        res = self.client.post("/api/infer", json={"fasta_text": payload})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Alignment requires at least 4 sequences for recombination detection"})

    # --- HTTP 422: Informative Sites < 10 ---
    def test_11_insufficient_informative_sites_sarbecovirus_5(self) -> None:
        sar_path = SAMPLE_DIR / "sarbecovirus_5_aligned.fasta"
        raw = sar_path.read_text(encoding="utf-8")
        res = self.client.post("/api/infer", json={"fasta_text": raw, "fasta_filename": "sarbecovirus_5_aligned.fasta"})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(
            res.json(),
            {"detail": "Insufficient informative polymorphic sites (found 9, required >= 10; only 9 parsimony-informative sites)"},
        )

    def test_12_insufficient_informative_sites_synthetic(self) -> None:
        # 4 sequences with exactly 3 informative polymorphic sites
        seq1 = "A" * 100
        seq2 = "A" * 100
        seq3 = "T" * 3 + "A" * 97
        seq4 = "T" * 3 + "A" * 97
        payload = f">s1\n{seq1}\n>s2\n{seq2}\n>s3\n{seq3}\n>s4\n{seq4}\n"
        res = self.client.post("/api/infer", json={"fasta_text": payload})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(
            res.json(),
            {"detail": "Insufficient informative polymorphic sites (found 3, required >= 10; only 3 parsimony-informative sites)"},
        )

    # --- HTTP 422: All-Gap Sequences ---
    def test_13_all_gap_sequence(self) -> None:
        payload = ">s1\nACGTACGTAC\n>seq_gap\n----------\n>s3\nACGTACGTAC\n>s4\nACGTACGTAC\n"
        res = self.client.post("/api/infer", json={"fasta_text": payload})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Sequence 'seq_gap' contains only gap characters"})

    def test_14_all_gap_sequence_mixed_chars(self) -> None:
        payload = ">s1\nACGTACGTAC\n>s2\n-.~.-.~.-.\n>s3\nACGTACGTAC\n>s4\nACGTACGTAC\n"
        res = self.client.post("/api/infer", json={"fasta_text": payload})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json(), {"detail": "Sequence 's2' contains only gap characters"})

    # --- HTTP 200: Valid Empirical Alignments ---
    def test_15_valid_empirical_neisseria(self) -> None:
        nei_path = SAMPLE_DIR / "test_neisseria.fasta"
        raw = nei_path.read_text(encoding="utf-8")
        res = self.client.post("/api/infer", json={"fasta_text": raw, "fasta_filename": "test_neisseria.fasta"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["alignment"]["n_sequences"], 4)
        self.assertEqual(data["alignment"]["genome_length_bp"], 787)
        self.assertEqual(data["alignment"]["n_informative_cols"], 62)
        self.assertIn("consolidated_events", data)
        self.assertEqual(len(data["consolidated_events"]), 1)
        ev = data["consolidated_events"][0]
        self.assertEqual(ev["most_plausible_bp_orig"], 204)
        self.assertEqual(ev["signal_attribution"], "flank_kl_heuristic_only")
        self.assertTrue(any("M_tx=62 < 314" in w for w in data["warnings"]))

    def test_16_valid_empirical_sarbecovirus(self) -> None:
        sar_path = SAMPLE_DIR / "sarbecovirus_aligned.fasta"
        raw = sar_path.read_text(encoding="utf-8")
        res = self.client.post("/api/infer", json={"fasta_text": raw, "fasta_filename": "sarbecovirus_aligned.fasta"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["alignment"]["n_sequences"], 19)
        self.assertEqual(data["alignment"]["genome_length_bp"], 30619)
        self.assertLess(data["timings"]["total_ms"], 5000.0)
        self.assertTrue(all(ev["signal_attribution"] == "flank_kl_heuristic_only" for ev in data["consolidated_events"]))

    def test_17_valid_rna_uracil_support(self) -> None:
        nei_path = SAMPLE_DIR / "test_neisseria.fasta"
        raw_rna = nei_path.read_text(encoding="utf-8").replace("T", "U").replace("t", "u")
        res = self.client.post("/api/infer", json={"fasta_text": raw_rna, "fasta_filename": "neisseria_rna.fasta"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["alignment"]["genome_length_bp"], 787)
        self.assertTrue(data["alignment"]["is_rna"])
        self.assertTrue(any("RNA input detected (U->T normalized)" in w for w in data["warnings"]))
        self.assertIn("is_rna=True (U->T normalized)", data["timings"]["stages"][0]["notes"])

    def test_18_valid_gap_tolerant_fallback(self) -> None:
        rng = random.Random(123)
        L = 2400
        a_base = [rng.choice("ACGT") for _ in range(L)]
        b_base = [rng.choice("ACGT") for _ in range(L)]

        def make_seq(tmpl: list[str], n_mut: int, gap_idx: int) -> str:
            arr = list(tmpl)
            for pos in rng.sample(range(L), n_mut):
                arr[pos] = rng.choice([x for x in "ACGT" if x != arr[pos]])
            arr[gap_idx::6] = ["-"] * len(arr[gap_idx::6])
            return "".join(arr)

        a0 = make_seq(a_base, 25, 0)
        a1 = make_seq(a_base, 25, 1)
        a2 = make_seq(a_base, 25, 2)
        b0 = make_seq(b_base, 25, 3)
        b1 = make_seq(b_base, 25, 4)
        b2 = make_seq(b_base, 25, 5)
        recomb = a0[:800] + b0[800:1600] + a0[1600:]
        gapped_fasta = "\n".join(
            [
                f">A0\n{a0}",
                f">A1\n{a1}",
                f">A2\n{a2}",
                f">B0\n{b0}",
                f">B1\n{b1}",
                f">B2\n{b2}",
                f">Recomb_Gapped\n{recomb}",
            ]
        )
        res = self.client.post("/api/infer", json={"fasta_text": gapped_fasta, "fasta_filename": "gapped.fasta", "gt_text": "800, 1600"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(any("Gap-tolerant fallback engaged" in w for w in data["warnings"]))
        self.assertIn("gap-tolerant fallback", data["timings"]["stages"][1]["notes"])
        self.assertGreaterEqual(len(data["consolidated_events"]), 2)
        m30 = next(m for m in data["diagnostics"]["tolerance_metrics"] if m["tolerance_bp"] == 30)
        self.assertEqual(m30["tp"], 2)

    # --- HTTP 500 Prevention Invariant ---
    def test_19_no_unhandled_500_tracebacks(self) -> None:
        adversarial_inputs = [
            "random junk without fasta headers",
            ">header_only_no_seq",
            ">s1\n>s2\n>s3\n>s4\n",
            ">s1\n" + "A" * 50 + "\n>s2\n" + "C" * 50,  # 2 sequences
        ]
        for adv in adversarial_inputs:
            res = self.client.post("/api/infer", json={"fasta_text": adv})
            self.assertIn(res.status_code, (400, 422))
            self.assertIn("detail", res.json())
            self.assertIsInstance(res.json()["detail"], str)


if __name__ == "__main__":
    unittest.main()
