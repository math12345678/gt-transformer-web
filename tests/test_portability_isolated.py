"""
Standalone Portability and Isolated Clone Verification Test Suite
Milestone 2: Standalone Packaging & Portability (Feature F5, F6, F7, F8)

Verifies:
1. Local gt_core imports and functions cleanly without parent directory on sys.path.
2. Local checkpoints (web/checkpoints/ckpt_N10000_extreme/params.pkl) load and warm up.
3. Local sample datasets (web/sample_data/) and benchmark data (web/benchmark_data/) are discovered.
4. Live inference runs standalone in an isolated directory disconnected from gt-transformer/.
5. End-to-end ./run_server.sh boots and serves /api/health, /api/presets, and /api/infer.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
import json

WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestStandalonePortability(unittest.TestCase):
    """Test suite ensuring gt-transformer-web is completely decoupled and portable."""

    def test_01_bundled_gt_core_integrity(self) -> None:
        """Verify all 7 core inference modules are present in web/gt_core and importable."""
        from gt_core import (
            data as D,
            decoder as DEC,
            haldec as H,
            model as M,
            pipeline as P,
            report as REP,
            train as T,
        )

        self.assertIsNotNone(P.informative_columns)
        self.assertIsNotNone(M.Config)
        self.assertIsNotNone(T.make_score)
        self.assertIsNotNone(H.boundary_zscore_flank)
        self.assertIsNotNone(H.flank_soft_switch_detail)
        self.assertIsNotNone(REP.consolidate_peaks)
        self.assertIsNotNone(D.CACHE)
        self.assertIsNotNone(DEC.gamete_pairs)

        # Verify critical bugfix in haldec.py:491
        import inspect
        lines, _ = inspect.getsourcelines(H.flank_soft_switch_detail)
        has_fix = any("eff_min = max(3, min_seg)" in line for line in lines)
        self.assertTrue(has_fix, "Critical bugfix eff_min = max(3, min_seg) is missing in gt_core/haldec.py")

    def test_02_bundled_checkpoint_and_data_integrity(self) -> None:
        """Verify bundled checkpoints, sample FASTA datasets, and benchmark data exist."""
        ckpt_path = WEB_ROOT / "checkpoints" / "ckpt_N10000_extreme" / "params.pkl"
        self.assertTrue(ckpt_path.is_file(), f"Bundled checkpoint missing: {ckpt_path}")
        self.assertGreater(ckpt_path.stat().st_size, 2_000_000, "Checkpoint file truncated")

        # Verify key sample FASTAs
        required_samples = [
            "d005_L3536_s96_r001.fas",
            "d048_L4577_s79_r001.fas",
            "d047_L15068_s27_ctrl.fas",
            "d00005_L7268_s23_ctrl.fas",
            "ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl.fas",
            "sarbecovirus_aligned.fasta",
            "test_neisseria.fasta",
        ]
        sample_dir = WEB_ROOT / "sample_data"
        for name in required_samples:
            sample_file = sample_dir / name
            self.assertTrue(sample_file.is_file(), f"Bundled sample FASTA missing: {name}")

        # Verify benchmark summaries
        bench_dir = WEB_ROOT / "benchmark_data"
        for name in ["scaling_results_summary.csv", "out_of_range_results_summary.csv", "eval_summary_N10000.json", "eval_comparison_before_after.json"]:
            self.assertTrue((bench_dir / name).is_file(), f"Bundled benchmark data missing: {name}")

    def test_03_inference_adapter_standalone_discovery(self) -> None:
        """Verify inference adapter discovers bundled checkpoints and presets without parent dir."""
        import app as web_app
        from schemas import InferenceConfig

        ckpts = web_app.engine.list_checkpoints()
        self.assertGreaterEqual(len(ckpts), 1)
        extreme_ckpt = next((c for c in ckpts if c["id"] == "ckpt_N10000_extreme"), None)
        self.assertIsNotNone(extreme_ckpt)
        self.assertTrue(extreme_ckpt["recommended"])

        presets = web_app.engine.list_presets()
        self.assertGreaterEqual(len(presets), 3)
        d005_preset = next((p for p in presets if p["id"] == "d005_L3536_s96_r001"), None)
        self.assertIsNotNone(d005_preset)

        # Run inference using bundled preset
        cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
        res = web_app.engine.run_inference(cfg, preset_id="d005_L3536_s96_r001")
        self.assertEqual(res.engine_mode, "live")
        self.assertGreater(len(res.consolidated_events), 0)

    def test_04_isolated_clone_execution(self) -> None:
        """Copy web/ to an isolated /tmp directory with no parent monorepo and verify server & API."""
        with tempfile.TemporaryDirectory(prefix="gt_isolated_test_") as tmp_dir:
            isolated_web = Path(tmp_dir) / "gt-transformer-web"
            
            # Copy web directory contents, excluding .git and .agents
            def ignore_patterns(path, names):
                ignored = set()
                if ".git" in names:
                    ignored.add(".git")
                if ".agents" in names:
                    ignored.add(".agents")
                if "__pycache__" in names:
                    ignored.add("__pycache__")
                return ignored

            shutil.copytree(WEB_ROOT, isolated_web, ignore=ignore_patterns)

            # Ensure parent of isolated directory contains NO monorepo files
            self.assertFalse((isolated_web.parent / "pipeline.py").exists())
            self.assertFalse((isolated_web.parent / "ckpt_N10000_extreme").exists())

            # Test Python interpreter locator
            test_py = sys.executable

            # Test running standalone inference script inside isolated directory
            verify_script = (
                "import sys\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, '.')\n"
                "from gt_core import pipeline as P, haldec as H, model as M, train as T\n"
                "import app as web_app\n"
                "from schemas import InferenceConfig\n"
                "ckpts = web_app.engine.list_checkpoints()\n"
                "assert len(ckpts) >= 1, 'No checkpoints discovered'\n"
                "presets = web_app.engine.list_presets()\n"
                "assert len(presets) >= 1, 'No presets discovered'\n"
                "cfg = InferenceConfig(checkpoint='ckpt_N10000_extreme', scoring_mode='flank_kl')\n"
                "res = web_app.engine.run_inference(cfg, preset_id='d005_L3536_s96_r001')\n"
                "assert len(res.consolidated_events) > 0, 'No events generated'\n"
                "print('ISOLATED_INFERENCE_OK')\n"
            )

            result = subprocess.run(
                [test_py, "-c", verify_script],
                cwd=str(isolated_web),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"Isolated inference script failed with code {result.returncode}:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
            )
            self.assertIn("ISOLATED_INFERENCE_OK", result.stdout)

            # Test ./run_server.sh boot and HTTP endpoints
            port = get_free_port()
            env = dict(os.environ)
            env["PORT"] = str(port)
            env["HOST"] = "127.0.0.1"
            env["PYTHON"] = test_py

            server_proc = subprocess.Popen(
                ["./run_server.sh"],
                cwd=str(isolated_web),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            server_ready = False
            base_url = f"http://127.0.0.1:{port}"
            try:
                for _ in range(30):
                    time.sleep(0.5)
                    try:
                        req = Request(f"{base_url}/api/health")
                        with urlopen(req, timeout=2) as resp:
                            if resp.status == 200:
                                data = json.loads(resp.read().decode("utf-8"))
                                if data.get("status") == "ok":
                                    server_ready = True
                                    break
                    except Exception:
                        pass

                self.assertTrue(server_ready, "Server failed to respond to /api/health within 15 seconds")

                # Test /api/presets
                req_presets = Request(f"{base_url}/api/presets")
                with urlopen(req_presets, timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    presets_data = json.loads(resp.read().decode("utf-8"))
                    self.assertGreaterEqual(len(presets_data.get("presets", [])), 1)

                # Test /api/infer
                infer_payload = json.dumps({
                    "preset_id": "d005_L3536_s96_r001",
                    "checkpoint": "ckpt_N10000_extreme",
                    "scoring_mode": "hybrid_flank",
                }).encode("utf-8")
                req_infer = Request(
                    f"{base_url}/api/infer",
                    data=infer_payload,
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(req_infer, timeout=60) as resp:
                    self.assertEqual(resp.status, 200)
                    infer_data = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(infer_data.get("status"), "ok")
                    self.assertGreater(len(infer_data.get("consolidated_events", [])), 0)

            finally:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                if server_proc.stdout:
                    server_proc.stdout.close()
                if server_proc.stderr:
                    server_proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
