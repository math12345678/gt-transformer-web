# Project: gt-transformer-web Production Hardening

## Architecture
`gt-transformer-web` is a standalone browser-based recombination detection workbench providing:
- JAX-accelerated Window Transformer inference and statistical flanking Kullback-Leibler (KL) divergence analysis.
- Consolidated 3-level event extraction (Segment Tracts, Candidate Breakpoints, Breakpoint-to-Tract Links).
- Desktop Win32 RDP5 / OpenRDP styling with interactive alignment viewers, coordinate translation, and multi-format exports.

The system is decomposed into:
- **Core Inference Engine (`web/gt_core/` & `web/services/inference_adapter.py`)**: Vectorized feature extraction, window-stratified null calibration, dynamic power-of-2 chunk bucketing, and 3-level event consolidation.
- **HTTP Application & Validation Gateway (`web/app.py`, `web/services/fasta_utils.py`, `web/schemas.py`)**: Structured HTTP 400/422 validation, async thread-offloaded inference, locked atomic JSON persistence, and LRU alignment caching.
- **Frontend Workbench (`web/templates/index.html`, `web/static/js/app.js`, `web/static/css/styles.css`)**: Win32 desktop UI, inline error banners, non-blocking AbortController cancellation, alignment navigation, and export generators.
- **Automated Verification Harness (`web/tests/`)**: 6-battery stress test suite and Playwright headless Chromium E2E verification.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Window-Stratified Null Standardization | Standardize flanking KL by effective window width $w(b)$ to eliminate edge false positives | M1 | R1, Survey 1 |
| F2 | Random & Shuffled Null Elimination | 0 events on unlinked column-permuted FASTAs; >=14/15 clean zero on pure random FASTAs | M1 | R1, Survey 1 |
| F3 | Edge Breakpoint Deadzone Relief | Dynamic adaptive `eff_min_seg` down to 6 bp to recover $bp=80, 150, L-150, L-80$ within $\pm 30$ bp | M1 | R1, Survey 1 |
| F4 | Strict Hybrid Confirmation | Require $\text{min\_sw} \ge 1.0$ and corroborated evidence for transformer-only peaks | M1 | R1, Survey 1 |
| F5 | Bundled Core Engine (`gt_core/`) | Self-contained packaging of 7 inference modules in `web/gt_core/` | M2 | R4, Survey 2 |
| F6 | Bundled Checkpoint & Data | Self-contained `web/checkpoints/ckpt_N10000_extreme/`, `benchmark_data/`, `sample_data/` | M2 | R4, Survey 2 |
| F7 | Decoupled Runtime & Multi-Discovery | Remove hard `../` imports; dynamic python detection in `./run_server.sh`; optional `../` discovery | M2 | R4, Survey 2 |
| F8 | Standalone Clone Portability | Verify clean execution in isolated `/tmp` directory without parent files | M2 | R4, Survey 2 |
| F9 | Structured Input Validation Gateway | Reject empty, binary, unaligned (unequal lengths), all-gap, $S < 4$, sites $< 10$ with HTTP 400/422 | M3 | R2, Survey 2 |
| F10 | Short Alignment & RNA Support | Linear zero-padding for $M_{tx} < 314$; RNA U->T conversion; gap-tolerant fallback | M3 | R2, Survey 2 |
| F11 | Diagnostic Honesty & Attribution | Surface OOD warnings and signal attribution in API schemas and UI diagnostics | M3 | R2, Survey 2 |
| F12 | CPU Inference Latency Optimization | Dynamic power-of-2 chunk bucketing `[32, 64, 128, 256, 512]` achieving $< 5$s CPU on 12x3000 bp | M4 | R3, Survey 3 |
| F13 | Concurrency Safety & File Locking | `fcntl.flock(LOCK_EX)` + atomic temp-file replace for annotations and timings | M4 | R3, Survey 3 |
| F14 | Thread-Safe Bounded LRU Cache | `ThreadSafeLRUCache` with `maxsize = 8` using `OrderedDict` + `threading.Lock` | M4 | R3, Survey 3 |
| F15 | Asynchronous UX & Inline Banners | Replace `alert()` with inline `#globalErrorBanner`; add `AbortController` cancellation | M4 | R3, Survey 3 |
| F16 | 6-Battery Automated Test Suite | Batteries 1-5 covering null, empirical, geometry, divergence, and concurrency | M5 (E2E) | R5, Survey 3 |
| F17 | Playwright Headless Chromium E2E | 1440x900 and 1024x768 viewports, 0 console errors, options update, download verification | M5 (E2E) | R5, Survey 3 |
| F18 | Regression & Test Invariants | 100% pass on core tests and live web tests; clean git status | M5 (E2E) | R5, Survey 3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Statistical Calibration & Recall | F1, F2, F3, F4 (haldec.py, pipeline.py, calibration math) | none | DONE (Gate 1 Iter 3 PASS) |
| M2 | Standalone Packaging & Portability | F5, F6, F7, F8 (gt_core/, checkpoints/, run_server.sh) | M1 | IN_PROGRESS |
| M3 | Empirical Validation & Diagnostics | F9, F10, F11 (app.py, fasta_utils.py, schemas.py) | M2 | PLANNED |
| M4 | Latency, Concurrency & UX | F12, F13, F14, F15 (inference_adapter.py, timing.py, app.js, index.html) | M3 | PLANNED |
| M5 | 6-Battery Stress Testing & Playwright E2E | F16, F17, F18 (tests/test_battery_*.py, Playwright E2E, audit) | M4 | PLANNED |

## Interface Contracts
### `web/services/inference_adapter.py` ↔ `web/gt_core/haldec.py`
- `boundary_zscore_flank(H, L_w=250, beta=20.0, min_seg=50, power=1.0, rng=None, n_perm=3)`:
  - Input: Binarized haplotype matrix $H \in \{0, 1\}^{M \times S}$.
  - Returns: `(z, raw_score, null_mean_prof)` where `z` is window-stratified standardized z-score array of length $M+1$.
- `flank_soft_switch_detail(H, b, ...)`:
  - Supports dynamic `min_seg` down to 6 for sensitive edge breakpoint recovery.

### `web/app.py` ↔ `web/services/fasta_utils.py`
- `parse_and_validate_fasta(fasta_text, max_gap_frac=0.25)`:
  - Returns: `(headers, seqs, is_rna, gap_status)`.
  - Raises: `ValueError` with descriptive message on empty payload, binary/null-byte content, unaligned headers (unequal lengths), all-gap sequences, $S < 4$, or informative sites $< 10$.

### `web/schemas.py` ↔ Frontend & API Clients
- `InferenceResponse`:
  - `warnings: list[str] = Field(default_factory=list)`
- `ConsolidatedEvent`:
  - `signal_attribution: str = "hybrid_flank"` ("hybrid_flank", "flank_kl_heuristic_only", "transformer_corroborated")
  - `attribution_note: Optional[str] = None`

### `web/services/timing.py` & `web/app.py` (Persistence)
- `locked_atomic_json_update(filepath: Path, update_fn: Callable[[list[dict]], list[dict]]) -> list[dict]`:
  - Acquires `fcntl.flock(LOCK_EX)` on `.lock` sidecar, reads existing content (or `[]` if missing/corrupt), calls `update_fn`, writes to temporary file in same directory, and executes atomic `os.replace`.

## Code Layout
- `web/gt_core/`: Bundled core inference modules (`pipeline.py`, `model.py`, `train.py`, `haldec.py`, `report.py`, `data.py`, `decoder.py`).
- `web/checkpoints/`: Bundled model checkpoints (`ckpt_N10000_extreme/params.pkl`).
- `web/sample_data/`: Sample FASTAs for instant testing (`d005`, `test_neisseria`, `sarbecovirus_aligned`).
- `web/benchmark_data/`: Benchmark CSV and JSON summaries.
- `web/services/`: Engine adapter, FASTA utilities, benchmark loader, timing logger, thread-safe LRU cache.
- `web/templates/` & `web/static/`: Desktop Win32 RDP5/OpenRDP UI, error banners, cancellation controls, and styling.
- `web/tests/`: Unit tests, 6-battery automated stress tests, and Playwright headless Chromium E2E tests.
- `web/run_server.sh`: Self-contained boot script with dynamic Python interpreter resolution.
