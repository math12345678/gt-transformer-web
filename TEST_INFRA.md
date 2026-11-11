# E2E Test Infra: gt-transformer-web

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation internals.
- Derived directly from `ORIGINAL_REQUEST.md` and user-facing acceptance criteria.
- Methodology: Category-Partition + BVA + Pairwise + Real-World Workload Testing across 6 batteries.

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | Window-Stratified Null Standardization | R1 § Statistical Calibration | 5 | 5 | ✓ |
| 2 | Column-Shuffled & Random FASTA Elimination | R1 § Null False-Positive | 5 | 5 | ✓ |
| 3 | Edge Breakpoint Deadzone Relief & Recall | R1 § Edge Sensitivity | 5 | 5 | ✓ |
| 4 | Biological Realism & Hybrid Corroboration | R1 § Model Sensitivity | 5 | 5 | ✓ |
| 5 | Standalone Bundling & /tmp Clone Run | R4 § Self-Contained Repo | 5 | 5 | ✓ |
| 6 | Empirical Ingestion & HTTP 400/422 Validation | R2 § Structured Validation | 5 | 5 | ✓ |
| 7 | Short Alignment Linear Padding & RNA Support | R2 § Empirical Support | 5 | 5 | ✓ |
| 8 | Diagnostic Honesty & Signal Attribution | R2 § Diagnostic Honesty | 5 | 5 | ✓ |
| 9 | CPU Inference Latency (< 5s on 12x3000 bp) | R3 § Interactive Latency | 5 | 5 | ✓ |
| 10 | Concurrency Safety & File Locking | R3 § Concurrency Safety | 5 | 5 | ✓ |
| 11 | Thread-Safe Bounded LRU Cache (<= 8) | R3 § LRU Cache | 5 | 5 | ✓ |
| 12 | Non-blocking UX & Error Banners | R3 § Asynchronous UX | 5 | 5 | ✓ |
| 13 | 6-Battery Automated Test Suite | R5 § Stress Testing | 5 | 5 | ✓ |
| 14 | Playwright Headless Chromium E2E (Dual Viewport) | R5 § Headless Browser | 5 | 5 | ✓ |

## Test Architecture
- Test runner: Python standard unittest via `unittest discover -s tests -v` inside `web/` and `../.venv/bin/python`.
- Automated 6-battery test modules:
  - `web/tests/test_battery_1_null.py`: 15 random FASTAs (GC 30%, 50%, 70%), shuffled FASTAs (`d048`, `d005`, `ext_0076`), clonal controls (`d00005`, `d047`, `ext_0101`), convergent homoplasy.
  - `web/tests/test_battery_2_empirical.py`: HTTP uploads of `test_neisseria.fasta`, `sarbecovirus_aligned.fasta`, RNA alignments, gapped alignments; HTTP 400/422 validation on `CRF_07_test.fasta`, `sarbecovirus_5_aligned.fasta`, empty, unaligned, binary payloads.
  - `web/tests/test_battery_3_geometry.py`: Planted edge breakpoints ($bp=80, 150, L-150, L-80$), short tracts ($\ge 300$ bp), sibling carriers, orphan donors.
  - `web/tests/test_battery_4_divergence.py`: Divergence sensitivity sweep and benchmark consistency.
  - `web/tests/test_battery_5_concurrency.py`: 5 concurrent inference + window + export requests, lock integrity, LRU cache bounded to $\le 8$, and CPU latency $< 5,000$ ms.
  - `web/tests/test_battery_6_playwright.py`: Headless Chromium E2E testing at `1440x900` and `1024x768` viewports (0 console errors, 0 failed network requests, preset switching, file upload, error banner, alignment sync, .ini options, all 5 file downloads: TSV, 3 CSVs, Run JSON).
- Directory layout:
  - `web/tests/`

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full Sarbecovirus Analysis (19 seqs x 30,619 bp) | F9, F10, F11, F12, F14 | High |
| 2 | Neisseria Short Alignment Diagnostic (4 seqs x 787 bp) | F9, F10, F11 | Medium |
| 3 | Malformed / Adversarial HTTP Uploads (Empty, Unaligned, Binary) | F9, F15 | Medium |
| 4 | High Concurrency Load (5 concurrent users + exports) | F12, F13, F14 | High |
| 5 | Full Browser End-to-End Workflow with File Exports | F11, F14, F15, F17 | High |

## Coverage Thresholds
- Tier 1: >= 5 per feature
- Tier 2: >= 5 per feature (where boundaries exist)
- Tier 3: pairwise coverage of major feature interactions
- Tier 4: >= 5 realistic application scenarios
