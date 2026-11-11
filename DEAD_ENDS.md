# Dead Ends Log

| Iteration | Approach Tried | Why It Failed | Files Touched |
|-----------|---------------|---------------|---------------|
| M1 Iter 1 | Global sequence identity cutoff (`compute_max_pairwise_identity < 0.45`) + 8-bin quantile stratification in `boundary_zscore_flank` + hard `eff_min_seg >= 6` + hard `min_sw >= 1.0` | 1. Global identity cutoff fails on column-shuffled biological FASTAs (where identity > 70%), causing null scores to spike to z=19.15 and triggering test_02 regression.<br>2. Discrete quantile binning with n_perm=2 leads to noisy MAD estimates in edge strata.<br>3. Hard eff_min_seg >= 6 misses edge breakpoints (bp = L-80) with <= 5 informative sites.<br>4. Hard min_sw >= 1.0 dilutes 300 bp short tracts in 1000 bp windows and completely blinds detection to orphan donors. | `haldec.py`, `web/services/inference_adapter.py` |
