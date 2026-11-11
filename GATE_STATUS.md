# Gate Status — gt-transformer-web Hardening

## Gate — Milestone 1 Iteration 3
| Agent | Role | Verdict | Source | Notes |
|-------|------|---------|--------|-------|
| worker_m1_iter3_1 | teamwork_preview_worker | DONE | handoff.md | Fixed haldec.py:491 eff_min, distinct parent check, multi-site contiguous tract evidence, negative logit veto, M_tx < 314 short alignment gate, schema field fix |
| reviewer_m1_iter3_1 | teamwork_preview_reviewer | APPROVE | handoff.md | 4/4 web live PASS, 5/5 battery 1 null PASS (0 FPs on ext_0101, homoplasy, permuted), 5/5 core suites PASS, 0 shortcuts/bypasses |
| reviewer_m1_iter3_2 | teamwork_preview_reviewer | APPROVE | handoff.md | Distinct parent check + contiguous tract support verified, test_neisseria preserved at bp=204 with honest flank_kl attribution |
| challenger_m1_iter3_2 | teamwork_preview_challenger | APPROVE | handoff.md | Planted edges 72.5% recall, short tracts 88.9% recall, d048 (2 events) & d005 (3 events) preserved without regression |
| auditor_m1_iter3_1 | teamwork_preview_auditor | CLEAN | handoff.md | Zero bypasses, zero hardcoding, authentic vector math verified |

Gate Result: **PASS**

All Milestone 1 criteria satisfied:
- Pure i.i.d. random FASTAs: 15/15 runs clean (0 FPs)
- Column-permuted biological FASTAs (d048, d005, ext_0076): 0 events under flank_kl and hybrid_flank
- Clonal negative controls (d00005, d047, ext_0101): 0 events
- Convergent homoplasy stress (15 unlinked SNPs): 0 events
- Biological sensitivity preserved (test_neisseria bp=204, d048 2 events, d005 3 events)
- Core coordinate and likelihood invariants: 100% PASS

---

## Gate — Milestone 1 Iteration 2
| Agent | Role | Verdict | Source | Notes |
|-------|------|---------|--------|-------|
| worker_m1_iter2_gen2 | teamwork_preview_worker | DONE | handoff.md | 4/4 web live tests pass, 5/5 core suites pass, 15/15 random nulls clean 0 |
| reviewer_m1_iter2_1 | teamwork_preview_reviewer | REQUEST_CHANGES | handoff.md | ext_0101 emits 2 FPs, convergent homoplasy emits 4 FPs (M_tx < 314 & z >= 8.0 filter bypass) |
| reviewer_m1_iter2_2 | teamwork_preview_reviewer | APPROVE | handoff.md | Mathematical model sound, deadzone relief b>=3 verified, 0 NaNs/IndexErrors |
| challenger_m1_iter2_1 | teamwork_preview_challenger | REJECT | handoff.md | Confirms ext_0101 2 FPs and homoplasy 4 FPs; no bypass found |
| challenger_m1_iter2_2_repl | teamwork_preview_challenger | APPROVE | handoff.md | 75% edge recall, 88.9% short tract recall, 100% carrier ID, z>35 on deadzone boundaries |
| auditor_m1_iter2_1 | teamwork_preview_auditor | CLEAN | handoff.md | Zero shortcuts/bypasses, zero hardcoding, authentic vector math |

Gate Result: **FAIL** (reviewer_m1_iter2_1 REQUEST_CHANGES, challenger_m1_iter2_1 REJECT)
