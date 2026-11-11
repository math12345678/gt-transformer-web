"""
Adversarial Stress Verification Script for Milestone 1 Iteration 3
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

import app as web_app
from schemas import InferenceConfig
from services.fasta_utils import parse_fasta_text

def test_seed42_permutations():
    print("=" * 60)
    print("CHALLENGE 1: Seed 42 Column Permutation on d048 and d005")
    print("=" * 60)
    
    workspace_root = Path(__file__).resolve().parents[4]
    aln_targets = [
        ("d048_L4577_s79_r001", "WINDOWS_RDP5_PACKAGE/01_ALIGNMENTS/d048_L4577_s79_r001.fas"),
        ("d005_L3536_s96_r001", "WINDOWS_RDP5_PACKAGE/01_ALIGNMENTS/d005_L3536_s96_r001.fas"),
    ]
    
    for name, rel_path in aln_targets:
        aln_file = workspace_root / rel_path
        if not aln_file.exists():
            aln_file = Path(__file__).resolve().parents[2] / rel_path.split("gt-transformer/")[-1]
        assert aln_file.exists(), f"File {aln_file} not found"
        
        raw_text = aln_file.read_text(encoding="utf-8", errors="replace")
        names, seqs = parse_fasta_text(raw_text)
        L = len(seqs[0])
        
        for seed in [42, 123, 777]:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(L)
            shuf_text = "\n".join(
                f">{names[i]}\n{''.join(seqs[i][p] for p in perm)}"
                for i in range(len(names))
            )
            
            # flank_kl
            cfg_kl = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="flank_kl")
            res_kl = web_app.engine.run_inference(cfg_kl, fasta_text=shuf_text, fasta_filename=f"{name}_perm{seed}_kl.fas")
            kl_events = len(res_kl.consolidated_events)
            
            # hybrid_flank
            cfg_hyb = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
            res_hyb = web_app.engine.run_inference(cfg_hyb, fasta_text=shuf_text, fasta_filename=f"{name}_perm{seed}_hyb.fas")
            hyb_events = len(res_hyb.consolidated_events)
            
            print(f"[{name} - seed {seed}] kl_events: {kl_events}, hyb_events: {hyb_events}, max_z: {max(res_kl.track_profile.flank_kl_z):.2f}")
            assert kl_events == 0, f"FAILED: {name} under seed {seed} produced {kl_events} flank_kl events!"
            assert hyb_events == 0, f"FAILED: {name} under seed {seed} produced {hyb_events} hybrid_flank events!"

    print(">> Challenge 1 PASS: All column-permuted alignments under seed 42, 123, 777 produce 0 events.")

def test_ext_0101_detailed_trace():
    print("\n" + "=" * 60)
    print("CHALLENGE 2: Clonal Control ext_0101 Detailed Trace")
    print("=" * 60)
    cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
    res = web_app.engine.run_inference(cfg, preset_id="ext_0101_L3291_s27_low_mu_ctrl_mu01_ctrl")
    
    n_events = len(res.consolidated_events)
    print(f"ext_0101 consolidated events: {n_events}")
    print(f"Alignment: M_tx={res.alignment.n_informative_cols}, seqs={res.alignment.n_sequences}, length={res.alignment.length}")
    print(f"Max flank_kl_z: {max(res.track_profile.flank_kl_z):.2f}")
    assert n_events == 0, f"FAILED: ext_0101 produced {n_events} false-positive consolidated events!"
    print(">> Challenge 2 PASS: ext_0101 produces exactly 0 consolidated events.")

def test_convergent_homoplasy_detailed():
    print("\n" + "=" * 60)
    print("CHALLENGE 3: Convergent Homoplasy Stress (seeds 777, 888)")
    print("=" * 60)
    S = 16
    L = 3000
    BASES = np.array(["A", "C", "G", "T"])
    
    for seed in [777, 888]:
        rng = np.random.default_rng(seed)
        root = rng.choice(BASES, size=L)
        mat = np.tile(root, (S, 1))
        
        clades = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        for clade in clades:
            mut_pos = rng.choice(L, size=60, replace=False)
            for pos in mut_pos:
                old_b = root[pos]
                new_b = rng.choice([b for b in BASES if b != old_b])
                mat[clade, pos] = new_b
                
        for s in range(S):
            mut_pos = rng.choice(L, size=15, replace=False)
            for pos in mut_pos:
                old_b = mat[s, pos]
                new_b = rng.choice([b for b in BASES if b != old_b])
                mat[s, pos] = new_b
                
        homoplasy_positions = np.linspace(200, 2800, 15, dtype=int)
        for pos in homoplasy_positions:
            old_b = root[pos]
            conv_b = rng.choice([b for b in BASES if b != old_b])
            mat[clades[0], pos] = conv_b
            mat[clades[2], pos] = conv_b
            
        fasta_text = "\n".join(f">seq_{s:02d}\n" + "".join(mat[s]) for s in range(S))
        cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
        res = web_app.engine.run_inference(cfg, fasta_text=fasta_text, fasta_filename=f"homoplasy_{seed}.fasta")
        
        n_events = len(res.consolidated_events)
        max_z = float(np.max(res.track_profile.flank_kl_z))
        print(f"[Homoplasy seed {seed}] Events: {n_events}, max_z: {max_z:.2f}, M_tx: {res.alignment.n_informative_cols}")
        assert n_events == 0, f"FAILED: Convergent homoplasy seed {seed} produced {n_events} events!"

    print(">> Challenge 3 PASS: Convergent homoplasy tests produce exactly 0 consolidated events.")

def test_real_recombinants_preserved():
    print("\n" + "=" * 60)
    print("CHALLENGE 4: Real Recombinant Preservation & Signal Attribution")
    print("=" * 60)
    cfg = InferenceConfig(checkpoint="ckpt_N10000_extreme", scoring_mode="hybrid_flank")
    
    # 1. test_neisseria
    res_neisseria = web_app.engine.run_inference(cfg, preset_id="test_neisseria")
    print(f"test_neisseria events: {len(res_neisseria.consolidated_events)}")
    assert len(res_neisseria.consolidated_events) >= 1, "FAILED: test_neisseria lost recombinant event!"
    ev = res_neisseria.consolidated_events[0]
    print(f"  Neisseria event 0: bp={ev.start_bp}-{ev.end_bp}, attribution={ev.signal_attribution}, note='{ev.attribution_note}'")
    assert ev.signal_attribution == "flank_kl_heuristic_only", f"Unexpected attribution: {ev.signal_attribution}"
    assert "M_tx=62 < 314" in (ev.attribution_note or ""), f"Expected M_tx < 314 in note, got: {ev.attribution_note}"
    
    # 2. d048
    res_d048 = web_app.engine.run_inference(cfg, preset_id="d048_L4577_s79_r001")
    print(f"d048 events: {len(res_d048.consolidated_events)}")
    assert len(res_d048.consolidated_events) >= 2, f"FAILED: d048 lost events! Got {len(res_d048.consolidated_events)}"
    
    # 3. d005
    res_d005 = web_app.engine.run_inference(cfg, preset_id="d005_L3536_s96_r001")
    print(f"d005 events: {len(res_d005.consolidated_events)}")
    assert len(res_d005.consolidated_events) >= 3, f"FAILED: d005 lost events! Got {len(res_d005.consolidated_events)}"

    print(">> Challenge 4 PASS: Real recombinants and diagnostics fully preserved.")

if __name__ == "__main__":
    web_app.engine.warmup("ckpt_N10000_extreme")
    test_seed42_permutations()
    test_ext_0101_detailed_trace()
    test_convergent_homoplasy_detailed()
    test_real_recombinants_preserved()
    print("\nALL ADVERSARIAL CHALLENGES PASSED SUCCESSFULLY!")
