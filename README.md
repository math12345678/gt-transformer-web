# gt-transformer-web

Web-accessible recombination detection and diagnostic workbench for `gt-transformer`, modeled on **RDP5** and **OpenRDP**.

## Interface Layout (RDP5 / OpenRDP Design)

- **Top Toolbar & Informative-Site Overview Strip** — `Open` (FASTA upload + optional `.json`/`.csv`/SantaSim `.txt` ground truth), `Run`, `Options` (OpenRDP-style `.ini` parameter dialog), dataset selector, and export buttons (`Save OpenRDP TSV`, `Save Breakpoints CSV`, `Save Sequence Calls`, `Timings CSV`, `Save JSON`).
- **Top-Left Pane (Multiple Sequence Alignment Viewer)** — Displays actual nucleotides from the loaded FASTA with parsimony-informative sites highlighted, sequence list sidebar, coordinate jumper, and carrier/parent highlighting.
- **Top-Right Pane (Tabbed Analysis & Diagnostics)**
  - `Overview`: Consolidated breakpoint table (`Event #`, `Method`, `BP`, `Span`, `Peaks`, `flank_kl z`, `TF logit`, `Carriers`, `Ext Parents L→R`, `Status`).
  - `Matrices / Parents`: Level-2 recombinant carrier switch scores (`p_switch`) and Level-3 ranked parental-like candidates.
  - `TP / FP / FN`: Tolerant matching ($\pm 0, \pm 3, \pm 10, \pm 30\text{ bp}$) against ground truth and matched uniform random controls.
  - `Timings`: Stage-by-stage `time.perf_counter()` latency & throughput (`bp/s`, `cols/s`) plus persistent run history.
  - `10k Validation`: Scaling and out-of-range benchmark summaries loaded from disk (`scaling_results_summary.csv`, `out_of_range_results_summary.csv`).
  - `Notes`: Structured diagnostic annotations per event.
- **Bottom-Left Pane (Breakpoint Scan Plot & Side Controls)** — `flank_kl` z-score profile, permutation-null threshold band, Window Transformer probability trace, top GT/Predicted tick strips, cursor coordinate readout (`bp : z`), method/checkpoint selectors, and `Run`/`Options...` controls.
- **Bottom-Right Pane (Mosaic Sequence Map)** — Horizontal carrier sequence bars colored by inferred left and right external parental tracts with breakpoint and confidence spans.

## Architecture

- `app.py` — FastAPI server & HTTP endpoints (`/`, `/api/health`, `/api/presets`, `/api/benchmarks`, `/api/alignment/window`, `/api/infer`, `/api/timings`, `/api/timings/export.csv`, `/api/annotations`)
- `schemas.py` — Pydantic request/response contracts
- `services/inference_adapter.py` — `GTTransformerInferenceEngine` wired directly to `../pipeline.py`, `../model.py`, `../train.py`, `../haldec.py`, `../report.py` and saved checkpoints (`ckpt_N10000_extreme`, `ckpt_N10000`, `ckpt_N8000`, `ckpt_N4000`, `ckpt_N2000`, `ckpt_N1000`, `ckpt/model_4x_data.pkl`)
- `services/timing.py` — `TimingRecorder` and `TimingHistoryStore`
- `services/fasta_utils.py` — FASTA parser and ground-truth log parser
- `services/diagnostics.py` — Greedy closest-first tolerant matching & TP/FP/FN breakdown
- `services/benchmarks.py` — Loader for 10,000-alignment and extreme evaluation summaries

## Running

```bash
./run_server.sh
```

Then open `http://127.0.0.1:8050`.

## Testing

```bash
../.venv/bin/python -m unittest discover -s tests -v
```
