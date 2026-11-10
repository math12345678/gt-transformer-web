"""
FASTA alignment reader/validator, alignment summary statistics, and
ground-truth log parser (JSON, CSV, SantaSim recombination_events_*.txt, or coordinate lists).
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any


def parse_fasta_text(fasta_text: str) -> tuple[list[str], list[str]]:
    """Parse FASTA string into equal-length (sequence_names, sequences).

    Raises ValueError if the input is empty or not valid FASTA.
    """
    names: list[str] = []
    seqs_parts: list[list[str]] = []
    cur: list[str] | None = None

    for raw_line in fasta_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            header = line[1:].strip() or f"seq_{len(names)}"
            names.append(header)
            seqs_parts.append([])
            cur = seqs_parts[-1]
        else:
            if cur is None:
                raise ValueError(
                    "Invalid FASTA format: sequence data encountered before first '>' header."
                )
            cleaned = re.sub(r"\s+", "", line).upper()
            cur.append(cleaned)

    if not names:
        raise ValueError("No FASTA records ('>' headers) found in alignment.")

    seqs = ["".join(parts) for parts in seqs_parts]
    if any(len(s) == 0 for s in seqs):
        raise ValueError("One or more FASTA records contain an empty sequence.")

    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        max_len = max(lengths)
        seqs = [s.ljust(max_len, "N") for s in seqs]

    return names, seqs


def compute_alignment_base_stats(seqs: list[str]) -> tuple[float, float]:
    """Return (gc_content_pct, ambiguity_pct) over the full alignment."""
    n_seq = len(seqs)
    length = len(seqs[0]) if seqs else 0
    total_bases = max(n_seq * length, 1)
    gc_count = 0
    non_acgt = 0
    for s in seqs:
        gc_count += s.count("G") + s.count("C")
        acgt = s.count("A") + s.count("C") + s.count("G") + s.count("T")
        non_acgt += len(s) - acgt
    return round(gc_count / total_bases * 100.0, 2), round(non_acgt / total_bases * 100.0, 2)


def parse_ground_truth_payload(
    raw_text: str,
    filename: str = "",
    genome_length_bp: int = 10000,
) -> list[dict[str, Any]]:
    """Parse uploaded or pasted ground-truth breakpoint logs.

    Supports:
      1. JSON list of event objects (matching `events.json` or `ground_truth_events.json`)
      2. SantaSim `recombination_events_*.txt` format (`EventNum * [bp1, bp2] * Generation`)
      3. CSV/TSV with columns like `breakpoint_orig`, `breakpoint`, `bp`, `Begin`, `End`
      4. Plain text / comma / newline separated list of integer base-pair positions
    """
    text = raw_text.strip()
    if not text:
        return []

    # 1. Try JSON first
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("events", data.get("ground_truth_events", [data]))
            items: list[dict[str, Any]] = []
            for idx, ev in enumerate(data, 1):
                if isinstance(ev, (int, float)):
                    bp = int(ev)
                    if 0 < bp < genome_length_bp:
                        items.append(
                            {
                                "gt_bp_orig": bp,
                                "gt_boundary_tx": None,
                                "gt_event_id": idx,
                                "event_type": "manual",
                                "generation": None,
                                "carrier_seqs": [],
                            }
                        )
                    continue
                if not isinstance(ev, dict):
                    continue
                ev_id = ev.get("event_id", idx)
                ev_type = str(ev.get("event_type", "unknown"))
                gen = ev.get("generation")
                bps = ev.get("breakpoints_orig") or ev.get("real_breakpoints") or ev.get("breakpoints") or []
                if isinstance(bps, (int, float)):
                    bps = [int(bps)]
                txs = ev.get("boundaries_tx") or []
                carriers = ev.get("carrier_seqs") or ev.get("recombinants") or []
                for b_i, bp_val in enumerate(bps):
                    bp_int = int(bp_val)
                    if bp_int <= 0 or bp_int >= genome_length_bp:
                        continue
                    tx_val = int(txs[b_i]) if b_i < len(txs) and txs[b_i] is not None else None
                    items.append(
                        {
                            "gt_bp_orig": bp_int,
                            "gt_boundary_tx": tx_val,
                            "gt_event_id": ev_id,
                            "event_type": ev_type,
                            "generation": int(gen) if gen is not None else None,
                            "carrier_seqs": [int(c) for c in carriers if str(c).isdigit()],
                        }
                    )
            return sorted(items, key=lambda x: x["gt_bp_orig"])
        except Exception:
            pass

    # 2. Try SantaSim raw `recombination_events_*.txt` format (`123*[450, 1200]*1500`)
    if "*" in text and "[" in text:
        ss_items: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or "EventNum" in line:
                continue
            parts = line.split("*")
            if len(parts) < 3:
                continue
            try:
                ev_id = int(parts[0].strip())
                gen = int(parts[2].strip())
                logged = [
                    int(x.strip())
                    for x in parts[1].strip("[]").split(",")
                    if x.strip().lstrip("-").isdigit()
                ]
                real = [b for b in logged if 0 < b < genome_length_bp]
                ev_type = "single" if len(real) == 1 and len(logged) >= 2 else ("double" if len(real) == 2 else "event")
                for bp_int in real:
                    ss_items.append(
                        {
                            "gt_bp_orig": bp_int,
                            "gt_boundary_tx": None,
                            "gt_event_id": ev_id,
                            "event_type": ev_type,
                            "generation": gen,
                            "carrier_seqs": [],
                        }
                    )
            except ValueError:
                continue
        if ss_items:
            return sorted(ss_items, key=lambda x: x["gt_bp_orig"])

    # 3. Try CSV/TSV if headers exist
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and ("," in lines[0] or "\t" in lines[0]):
        try:
            delim = "\t" if "\t" in lines[0] else ","
            reader = csv.DictReader(io.StringIO(text), delimiter=delim)
            if reader.fieldnames:
                items = []
                for idx, row in enumerate(reader, 1):
                    lower_map = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                    bp_candidates: list[int] = []
                    for col_key in (
                        "breakpoint_orig",
                        "breakpoints_orig",
                        "most_plausible_bp_orig",
                        "breakpoint",
                        "bp",
                        "position",
                        "begin",
                        "end",
                    ):
                        if col_key in lower_map and lower_map[col_key]:
                            for tok in re.split(r"[;,/\s]+", lower_map[col_key]):
                                if tok.isdigit():
                                    bp_candidates.append(int(tok))
                    for bp_int in bp_candidates:
                        if 0 < bp_int < genome_length_bp:
                            items.append(
                                {
                                    "gt_bp_orig": bp_int,
                                    "gt_boundary_tx": None,
                                    "gt_event_id": lower_map.get("event_id", idx),
                                    "event_type": lower_map.get("event_type", "csv_import"),
                                    "generation": None,
                                    "carrier_seqs": [],
                                }
                            )
                if items:
                    return sorted(items, key=lambda x: x["gt_bp_orig"])
        except Exception:
            pass

    # 4. Fallback: extract integer coordinates from plain text
    nums = [int(m) for m in re.findall(r"\b\d+\b", text)]
    valid_nums = sorted({n for n in nums if 0 < n < genome_length_bp})
    return [
        {
            "gt_bp_orig": bp,
            "gt_boundary_tx": None,
            "gt_event_id": idx,
            "event_type": "manual_coordinate",
            "generation": None,
            "carrier_seqs": [],
        }
        for idx, bp in enumerate(valid_nums, 1)
    ]
