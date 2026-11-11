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


GAP_CHARACTERS = set("-.~?")


class AlignmentValidationError(ValueError):
    """Base exception for alignment validation errors carrying HTTP status codes."""

    def __init__(self, detail: str, status_code: int = 422):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class EmptyPayloadError(AlignmentValidationError):
    """HTTP 400: Empty or whitespace-only alignment input."""

    def __init__(self, detail: str = "Empty alignment payload"):
        super().__init__(detail=detail, status_code=400)


class BinaryPayloadError(AlignmentValidationError):
    """HTTP 400: Binary, null-byte, or non-text alignment content."""

    def __init__(self, detail: str = "Binary or non-text alignment content detected"):
        super().__init__(detail=detail, status_code=400)


class UnequalLengthError(AlignmentValidationError):
    """HTTP 422: Sequences have unequal lengths (unaligned)."""

    def __init__(self, detail: str = "Alignment sequences must have identical lengths"):
        super().__init__(detail=detail, status_code=422)


class InsufficientSequencesError(AlignmentValidationError):
    """HTTP 422: Alignment has fewer than 4 sequences."""

    def __init__(self, detail: str = "Alignment requires at least 4 sequences for recombination detection"):
        super().__init__(detail=detail, status_code=422)


class InsufficientInformativeSitesError(AlignmentValidationError):
    """HTTP 422: Alignment has fewer than 10 informative polymorphic sites."""

    def __init__(self, found: int, required: int = 10):
        super().__init__(
            detail=f"Insufficient informative polymorphic sites (found {found}, required >= {required}; only {found} parsimony-informative sites)",
            status_code=422,
        )


class AllGapSequenceError(AlignmentValidationError):
    """HTTP 422: A sequence contains only gap characters."""

    def __init__(self, seq_name: str):
        super().__init__(
            detail=f"Sequence '{seq_name}' contains only gap characters",
            status_code=422,
        )


def count_parsimony_informative_sites(
    seqs: list[str],
    max_gap_frac: float = 0.25,
) -> tuple[int, str]:
    """Count informative polymorphic sites across aligned sequences.

    Returns (count, mode) where mode is 'strict_acgt' or 'gap_tolerant'.
    """
    S = len(seqs)
    L = len(seqs[0]) if seqs else 0
    acgt = set("ACGT")
    strict_count = 0
    gap_count = 0

    for c in range(L):
        chars = [s[c] for s in seqs]
        non_acgt = sum(1 for ch in chars if ch not in acgt)
        # 1. Strict ACGT check
        if non_acgt == 0:
            counts: dict[str, int] = {}
            for ch in chars:
                counts[ch] = counts.get(ch, 0) + 1
            if len(counts) >= 2 and sorted(counts.values(), reverse=True)[1] >= 2:
                strict_count += 1
        # 2. Gap-tolerant check
        if non_acgt / max(S, 1) <= max_gap_frac:
            acgt_counts: dict[str, int] = {}
            for ch in chars:
                if ch in acgt:
                    acgt_counts[ch] = acgt_counts.get(ch, 0) + 1
            ordered = sorted(acgt_counts.values(), reverse=True)
            if len(ordered) >= 2 and ordered[1] >= 2:
                gap_count += 1

    if strict_count >= 30:
        return strict_count, "strict_acgt"
    if gap_count > strict_count:
        return gap_count, "gap_tolerant"
    return strict_count, "strict_acgt"


def parse_and_validate_fasta(
    fasta_text: str | bytes,
    max_gap_frac: float = 0.25,
    min_sequences: int = 4,
    min_informative: int = 10,
) -> tuple[list[str], list[str], bool, dict[str, Any]]:
    """Parse and rigorously validate FASTA alignment payload.

    Returns:
        (headers, seqs, is_rna, gap_status)
    Raises:
        EmptyPayloadError: Payload is empty or whitespace (HTTP 400).
        BinaryPayloadError: Null bytes or non-text binary data detected (HTTP 400).
        AlignmentValidationError: Invalid FASTA structure (HTTP 400/422).
        UnequalLengthError: Sequences have unequal lengths (HTTP 422).
        InsufficientSequencesError: Panel size S < min_sequences (HTTP 422).
        InsufficientInformativeSitesError: Informative sites < min_informative (HTTP 422).
        AllGapSequenceError: Any sequence contains only gap characters (HTTP 422).
    """
    if fasta_text is None:
        raise EmptyPayloadError("Empty alignment payload")

    if isinstance(fasta_text, bytes):
        if len(fasta_text.strip()) == 0:
            raise EmptyPayloadError("Empty alignment payload")
        if b"\x00" in fasta_text:
            raise BinaryPayloadError("Binary or non-text alignment content detected")
        try:
            text = fasta_text.decode("utf-8")
        except UnicodeDecodeError:
            raise BinaryPayloadError("Binary or non-text alignment content detected")
    else:
        if len(fasta_text.strip()) == 0:
            raise EmptyPayloadError("Empty alignment payload")
        if "\x00" in fasta_text:
            raise BinaryPayloadError("Binary or non-text alignment content detected")
        text = fasta_text

    # Non-printable control character check (excluding \r, \n, \t)
    if any(ord(c) < 32 and c not in "\r\n\t" for c in text):
        raise BinaryPayloadError("Binary or non-text alignment content detected")

    names: list[str] = []
    seqs_parts: list[list[str]] = []
    cur: list[str] | None = None
    saw_u = False

    for raw_line in text.splitlines():
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
                raise AlignmentValidationError(
                    "Invalid FASTA format: sequence data encountered before first '>' header.",
                    status_code=400,
                )
            cleaned = re.sub(r"\s+", "", line).upper()
            if "U" in cleaned:
                saw_u = True
                cleaned = cleaned.replace("U", "T")
            cur.append(cleaned)

    if not names:
        raise AlignmentValidationError(
            "No FASTA records ('>' headers) found in alignment.",
            status_code=400,
        )

    seqs = ["".join(parts) for parts in seqs_parts]
    if any(len(s) == 0 for s in seqs):
        raise AlignmentValidationError(
            "One or more FASTA records contain an empty sequence.",
            status_code=422,
        )

    # 1. Unaligned headers / unequal sequence lengths
    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise UnequalLengthError("Alignment sequences must have identical lengths")

    # 2. All-gap sequences
    for name, s in zip(names, seqs):
        if set(s).issubset(GAP_CHARACTERS):
            raise AllGapSequenceError(name)

    # 3. Panel size < 4 sequences
    S = len(seqs)
    if S < min_sequences:
        raise InsufficientSequencesError("Alignment requires at least 4 sequences for recombination detection")

    # 4. Informative sites < 10
    n_inf, mode = count_parsimony_informative_sites(seqs, max_gap_frac=max_gap_frac)
    if n_inf < min_informative:
        raise InsufficientInformativeSitesError(found=n_inf, required=min_informative)

    L = len(seqs[0]) if seqs else 0
    gap_status = {
        "genome_length_bp": L,
        "n_sequences": S,
        "n_informative": n_inf,
        "mode": mode,
        "is_rna": saw_u,
        "max_gap_frac": max_gap_frac,
    }
    return names, seqs, saw_u, gap_status


def parse_fasta_text(fasta_text: str) -> tuple[list[str], list[str]]:
    """Parse FASTA string into equal-length (sequence_names, sequences).

    Raises AlignmentValidationError / ValueError if the input is empty or not valid FASTA.
    """
    if not fasta_text or not fasta_text.strip():
        raise EmptyPayloadError("Empty alignment payload")
    if "\x00" in fasta_text:
        raise BinaryPayloadError("Binary or non-text alignment content detected")
    if any(ord(c) < 32 and c not in "\r\n\t" for c in fasta_text):
        raise BinaryPayloadError("Binary or non-text alignment content detected")

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
                raise AlignmentValidationError(
                    "Invalid FASTA format: sequence data encountered before first '>' header.",
                    status_code=400,
                )
            cleaned = re.sub(r"\s+", "", line).upper().replace("U", "T")
            cur.append(cleaned)

    if not names:
        raise AlignmentValidationError(
            "No FASTA records ('>' headers) found in alignment.",
            status_code=400,
        )

    seqs = ["".join(parts) for parts in seqs_parts]
    if any(len(s) == 0 for s in seqs):
        raise AlignmentValidationError(
            "One or more FASTA records contain an empty sequence.",
            status_code=422,
        )

    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise UnequalLengthError("Alignment sequences must have identical lengths")

    for name, s in zip(names, seqs):
        if set(s).issubset(GAP_CHARACTERS):
            raise AllGapSequenceError(name)

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
