"""
gt_core: Bundled core GT-Transformer inference and calibration package.

Self-contained packaging of core inference modules for standalone web application:
- pipeline: Informative columns, binarisation, coordinate unshrinking
- model: JAX Window Transformer architecture, Config
- train: Model compilation and scoring (make_score)
- haldec: Statistical flanking KL divergence, null calibration, soft-switch detail
- report: 3-level event consolidation (consolidate_peaks)
- data: Dataset utilities and constants
- decoder: Recombination decoder and tract extraction
"""
from __future__ import annotations

import sys

# Submodules
from . import data
from . import decoder
from . import haldec
from . import model
from . import pipeline
from . import report
from . import train

# Register submodules into sys.modules under un-namespaced names IF NOT already present,
# ensuring legacy scripts and tests (e.g. `import pipeline as P`) continue to function
# seamlessly without duplicate module instances.
sys.modules.setdefault("data", data)
sys.modules.setdefault("decoder", decoder)
sys.modules.setdefault("haldec", haldec)
sys.modules.setdefault("model", model)
sys.modules.setdefault("pipeline", pipeline)
sys.modules.setdefault("report", report)
sys.modules.setdefault("train", train)

__all__ = [
    "data",
    "decoder",
    "haldec",
    "model",
    "pipeline",
    "report",
    "train",
]

__version__ = "1.0.0"
