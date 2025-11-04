# src/system/Model/utils/repro.py
from __future__ import annotations
import os, random
import numpy as np
import torch
import logging
from typing import Optional

_log = logging.getLogger(__name__)


def configure_reproducibility(
    *,
    seed_python: int,
    seed_numpy: int,
    seed_torch: int,
    device_pref: Optional[str] = None,  # "cuda" | "mps" | "cpu" | None(auto)
    strict: Optional[bool] = None,  # if None, read AI_INV_STRICT env (default True)
) -> str:
    """
    Set global seeds and deterministic options *before* any model/loaders are constructed.
    Returns the selected device name ("cuda"|"mps"|"cpu").
    """
    # ---- resolve device target (do NOT force move; this is just for logging/knobs) ----
    env_pref = os.getenv("AI_INV_DEVICE", "").lower() or None
    pref = (device_pref or env_pref or "").lower()

    if pref == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif pref == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        # auto preference: cuda > mps > cpu
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # ---- seeds ----
    random.seed(seed_python)
    np.random.seed(seed_numpy)
    torch.manual_seed(seed_torch)
    # harmless on non-CUDA:
    torch.cuda.manual_seed_all(seed_torch)

    # ---- strict determinism toggle ----
    # default: True unless explicitly disabled via env
    if strict is None:
        strict = os.getenv("AI_INV_STRICT", "1") not in ("0", "false", "False")

    if strict:
        # Deterministic algorithms where supported (raises if a non-deterministic op is used)
        torch.use_deterministic_algorithms(True)

    # ---- CUDA-only knobs (guarded). Use *new* TF32 API to avoid deprecation warning. ----
    if device == "cuda":
        # New API: prefer IEEE-754 FP32 math to avoid TF32 stochasticity differences
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        except Exception:
            pass
        try:
            torch.backends.cudnn.conv.fp32_precision = "ieee"  # or "tf32" if you want TF32
        except Exception:
            pass

        # cuDNN deterministic vs benchmark
        try:
            import torch.backends.cudnn as cudnn

            cudnn.benchmark = False
            cudnn.deterministic = bool(strict)
        except Exception:
            pass

        # cublas workspace for bitwise repeatability on CUDA
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # NOTE: On MPS/CPU we do not touch CUDA backends at all, which avoids the warning.

    _log.debug(
        "[repro] device=%s seeds(py=%d, np=%d, torch=%d) strict=%s",
        device,
        seed_python,
        seed_numpy,
        seed_torch,
        strict,
    )
    return device

def _seed_worker(worker_id: int):
    """
    Ensures each worker has a deterministic RNG state derived from the main generator.
    Works cross-platform (Windows uses spawn).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)