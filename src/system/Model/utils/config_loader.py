# src/system/Model/utils/config_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable
import os
import yaml


def find_project_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for _ in range(6):
        if (cur / "experiments").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    # Fallback: current working directory
    return Path.cwd()


ROOT = find_project_root()
EXP_DIR = ROOT / "experiments"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config layer not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override composed YAML with environment variables set by the GUI.

    Supported env vars:
      - AI_INV_TICKERS: comma-separated, e.g. "AAPL,MSFT,AMZN"
      - AI_INV_START:   ISO date "YYYY-MM-DD"
      - AI_INV_END:     ISO date "YYYY-MM-DD"
    """
    tickers_env = os.getenv("AI_INV_TICKERS", "").strip()
    start_env = os.getenv("AI_INV_START", "").strip()
    end_env = os.getenv("AI_INV_END", "").strip()

    if tickers_env:
        tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
        if tickers:
            cfg.setdefault("universe", {})
            cfg["universe"]["tickers"] = tickers

    if start_env or end_env:
        cfg.setdefault("dates", {})
        if start_env:
            cfg["dates"]["start"] = start_env
        if end_env:
            cfg["dates"]["end"] = end_env

    return cfg


def build_config_dict(
    *,
    base: str = "base",
    feature_layers: Iterable[str] = (),
    model: str = "lstm",
    allocator: str = "softmax",
    experiment_name: str | None = None,
) -> Dict[str, Any]:
    """
    Compose a nested config dict from layered YAMLs:

      experiments/base.yaml
      experiments/features/<feat>.yaml  for each feat in feature_layers
      experiments/model/<model>.yaml
      experiments/allocator/<allocator>.yaml

    Environment overrides (set by the GUI) are applied at the end:
      AI_INV_TICKERS, AI_INV_START, AI_INV_END

    Returns a plain dict. View/Controller layer is responsible for turning it
    into an ExperimentConfig.
    """
    cfg: Dict[str, Any] = {}

    # 1) base
    cfg = _deep_update(cfg, _load_yaml(EXP_DIR / f"{base}.yaml"))

    # 2) features
    for feat in feature_layers:
        cfg = _deep_update(cfg, _load_yaml(EXP_DIR / "features" / f"{feat}.yaml"))

    # 3) model
    cfg = _deep_update(cfg, _load_yaml(EXP_DIR / "model" / f"{model}.yaml"))

    # 4) allocator
    cfg = _deep_update(cfg, _load_yaml(EXP_DIR / "allocator" / f"{allocator}.yaml"))

    # 5) auto experiment name if not provided
    if experiment_name is None:
        feats_slug = "_".join(feature_layers) if feature_layers else "nofeats"
        exp_name = f"{feats_slug}_{model}_{allocator}"
    else:
        exp_name = experiment_name

    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = exp_name

    # artifacts_root can be set in base.yaml; default if missing:
    cfg["experiment"].setdefault("artifacts_root", "artifacts")

    # 6) apply environment overrides from GUI (tickers/dates)
    cfg = _apply_env_overrides(cfg)

    return cfg
