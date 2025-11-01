# src/system/Views/cli.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json

# --- Model imports ---
from system.Model.experiment_config import ExperimentConfig
from system.Model.artifact_store import ArtifactStore
from system.Model.data.data_sources.prices import PriceDataSource
from system.Controller.cache_manager import CacheManager
from system.Model.data.trading_calendar import TradingCalendar
from system.Model.data.data_validator import DataValidator
from system.Model.features.technical_features import TechnicalFeatureBuilder
from system.Model.backtesting.backtester import Backtester

# --- Controller imports ---
from system.Controller.price_manager import PriceManager
from system.Controller.feature_pipeline import FeaturePipeline
from system.Controller.training_manager import TrainingManager
from system.Controller.allocation_manager import AllocationManager
from system.Controller.run_manager import RunManager


def _build_services(cfg: ExperimentConfig) -> tuple[RunManager, ArtifactStore]:
    """
    Construct the ArtifactStore and all Phase-1 managers, then assemble RunManager.
    The RunManager will call store.new_run(...) when runner.run() is called.
    """
    store = ArtifactStore(cfg.artifacts_root)

    # --- shared services (Model layer) ---
    cache = CacheManager(store)
    calendar = TradingCalendar()
    validator = DataValidator()
    source = PriceDataSource()

    # --- controllers ---
    price_mgr = PriceManager(source=source, cache=cache, calendar=calendar, validator=validator)
    feat_pipe = FeaturePipeline(cache=cache, technical_builder=TechnicalFeatureBuilder())
    trainer_mgr = TrainingManager(store=store)
    alloc_mgr = AllocationManager(store=store)
    backtester = Backtester(
        transaction_cost_bps=cfg.backtest.transaction_cost_bps,
        long_only=cfg.backtest.long_only,
    )

    runner = RunManager(
        cfg=cfg,
        store=store,
        price_manager=price_mgr,
        feature_pipeline=feat_pipe,
        training_manager=trainer_mgr,
        allocation_manager=alloc_mgr,
        backtester=backtester,
        cache=cache,
        calendar=calendar,
    )
    return runner, store


def cmd_run(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.from_yaml(Path(args.config))
    runner, _ = _build_services(cfg)
    runner.run()  # orchestrates Phase-1: prices -> tech features -> LSTM -> softmax -> backtest


def cmd_list_runs(args: argparse.Namespace) -> None:
    # If a config is provided, honor its artifacts_root; else default to 'artifacts/'
    root = Path(args.artifacts_root) if args.artifacts_root else Path("artifacts")
    store = ArtifactStore(root, create=True)
    runs = list(store.list_runs())
    if not runs:
        print("(no runs found)")
        return
    for r in runs:
        print(r)


def cmd_show_metrics(args: argparse.Namespace) -> None:
    root = Path(args.artifacts_root) if args.artifacts_root else Path("artifacts")
    store = ArtifactStore(root, create=True)

    run_id = args.run_id or store.latest_run()
    if not run_id:
        print("No runs available.")
        return

    store.ensure_existing_run(run_id)
    metrics_path = store.backtest_metrics_path()
    if not metrics_path.exists():
        print(f"Metrics not found for run: {run_id}")
        return

    metrics = json.loads(metrics_path.read_text())
    print(f"Run: {run_id}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


def cmd_clean_cache(args: argparse.Namespace) -> None:
    # Clean cache for a specific run-id (or latest if omitted)
    root = Path(args.artifacts_root) if args.artifacts_root else Path("artifacts")
    store = ArtifactStore(root, create=True)
    run_id = args.run_id or store.latest_run()
    if not run_id:
        print("No runs available to clean.")
        return

    store.ensure_existing_run(run_id)
    cache = CacheManager(store)

    # Phase-1: prices + technical features
    # (add more invalidations later as you add caches)
    try:
        cache.invalidate_prices()
        print(f"[ok] Cleared price cache for run {run_id}")
    except Exception as e:
        print(f"[warn] Could not clear price cache: {e}")

    # optional: implement this helper in CacheManager if you want to clean features too
    try:
        if hasattr(cache, "invalidate_technical_features"):
            cache.invalidate_technical_features()  # tiny method you can add
            print(f"[ok] Cleared technical features for run {run_id}")
    except Exception as e:
        print(f"[warn] Could not clear technical features: {e}")


def cmd_prices(args):
    cfg = ExperimentConfig.from_yaml(Path(args.config))
    store = ArtifactStore()  # uses your defaults; adjust if you store paths in cfg
    pm = PriceManager(store)
    # TODO: replace with your actual API if different
    # Many implementations expose a single 'run' or 'fetch_and_cache' method.
    if hasattr(pm, "run"):
        pm.run(cfg)
    elif hasattr(pm, "fetch_and_cache"):
        pm.fetch_and_cache(cfg)
    else:
        raise RuntimeError("PriceManager missing expected run()/fetch_and_cache() method")


def cmd_features(args):
    cfg = ExperimentConfig.from_yaml(Path(args.config))
    store = ArtifactStore()
    fp = FeaturePipeline(store)
    # TODO: replace with your actual API if different
    if hasattr(fp, "build_technical"):
        fp.build_technical(cfg)
    else:
        raise RuntimeError(
            "FeaturePipeline missing expected run()/build_from_cached_prices() method"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    ap = argparse.ArgumentParser(
        prog="ai-investor-cli", description="AI_Investor_System CLI (Phase-1)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run", help="Execute a Phase-1 experiment end-to-end")
    p_run.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    p_run.set_defaults(func=cmd_run)

    # list-runs
    p_ls = sub.add_parser("list-runs", help="List runs in artifacts root")
    p_ls.add_argument(
        "--artifacts-root", type=Path, default=None, help="Root folder (default: ./artifacts)"
    )
    p_ls.set_defaults(func=cmd_list_runs)

    # show-metrics
    p_metrics = sub.add_parser(
        "show-metrics", help="Print metrics.json for a run (default: latest)"
    )
    p_metrics.add_argument("--artifacts-root", type=Path, default=None)
    p_metrics.add_argument("--run-id", type=str, default=None)
    p_metrics.set_defaults(func=cmd_show_metrics)

    # clean-cache
    p_clean = sub.add_parser("clean-cache", help="Delete caches for a run (default: latest)")
    p_clean.add_argument("--artifacts-root", type=Path, default=None)
    p_clean.add_argument("--run-id", type=str, default=None)
    p_clean.set_defaults(func=cmd_clean_cache)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
