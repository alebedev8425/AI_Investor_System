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
from system.Controller.reporting_manager import ReportingManager


# --- Controller imports ---
from system.Controller.price_manager import PriceManager
from system.Controller.feature_pipeline import FeaturePipeline
from system.Controller.training_manager import TrainingManager
from system.Controller.allocation_manager import AllocationManager
from system.Controller.run_manager import RunManager

# --- Utility imports ---
from system.Model.utils.repro import configure_reproducibility


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
    reporting_mgr = ReportingManager(store=store)

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
        reporting=reporting_mgr,
    )
    return runner, store


def cmd_run(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.from_yaml(Path(args.config))

    device = configure_reproducibility(
        seed_python=cfg.seeds.python,
        seed_numpy=cfg.seeds.numpy,
        seed_torch=cfg.seeds.torch,
        device_pref=None,  # let env AI_INV_DEVICE choose, else auto
        strict=None,  # let env AI_INV_STRICT choose
    )

    logging.getLogger(__name__).info(
        f"[repro] device={device} | "
        f"seeds(py={cfg.seeds.python}, np={cfg.seeds.numpy}, torch={cfg.seeds.torch})"
    )

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


def cmd_report(args: argparse.Namespace) -> None:
    root = Path(args.artifacts_root) if args.artifacts_root else Path("artifacts")
    store = ArtifactStore(root, create=True)
    run_id = args.run_id or store.latest_run()
    if not run_id:
        print("No runs available.")
        return
    store.ensure_existing_run(run_id)
    rm = ReportingManager(store)
    out = rm.build_single()
    print(str(out))


def cmd_compare(args: argparse.Namespace) -> None:
    root = Path(args.artifacts_root) if args.artifacts_root else Path("artifacts")
    store = ArtifactStore(root, create=True)
    a = args.a
    b = args.b
    if not a or not b:
        print("compare requires --a RUN_ID_A and --b RUN_ID_B")
        return
    rm = ReportingManager(store)  # uses store.run_id for output location
    # choose which run folder to host the compare report; here we use 'b' by convention
    store.ensure_existing_run(b)
    out = rm.build_compare(a, b)
    print(str(out))


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

    p_rep = sub.add_parser("report", help="Build HTML report for a run (default: latest)")
    p_rep.add_argument("--artifacts-root", type=Path, default=None)
    p_rep.add_argument("--run-id", type=str, default=None)
    p_rep.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare", help="Build comparison HTML for two runs")
    p_cmp.add_argument("--artifacts-root", type=Path, default=None)
    p_cmp.add_argument("--a", type=str, required=True)
    p_cmp.add_argument("--b", type=str, required=True)
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
