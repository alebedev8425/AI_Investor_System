# src/system/controllers/run_manager.py
from __future__ import annotations

from dataclasses import asdict
from typing import Optional
import logging
import pandas as pd

from system.Model.experiment_config import ExperimentConfig
from system.Model.artifact_store import ArtifactStore

from system.Controller.price_manager import PriceManager
from system.Controller.feature_pipeline import FeaturePipeline
from system.Controller.training_manager import TrainingManager
from system.Controller.allocation_manager import AllocationManager
from system.Model.evaluation.backtester import Backtester
from system.Model.data.trading_calendar import TradingCalendar
from system.Controller.cache_manager import CacheManager
from system.Controller.reporting_manager import ReportingManager
from system.Model.evaluation.prediction_metrics import PredictionEvaluator


class RunManager:
    """
    Orchestrates the Phase-1 pipeline:
      ingest_prices -> build_technical -> train -> allocate -> backtest

    All model logic lives in the imported services, or in controllers of imported services. This class sequences them,
    snapshots config into ArtifactStore, and persists run outputs.
    """

    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        store: ArtifactStore,
        price_manager: PriceManager,
        feature_pipeline: FeaturePipeline,
        training_manager: TrainingManager,
        allocation_manager: AllocationManager,
        backtester: Backtester,
        cache: CacheManager,
        reporting: ReportingManager,
        calendar: Optional[TradingCalendar] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cfg = cfg
        self._store = store
        self._prices = price_manager
        self._features = feature_pipeline
        self._trainer = training_manager
        self._allocator = allocation_manager
        self._backtester = backtester
        self._cache = cache
        self._reporting = reporting
        self._cal = calendar
        self._log = logger or logging.getLogger(__name__)

    # public API

    def run(self) -> bool:
        """Execute the configured Phase-1 pipeline."""
        # 0) Create a run folder + manifest
        self._bootstrap_run()

        # 1) Ingest prices
        if self._cfg.pipelines.ingest_prices:
            prices = self._prices.ingest_prices(
                tickers=self._cfg.tickers,
                start=self._cfg.start,
                end=self._cfg.end,
                force_refresh=False,
                adjust=True,
            )
        else:
            # If skipping ingest, load from cache
            prices = self._cache.load_prices()
            self._log.info("Loaded prices from cache: %d rows", len(prices))

        # incase of empty / invalid symbols for tickers in gui
        if prices is None or prices.empty:
            self._log.error(
                "No price data fetched for the requested universe: %s. "
                "Check symbol spelling (e.g., 'AAPL' not 'APPL') or date range.",
                ",".join(self._cfg.tickers),
            )
            return False  # stop the run gracefully

        # 2) Features (technical + optional sentiment/events/corr/graph)
        if any(
            [
                self._cfg.pipelines.features_technical,
                self._cfg.pipelines.features_sentiment,
                self._cfg.pipelines.features_events,
                self._cfg.pipelines.features_correlation,
                self._cfg.pipelines.features_graph,
            ]
        ):
            feats = self._features.build_features(self._cfg, prices, overwrite=False)
        else:
            feats = self._cache.load_technical_features()
            self._log.info("Loaded features from cache: %d rows", len(feats))

        # 3) Train & predict
        if self._cfg.pipelines.train_model:
            preds = self._trainer.run(self._cfg)  # persists predictions internally too
        else:
            preds = self._store.load_csv(self._store.predictions_path())
            self._log.info("Loaded predictions from artifacts: %d rows", len(preds))

        # 3b) Prediction-quality metrics on the test set
        try:
            evaluator = PredictionEvaluator(horizon_col="target_5d", pred_col="pred_5d")

            # feats = full engineered table, includes target_5d
            # preds = test-set predictions ['date','ticker','pred_5d']
            pred_result = evaluator.evaluate(preds_df=preds, labels_df=feats)

            if pred_result.metrics:
                self._store.save_json(
                    pred_result.metrics,
                    self._store.prediction_metrics_path(),
                )
                self._log.info(
                    "Prediction metrics: MAE=%.6f  RMSE=%.6f  IC=%.3f  Hit=%.2f%%",
                    pred_result.metrics.get("mae", float("nan")),
                    pred_result.metrics.get("rmse", float("nan")),
                    pred_result.metrics.get("ic_mean", float("nan")),
                    100.0 * pred_result.metrics.get("hit_rate", float("nan")),
                )
            else:
                self._log.warning(
                    "Prediction metrics: no overlapping data between preds and labels."
                )
        except Exception as e:
            self._log.warning("Prediction evaluation failed: %s", e)

        # 4) Allocate
        if self._cfg.pipelines.allocate:
            # keep calendar optional; AllocationManager may ignore it for softmax
            weights = self._allocator.run(self._cfg, preds)
        else:
            weights = self._store.load_csv(self._store.weights_path())
            self._log.info("Loaded weights from artifacts: %d rows", len(weights))

        # 5) Backtest
        if self._cfg.pipelines.backtest:
            returns = self._backtester._compute_returns_for_backtest(prices)

            # backtest on test period only where we have actual predictions
            weights = weights.copy()
            weights["date"] = pd.to_datetime(weights["date"])
            test_start = weights["date"].min()
            test_end = weights["date"].max()

            returns_bt = returns.copy()
            returns_bt["date"] = pd.to_datetime(returns_bt["date"])
            returns_bt = returns_bt[
                (returns_bt["date"] >= test_start) & (returns_bt["date"] <= test_end)
            ].copy()

            self._log.info(
                "Backtest period is: %s to %s, based on weights date range (%d rows).",
                test_start.date().isoformat(),
                test_end.date().isoformat(),
                len(returns_bt),
            )

            daily, metrics = self._backtester.run(weights=weights, returns=returns_bt)

            # (A) normalized portfolio daily returns for reporting: ['date','ret']
            daily_out = daily[["date", "port_ret"]].rename(columns={"port_ret": "ret"})
            self._store.save_csv(daily_out, self._store.backtest_returns_path(), index=False)

            # (B) optional: keep the richer daily frame too (turnover/TC etc.)
            self._store.save_csv(
                daily, self._store.backtest_returns_path("daily_portfolio.csv"), index=False
            )

            # metrics (now includes n_days in Backtester)
            self._store.save_json(metrics, self._store.backtest_metrics_path())

            self._log.info(
                "Backtest complete. Days=%d  Sharpe~=%.3f  CumRet=%.2f%%",
                metrics.get("n_days", len(daily_out)),
                metrics.get("sharpe_like", 0.0),
                100.0 * metrics.get("cumulative_return", 0.0),
            )
        else:
            self._log.info("Backtest step skipped by configuration.")

        # after backtest
        if getattr(self._cfg.pipelines, "report", False) and self._reporting:
            out = self._reporting.build_single()
            self._log.info("Report written: %s", out)

        if getattr(self._cfg.pipelines, "compare", False) and self._reporting:
            baseline = getattr(getattr(self._cfg, "compare", None), "baseline_run_id", None)
            if baseline:
                out = self._reporting.build_compare(baseline, self._store.run_id)
                self._log.info("Comparison report written: %s", out)

        return True

    # ---------- helpers ----------

    def _bootstrap_run(self) -> None:
        """Create run folder and write manifest with a JSON-serializable config snapshot."""
        if getattr(self._store, "_run_id", None) is None:
            snap = self._cfg_to_jsonable(self._cfg)
            self._store.new_run(self._cfg.experiment_name, config_snapshot=snap)
            self._log.info("New run created: %s", self._store.run_id)

    @staticmethod
    def _cfg_to_jsonable(cfg: ExperimentConfig) -> dict:
        # Start from a plain dict of the dataclass
        d = asdict(cfg)

        # Normalize non-JSON types
        # - Path -> str
        # - date/datetime -> ISO string
        # (nested dataclasses from .pipelines/.model/.allocator/.backtest/.seeds
        # already contain primitives only)
        d["artifacts_root"] = str(cfg.artifacts_root)
        d["start"] = cfg.start.isoformat()
        d["end"] = cfg.end.isoformat()

        # Tickers as upper-case strings for consistency in the manifest
        d["tickers"] = [t.upper() for t in cfg.tickers]

        return d
