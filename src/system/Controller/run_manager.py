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
from system.Model.backtesting.backtester import Backtester
from system.Model.data.trading_calendar import TradingCalendar
from system.Controller.cache_manager import CacheManager
from system.Controller.reporting_manager import ReportingManager


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

    def run(self) -> None:
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

        # 2) Technical features
        if self._cfg.pipelines.features_technical:
            feats = self._features.build_technical(prices, overwrite=False)
        else:
            feats = self._cache.load_technical_features()
            self._log.info("Loaded technical features from cache: %d rows", len(feats))

        # 3) Train & predict (Phase-1: LSTM only is fine; TrainingManager encapsulates that)
        if self._cfg.pipelines.train_model:
            preds = self._trainer.run(self._cfg)  # persists predictions internally too
        else:
            preds = self._store.load_csv(self._store.predictions_path())
            self._log.info("Loaded predictions from artifacts: %d rows", len(preds))

        # 4) Allocate (Phase-1: softmax via AllocationManager)
        if self._cfg.pipelines.allocate:
            # keep calendar optional; AllocationManager may ignore it for softmax
            weights = self._allocator.run_softmax(self._cfg, preds)
        else:
            weights = self._store.load_csv(self._store.weights_path())
            self._log.info("Loaded weights from artifacts: %d rows", len(weights))

        # 5) Backtest
        if self._cfg.pipelines.backtest:
            returns = self._compute_returns_for_backtest(prices)
            daily, metrics = self._backtester.run(weights=weights, returns=returns)

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

    @staticmethod
    def _compute_returns_for_backtest(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build tidy daily returns for backtest.
        Uses 'adj_close' if present, else 'close'. Expects columns: ['date','ticker',...].
        Output columns: ['date','ticker','ret'] with date as Timestamp (normalized to midnight).
        """
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        if price_col not in df.columns:
            raise ValueError("Prices missing both 'adj_close' and 'close' columns.")

        df = df.sort_values(["ticker", "date"])
        df["ret"] = df.groupby("ticker")[price_col].pct_change().fillna(0.0)
        return df[["date", "ticker", "ret"]].reset_index(drop=True)
