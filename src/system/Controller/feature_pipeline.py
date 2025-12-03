# src/system/Controller/feature_pipeline.py

from __future__ import annotations

import logging
from functools import reduce
from typing import Optional, List

import pandas as pd

from system.Controller.cache_manager import CacheManager
from system.Model.features.technical_features import TechnicalFeatureBuilder
from system.Model.features.correlation import CorrelationFeatureBuilder
from system.Model.features.graph import GraphFeatureBuilder
from system.Model.data.data_sources.sentiment import SentimentDataSource
from system.Model.data.data_sources.event import EventDataSource
from system.Model.artifact_store import ArtifactStore
from system.Model.experiment_config import ExperimentConfig
from system.Model.data.data_validator import DataValidator
from system.Model.features.stat_vol import StatVolFeatureBuilder


class FeaturePipeline:
    """
    Central feature builder.

    Uses CacheManager for all feature blocks:
      - prices_key      -> ensure_technical(...)
      - sentiment_key   -> ensure_sentiment(...)
      - events_key      -> ensure_events(...)
      - corr_key        -> ensure_correlation(...)
      - graph_key       -> ensure_graph(...)

    Output:
      - merged feature table saved to store.technical_features_path()
        (kept for backward compatibility; contains all enabled features).
    """

    _MIN_NON_NULL_FRAC = 0.05  # drop columns with <5% coverage

    def __init__(
        self,
        *,
        cache: CacheManager,
        store: ArtifactStore,
        technical_builder: TechnicalFeatureBuilder,
        sentiment_source: Optional[SentimentDataSource] = None,
        event_source: Optional[EventDataSource] = None,
        corr_builder: Optional[CorrelationFeatureBuilder] = None,
        graph_builder: Optional[GraphFeatureBuilder] = None,
        stat_vol_builder: Optional[StatVolFeatureBuilder] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cache = cache
        self._store = store
        self._tech = technical_builder
        self._sent_src = sentiment_source
        self._evt_src = event_source
        self._corr = corr_builder
        self._graph = graph_builder
        self._stat = stat_vol_builder
        self._log = logger or logging.getLogger(__name__)
        self._validator = DataValidator(logger=self._log)

    # -------- public entrypoint --------

    def build_features(
        self,
        cfg: ExperimentConfig,
        prices: pd.DataFrame,
        *,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        if prices is None or prices.empty:
            raise ValueError("FeaturePipeline.build_features: received empty prices DataFrame")

        tickers: List[str] = sorted({str(t).upper() for t in prices["ticker"].unique()})
        start = pd.to_datetime(prices["date"]).min().date()
        end = pd.to_datetime(prices["date"]).max().date()

        # Base grid
        base = (
            prices[["date", "ticker"]]
            .copy()
            .assign(date=pd.to_datetime(prices["date"]))
            .drop_duplicates()
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )
        blocks: List[pd.DataFrame] = [base]
        tech: pd.DataFrame | None = None

        # Shared price key (must match PriceManager / CacheManager usage)
        price_key = self._cache.prices_key(
            tickers=tickers,
            start=start,
            end=end,
            adjust=True,
        )

        # --- technical ---
        if cfg.pipelines.features_technical:
            tech = self._build_technical_from_cache(
                price_key=price_key,
                prices=prices,
                overwrite=overwrite,
            )
            if not tech.empty:
                blocks.append(tech)
            else:
                self._log.warning("[features] Technical features empty for this config/key.")

        # --- stat vol ---
        if cfg.pipelines.features_stat_vol and self._stat is not None and tech is not None:
            stat = self._build_stat_vol(tech)
            if not stat.empty:
                blocks.append(stat)
            else:
                self._log.warning("[features] Stat vol features empty for this config/key.")

        # --- sentiment ---
        if cfg.pipelines.features_sentiment and self._sent_src is not None:
            sent = self._build_sentiment_from_cache(
                tickers=tickers,
                start=start,
                end=end,
                overwrite=overwrite,
            )
            if not sent.empty:
                blocks.append(sent)

        # --- events ---
        if cfg.pipelines.features_events and self._evt_src is not None:
            ev = self._build_events_from_cache(
                tickers=tickers,
                start=start,
                end=end,
                overwrite=overwrite,
            )
            if not ev.empty:
                blocks.append(ev)

        # precompute returns once if needed
        need_ret = (cfg.pipelines.features_correlation and self._corr is not None) or (
            cfg.pipelines.features_graph and self._graph is not None
        )
        returns = self._compute_returns(prices) if need_ret else None

        # --- correlation ---
        if cfg.pipelines.features_correlation and self._corr is not None and returns is not None:
            corr = self._build_corr_from_cache(
                price_key=price_key,
                returns=returns,
                overwrite=overwrite,
            )
            if not corr.empty:
                blocks.append(corr)

        # --- graph ---
        if cfg.pipelines.features_graph and self._graph is not None and returns is not None:
            gfe = self._build_graph_from_cache(
                price_key=price_key,
                returns=returns,
                overwrite=overwrite,
            )
            if not gfe.empty:
                blocks.append(gfe)

        # --- merge ---
        feats = self._merge_blocks(blocks)

        # --- COMPREHENSIVE VALIDATION ---
        feats = self._validator.validate_features(
            feats, target_col="target_5d", structural_only=True
        )

        # Generate quality report for debugging - temporary
        quality_report = self._validator.generate_quality_report(feats)
        self._store.save_json(quality_report, self._store.features_dir / "data_quality_report.json")
        self._log.info("Data quality report saved")

        # --- drop sparse/empty feature columns ---
        feats = self._drop_sparse_columns(
            feats,
            min_non_null_frac=self._MIN_NON_NULL_FRAC,
        )

        # --- log sentiment coverage if any survive ---
        sent_cols = [c for c in feats.columns if c.startswith("sent_")]
        if sent_cols:
            non_null = feats[sent_cols].notna().sum()
            msg = ", ".join(f"{c}={int(non_null[c])}" for c in sent_cols)
            self._log.info("[features] Sentiment coverage (non-null counts): %s", msg)

        # --- persist canonical feature table ---
        self._store.save_csv(feats, self._store.technical_features_path(), index=False)
        self._log.info(
            "Saved merged features: %d rows, %d columns",
            len(feats),
            len(feats.columns),
        )

        return feats

    # -------- cache-backed builders --------

    def _build_technical_from_cache(
        self,
        *,
        price_key: str,
        prices: pd.DataFrame,
        overwrite: bool,
    ) -> pd.DataFrame:
        df = self._cache.ensure_technical(
            price_key=price_key,
            df_prices=prices,
            builder=self._tech,
            overwrite=overwrite,
        )
        return DataValidator.normalize_block(df)

    def _build_sentiment_from_cache(
        self,
        *,
        tickers: List[str],
        start,
        end,
        overwrite: bool,
    ) -> pd.DataFrame:
        # fetch_fn closes over tickers/start/end
        def fetch():
            return self._sent_src.get_daily_sentiment(
                tickers=tickers,
                start_date=str(start),
                end_date=str(end),
            )

        df = self._cache.ensure_sentiment(
            tickers=tickers,
            start=start,
            end=end,
            fetch_fn=fetch,
            force_refresh=overwrite,
            # if your SentimentDataSource exposes a source name, plug it here
            source="alphavantage",
        )
        df = DataValidator.normalize_block(df)
        if not df.empty:
            self._store.save_csv(df, self._store.sentiment_features_path(), index=False)
        return df

    def _build_events_from_cache(
        self,
        *,
        tickers: List[str],
        start,
        end,
        overwrite: bool,
    ) -> pd.DataFrame:
        def fetch():
            return self._evt_src.get_daily_events(
                tickers=tickers,
                start_date=str(start),
                end_date=str(end),
            )

        df = self._cache.ensure_events(
            tickers=tickers,
            start=start,
            end=end,
            fetch_fn=fetch,
            force_refresh=overwrite,
            source="alphavantage",
        )
        df = DataValidator.normalize_block(df)
        if not df.empty:
            self._store.save_csv(df, self._store.event_features_path(), index=False)
        return df

    def _build_corr_from_cache(
        self,
        *,
        price_key: str,
        returns: pd.DataFrame,
        overwrite: bool,
    ) -> pd.DataFrame:
        df = self._cache.ensure_correlation(
            price_key=price_key,
            df_returns=returns,
            builder=self._corr,
            overwrite=overwrite,
        )
        return DataValidator.normalize_block(df)

    def _build_graph_from_cache(
        self,
        *,
        price_key: str,
        returns: pd.DataFrame,
        overwrite: bool,
    ) -> pd.DataFrame:
        df = self._cache.ensure_graph(
            price_key=price_key,
            df_returns=returns,
            builder=self._graph,
            overwrite=overwrite,
        )
        return DataValidator.normalize_block(df)

    # no need to cache since stat vol depends on technical features only, and runs quickly
    def _build_stat_vol(self, tech: pd.DataFrame) -> pd.DataFrame:
        df = self._stat.build(tech)
        return DataValidator.normalize_block(df)

    # -------- utility helpers --------

    @staticmethod
    def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        if price_col not in df.columns:
            raise ValueError("Prices missing both 'adj_close' and 'close' columns.")

        df = df.sort_values(["ticker", "date"])
        df["ret"] = df.groupby("ticker")[price_col].pct_change().fillna(0.0)
        return df[["date", "ticker", "ret"]].reset_index(drop=True)

    @staticmethod
    def _merge_blocks(blocks: List[pd.DataFrame]) -> pd.DataFrame:
        clean: List[pd.DataFrame] = []
        for b in blocks:
            if b is None or b.empty:
                continue
            cols = ["date", "ticker"] + [c for c in b.columns if c not in ("date", "ticker")]
            bb = b[cols].copy()
            bb["date"] = pd.to_datetime(bb["date"])
            bb["ticker"] = bb["ticker"].astype(str)
            clean.append(bb)

        if not clean:
            return pd.DataFrame(columns=["date", "ticker"])

        def _merge(l: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
            return pd.merge(l, r, on=["date", "ticker"], how="left")

        out = reduce(_merge, clean)
        return out.sort_values(["date", "ticker"]).reset_index(drop=True)

    def _drop_sparse_columns(
        self,
        feats: pd.DataFrame,
        *,
        min_non_null_frac: float,
    ) -> pd.DataFrame:
        if feats.empty:
            return feats

        n = len(feats)
        protected = {"date", "ticker"}
        drop: List[str] = []

        for c in feats.columns:
            if c in protected:
                continue
            frac = feats[c].notna().sum() / float(n)
            if frac == 0.0 or frac < min_non_null_frac:
                drop.append(c)

        if drop:
            self._log.warning(
                "[features] Dropping %d sparse/empty feature columns (coverage < %.1f%%): %s",
                len(drop),
                min_non_null_frac * 100.0,
                drop,
            )
            feats = feats.drop(columns=drop)

        return feats
