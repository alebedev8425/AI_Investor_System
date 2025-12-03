# src/system/Controller/cache_manager.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
import json
import hashlib
import logging
import shutil


from system.Model.artifact_store import ArtifactStore


def _key(obj) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:16]


class CacheManager:
    """
    Phase-1: minimal cache manager for price data.
    Later add more caches as needed.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._log = logging.getLogger(__name__)

    # ---------- cache keys ----------
    def prices_key(
        self, tickers: Iterable[str], start, end, *, adjust: bool, source: str = "yfinance"
    ) -> str:
        tickers = sorted([str(t).upper() for t in tickers])
        return _key(
            {
                "type": "prices",
                "tickers": tickers,
                "start": str(start),
                "end": str(end),
                "adjust": bool(adjust),
                "source": source,
            }
        )

    def technical_key(self, price_key: str, *, use_adj: bool, ma_windows: tuple[int, ...]) -> str:
        return _key(
            {
                "type": "technical",
                "price_key": price_key,
                "use_adj": bool(use_adj),
                "ma_windows": list(ma_windows),
            }
        )

    def sentiment_key(
        self,
        tickers,
        start,
        end,
        source: str = "alphavantage",
    ) -> str:
        tickers = sorted([str(t).upper() for t in tickers])
        return _key(
            {
                "type": "sentiment",
                "tickers": tickers,
                "start": str(start),
                "end": str(end),
                "source": source,
            }
        )

    def events_key(
        self,
        tickers,
        start,
        end,
        source: str = "alphavantage",
    ) -> str:
        tickers = sorted([str(t).upper() for t in tickers])
        return _key(
            {
                "type": "events",
                "tickers": tickers,
                "start": str(start),
                "end": str(end),
                "source": source,
            }
        )

    def corr_key(
        self,
        price_key: str,
        window: int,
        min_periods: int,
        corr_threshold: float,
    ) -> str:
        return _key(
            {
                "type": "corr_features",
                "price_key": price_key,
                "window": int(window),
                "min_periods": int(min_periods),
                "corr_threshold": float(corr_threshold),
            }
        )

    def graph_key(
        self,
        price_key: str,
        window: int,
        min_periods: int,
        corr_threshold: float,
    ) -> str:
        return _key(
            {
                "type": "graph_features",
                "price_key": price_key,
                "window": int(window),
                "min_periods": int(min_periods),
                "corr_threshold": float(corr_threshold),
            }
        )

    # ---------- shared cache I/O ----------
    def shared_exists(self, path: Path) -> bool:
        return path.exists()

    def save_shared_csv(
        self, df: pd.DataFrame, csv_path: Path, meta_path: Path, meta: dict
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(csv_path)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def ensure_prices(
        self, *, tickers, start, end, adjust: bool, fetch_fn, force_refresh: bool = False
    ) -> pd.DataFrame:
        pkey = self.prices_key(tickers, start, end, adjust=adjust)
        shared_csv = self._store.shared_prices_path(pkey)
        shared_meta = self._store.shared_prices_meta(pkey)
        run_csv = self._store.price_cache_path()

        # ---- CACHE HIT ----
        if (not force_refresh) and self.shared_exists(shared_csv):
            self._log.info("[cache:prices] HIT key=%s -> %s", pkey, shared_csv.name)
            self._store.materialize_to_run(shared_csv, run_csv)
            self._store.write_run_meta(
                "prices",
                {
                    "key": pkey,
                    "source": "shared",
                    "shared_csv": str(shared_csv),
                    "tickers": sorted([t.upper() for t in tickers]),
                    "start": str(start),
                    "end": str(end),
                    "adjust": bool(adjust),
                },
            )
            return pd.read_csv(run_csv)

        # ---- CACHE MISS ----
        self._log.info("[cache:prices] MISS key=%s (fetching)", pkey)
        df = fetch_fn(tickers=tickers, start=start, end=end, adjust=adjust)
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        # write shared + meta
        self.save_shared_csv(
            df,
            shared_csv,
            shared_meta,
            meta={
                "key": pkey,
                "tickers": sorted([t.upper() for t in tickers]),
                "start": str(start),
                "end": str(end),
                "adjust": bool(adjust),
            },
        )

        # hydrate into run + write run meta
        self._store.materialize_to_run(shared_csv, run_csv)
        self._store.write_run_meta(
            "prices",
            {
                "key": pkey,
                "source": "fresh",
                "shared_csv": str(shared_csv),
                "tickers": sorted([t.upper() for t in tickers]),
                "start": str(start),
                "end": str(end),
                "adjust": bool(adjust),
            },
        )
        return df

    def ensure_technical(
        self, *, price_key: str, df_prices: pd.DataFrame, builder, overwrite: bool = False
    ) -> pd.DataFrame:
        tkey = self.technical_key(price_key, use_adj=builder.use_adj, ma_windows=builder.ma_windows)
        shared_csv = self._store.shared_technical_path(tkey)
        shared_meta = self._store.shared_technical_meta(tkey)
        run_csv = self._store.technical_features_path()

        # ---- CACHE HIT ----
        if (not overwrite) and self.shared_exists(shared_csv):
            self._log.info("[cache:technical] HIT key=%s -> %s", tkey, shared_csv.name)
            self._store.materialize_to_run(shared_csv, run_csv)
            self._store.write_run_meta(
                "technical",
                {
                    "key": tkey,
                    "source": "shared",
                    "price_key": price_key,
                    "use_adj": bool(builder.use_adj),
                    "ma_windows": list(builder.ma_windows),
                    "shared_csv": str(shared_csv),
                },
            )
            return pd.read_csv(run_csv)

        # ---- CACHE MISS ----
        self._log.info("[cache:technical] MISS key=%s (building)", tkey)
        feats = builder.build(df_prices).sort_values(["date", "ticker"]).reset_index(drop=True)

        # write shared + meta
        self.save_shared_csv(
            feats,
            shared_csv,
            shared_meta,
            meta={
                "key": tkey,
                "price_key": price_key,
                "use_adj": bool(builder.use_adj),
                "ma_windows": list(builder.ma_windows),
            },
        )

        # hydrate into run + write run meta
        self._store.materialize_to_run(shared_csv, run_csv)
        self._store.write_run_meta(
            "technical",
            {
                "key": tkey,
                "source": "fresh",
                "price_key": price_key,
                "use_adj": bool(builder.use_adj),
                "ma_windows": list(builder.ma_windows),
                "shared_csv": str(shared_csv),
            },
        )
        return feats

    def ensure_sentiment(
        self,
        *,
        tickers,
        start,
        end,
        fetch_fn,
        force_refresh: bool = False,
        source: str = "alphavantage",
    ) -> pd.DataFrame:
        skey = self.sentiment_key(tickers, start, end, source=source)
        shared_csv = self._store.shared_sentiment_path(skey)
        shared_meta = self._store.shared_sentiment_meta(skey)
        run_csv = self._store.sentiment_features_path()

        if (not force_refresh) and self.shared_exists(shared_csv):
            self._log.info("[cache:sentiment] HIT key=%s -> %s", skey, shared_csv.name)
            self._store.materialize_to_run(shared_csv, run_csv)
            self._store.write_run_meta(
                "sentiment",
                {
                    "key": skey,
                    "source": "shared",
                    "shared_csv": str(shared_csv),
                },
            )
            return pd.read_csv(run_csv)

        self._log.info("[cache:sentiment] MISS key=%s (fetching)", skey)
        df = fetch_fn()
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        self.save_shared_csv(
            df,
            shared_csv,
            shared_meta,
            meta={
                "key": skey,
                "tickers": sorted([str(t).upper() for t in tickers]),
                "start": str(start),
                "end": str(end),
                "source": source,
            },
        )

        self._store.materialize_to_run(shared_csv, run_csv)
        self._store.write_run_meta(
            "sentiment",
            {
                "key": skey,
                "source": "fresh",
                "shared_csv": str(shared_csv),
            },
        )
        return df

    def ensure_events(
        self,
        *,
        tickers,
        start,
        end,
        fetch_fn,
        force_refresh: bool = False,
        source: str = "alphavantage",
    ) -> pd.DataFrame:
        ekey = self.events_key(tickers, start, end, source=source)
        shared_csv = self._store.shared_events_path(ekey)
        shared_meta = self._store.shared_events_meta(ekey)
        run_csv = self._store.event_features_path()

        if (not force_refresh) and self.shared_exists(shared_csv):
            self._log.info("[cache:events] HIT key=%s -> %s", ekey, shared_csv.name)
            self._store.materialize_to_run(shared_csv, run_csv)
            self._store.write_run_meta(
                "events",
                {
                    "key": ekey,
                    "source": "shared",
                    "shared_csv": str(shared_csv),
                },
            )
            return pd.read_csv(run_csv)

        self._log.info("[cache:events] MISS key=%s (fetching)", ekey)
        df = fetch_fn()
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        self.save_shared_csv(
            df,
            shared_csv,
            shared_meta,
            meta={
                "key": ekey,
                "tickers": sorted([str(t).upper() for t in tickers]),
                "start": str(start),
                "end": str(end),
                "source": source,
            },
        )

        self._store.materialize_to_run(shared_csv, run_csv)
        self._store.write_run_meta(
            "events",
            {
                "key": ekey,
                "source": "fresh",
                "shared_csv": str(shared_csv),
            },
        )
        return df

    def ensure_correlation(
        self,
        *,
        price_key: str,
        df_returns: pd.DataFrame,
        builder,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        ckey = self.corr_key(
            price_key,
            builder.cfg.window,
            builder.cfg.min_periods,
            builder.cfg.corr_threshold,
        )
        shared_csv = self._store.shared_corr_path(ckey)
        shared_meta = self._store.shared_corr_meta(ckey)
        run_csv = self._store.corr_features_path()

        if (not overwrite) and self.shared_exists(shared_csv):
            self._log.info("[cache:corr] HIT key=%s -> %s", ckey, shared_csv.name)
            self._store.materialize_to_run(shared_csv, run_csv)
            self._store.write_run_meta(
                "correlation",
                {
                    "key": ckey,
                    "source": "shared",
                    "shared_csv": str(shared_csv),
                    "price_key": price_key,
                },
            )
            return pd.read_csv(run_csv)

        self._log.info("[cache:corr] MISS key=%s (building)", ckey)
        feats = builder.build(df_returns)
        feats = feats.sort_values(["date", "ticker"]).reset_index(drop=True)

        self.save_shared_csv(
            feats,
            shared_csv,
            shared_meta,
            meta={
                "key": ckey,
                "price_key": price_key,
                "window": builder.cfg.window,
                "min_periods": builder.cfg.min_periods,
                "corr_threshold": float(builder.cfg.corr_threshold),
            },
        )

        self._store.materialize_to_run(shared_csv, run_csv)
        self._store.write_run_meta(
            "correlation",
            {
                "key": ckey,
                "source": "fresh",
                "shared_csv": str(shared_csv),
                "price_key": price_key,
            },
        )
        return feats

    def ensure_graph(
        self,
        *,
        price_key: str,
        df_returns: pd.DataFrame,
        builder,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        gkey = self.graph_key(
            price_key,
            builder.cfg.window,
            builder.cfg.min_periods,
            builder.cfg.corr_threshold,
        )
        shared_csv = self._store.shared_graph_path(gkey)
        shared_meta = self._store.shared_graph_meta(gkey)
        run_csv = self._store.graph_features_path()

        if (not overwrite) and self.shared_exists(shared_csv):
            self._log.info("[cache:graph] HIT key=%s -> %s", gkey, shared_csv.name)
            self._store.materialize_to_run(shared_csv, run_csv)
            self._store.write_run_meta(
                "graph",
                {
                    "key": gkey,
                    "source": "shared",
                    "shared_csv": str(shared_csv),
                    "price_key": price_key,
                },
            )
            return pd.read_csv(run_csv)

        self._log.info("[cache:graph] MISS key=%s (building)", gkey)
        feats = builder.build(df_returns)
        feats = feats.sort_values(["date", "ticker"]).reset_index(drop=True)

        self.save_shared_csv(
            feats,
            shared_csv,
            shared_meta,
            meta={
                "key": gkey,
                "price_key": price_key,
                "window": builder.cfg.window,
                "min_periods": builder.cfg.min_periods,
                "corr_threshold": float(builder.cfg.corr_threshold),
            },
        )

        self._store.materialize_to_run(shared_csv, run_csv)
        self._store.write_run_meta(
            "graph",
            {
                "key": gkey,
                "source": "fresh",
                "shared_csv": str(shared_csv),
                "price_key": price_key,
            },
        )
        return feats

    # ---- prices ----
    def prices_exists(self) -> bool:
        return self._store.price_cache_path().exists()

    def load_prices(self) -> pd.DataFrame:
        path = self._store.price_cache_path()
        return self._store.load_csv(path)

    def save_prices(self, df: pd.DataFrame) -> None:
        path = self._store.price_cache_path()
        self._store.save_csv(df, path, index=False)

    def invalidate_prices(self) -> None:
        p = self._store.price_cache_path()
        if p.exists():
            p.unlink()

    # ---- technical features ----
    def technical_features_exists(self) -> bool:
        return self._store.technical_features_path().exists()

    def load_technical_features(self) -> pd.DataFrame:
        return self._store.load_csv(self._store.technical_features_path())

    def save_technical_features(self, df: pd.DataFrame) -> None:
        self._store.save_csv(df, self._store.technical_features_path(), index=False)

    def invalidate_technical_features(self) -> None:
        p = self._store.technical_features_path()
        if p.exists():
            p.unlink()

        # ---- cache clearing helpers ----

    def clear_shared_cache(self) -> None:
        """
        Delete the shared cache directory (_shared_cache) under artifacts_root.
        Keeps individual run folders intact.
        """
        root = self._store.shared_cache_root
        if root.exists():
            shutil.rmtree(root)
            self._log.info("Cleared shared cache at %s", root)
        else:
            self._log.info("Shared cache directory does not exist: %s", root)
