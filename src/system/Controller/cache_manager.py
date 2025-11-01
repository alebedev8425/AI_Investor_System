# src/system/data/cache_manager.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
import json
import hashlib
import logging


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
