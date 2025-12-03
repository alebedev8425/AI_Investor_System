# src/system/Model/data/data_sources/sentiment.py
from __future__ import annotations

import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests


@dataclass
class SentimentConfig:
    """
    Configuration for fetching and caching sentiment features.

    Data sources (priority):
    1. Local CSV cache in cache_dir
    2. Alpha Vantage NEWS_SENTIMENT API (if ALPHAVANTAGE_API_KEY set)
    """

    cache_dir: Path = Path("artifacts/external/sentiment")
    alpha_vantage_api_key: str = "8MDC834CHK1L15Y7"
    alpha_vantage_endpoint: str = "https://www.alphavantage.co/query"
    # how far back to look in days if caller doesn't slice dates tightly
    max_days: int = 30
    # polite sleep between API calls (seconds)
    api_pause: float = 1.1


class SentimentDataSource:
    """
    Provides daily per-ticker sentiment features.

    Output format:
        date, ticker, sent_mean, sent_median, sent_count

    Behavior:
    - If a suitable CSV exists in cache_dir, load it.
    - Else, if alpha_vantage_api_key is available, fetch from Alpha Vantage's NEWS_SENTIMENT,
      aggregate to daily per ticker, store CSV, and return.
    - Else, raise a clear error telling the user what to do.
    """

    def __init__(self, config: SentimentConfig | None = None):
        if config is None:
            config = SentimentConfig()
        if config.alpha_vantage_api_key is None:
            config.alpha_vantage_api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        self.cfg = config
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------- public API --------------

    def get_daily_sentiment(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        tickers = sorted({t.upper() for t in tickers})
        if not tickers:
            return pd.DataFrame(
                columns=["date", "ticker", "sent_mean", "sent_median", "sent_count"]
            )

        cache_path = self._cache_path(tickers, start_date, end_date)
        if cache_path.exists():
            return self._load_cached(cache_path)

        if not self.cfg.alpha_vantage_api_key:
            raise RuntimeError(
                "No sentiment CSV cache found and ALPHAVANTAGE_API_KEY is not set.\n"
                "Either:\n"
                " - Set ALPHAVANTAGE_API_KEY in your environment, or\n"
                " - Precompute sentiment CSVs into artifacts/external/sentiment."
            )

        df = self._fetch_alpha_vantage_news_sentiment(tickers, start_date, end_date)
        if not df.empty:
            df.to_csv(cache_path, index=False)
        return df

    # -------------- internals --------------

    def _cache_path(self, tickers: List[str], start_date: str, end_date: str) -> Path:
        key = ",".join(tickers) + f"|{start_date}|{end_date}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return self.cfg.cache_dir / f"sentiment_{h}.csv"

    def _load_cached(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _fetch_alpha_vantage_news_sentiment(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch and aggregate Alpha Vantage NEWS_SENTIMENT into daily per-ticker scores.

        Strategy:
        - For each ticker, walk [start_date, end_date] in chunks of `self.cfg.max_days`.
        - For each chunk, call NEWS_SENTIMENT with time_from/time_to.
        - Deduplicate articles across overlapping windows.
        - Aggregate:
              sent_mean   = mean(relevance_score * overall_sentiment_score)
              sent_median = median(...)
              sent_count  = #articles
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        max_span = int(self.cfg.max_days)

        records: list[dict] = []
        seen_ids: set[tuple] = set()  # to dedupe: (ticker, time_published, url/title)

        for t in tickers:
            cursor = start_dt

            while cursor <= end_dt:
                window_end = min(cursor + pd.Timedelta(days=max_span - 1), end_dt)

                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": t,
                    "apikey": self.cfg.alpha_vantage_api_key,
                    "time_from": cursor.strftime("%Y%m%dT0000"),
                    "time_to": window_end.strftime("%Y%m%dT2359"),
                    "sort": "EARLIEST",  # walk forward
                    "limit": 1000,  # be explicit; AV will cap as allowed
                }

                try:
                    r = requests.get(self.cfg.alpha_vantage_endpoint, params=params, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                except Exception as e:
                    print(
                        f"[sentiment] Alpha Vantage request failed for {t} [{cursor}–{window_end}]: {e}"
                    )
                    time.sleep(self.cfg.api_pause)
                    cursor = window_end + pd.Timedelta(days=1)
                    continue

                feed = data.get("feed", [])
                if not isinstance(feed, list):
                    msg = data.get("note") or data.get("Information") or str(data)[:200]
                    print(f"[sentiment] No feed for {t} [{cursor}–{window_end}]. Response: {msg}")
                    time.sleep(self.cfg.api_pause)
                    cursor = window_end + pd.Timedelta(days=1)
                    continue

                for item in feed:
                    ts = item.get("time_published")
                    if not ts:
                        continue

                    # Alpha Vantage format: 'YYYYMMDDTHHMMSS'
                    d_str = ts[:8]
                    try:
                        d = pd.to_datetime(d_str).date()
                    except Exception:
                        continue

                    if not (start_dt.date() <= d <= end_dt.date()):
                        continue

                    url = item.get("url") or ""
                    title = item.get("title") or ""
                    uid = (t, ts, url or title)
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)

                    try:
                        ovr = float(item.get("overall_sentiment_score", 0.0))
                        rel = float(item.get("relevance_score", 1.0))
                    except Exception:
                        ovr, rel = 0.0, 1.0

                    score = ovr * rel
                    records.append(
                        {
                            "date": d,
                            "ticker": t,
                            "score": score,
                        }
                    )

                time.sleep(self.cfg.api_pause)
                cursor = window_end + pd.Timedelta(days=1)

        if not records:
            # keep contract: same columns, empty
            return pd.DataFrame(
                columns=["date", "ticker", "sent_mean", "sent_median", "sent_count"]
            )

        df = pd.DataFrame(records)

        # Aggregate per (date, ticker)
        agg = (
            df.groupby(["date", "ticker"])["score"]
            .agg(sent_mean="mean", sent_median="median", sent_count="size")
            .reset_index()
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )

        return agg
