# src/system/Model/data/data_sources/event.py
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
class EventConfig:
    cache_dir: Path = Path("artifacts/external/events")
    # Do NOT hardcode a key; prefer env (set ALPHAVANTAGE_API_KEY in your shell)
    alpha_vantage_api_key: str = ""
    alpha_vantage_endpoint: str = "https://www.alphavantage.co/query"
    api_pause: float = 12.0  # AV free tier ≈ 5 req/min. Use >=12s to be safe.
    retries: int = 2  # small retry count for throttle bursts
    pad_days: int = 10  # include nearby events for days_to_earnings


class EventDataSource:
    """
    Output schema:
      date (datetime64[ns]), ticker (str), has_earnings (0/1), days_to_earnings (int or 999)
    """

    def __init__(self, config: EventConfig | None = None):
        cfg = config or EventConfig()
        # Env overrides any default/missing
        key_env = os.environ.get("ALPHAVANTAGE_API_KEY")
        if key_env:
            cfg.alpha_vantage_api_key = key_env
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------- public --------------

    def get_daily_events(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        tickers = sorted({str(t).upper() for t in tickers})
        if not tickers:
            return pd.DataFrame(columns=["date", "ticker", "has_earnings", "days_to_earnings"])

        cache_path = self._cache_path(tickers, start_date, end_date)
        if cache_path.exists():
            return self._load_cached(cache_path)

        if not self.cfg.alpha_vantage_api_key:
            raise RuntimeError(
                "Events cache miss and ALPHAVANTAGE_API_KEY not set.\n"
                "Set ALPHAVANTAGE_API_KEY in your environment or precompute CSVs."
            )

        df = self._fetch_alpha_vantage_earnings(tickers, start_date, end_date)
        if not df.empty:
            df.to_csv(cache_path, index=False)
        return df

    # -------------- internals --------------

    def _cache_path(self, tickers: List[str], start_date: str, end_date: str) -> Path:
        key = ",".join(tickers) + f"|{start_date}|{end_date}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return self.cfg.cache_dir / f"events_{h}.csv"

    def _load_cached(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ---- Alpha Vantage helpers ----

    def _av_json(self, **params) -> dict:
        """
        Wrap requests + detect AV throttle/error payloads.
        Retries with backoff when we see 'Note'/'Information'/'Error Message'.
        """
        last_err = None
        for i in range(self.cfg.retries + 1):
            try:
                r = requests.get(self.cfg.alpha_vantage_endpoint, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                if any(k in data for k in ("Note", "Information", "Error Message")):
                    msg = data.get("Note") or data.get("Information") or data.get("Error Message")
                    last_err = RuntimeError(msg or "Alpha Vantage API error")
                    time.sleep(self.cfg.api_pause)
                    continue
                return data
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.api_pause)
        raise last_err or RuntimeError("Alpha Vantage request failed after retries")

    def _extract_dates(self, items: list[dict]) -> list[pd.Timestamp.date]:
        """
        Normalize multiple possible date keys across AV responses.
        """
        dates: list = []
        for it in items:
            for key in ("reportedDate", "reportDate", "fiscalDateEnding", "calendarDate"):
                d = it.get(key)
                if not d:
                    continue
                try:
                    dates.append(pd.to_datetime(d).date())
                    break
                except Exception:
                    pass
        return sorted(set(dates))

    def _earnings_history(self, symbol: str) -> list:
        data = self._av_json(
            function="EARNINGS",
            symbol=symbol,
            apikey=self.cfg.alpha_vantage_api_key,
        )
        # Typical container is 'quarterlyEarnings'
        return self._extract_dates(data.get("quarterlyEarnings", []) or [])

    def _earnings_calendar(self, symbol: str) -> list:
        # Upcoming (and sometimes recent)
        data = self._av_json(
            function="EARNINGS_CALENDAR",
            symbol=symbol,
            apikey=self.cfg.alpha_vantage_api_key,
        )
        return self._extract_dates(data.get("earningsCalendar", []) or [])

    def _fetch_alpha_vantage_earnings(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        all_rows: list[dict] = []

        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        pad = int(self.cfg.pad_days)
        pad_start = (pd.to_datetime(start) - pd.Timedelta(days=pad)).date()
        pad_end = (pd.to_datetime(end) + pd.Timedelta(days=pad)).date()

        for t in tickers:
            try:
                hist = self._earnings_history(t)
            except Exception as e:
                print(f"[events] EARNINGS history failed for {t}: {e}")
                hist = []
            time.sleep(self.cfg.api_pause)

            try:
                cal = self._earnings_calendar(t)
            except Exception as e:
                print(f"[events] EARNINGS_CALENDAR failed for {t}: {e}")
                cal = []
            time.sleep(self.cfg.api_pause)

            events = [d for d in set(hist + cal) if pad_start <= d <= pad_end]
            events.sort()

            if not events:
                # still emit rows? we only need days_to in range, but without events it's 999 everywhere.
                # Skip to reduce useless rows; downstream merge will just have NaNs if we emit nothing.
                continue

            for d in pd.date_range(start, end, freq="D").date:
                deltas = [(ed - d).days for ed in events]
                nearest = min(deltas, key=lambda x: abs(x)) if deltas else 999
                days_to = nearest if isinstance(nearest, int) and abs(nearest) <= pad else 999
                all_rows.append(
                    {
                        "date": pd.to_datetime(d),  # ensure ts for merge
                        "ticker": t,
                        "has_earnings": 1 if d in events else 0,
                        "days_to_earnings": int(days_to),
                    }
                )

        if not all_rows:
            return pd.DataFrame(columns=["date", "ticker", "has_earnings", "days_to_earnings"])

        df = pd.DataFrame(all_rows).sort_values(["date", "ticker"]).reset_index(drop=True)
        return df
