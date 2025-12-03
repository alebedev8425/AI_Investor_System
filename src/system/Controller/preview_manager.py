# src/system/Controller/preview_manager.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Tuple, List
import json
import pandas as pd

from system.Model.artifact_store import ArtifactStore


class PreviewManager:
    """
    Read-only previews for Utilities menu:
      - preview_prices()
      - preview_technical_features()
      - preview_predictions()
      - load_universe_defaults()  (tickers/start/end from latest manifest)
    All file access is via ArtifactStore.
    """

    def __init__(self, store: ArtifactStore):
        self._root_store = store  # may or may not have a run_id set

    # ----- latest run helpers -----
    def latest_run_id(self) -> Optional[str]:
        return self._root_store.latest_run()

    def list_runs(self) -> list[str]:
        """Controller façade to enumerate available runs."""
        return list(self._root_store.list_runs())

    def _store_for(self, run_id: str) -> ArtifactStore:
        s = ArtifactStore(self._root_store.root, create=True)
        s.ensure_existing_run(run_id)
        return s

    def _read_csv_head(self, path: Path, limit: int) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(str(path))
        df = pd.read_csv(path)
        if limit and len(df) > limit:
            df = df.head(limit)
        return df

    # ----- previews -----
    def preview_prices(self, limit: int = 500) -> pd.DataFrame:
        rid = self.latest_run_id()
        if not rid:
            raise RuntimeError("No runs found.")
        s = self._store_for(rid)
        return self._read_csv_head(s.price_cache_path(), limit)

    def preview_technical_features(self, limit: int = 500) -> pd.DataFrame:
        rid = self.latest_run_id()
        if not rid:
            raise RuntimeError("No runs found.")
        s = self._store_for(rid)
        return self._read_csv_head(s.technical_features_path(), limit)

    def preview_predictions(self, limit: int = 500) -> pd.DataFrame:
        rid = self.latest_run_id()
        if not rid:
            raise RuntimeError("No runs found.")
        s = self._store_for(rid)
        return self._read_csv_head(s.predictions_path(), limit)

    # ----- universe defaults (tickers/dates) -----
    def load_universe_defaults(self) -> Tuple[List[str], Optional[str], Optional[str]]:
        """
        Returns (tickers, start, end) for the latest run.
        Order of preference:
          1) data/prices.meta.json written by CacheManager.ensure_prices()
          2) manifest.json -> config.* (robust fallback)
        Missing values come back as [] / None.
        """
        rid = self.latest_run_id()
        if not rid:
            return [], None, None

        s = self._store_for(rid)

        # 1) Prefer run-level prices meta (authoritative)
        meta_path = s.run_meta_path("prices")  # artifacts/<run>/data/prices.meta.json
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                tickers = self._coerce_tickers(m.get("tickers", []))
                start = self._coerce_date(m.get("start"))
                end = self._coerce_date(m.get("end"))
                if tickers or start or end:
                    return tickers, start, end
            except Exception:
                pass  # fall through to manifest

        # 2) Fallback to manifest.json (robust key search)
        man = s.manifest_path
        if man.exists():
            try:
                d = json.loads(man.read_text())
                cfg = d.get("config") or {}

                # tolerant nested getter
                def _pick(root: Any, path: List[str], default=None):
                    cur = root
                    for k in path:
                        if not isinstance(cur, dict) or k not in cur:
                            return default
                        cur = cur[k]
                    return cur

                # Try several common layouts for tickers/start/end
                cand_paths_t = [
                    ["universe", "tickers"],
                    ["data", "universe", "tickers"],
                    ["prices", "universe", "tickers"],
                    ["universe"],  # if it's already a list
                ]
                cand_paths_s = [
                    ["universe", "start"],
                    ["data", "universe", "start"],
                    ["prices", "universe", "start"],
                    ["start"],
                ]
                cand_paths_e = [
                    ["universe", "end"],
                    ["data", "universe", "end"],
                    ["prices", "universe", "end"],
                    ["end"],
                ]

                raw_t = None
                for p in cand_paths_t:
                    v = _pick(cfg, p, None)
                    if v is not None:
                        raw_t = v
                        break

                start = None
                for p in cand_paths_s:
                    v = _pick(cfg, p, None)
                    if v is not None:
                        start = self._coerce_date(v)
                        break

                end = None
                for p in cand_paths_e:
                    v = _pick(cfg, p, None)
                    if v is not None:
                        end = self._coerce_date(v)
                        break

                tickers = self._coerce_tickers(raw_t or [])

                return tickers, start, end
            except Exception:
                pass

        # Nothing found
        return [], None, None

    def recommended_tickers(self, limit: int = 31) -> list[str]:
        """
        Return up to `limit` suggested tickers. Prefers the latest run’s recorded universe
        (prices meta or manifest); falls back to a default liquid list. Uppercase, unique, stable order.
        """
        import json

        seen: set[str] = set()
        picks: list[str] = []

        rid = self.latest_run_id()
        if rid:
            s = self._store_for(rid)
            meta = s.run_meta_path("prices")
            if meta.exists():
                try:
                    m = json.loads(meta.read_text())
                    for t in self._coerce_tickers(m.get("tickers", [])):
                        u = t.upper()
                        if u not in seen:
                            seen.add(u)
                            picks.append(u)
                            if len(picks) >= limit:
                                return picks
                except Exception:
                    pass

            man = s.manifest_path
            if man.exists():
                try:
                    d = json.loads(man.read_text())
                    cfg = d.get("config") or {}
                    raw = (cfg.get("universe", {}) or {}).get("tickers") or d.get("tickers") or []
                    for t in self._coerce_tickers(raw):
                        u = t.upper()
                        if u not in seen:
                            seen.add(u)
                            picks.append(u)
                            if len(picks) >= limit:
                                return picks
                except Exception:
                    pass

        defaults = [
            # Tech / Communication
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "ORCL",
            "CSCO",
            # Financials
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            # Energy
            "XOM",
            "CVX",
            "COP",
            # Healthcare
            "UNH",
            "LLY",
            "JNJ",
            "PFE",
            "MRK",
            # Consumer Staples
            "KO",
            "PEP",
            "WMT",
            "COST",
            # Consumer Discretionary
            "HD",
            "MCD",
            "NKE",
            "DIS",
            # Industrials / Others
            "CAT",
            "BA",
        ]

        for t in defaults:
            if t not in seen:
                seen.add(t)
                picks.append(t)
                if len(picks) >= limit:
                    break

        return picks[:limit]

    @staticmethod
    def _coerce_tickers(val) -> List[str]:
        if val is None:
            return []
        if isinstance(val, str):
            return [t.strip().upper() for t in val.split(",") if t.strip()]
        if isinstance(val, list):
            return [str(t).strip().upper() for t in val if str(t).strip()]
        return []

    @staticmethod
    def _coerce_date(val) -> Optional[str]:
        if not val:
            return None
        # accept already-good "YYYY-MM-DD" or anything truthy; GUI will parse
        return str(val)
