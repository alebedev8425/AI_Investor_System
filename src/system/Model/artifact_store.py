# src/system/Model/artifact_store.py  (add near the top of the class)
from __future__ import annotations

import json
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional, Union, Iterable
import pandas as pd

from hashlib import sha1
import shutil


class ArtifactStore:
    """
    ArtifactStore

    Responsibilities:
    - Centralize all filesystem locations and file naming conventions.
    - Create a per-run directory and write a manifest.json (config snapshot, timestamps, etc.).
    - Provide convenience helpers to save/load CSV/JSON artifacts:
        • Price cache (data/)
        • Engineered features (features/)
        • Model checkpoints & scalers (models/)
        • Predictions (predictions/)
        • Portfolio weights (weights/)
        • Backtest results and metrics (backtests/)
        • Reports (reports/)
        • Logs (logs/)
    - Offer discoverability utilities (list_runs, latest_run).

    Design goals:
    - Minimal surface area but easy to extend later.
    - OS-safe paths using pathlib.
    - Atomic writes for JSON/CSV to avoid half-written files on crash.
    """

    def __init__(
        self, root: Union[str, Path], run_id: Optional[str] = None, create: bool = True
    ) -> None:
        """
        :param root: root directory where all artifacts live (e.g., artifacts/)
        :param run_id: optional run identifier (if None, call `new_run(...)` to create one)
        :param create: if True, create root dir if missing
        """
        self.root = Path(root).resolve()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)

        self._run_id = run_id  # populated when you call new_run or pass in existing
        if run_id is not None:
            # Don’t eagerly create; assume this is an existing run folder to be read
            pass

    # ----- basic properties -----
    # use @property to prevent external mutation, and create getters/setters

    @property
    def run_id(self) -> str:
        if not self._run_id:
            raise RuntimeError(
                "ArtifactStore.run_id is not set. Call new_run() or pass run_id at construction."
            )
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    # subdirectories for a given run
    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def features_dir(self) -> Path:
        return self.run_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.run_dir / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def weights_dir(self) -> Path:
        return self.run_dir / "weights"

    @property
    def backtests_dir(self) -> Path:
        return self.run_dir / "backtests"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    # ------------- lifecycle -------------

    def new_run(self, name_hint: str, config_snapshot: Dict[str, Any]) -> str:
        """
        Create a new run folder with a unique run ID and write manifest.json

        :param name_hint: something human-readable like "exp_baseline"
        :param config_snapshot: dict of the resolved configuration used for this run
        :return: the new run_id
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name_hint)
        self._run_id = f"{timestamp}_{safe_name}"
        self._ensure_run_dirs()
        self._write_manifest(
            {
                "run_id": self.run_id,
                "name_hint": name_hint,
                "created_utc": timestamp,
                "config": config_snapshot,
                "versions": self._collect_versions(),
            }
        )
        return self.run_id

    def ensure_existing_run(self, run_id: str) -> None:
        """
        Point the store at an existing run (no creation). Useful for reading artifacts later.
        """
        run_path = self.root / run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run folder does not exist: {run_path}")
        self._run_id = run_id

    # ----- path helpers -----
    # used to help find paths of specific artificat subdirectories

    # data
    def price_cache_path(self) -> Path:
        return self.data_dir / "prices.csv"  # OHLCV for data universe in config

    # features
    def technical_features_path(self) -> Path:
        return self.features_dir / "technical_features.csv"

    def scaler_path(self, name: str = "standard_scaler.pkl") -> Path:
        return self.models_dir / name

    # models
    def model_checkpoint_path(self, name: str = "lstm.ckpt") -> Path:
        return self.models_dir / name

    # predictions (per-date is flexible; Phase 1 keeps a single CSV)
    def predictions_path(self, name: str = "predictions.csv") -> Path:
        return self.predictions_dir / name

    # weights
    def weights_path(self, name: str = "weights.csv") -> Path:
        return self.weights_dir / name

    # backtests
    def backtest_returns_path(self, name: str = "daily_returns.csv") -> Path:
        return self.backtests_dir / name

    def backtest_metrics_path(self, name: str = "metrics.json") -> Path:
        return self.backtests_dir / name

    # reports
    def report_pdf_path(self, name: str = "report.pdf") -> Path:
        return self.reports_dir / name

    # logs
    def log_file_path(self, name: str = "run.log") -> Path:
        return self.logs_dir / name

    # ----- Input/Output helpers  -----

    def _json_default(o):
        from pathlib import Path
        from datetime import date, datetime

        try:
            import numpy as np
            import pandas as pd
        except Exception:
            np = pd = None  # optional

        if isinstance(o, (date, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if pd is not None and isinstance(o, pd.Timestamp):
            return o.isoformat()
        if np is not None and isinstance(o, (np.integer,)):
            return int(o)
        if np is not None and isinstance(o, (np.floating,)):
            return float(o)
        # last-resort fallback:
        return str(o)

    def save_json(self, obj: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=self._json_default)
        os.replace(tmp, path)

    def load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_csv(self, df, path: Path, index: bool = False) -> None:
        if pd is None:
            raise RuntimeError("pandas is required for save_csv/load_csv. Please install pandas.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=index)
        os.replace(tmp, path)

    def load_csv(self, path: Path):
        if pd is None:
            raise RuntimeError("pandas is required for save_csv/load_csv. Please install pandas.")
        return pd.read_csv(path)

    # ------------- discoverability -------------

    # use generator function for memory efficiency, and to allow the caller to stop early
    def list_runs(self) -> Iterable[str]:
        """
        List available run IDs under root. A run is any subdirectory that contains a manifest.json.
        """
        if not self.root.exists():
            return []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / "manifest.json").exists():
                yield child.name

    def latest_run(self) -> Optional[str]:
        """
        Return the latest run id (UTC timestamp prefix makes this the latest by time).
        """
        runs = list(self.list_runs())
        return runs[-1] if runs else None

    # ------------- internals -------------

    def _collect_versions(self) -> Dict[str, str]:
        import platform

        vers = {
            "python": platform.python_version(),
        }
        try:
            import pandas as _pd

            vers["pandas"] = _pd.__version__
        except Exception:
            pass
        try:
            import numpy as _np

            vers["numpy"] = _np.__version__
        except Exception:
            pass
        try:
            import torch as _torch

            vers["torch"] = _torch.__version__
        except Exception:
            pass
        try:
            import yfinance as _yf

            vers["yfinance"] = getattr(_yf, "__version__", "unknown")
        except Exception:
            pass
        return vers

    def _ensure_run_dirs(self) -> None:  # type: ignore[no-redef]
        # create run directory structure if missing
        for d in (
            self.data_dir,
            self.features_dir,
            self.models_dir,
            self.predictions_dir,
            self.weights_dir,
            self.backtests_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        self.save_json(manifest, self.manifest_path)

    def _require_run_id(self) -> None:
        if not self._run_id:
            raise RuntimeError("run_id is not set. Call new_run() first.")

    # ------------- shared cache (across runs) -------------

    @property
    def shared_cache_root(self) -> Path:
        # a single cache for *all* runs under this artifacts root
        p = self.root / "_shared_cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def shared_prices_dir(self) -> Path:
        p = self.shared_cache_root / "prices"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def shared_technical_dir(self) -> Path:
        p = self.shared_cache_root / "technical"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _shared_path(self, base: Path, key: str, ext: str) -> Path:
        return base / f"{key}.{ext}"

    def shared_prices_path(self, key: str) -> Path:
        return self._shared_path(self.shared_prices_dir, key, "csv")

    def shared_prices_meta(self, key: str) -> Path:
        return self._shared_path(self.shared_prices_dir, key, "meta.json")

    def shared_technical_path(self, key: str) -> Path:
        return self._shared_path(self.shared_technical_dir, key, "csv")

    def shared_technical_meta(self, key: str) -> Path:
        return self._shared_path(self.shared_technical_dir, key, "meta.json")

    def materialize_to_run(self, shared_csv: Path, run_csv: Path) -> None:
        run_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared_csv, run_csv)

    def run_meta_path(self, section: str) -> Path:
        if section == "prices":
            return self.data_dir / "prices.meta.json"
        if section == "technical":
            return self.features_dir / "technical.meta.json"
        return self.run_dir / f"{section}.meta.json"

    def write_run_meta(self, section: str, meta: dict) -> None:
        self.save_json(meta, self.run_meta_path(section))
