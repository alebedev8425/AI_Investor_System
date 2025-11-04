# src/system/Controller/reporting_manager.py
from __future__ import annotations
from pathlib import Path
from system.Model.artifact_store import ArtifactStore
from system.Model.reporting.report_builder import RunInputs, build_run_report, build_compare_report

class ReportingManager:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def build_single(self) -> Path:
        """Builds report for the store.run_id. Returns the HTML path."""
        html = self._store.report_html_path()
        assets = self._store.report_assets_dir()
        rin = RunInputs(
            daily_path=self._store.backtest_returns_path(),
            metrics_path=self._store.backtest_metrics_path(),
            manifest_path=self._store.manifest_path,
            assets_dir=assets,
        )
        build_run_report(rin, html)
        return html

    def build_compare(self, run_a: str, run_b: str) -> Path:
        """Builds a comparison report for two runs. Returns the compare HTML path."""
        # Use fresh store handles pointed at each run
        sa = ArtifactStore(self._store.root); sa.ensure_existing_run(run_a)
        sb = ArtifactStore(self._store.root); sb.ensure_existing_run(run_b)

        tag = f"{run_a}_vs_{run_b}"
        html = self._store.compare_html_path(tag)
        assets = self._store.compare_assets_dir(tag)

        a_in = RunInputs(sa.backtest_returns_path(), sa.backtest_metrics_path(), sa.manifest_path, assets)
        b_in = RunInputs(sb.backtest_returns_path(), sb.backtest_metrics_path(), sb.manifest_path, assets)

        build_compare_report(a_in, b_in, html, label_a=run_a, label_b=run_b)
        return html