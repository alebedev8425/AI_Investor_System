# src/system/View/gui.py
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List
from pathlib import Path
import sys
import pandas as pd
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QShortcut, QKeySequence, QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtGui import QTextOption
import qdarktheme

# TYPE HINTS ONLY (no runtime import of Controllers here)
if TYPE_CHECKING:
    from system.Controller.reporting_manager import ReportingManager
    from system.Controller.preview_manager import PreviewManager
    from system.Controller.cache_manager import CacheManager


def find_project_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for _ in range(6):
        if (cur / "experiments").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


PROJ_ROOT = find_project_root()


class Runner(QtCore.QObject):
    """
    Thin wrapper to execute the CLI entry points in a separate process
    while streaming output back to the GUI.
    """

    line = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.proc: QtCore.QProcess | None = None

    def _start(self, args: list[str], env_overrides: dict[str, str] | None = None) -> None:
        if self.proc:
            self.proc.kill()
            self.proc.deleteLater()

        self.proc = QtCore.QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setWorkingDirectory(str(PROJ_ROOT))
        self.proc.setArguments(args)

        env = QtCore.QProcessEnvironment.systemEnvironment()
        if env_overrides:
            for k, v in env_overrides.items():
                env.insert(k, v)
        self.proc.setProcessEnvironment(env)

        self.proc.readyReadStandardOutput.connect(
            lambda: self._drain(self.proc.readAllStandardOutput())
        )
        self.proc.readyReadStandardError.connect(
            lambda: self._drain(self.proc.readAllStandardError())
        )
        self.proc.finished.connect(lambda code, _status: self.finished.emit(code))
        self.proc.start()

    def _drain(self, qba: QtCore.QByteArray) -> None:
        text = bytes(qba).decode("utf-8", errors="replace")
        for ln in text.splitlines():
            self.line.emit(ln)

    # ---- public: run composed config ----
    def run_recipe(
        self,
        features: list[str],
        model: str,
        allocator: str,
        *,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        args = [
            "-m",
            "system.View.cli",
            "recipe",
            "--features",
            ",".join(features),
            "--model",
            model,
            "--allocator",
            allocator,
        ]
        env_overrides: dict[str, str] = {}
        if tickers:
            env_overrides["AI_INV_TICKERS"] = ",".join(tickers)
        if start:
            env_overrides["AI_INV_START"] = start
        if end:
            env_overrides["AI_INV_END"] = end
        self._start(args, env_overrides=env_overrides)

    # ---- public: compare two runs ----
    def run_compare(self, run_a: str, run_b: str) -> None:
        args = ["-m", "system.View.cli", "compare", "--a", run_a, "--b", run_b]
        self._start(args)


class MainWin(QtWidgets.QMainWindow):
    """
    Pure View:
      - No Model imports
      - Talks to Controllers via DI (reporting/preview/cache)
      - Renders UI, gathers inputs, shows outputs
    """

    MIN_BUSINESS_DAYS = 504  # ~24 months (approx business days)
    RECOMM_BUSINESS_DAYS = 504

    def __init__(
        self, reporting: "ReportingManager", preview: "PreviewManager", cache: "CacheManager"
    ):
        super().__init__()
        self._reporting = reporting
        self._preview = preview
        self._cache = cache

        self.setWindowTitle("AI Investor System — Runner")
        self.resize(1200, 760)
        self._pending_compare_path: Path | None = None
        self._last_report_path: Path | None = None

        # ---------- central area ----------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # ----- left controls -----
        self.chk_tech = QtWidgets.QCheckBox("Technical")
        self.chk_tech.setChecked(True)
        self.chk_tech.setEnabled(False)
        self.chk_sent = QtWidgets.QCheckBox("Sentiment")
        self.chk_event = QtWidgets.QCheckBox("Events")
        self.chk_corr = QtWidgets.QCheckBox("Correlations")
        self.chk_graph = QtWidgets.QCheckBox("Graph")

        self.chk_sent.setVisible(False)
        self.chk_event.setVisible(False)

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["baseline", "lstm", "transformer", "gnn"])
        self.cmb_model.currentTextChanged.connect(self._on_model_changed)
        self.cmb_alloc = QtWidgets.QComboBox()
        self.cmb_alloc.addItems(["softmax", "mean-variance", "rl"])

        # --- NEW: Tickers list (checkable, top-10 suggestions) ---
        self.lst_tickers = QtWidgets.QListWidget()
        self.lst_tickers.setUniformItemSizes(True)
        self.lst_tickers.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.lst_tickers.setMinimumHeight(160)

        # Dates
        self.date_start = QtWidgets.QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_start.setDate(QtCore.QDate.currentDate().addMonths(-24))

        self.date_end = QtWidgets.QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setDate(QtCore.QDate.currentDate())

        self.btn_load_universe = QtWidgets.QPushButton("Load from latest run")
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_clear = QtWidgets.QPushButton("Clear log")

        left = QtWidgets.QFrame()
        left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        lv = QtWidgets.QVBoxLayout(left)

        gb_feat = QtWidgets.QGroupBox("Features")
        v = QtWidgets.QVBoxLayout(gb_feat)
        for w in (self.chk_tech, self.chk_sent, self.chk_event, self.chk_corr, self.chk_graph):
            v.addWidget(w)

        gb_model = QtWidgets.QGroupBox("Model")
        v2 = QtWidgets.QVBoxLayout(gb_model)
        v2.addWidget(self.cmb_model)
        gb_alloc = QtWidgets.QGroupBox("Allocator")
        v3 = QtWidgets.QVBoxLayout(gb_alloc)
        v3.addWidget(self.cmb_alloc)

        gb_ud = QtWidgets.QGroupBox("Universe & Dates")
        v4 = QtWidgets.QFormLayout(gb_ud)
        v4.addRow("Tickers (choose):", self.lst_tickers)
        v4.addRow("Start date:", self.date_start)
        v4.addRow("End date:", self.date_end)
        v4.addRow("", self.btn_load_universe)

        lv.addWidget(gb_feat)
        lv.addWidget(gb_model)
        lv.addWidget(gb_alloc)
        lv.addWidget(gb_ud)
        lv.addStretch(1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)

        # ----- right: tabs -----
        tabs = QtWidgets.QTabWidget()

        # Report tab
        rep_tab = QtWidgets.QWidget()
        rep_lay = QtWidgets.QVBoxLayout(rep_tab)
        rep_top = QtWidgets.QHBoxLayout()
        self.btn_view_report = QtWidgets.QPushButton("Open latest report")
        self.btn_export_pdf = QtWidgets.QPushButton("Export to PDF")
        self.btn_export_pdf.setEnabled(False)
        self.btn_export_pdf.setVisible(False)
        rep_top.addWidget(self.btn_view_report)
        rep_top.addWidget(self.btn_export_pdf)
        rep_top.addStretch(1)
        self.report = QtWidgets.QTextBrowser()
        self.report.setOpenExternalLinks(True)
        self.report.setStyleSheet("""
            QTextBrowser {
            background: #ffffff; color: #111;      /* white paper */
            border: 1px solid #2e3440; border-radius: 8px;
            }
            """)
        rep_lay.addLayout(rep_top)
        rep_lay.addWidget(self.report, 1)
        tabs.addTab(rep_tab, "Report")

        # Compare tab
        cmp_tab = QtWidgets.QWidget()
        cmp_lay = QtWidgets.QVBoxLayout(cmp_tab)
        pick = QtWidgets.QHBoxLayout()
        self.cmb_run_a = QtWidgets.QComboBox()
        self.cmb_run_b = QtWidgets.QComboBox()
        self.btn_refresh_runs = QtWidgets.QPushButton("Refresh runs")
        self.btn_build_compare = QtWidgets.QPushButton("Build compare")
        pick.addWidget(QtWidgets.QLabel("A:"))
        pick.addWidget(self.cmb_run_a, 1)
        pick.addSpacing(12)
        pick.addWidget(QtWidgets.QLabel("B:"))
        pick.addWidget(self.cmb_run_b, 1)
        pick.addSpacing(12)
        pick.addWidget(self.btn_refresh_runs)
        pick.addWidget(self.btn_build_compare)
        self.compare_view = QtWidgets.QTextBrowser()
        self.compare_view.setOpenExternalLinks(True)
        self.compare_view.setStyleSheet("""
            QTextBrowser {
            background: #ffffff; color: #111;
            border: 1px solid #2e3440; border-radius: 8px;
            }
            """)
        cmp_lay.addLayout(pick)
        cmp_lay.addWidget(self.compare_view, 1)
        tabs.addTab(cmp_tab, "Compare")

        pick.addStretch(1)

        # Compare pickers: prevent huge minimum width from long run IDs
        for cmb in (self.cmb_run_a, self.cmb_run_b):
            cmb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(22)  # ~22 chars; tune if you want
            cmb.setMinimumWidth(0)  # allow shrinking
            cmb.setMaximumWidth(520)  # hard cap so dock never gets shoved
            cmb.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        split_top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split_top.addWidget(left)
        split_top.addWidget(tabs)
        split_top.setChildrenCollapsible(False)  # stops collapsing to zero
        split_top.setStretchFactor(0, 0)  # left panel fixed-ish
        split_top.setStretchFactor(1, 1)  # tabs expand
        split_top.setSizes([360, 900])

        c_lay = QtWidgets.QVBoxLayout(central)
        c_lay.addWidget(split_top, 1)

        # ---------- log dock ----------
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(3000)
        self.out.setPlaceholderText("Logs will appear here…")
        dock = QtWidgets.QDockWidget("Log", self)
        dock.setWidget(self.out)
        dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        dock.setMinimumWidth(320)
        self._log_dock = dock
        self._saved_log_width = 360

        # for gui color changes
        left.setObjectName("LeftPane")
        tabs.setObjectName("CenterTabs")
        self.report.setObjectName("ReportView")
        self.compare_view.setObjectName("CompareView")
        self.out.setObjectName("LogText")
        dock.setObjectName("LogDock")

        tabs.currentChanged.connect(
            lambda _i: [w.document().setTextWidth(980.0) for w in (self.report, self.compare_view)]
        )

        # hotkey: toggle log
        self._toggleLogShortcut = QShortcut(QKeySequence("F2"), self)
        self._toggleLogShortcut.activated.connect(self._toggle_log_panel)

        self.report.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.report.setMinimumWidth(0)
        self.compare_view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.compare_view.setMinimumWidth(0)

        for _w in (self.report, self.compare_view):
            _w.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
            _w.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            _w.setMinimumWidth(0)
            _w.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)

        # ---------- runner wiring ----------
        self.runner = Runner()
        self.runner.line.connect(lambda s: self.out.appendPlainText(s))
        self.runner.finished.connect(self._on_finished)

        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_clear.clicked.connect(lambda: self.out.setPlainText(""))

        # report actions
        self.btn_view_report.clicked.connect(self._load_latest_report_into_viewer)
        self.btn_export_pdf.clicked.connect(self._on_export_pdf_clicked)

        # compare actions
        self.btn_refresh_runs.clicked.connect(self._refresh_runs)
        self.btn_build_compare.clicked.connect(self._on_build_compare_clicked)

        # universe actions
        self.btn_load_universe.clicked.connect(self._load_universe_from_latest)

        # menus
        self._create_menus()

        # initial populate
        self._refresh_runs()
        self._populate_ticker_list()

        self.cmb_model.currentTextChanged.connect(self._on_model_changed)

        QtCore.QTimer.singleShot(
            0,
            lambda: self.resizeDocks(
                [self._log_dock], [self._saved_log_width], QtCore.Qt.Horizontal
            ),
        )

    # ---------- menus / utilities ----------
    def _create_menus(self) -> None:
        menubar = self.menuBar()

        util_menu = menubar.addMenu("Utilities")
        act_view_data = util_menu.addAction("View data")
        act_view_features = util_menu.addAction("View features")
        act_view_preds = util_menu.addAction("View predictions")
        util_menu.addSeparator()
        act_clean_shared = util_menu.addAction("Clean Shared Cache…")
        act_view_data.triggered.connect(self._on_view_data)
        act_view_features.triggered.connect(self._on_view_features)
        act_clean_shared.triggered.connect(self._on_clean_shared_cache)
        act_view_preds.triggered.connect(self._on_view_predictions)

        view_menu = menubar.addMenu("View")
        act_toggle_log = view_menu.addAction("Toggle Log (F2)")
        act_toggle_log.triggered.connect(
            lambda: self._log_dock.setVisible(not self._log_dock.isVisible())
        )

    def _toggle_log_panel(self) -> None:
        if self._log_dock.isVisible():
            # remember current width before hiding
            self._saved_log_width = self._log_dock.width() or self._saved_log_width
            self._log_dock.setVisible(False)
        else:
            self._log_dock.setVisible(True)
            # force dock to reclaim width on the right
            try:
                self.resizeDocks([self._log_dock], [self._saved_log_width], QtCore.Qt.Horizontal)
            except Exception:
                pass

        # ---------- menu action handlers ----------

    def _on_view_data(self) -> None:
        try:
            df = self._preview.preview_prices(limit=500)
            self._open_df_preview(df, title="Raw price data (latest run)")
        except RuntimeError:
            self._no_run_warning("raw data")
        except FileNotFoundError:
            rid = self._preview.latest_run_id() or "?"
            self._no_file_warning("data", rid)

    def _on_view_features(self) -> None:
        try:
            df = self._preview.preview_technical_features(limit=500)
            self._open_df_preview(df, title="Engineered features (latest run)")
        except RuntimeError:
            self._no_run_warning("features")
        except FileNotFoundError:
            rid = self._preview.latest_run_id() or "?"
            self._no_file_warning("features", rid)

    def _on_view_predictions(self) -> None:
        try:
            df = self._preview.preview_predictions(limit=500)
            self._open_df_preview(df, title="Model predictions (latest run)")
        except RuntimeError:
            self._no_run_warning("predictions")
        except FileNotFoundError:
            rid = self._preview.latest_run_id() or "?"
            self._no_file_warning("predictions", rid)

    def _on_clean_shared_cache(self) -> None:
        if not self._confirm(
            "Clean Shared Cache",
            "Delete artifacts/_shared_cache?\nThis removes cached prices/features shared across runs.",
        ):
            return
        try:
            self._cache.clear_shared_cache()
            self.out.appendPlainText("[ok] Cleared shared cache (artifacts/_shared_cache).")
            QtWidgets.QMessageBox.information(self, "Done", "Shared cache cleared.")
        except Exception as e:
            self.out.appendPlainText(f"[error] Failed to clear shared cache: {e}")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to clear shared cache:\n{e}")

    # ---------- UI helpers ----------
    def _open_df_preview(self, df: pd.DataFrame, *, title: str, max_rows: int = 500) -> None:
        if df is None or df.empty:
            QtWidgets.QMessageBox.information(self, "Empty", "No rows to display.")
            return
        if len(df) > max_rows:
            df = df.head(max_rows)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(900, 500)
        layout = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(dlg)
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        for i in range(len(df)):
            row_vals = df.iloc[i]
            for j, col in enumerate(df.columns):
                val = row_vals[col]
                item = QtWidgets.QTableWidgetItem("" if pd.isna(val) else str(val))
                table.setItem(i, j, item)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        layout.addWidget(table)
        btn = QtWidgets.QPushButton("Close", dlg)
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, alignment=QtCore.Qt.AlignRight)
        dlg.exec()

    def _confirm(self, title: str, text: str) -> bool:
        btn = QtWidgets.QMessageBox.question(
            self,
            title,
            text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return btn == QtWidgets.QMessageBox.Yes

    def _no_run_warning(self, what: str) -> None:
        msg = f"[info] No runs found. Run an experiment before viewing {what}."
        self.out.appendPlainText(msg)
        QtWidgets.QMessageBox.information(self, "No runs yet", msg)

    def _no_file_warning(self, what: str, run_name: str) -> None:
        msg = f"[info] No {what} file found under latest run: {run_name}"
        self.out.appendPlainText(msg)
        QtWidgets.QMessageBox.information(self, f"No {what} found", msg)

    # ---------- Ticker list ----------
    def _populate_ticker_list(self, precheck: List[str] | None = None, limit: int = 31) -> None:
        pre = {t.upper() for t in (precheck or [])}
        self.lst_tickers.clear()
        try:
            suggestions = self._preview.recommended_tickers(limit=limit)
        except Exception:
            suggestions = []

        def _add_item(t: str, checked: bool) -> None:
            it = QtWidgets.QListWidgetItem(t.upper())
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            self.lst_tickers.addItem(it)

        seen = set()
        for t in suggestions:
            seen.add(t.upper())
            _add_item(t, t.upper() in pre)

        for t in pre:
            if t not in seen:
                _add_item(t, True)

    def _selected_tickers(self) -> list[str]:
        out: list[str] = []
        for i in range(self.lst_tickers.count()):
            it = self.lst_tickers.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                out.append(it.text().upper())
        return out

    # ---------- feature/model/alloc ----------
    def _selected_features(self) -> list[str]:
        feats: list[str] = ["tech"]  # always include technical
        if self.chk_sent.isChecked():
            feats.append("sent")
        if self.chk_event.isChecked():
            feats.append("events")
        if self.chk_corr.isChecked():
            feats.append("corr")
        if self.chk_graph.isChecked():
            feats.append("graph")
        return feats

    def _on_model_changed(self, text: str) -> None:
        """
        Enforce model–feature coupling:

        - baseline / lstm / transformer:
            * Technical ON, locked
            * Corr/Graph visible but OFF and disabled
            * Sentiment/Events hidden or disabled (see below)
        - gnn:
            * Technical, Correlations, Graph ON and locked
            * Sentiment/Events kept off for now
        """
        model = text.lower()

        if model in ("baseline", "lstm", "transformer"):
            # Only technical features allowed
            self.chk_tech.setChecked(True)
            self.chk_tech.setEnabled(False)

            # Show corr/graph options but don't allow toggling
            for chk in (self.chk_corr, self.chk_graph):
                chk.setChecked(False)
                chk.setEnabled(False)

            # Sentiment / Events: keep off + disabled for now
            self.chk_sent.setChecked(False)
            self.chk_sent.setEnabled(False)
            self.chk_event.setChecked(False)
            self.chk_event.setEnabled(False)

        elif model == "gnn":
            # GNN: must use tech + corr + graph
            self.chk_tech.setChecked(True)
            self.chk_corr.setChecked(True)
            self.chk_graph.setChecked(True)

            for chk in (self.chk_tech, self.chk_corr, self.chk_graph):
                chk.setEnabled(False)

            # Keep sentiment/events off and disabled (we're not using them yet)
            self.chk_sent.setChecked(False)
            self.chk_sent.setEnabled(False)
            self.chk_event.setChecked(False)
            self.chk_event.setEnabled(False)

        else:
            # Fallback for any future model type: let user control everything
            for chk in (
                self.chk_tech,
                self.chk_sent,
                self.chk_event,
                self.chk_corr,
                self.chk_graph,
            ):
                chk.setEnabled(True)

    def _model_key(self) -> str:
        return self.cmb_model.currentText().lower()

    def _allocator_key(self) -> str:
        txt = self.cmb_alloc.currentText().lower()
        return "meanvar" if txt == "mean-variance" else txt

    def _date_str(self, qdate: QtCore.QDate) -> str:
        return qdate.toString("yyyy-MM-dd")

    # ---------- run ----------
    def _validate_date_range_or_prompt(self) -> bool:
        start = self._date_str(self.date_start.date())
        end = self._date_str(self.date_end.date())
        try:
            bdays = len(pd.bdate_range(start=start, end=end))
        except Exception:
            bdays = 0

        if bdays >= self.MIN_BUSINESS_DAYS:
            return True

        msg = (
            f"The selected date range has only ~{bdays} business days.\n\n"
            f"A minimum of {self.MIN_BUSINESS_DAYS} business days (~24 months) is required "
            f"to train and evaluate the model.\n\n"
            f"Would you like to auto-adjust Start to approximately 24 months before End?"
        )
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Date range too short")
        box.setText(msg)
        yes_btn = box.addButton("Auto-adjust", QtWidgets.QMessageBox.AcceptRole)
        box.exec()

        if box.clickedButton() is yes_btn:
            new_start = self.date_end.date().addMonths(-24)
            self.date_start.setDate(new_start)
            self.out.appendPlainText("[info] Adjusted start date to ~24 months before end.")
            return True

        self.out.appendPlainText("[warn] Run cancelled: date range too short.")
        return False

    def _on_run_clicked(self) -> None:
        feats = self._selected_features()
        model = self._model_key()
        alloc = self._allocator_key()
        tickers = self._selected_tickers()
        start = self._date_str(self.date_start.date())
        end = self._date_str(self.date_end.date())

        # --- Enforce model ↔ feature mapping (safety net) ---
        if model in ("baseline", "lstm", "transformer"):
            # Only technical features for these models
            feats = ["tech"]
        elif model == "gnn":
            # GNN always uses tech + corr + graph
            feats = ["tech", "corr", "graph"]

        if not self._validate_date_range_or_prompt():
            return
        if not feats:
            self.out.appendPlainText("[warn] Select at least one feature block.")
            return

        combo_str = (
            f"features={','.join(feats)} | model={model} | alloc={alloc} | "
            f"tickers={','.join(tickers) if tickers else '(default)'} | "
            f"start={start or '(default)'} | end={end or '(default)'}"
        )
        self.out.setPlainText(f"[RUN recipe] {combo_str}\n")

        self.runner.run_recipe(
            feats,
            model,
            alloc,
            tickers=tickers if tickers else None,
            start=start if start else None,
            end=end if end else None,
        )

    def _on_finished(self, code: int) -> None:
        self.out.appendPlainText("\n[OK] Finished." if code == 0 else f"\n[FAIL] Exit code {code}")

        if code == 0:
            self._load_latest_report_into_viewer()
            if self._pending_compare_path and self._pending_compare_path.exists():
                self._load_html_into(self.compare_view, self._pending_compare_path)
                self.out.appendPlainText(
                    f"[info] Loaded compare report: {self._pending_compare_path}"
                )
        elif code == 3:
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient data for training",
                (
                    "Your configuration produced too few samples to build training/validation/test sets.\n\n"
                    f"Please select a wider date range (≥ {self.MIN_BUSINESS_DAYS} business days; "
                    f"~{self.RECOMM_BUSINESS_DAYS} recommended) and try again."
                ),
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Run failed", "The run aborted early. Check the log for details."
            )

        self._pending_compare_path = None

    # ---------- reports ----------
    def _load_html_into(self, browser: QtWidgets.QTextBrowser, html_path: Path) -> None:
        browser.setSource(QtCore.QUrl.fromLocalFile(str(html_path)))

    def _load_latest_report_into_viewer(self) -> None:
        p = self._reporting.resolve_latest_report_path()
        if not p:
            self._last_report_path = None
            self.btn_export_pdf.setEnabled(False)
            self.btn_export_pdf.setVisible(False)
            self.out.appendPlainText("[info] No report.html found for latest run.")
            QtWidgets.QMessageBox.information(
                self, "No report", "No report.html found for latest run."
            )
            return

        self._load_html_into(self.report, p)
        self._last_report_path = p
        self.btn_export_pdf.setVisible(True)
        self.btn_export_pdf.setEnabled(True)
        self.out.appendPlainText(f"[info] Loaded report: {p}")

    def _on_export_pdf_clicked(self) -> None:
        html_path = self._last_report_path or self._reporting.resolve_latest_report_path()
        if not html_path or not html_path.exists():
            QtWidgets.QMessageBox.information(self, "No report", "Load a report before exporting.")
            return

        suggested = str(html_path.with_suffix(".pdf"))
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export report to PDF", suggested, "PDF Files (*.pdf)"
        )
        if not dest:
            return

        try:
            # 1) Reuse the *same* document the QTextBrowser is using.
            doc = self.report.document()

            # 2) Configure printer → PDF
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(dest)

            # 3) Make the document's page size match the printer *in points*.
            #    This is the critical part: use QPrinter.Point, NOT DevicePixel.
            page_rect = printer.pageRect(QPrinter.Point)
            doc.setPageSize(QtCore.QSizeF(page_rect.size()))
            doc.setTextWidth(page_rect.width())

            # 4) Print
            fn = getattr(doc, "print", None) or getattr(doc, "print_", None)
            if not fn:
                raise RuntimeError("QTextDocument has neither print nor print_")

            fn(printer)

            self.out.appendPlainText(f"[ok] Exported PDF: {dest}")
            QtWidgets.QMessageBox.information(self, "Exported", f"Saved PDF to:\n{dest}")
        except Exception as e:
            self.out.appendPlainText(f"[error] PDF export failed: {e}")
            QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not export:\n{e}")

    def _load_universe_from_latest(self) -> None:
        tickers, start, end = self._preview.load_universe_defaults()
        if not tickers and not start and not end:
            self._no_run_warning("universe/dates")
            return

        self._populate_ticker_list(precheck=tickers, limit=10)

        if start:
            self.date_start.setDate(QtCore.QDate.fromString(start, "yyyy-MM-dd"))
        if end:
            self.date_end.setDate(QtCore.QDate.fromString(end, "yyyy-MM-dd"))
        self.out.appendPlainText("[info] Loaded universe/dates from latest run manifest.")

    # ---------- compare tab ----------
    def _refresh_runs(self) -> None:
        runs = self._preview.list_runs()
        self.cmb_run_a.clear()
        self.cmb_run_b.clear()
        self.cmb_run_a.addItems(runs)
        self.cmb_run_b.addItems(runs)
        if runs:
            self.cmb_run_b.setCurrentIndex(len(runs) - 1)
            self.cmb_run_a.setCurrentIndex(max(0, len(runs) - 2))

    def _on_build_compare_clicked(self) -> None:
        a = self.cmb_run_a.currentText().strip()
        b = self.cmb_run_b.currentText().strip()
        if not a or not b or a == b:
            self.out.appendPlainText("[warn] Pick two different runs.")
            return

        self.out.appendPlainText(f"[COMPARE] Building {a} vs {b}")

        # Let Controller decide where compare.html lives (keeps View FS-agnostic)
        try:
            html = self._reporting.expected_compare_html_path(a, b)
        except AttributeError:
            # fallback: predictable tag (temporary until controller method added)
            tag = f"{a}_vs_{b}"
            html = (PROJ_ROOT / "artifacts" / "_comparisons" / tag / "compare.html").resolve()

        self._pending_compare_path = html
        self.out.appendPlainText(str(html))
        self.runner.run_compare(a, b)


def main() -> None:
    # Local, late imports keep View module free of Model at import time.
    from system.Model.artifact_store import ArtifactStore
    from system.Controller.reporting_manager import ReportingManager
    from system.Controller.preview_manager import PreviewManager
    from system.Controller.cache_manager import CacheManager

    app = QtWidgets.QApplication(sys.argv)

    THEME_QSS = """
    /* ---------- Left setup pane (GREY) ---------- */
    #LeftPane {
        background: #f3f4f6;   /* light gray */
    }
    #LeftPane QGroupBox {
        background: transparent;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin-top: 12px;
    }
    #LeftPane QGroupBox::title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 6px;
        color: #374151;
        background: transparent;
    }
    #LeftPane QLabel,
    #LeftPane QCheckBox,
    #LeftPane QComboBox,
    #LeftPane QListWidget {
        color: #111827;
    }
    #LeftPane QComboBox, #LeftPane QListWidget, #LeftPane QLineEdit {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 6px;
    }
    #LeftPane QPushButton {
        background: #111827;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 6px 10px;
    }
    #LeftPane QPushButton:hover { background: #1f2937; }


    #CenterTabs QWidget { min-width: 0px; }  /* prevents child widgets from forcing a wide min size */

    /* ---------- Right log dock (BLACK) ---------- */
    #LogDock { background: #000000; color: #e5e7eb; }
    #LogDock::title { background: #000000; color: #e5e7eb; padding-left: 6px; }
    #LogText {
        background: #000000;
        color: #e5e7eb;
        border: none;
        selection-background-color: #374151;
    }

    /* Optional: splitter handle */
    QSplitter::handle { background: #d1d5db; }
    QSplitter::handle:hover { background: #9ca3af; }
    """

    # Apply dark base, then override the three zones
    qdarktheme.setup_theme("dark", additional_qss=THEME_QSS)

    artifacts_root = (find_project_root() / "artifacts").resolve()
    store = ArtifactStore(artifacts_root, create=True)
    reporting = ReportingManager(store)
    preview = PreviewManager(store)
    cache = CacheManager(store)

    w = MainWin(reporting, preview, cache)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
