# --- robust Qt bootstrap (macOS/Windows/Linux) ---
from __future__ import annotations
import os, sys, platform, importlib
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from PySide6 import QtGui
from PySide6.QtGui import QShortcut, QKeySequence


def find_project_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for _ in range(6):
        if (cur / "experiments").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


PROJ_ROOT = find_project_root()

# Map GUI selection -> existing YAML under experiments/
# Fill ONLY the combos you already have.
CONFIG_MAP = {
    "technical|lstm|softmax": "experiments/exp_tech_lstm_softmax.yaml",
    # "technical+sentiment|lstm|softmax": "experiments/exp_lstm_sent_softmax.yaml",
    # "technical|transformer|softmax": "experiments/exp_transformer_softmax.yaml",
    # "technical|lstm|mean-variance": "experiments/exp_lstm_mvo.yaml",
    # "technical|gnn|softmax": "experiments/exp_gnn_softmax.yaml",
    # "technical+sentiment+events+correlations+graph|transformer|rl": "experiments/exp_trans_full_rl.yaml",
}


class Runner(QtCore.QObject):
    line = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.proc: QtCore.QProcess | None = None

    def run(self, cfg_path: Path):
        if self.proc:
            self.proc.kill()
            self.proc.deleteLater()
        self.proc = QtCore.QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setWorkingDirectory(str(PROJ_ROOT))
        self.proc.setArguments(["-m", "system.View.cli", "run", "--config", str(cfg_path)])
        self.proc.readyReadStandardOutput.connect(
            lambda: self._drain(self.proc.readAllStandardOutput())
        )
        self.proc.readyReadStandardError.connect(
            lambda: self._drain(self.proc.readAllStandardError())
        )
        self.proc.finished.connect(lambda code, status: self.finished.emit(code))
        self.proc.start()

    def _drain(self, qba):
        text = bytes(qba).decode("utf-8", errors="replace")
        for ln in text.splitlines():
            self.line.emit(ln)

    def run_compare(self, run_a: str, run_b: str):
        if self.proc:
            self.proc.kill()
            self.proc.deleteLater()
        self.proc = QtCore.QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setWorkingDirectory(str(PROJ_ROOT))
        # use CLI compare to build compare report into _comparisons/<tag>/compare.html
        self.proc.setArguments(["-m", "system.View.cli", "compare", "--a", run_a, "--b", run_b])
        self.proc.readyReadStandardOutput.connect(
            lambda: self._drain(self.proc.readAllStandardOutput())
        )
        self.proc.readyReadStandardError.connect(
            lambda: self._drain(self.proc.readAllStandardError())
        )
        self.proc.finished.connect(lambda code, status: self.finished.emit(code))
        self.proc.start()


class MainWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Investor System — Runner")
        self.resize(1100, 720)
        self._pending_compare_path: Path | None = None

        # ---------- central area: left controls + right tabs ----------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # left controls (same as before)
        self.chk_tech = QtWidgets.QCheckBox("Technical")
        self.chk_tech.setChecked(True)
        self.chk_sent = QtWidgets.QCheckBox("Sentiment")
        self.chk_event = QtWidgets.QCheckBox("Events")
        self.chk_corr = QtWidgets.QCheckBox("Correlations")
        self.chk_graph = QtWidgets.QCheckBox("Graph")

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["lstm", "transformer", "gnn"])
        self.cmb_alloc = QtWidgets.QComboBox()
        self.cmb_alloc.addItems(["softmax", "mean-variance", "rl"])

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
        lv.addWidget(gb_feat)
        lv.addWidget(gb_model)
        lv.addWidget(gb_alloc)
        lv.addStretch(1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)

        # right tabs: Report / Compare
        tabs = QtWidgets.QTabWidget()
        # Report tab
        rep_tab = QtWidgets.QWidget()
        rep_lay = QtWidgets.QVBoxLayout(rep_tab)
        self.report = QtWidgets.QTextBrowser()
        self.report.setOpenExternalLinks(True)
        self.btn_view_report = QtWidgets.QPushButton("Open latest report")
        rep_lay.addWidget(self.btn_view_report)
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
        cmp_lay.addLayout(pick)
        cmp_lay.addWidget(self.compare_view, 1)
        tabs.addTab(cmp_tab, "Compare")

        # splitter: left controls | right tabs
        split_top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split_top.addWidget(left)
        split_top.addWidget(tabs)
        split_top.setSizes([300, 800])

        # central layout container
        c_lay = QtWidgets.QVBoxLayout(central)
        c_lay.addWidget(split_top, 1)

        # ---------- log in a dock (compact & togglable) ----------
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(3000)  # cap memory & height
        self.out.setPlaceholderText("Logs will appear here…")
        dock = QtWidgets.QDockWidget("Log", self)
        dock.setWidget(self.out)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        dock.setFixedHeight(160)  # keep it compact
        self._log_dock = dock

        # hotkey to toggle log
        self._toggleLogShortcut = QShortcut(QKeySequence("F2"), self)
        self._toggleLogShortcut.activated.connect(
            lambda: self._log_dock.setVisible(not self._log_dock.isVisible())
        )

        # ---------- runner wires ----------
        self.runner = Runner()
        self.runner.line.connect(lambda s: self.out.appendPlainText(s))
        self.runner.finished.connect(self._on_finished)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_clear.clicked.connect(lambda: self.out.setPlainText(""))

        # report actions
        self.btn_view_report.clicked.connect(self._load_latest_report_into_viewer)

        # compare actions
        self.btn_refresh_runs.clicked.connect(self._refresh_runs)
        self.btn_build_compare.clicked.connect(self._on_build_compare_clicked)

        # initial data
        self._refresh_runs()

    # ---------- same helpers as before, with minor add-ons ----------
    def _features_key(self) -> str:
        sel = []
        if self.chk_tech.isChecked():
            sel.append("technical")
        if self.chk_sent.isChecked():
            sel.append("sentiment")
        if self.chk_event.isChecked():
            sel.append("events")
        if self.chk_corr.isChecked():
            sel.append("correlations")
        if self.chk_graph.isChecked():
            sel.append("graph")
        sel.sort()
        return "+".join(sel)

    def _combo_key(self) -> str:
        return (
            f"{self._features_key()}|{self.cmb_model.currentText()}|{self.cmb_alloc.currentText()}"
        )

    def _config_path(self) -> Path | None:
        key = self._combo_key()
        rel = CONFIG_MAP.get(key)
        return (PROJ_ROOT / rel).resolve() if rel else None

    def _on_run_clicked(self):
        cfg = self._config_path()
        if not cfg or not cfg.exists():
            self.out.appendPlainText(f"[Under construction] No config for: {self._combo_key()}")
            self.out.appendPlainText("→ Add this combo to CONFIG_MAP with a valid YAML path.")
            return
        self.out.setPlainText(f"[RUN] {cfg}\n")
        self.runner.run(cfg)

    def _on_finished(self, code: int):
        self.out.appendPlainText("\n[OK] Finished." if code == 0 else f"\n[FAIL] Exit code {code}")
        if code == 0:
            # always refresh latest single-run report
            self._load_latest_report_into_viewer()

            # if we were building a compare, load it now
            if self._pending_compare_path and self._pending_compare_path.exists():
                self._load_html_into(self.compare_view, self._pending_compare_path)
                self.out.appendPlainText(
                    f"[info] Loaded compare report: {self._pending_compare_path}"
                )

        # clear pending path either way
        self._pending_compare_path = None

    def _artifacts_root(self) -> Path:
        return (PROJ_ROOT / "artifacts").resolve()

    def _list_runs(self) -> list[Path]:
        root = self._artifacts_root()
        if not root.exists():
            return []
        runs = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
        runs.sort()
        return runs

    def _latest_run_dir(self) -> Path | None:
        runs = self._list_runs()
        return runs[-1] if runs else None

    def _latest_report_html(self) -> Path | None:
        r = self._latest_run_dir()
        if not r:
            return None
        p = r / "reports" / "report.html"
        return p if p.exists() else None

    def _load_html_into(self, browser: QtWidgets.QTextBrowser, html_path: Path) -> None:
        browser.setSource(QtCore.QUrl.fromLocalFile(str(html_path)))

    def _load_latest_report_into_viewer(self):
        p = self._latest_report_html()
        if not p:
            self.out.appendPlainText("[info] No report.html found for latest run.")
            return
        self._load_html_into(self.report, p)
        self.out.appendPlainText(f"[info] Loaded report: {p}")

    # ----- compare tab -----
    def _refresh_runs(self):
        runs = [r.name for r in self._list_runs()]
        self.cmb_run_a.clear()
        self.cmb_run_b.clear()
        self.cmb_run_a.addItems(runs)
        self.cmb_run_b.addItems(runs)
        if runs:
            self.cmb_run_b.setCurrentIndex(len(runs) - 1)
            self.cmb_run_a.setCurrentIndex(max(0, len(runs) - 2))

    def _on_build_compare_clicked(self):
        a = self.cmb_run_a.currentText().strip()
        b = self.cmb_run_b.currentText().strip()
        if not a or not b or a == b:
            self.out.appendPlainText("[warn] Pick two different runs.")
            return
        self.out.appendPlainText(f"[COMPARE] Building {a} vs {b}")
        # expected output path
        tag = f"{a}_vs_{b}"
        html = self._artifacts_root() / "_comparisons" / tag / "compare.html"
        self._pending_compare_path = html
        self.out.appendPlainText(str(html))
        # kick off the build (do not try to load yet)
        self.runner.run_compare(a, b)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
