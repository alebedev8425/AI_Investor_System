# src/system/View/gui.py
from __future__ import annotations
import sys
from pathlib import Path
import os
import pathlib

# Pick the binding you installed
binding = "PySide6"
m = __import__(binding)

# Try common plugin locations inside the wheel
base = pathlib.Path(m.__file__).parent
candidates = [
    base / "Qt" / "plugins" / "platforms",
    base / "Qt6" / "plugins" / "platforms",
]
for p in candidates:
    if p.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(p))
        break

# (optional) make Qt verbose about plugin loading while debugging
# os.environ["QT_DEBUG_PLUGINS"] = "1"

os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")

from PySide6 import QtCore, QtWidgets


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
        self.proc.finished.connect(self.finished.emit)
        self.proc.start()

    def _drain(self, qba):
        text = bytes(qba).decode("utf-8", errors="replace")
        for ln in text.splitlines():
            self.line.emit(ln)


class MainWin(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Investor System — Runner")
        self.resize(900, 600)

        # Features
        self.chk_tech = QtWidgets.QCheckBox("Technical")
        self.chk_tech.setChecked(True)
        self.chk_sent = QtWidgets.QCheckBox("Sentiment")
        self.chk_event = QtWidgets.QCheckBox("Events")
        self.chk_corr = QtWidgets.QCheckBox("Correlations")
        self.chk_graph = QtWidgets.QCheckBox("Graph")
        feat_box = QtWidgets.QGroupBox("Feature bundle (NASDAQ universe)")
        v = QtWidgets.QVBoxLayout(feat_box)
        for w in (self.chk_tech, self.chk_sent, self.chk_event, self.chk_corr, self.chk_graph):
            v.addWidget(w)

        # Model
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["lstm", "transformer", "gnn"])
        mdl_box = QtWidgets.QGroupBox("Model")
        v2 = QtWidgets.QVBoxLayout(mdl_box)
        v2.addWidget(self.cmb_model)

        # Allocator
        self.cmb_alloc = QtWidgets.QComboBox()
        self.cmb_alloc.addItems(["softmax", "mean-variance", "rl"])
        alloc_box = QtWidgets.QGroupBox("Allocator")
        v3 = QtWidgets.QVBoxLayout(alloc_box)
        v3.addWidget(self.cmb_alloc)

        # Buttons
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_clear = QtWidgets.QPushButton("Clear log")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_run)
        row.addStretch(1)
        row.addWidget(self.btn_clear)

        # Output
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(feat_box, 2)
        top.addWidget(mdl_box, 1)
        top.addWidget(alloc_box, 1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(row)
        lay.addWidget(self.out, 10)

        # Runner
        self.runner = Runner()
        self.runner.line.connect(lambda s: self.out.appendPlainText(s))
        self.runner.finished.connect(self._on_finished)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_clear.clicked.connect(lambda: self.out.setPlainText(""))

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
        if not rel:
            return None
        return (PROJ_ROOT / rel).resolve()

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
