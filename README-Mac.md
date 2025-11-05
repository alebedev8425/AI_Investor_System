# AI_Investor_System — macOS Setup & Usage (Python 3.11, Apple Silicon Friendly)

This guide takes you from a fresh macOS setup to running the GUI and CLI, generating reports, and building comparisons.

---

## 1. Install Prerequisites
- Python 3.11 (universal2) from python.org

  - Visit https://www.python.org/downloads/macos/ and download the latest Python 3.11.x macOS “universal2” installer.
  - Run the installer and follow the prompts.
  - After installation, verify in Terminal:
    ```bash
    python3 --version
    ```
    (Expected: Python 3.11.x)

  - Optional but recommended: run the installed **Install Certificates.command** located in the Python 3.11 folder (e.g., `/Applications/Python 3.11/`). This fixes common SSL issues for pip.

- Git
  - If Git is missing, install Xcode Command Line Tools:
    ```bash
    xcode-select --install
    ```
  - Or install via Homebrew:
    ```bash
    brew install git
    ```

- Optional: Apple Silicon GPU Acceleration (MPS) for PyTorch
  - Current PyTorch wheels on macOS typically enable MPS automatically on Apple Silicon. If unavailable, CPU works fine.

---

## 2) Clone the Repository

Open Terminal, then:
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Paths with spaces are fine. Example:
```bash
cd "/Users/<you>/Documents/AI_Investor_System Research/AI_Investor_System"
```

---

## 3) Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verification:**
```bash
python --version
pip --version
```

**Upgrade packaging tools:**
```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 4) Install Dependencies

If the repo includes requirement files:
```bash
pip install -r requirements/base.txt
```

If not, install core libraries directly (CPU/MPS-friendly):
```bash
pip install PySide6 pandas numpy matplotlib pyyaml
```

**PyTorch (MPS or CPU; default macOS wheel supports MPS on Apple Silicon):**
```bash
pip install torch
```

**CPU-only wheel (optional):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 5) Project Structure (What You Should Expect)

- `experiments/` (YAML configs, e.g., `exp_tech_lstm_softmax.yaml`)
- `src/system/...` (GUI, CLI, report builders, controllers)
- `artifacts/` (auto-created for run outputs and comparisons)
- `.gitignore` (should exclude artifacts/, venvs, caches, etc.)

Your user path may include spaces (for example, `AI_Investor_System Research`). This is fine; quote paths when typing commands by hand.

---

## 6) Configure the GUI Combo → YAML Mapping

In file: `src/system/View/gui.py`  
Ensure the `CONFIG_MAP` dictionary includes your chosen GUI combination mapped to an existing YAML under `experiments/`. Example:  
`"technical|lstm|softmax": "experiments/exp_tech_lstm_softmax.yaml"`

If you add more YAMLs, add their combos to `CONFIG_MAP` accordingly.

---

## 7) Launch the GUI

From the repo root (with venv activated):
```bash
PYTHONPATH=src python -m system.View.gui
```

In the GUI:
- Select Features, Model, and Allocator.
- Click “Run” to execute the configuration.
- Click “Open latest report” to load `artifacts/<latest_run>/reports/report.html`.
- In the “Compare” tab, pick two runs (A and B) and click “Build compare” to create:  
  `artifacts/_comparisons/<A>_vs_<B>/compare.html`

**Tip:**
- If compare doesn’t appear immediately, it’s usually viewer caching. Click **Build compare** once more or switch tabs and return. The saved HTML in `_comparisons` is correct.

---

## 8) Run the CLI (Optional, No GUI)

Single run:
```bash
PYTHONPATH=src python -m system.View.cli run --config experiments/exp_tech_lstm_softmax.yaml
```

Compare:
```bash
PYTHONPATH=src python -m system.View.cli compare --a 20251104_214520_phase1_baseline --b 20251104_214206_phase1_baseline
```

Open the resulting HTML directly in your browser or via the GUI compare tab.

---

## 9) Outputs & Where to Find Them

**Per-run (auto-created):**
```
artifacts/<run_id>/manifest.json
artifacts/<run_id>/backtests/daily_returns.csv
artifacts/<run_id>/backtests/metrics.json
artifacts/<run_id>/reports/assets/*.png
artifacts/<run_id>/reports/report.html
```

**Comparisons:**
```
artifacts/_comparisons/<A>_vs_<B>/compare.html
artifacts/_comparisons/<A>_vs_<B>/assets/equity_compare.png
artifacts/_comparisons/<A>_vs_<B>/assets/drawdown_compare.png
```
