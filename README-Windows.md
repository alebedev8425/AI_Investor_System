# AI_Investor_System — Windows Setup & Usage (Python 3.11)

This guide walks you from a blank Windows machine to running the GUI and CLI, generating reports, and building comparisons.

===============================================================================

1. # Install Prerequisites
   • Python 3.11 (64-bit) from python.org

- Visit https://www.python.org/downloads/windows/ and download the latest Python 3.11.x "Windows installer (64-bit)".
- Run the installer:
  • Check the box "Add Python 3.11 to PATH".
  • Click "Customize installation" if you want, but default options are fine.
  • Finish the installation and close the installer.
- Verify in a new Command Prompt:
  python --version
  (Expected: Python 3.11.x)
  If it prints a different version, use the Python Launcher:
  py -3.11 --version

• Git for Windows

- Visit https://git-scm.com/download/win and run the installer.
- Accept defaults unless you have preferences.

• Optional: NVIDIA CUDA for GPU acceleration with PyTorch

- If you have an NVIDIA GPU and want CUDA:
  • Make sure your GPU and driver support the chosen CUDA toolkit.
  • You can install a CUDA-enabled PyTorch wheel later (below). If unsure, use CPU first.

=============================================================================== 2) Clone the Repository
===============================================================================
Open Command Prompt (or PowerShell), then:
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

Tip: If your path contains spaces, quoting is fine:
cd "C:\Users\<you>\Documents\AI_Investor_System Research\AI_Investor_System"

=============================================================================== 3) Create and Activate a Virtual Environment
===============================================================================
Command Prompt:
py -3.11 -m venv .venv
.\.venv\Scripts\activate

PowerShell:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

Verification:
python --version
pip --version

Upgrade packaging tools:
python -m pip install --upgrade pip setuptools wheel

=============================================================================== 4) Install Dependencies
===============================================================================
Run:
pip install -r requirements/base.txt
pip install -r requirements/torch/windows-cpu.txt

If not, install the core libraries explicitly (CPU-friendly defaults):
pip install PySide6 pandas numpy matplotlib pyyaml

PyTorch (CPU, simple and reliable):
pip install torch --index-url https://download.pytorch.org/whl/cpu

Optional: PyTorch with CUDA (replace cu124 with your CUDA version if needed):
pip install torch --index-url https://download.pytorch.org/whl/cu124

Note: If you see SSL issues, ensure your system certificates are up to date and consider:
python -m pip install --upgrade certifi

=============================================================================== 5) Project Structure (What You Should Expect)
===============================================================================
• experiments/ (YAML configs, e.g. exp_tech_lstm_softmax.yaml)
• src/system/... (GUI, CLI, report builders, controllers)
• artifacts/ (auto-created at runtime for run outputs and comparisons)
• .gitignore (should exclude artifacts/, venvs, caches, etc.)

You may see a folder name with a space in your user path (e.g., "AI_Investor_System Research"). This is fine; commands still work, but you may need to quote paths if typing manually.

=============================================================================== 6) Configure the GUI Combo → YAML Mapping
===============================================================================
In file: src/system/View/gui.py
There is a CONFIG_MAP dictionary mapping GUI selections to YAML files in experiments/. Make sure the combination you choose in the GUI exists in CONFIG_MAP and the YAML file exists in experiments/. For example:
"technical|lstm|softmax": "experiments/exp_tech_lstm_softmax.yaml"

If you add more YAMLs, add their combos to CONFIG_MAP accordingly.

=============================================================================== 7) Launch the GUI
===============================================================================
Using Command Prompt or PowerShell from the repo root (MAKE SURE VENV IS ACTIVATED):
set PYTHONPATH=src && python -m system.View.gui

(If using PowerShell, this works the same in a single line.)

In the GUI:
• Select your Features (e.g., Technical), Model (e.g., lstm), and Allocator (e.g., softmax).
• Click "Run" to execute the experiment.
• Click "Open latest report" to load artifacts/<latest*run>/reports/report.html.
• Go to "Compare" tab, pick two runs (A and B), then "Build compare" to create:
artifacts/\_comparisons/<A>\_vs*<B>/compare.html

Tips:
• If compare doesn’t show immediately, click Build compare again or re-open. This is often just viewer caching. The system writes the correct compare.html in the \_comparisons folder.

=============================================================================== 8) Run the CLI (Optional, No GUI)
===============================================================================
Run a single experiment:
set PYTHONPATH=src && python -m system.View.cli run --config experiments\exp_tech_lstm_softmax.yaml

Build a comparison between two runs (use folder names from artifacts/):
set PYTHONPATH=src && python -m system.View.cli compare --a 20251104_214520_phase1_baseline --b 20251104_214206_phase1_baseline

Open the resulting HTML (double-click in File Explorer or load it in the GUI compare tab).

=============================================================================== 9) Outputs & Where to Find Them
===============================================================================
Per-run outputs (created automatically):
artifacts/<run_id>/manifest.json
artifacts/<run_id>/backtests/daily_returns.csv
artifacts/<run_id>/backtests/metrics.json
artifacts/<run_id>/reports/assets/\*.png
artifacts/<run_id>/reports/report.html

Comparisons:
artifacts/_comparisons/<A>\_vs_<B>/compare.html
artifacts/_comparisons/<A>\_vs_<B>/assets/equity*compare.png
artifacts/\_comparisons/<A>\_vs*<B>/assets/drawdown_compare.png

=============================================================================== 10) Troubleshooting & Tips
===============================================================================
• “No config for combo” in GUI:

- Add the combo → YAML mapping to CONFIG_MAP and ensure the YAML exists in experiments/.
  • Compare appears stale:
- Click “Build compare” again or switch tabs and return. It’s usually a cache issue in the viewer. The HTML on disk is correct.
  • Dark background / light text for reports:
- The report builder CSS is already tuned for light text on dark backgrounds (as per the code you integrated). If you further customize CSS, keep body text and labels light.
  • QShortcut import:
- The code imports QShortcut and QKeySequence from PySide6.QtGui. This is correct on Windows.
  • Device selection (CPU/GPU):
- The code you shared logs device mps on macOS. On Windows you’ll see CUDA if installed, otherwise CPU. If you have custom device-selection logic elsewhere, ensure it falls back to CPU when no GPU is found.
  • Paths with spaces:
- Use quotes when typing paths by hand, or place the repo in a short path like C:\src\AI_Investor_System.
