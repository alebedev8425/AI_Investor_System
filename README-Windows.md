# AI_Investor_System — Windows Setup & Usage (Python 3.11)

This guide walks you from a blank Windows machine to running the GUI and CLI, generating reports, and building comparisons.

---

## 1. Install Prerequisites
- Python 3.11 (64-bit) from python.org

  - Visit https://www.python.org/downloads/windows/ and download the latest Python 3.11.x "Windows installer (64-bit)".
  - Run the installer:
    - Check the box "Add Python 3.11 to PATH".
    - Click "Customize installation" if you want, but default options are fine.
    - Finish the installation and close the installer.
  - Verify in a new Command Prompt:
    ```bat
    python --version
    ```
    (Expected: Python 3.11.x)  
    If it prints a different version, use the Python Launcher:
    ```bat
    py -3.11 --version
    ```

- Git for Windows

  - Visit https://git-scm.com/download/win and run the installer.
  - Accept defaults unless you have preferences.

- Optional: NVIDIA CUDA for GPU acceleration with PyTorch

  - If you have an NVIDIA GPU and want CUDA:
    - Make sure your GPU and driver support the chosen CUDA toolkit.
    - You can install a CUDA-enabled PyTorch wheel later (below). If unsure, use CPU first.

---

## 2) Clone the Repository

I recommend using VS code for simplicity.  
Download VS code from this link: https://code.visualstudio.com/download

Now, open vscode, and click the 2nd icon from the right, in the top right hand corner to open the terminal.

In GitHub, copy the HTTPS clone url from the big green CODE button.  

Then in the terminal, type "git clone ", and paste the HTTPS clone url  

Then, cd to the name of the repository cloned, it should be "AI_Investor_System"

Lastly, when located in the right directory, type "code ." to open the project in a new vs code window.



## 3) Create and Activate a Virtual Environment

Again, open the terminal in vs code, then:

**Command Prompt:**
```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

**PowerShell:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Verification:**
```bat
python --version
pip --version
```

**Upgrade packaging tools:**
```bat
python -m pip install --upgrade pip setuptools wheel
```

---

## 4) Install Dependencies

(Powershell is what is used if you click the 2nd from right button in top right of vscode)

**Run:**
```bat
pip install -r requirements/base.txt
pip install -r requirements/torch/windows-cpu.txt
```

If for any reason this does not work (it should) do the following:

Install the core libraries explicitly (CPU-friendly defaults):
```bat
pip install PySide6 pandas numpy matplotlib pyyaml
```

**PyTorch (CPU, simple and reliable):**
```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Optional: PyTorch with CUDA (replace cu124 with your CUDA version if needed):**
```bat
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Note:** If you see SSL issues, ensure your system certificates are up to date and consider:
```bat
python -m pip install --upgrade certifi
```

---

## 5) Project Structure (What You Should Expect)

- `experiments/` (YAML configs, e.g. `exp_tech_lstm_softmax.yaml`)
- `src/system/...` (GUI, CLI, report builders, controllers)
- `artifacts/` (auto-created at runtime for run outputs and comparisons)
- `.gitignore` (should exclude artifacts/, venvs, caches, etc.)

You may see a folder name with a space in your user path (e.g., "AI_Investor_System Research"). This is fine; commands still work, but you may need to quote paths if typing manually.

---

## 6) Configure the GUI Combo → YAML Mapping

In file: `src/system/View/gui.py`  
There is a CONFIG_MAP dictionary mapping GUI selections to YAML files in experiments/. Right now, only technical features, lstm, and softmax is implemented. For example:  
`"technical|lstm|softmax": "experiments/exp_tech_lstm_softmax.yaml"`

If you more YAMLs are added, their combos to CONFIG_MAP will be added accordingly.

---

## 7) Launch the GUI

Using PowerShell from the repo root (MAKE SURE VENV IS ACTIVATED):
```bat
$env:PYTHONPATH = "src"
python -m system.View.gui
```

In the GUI:
- Select your Features (e.g., Technical), Model (e.g., lstm), and Allocator (e.g., softmax).
- Click "Run" to execute the experiment.
- Click "Open latest report" to load `artifacts/<latest*run>/reports/report.html`.
- Go to "Compare" tab, pick two runs (A and B), then "Build compare" to create:  
  `artifacts/_comparisons/<A>_vs*<B>/compare.html`

**Tips:**
- If compare doesn’t show immediately, click Build compare again or re-open. This is often just viewer caching. The system writes the correct compare.html in the `_comparisons` folder.

---


## 8) Outputs & Where to Find Them

**Per-run outputs (created automatically):**
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
artifacts/_comparisons/<A>_vs_<B>/assets/equity*compare.png
artifacts/_comparisons/<A>_vs*<B>/assets/drawdown_compare.png
```
