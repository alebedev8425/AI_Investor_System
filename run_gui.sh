#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- CONFIG ----------
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
LOGDIR="$ROOT/artifacts/_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/qt_autofix_${STAMP}.log"

# Pin a stable range that works on macOS 13–15 with Py3.11 arm64/x86_64
PYSIDE_SPEC='PySide6>=6.8,<6.9'
SHIBOKEN_SPEC='shiboken6>=6.8,<6.9'

# Set REBUILD_VENV=1 to allow the script to nuke/rebuild the venv as a last resort
REBUILD_VENV="${REBUILD_VENV:-0}"

say() { echo -e "$*" | tee -a "$LOG" ; }
die() { say "\n[fatal] $*"; exit 1; }

ensure_venv() {
  [[ -x "$PY" ]] || die "venv missing at $VENV. Create it first: python3.11 -m venv .venv && source .venv/bin/activate"
}

qt_paths_env() {
  # Compute plugin/lib paths from the wheel itself
  local pydir
  pydir="$("$PY" - <<'PY'
import PySide6, pathlib
print(pathlib.Path(PySide6.__file__).resolve().parent)
PY
)"
  local plugins="$pydir/Qt/plugins"
  local libs="$pydir/Qt/lib"
  export QT_QPA_PLATFORM_PLUGIN_PATH="$plugins/platforms"
  export QT_PLUGIN_PATH="$plugins"
  export DYLD_FRAMEWORK_PATH="$libs"
  export DYLD_LIBRARY_PATH="$libs"
  # Make logs verbose when things go sideways
  export QT_DEBUG_PLUGINS="${QT_DEBUG_PLUGINS:-0}"
}

clear_conflicting_env() {
  unset QT_QPA_PLATFORM
  unset QT_PLUGIN_PATH
  unset QT_QPA_PLATFORM_PLUGIN_PATH
  unset DYLD_FRAMEWORK_PATH
  unset DYLD_LIBRARY_PATH
}

arch_check() {
  local sys_arch lib_path archs info
  sys_arch="$(uname -m)"
  lib_path="$("$PY" - <<'PY'
import PySide6, pathlib
print((pathlib.Path(PySide6.__file__).resolve().parent/"Qt/plugins/platforms/libqcocoa.dylib").as_posix())
PY
)"
  if [[ ! -f "$lib_path" ]]; then
    die "libqcocoa.dylib not found at $lib_path"
  fi

  # Prefer lipo for clean arch reporting
  archs="$(lipo -archs "$lib_path" 2>/dev/null || true)"
  if [[ -n "$archs" ]]; then
    say "[check] system arch: $sys_arch | lib archs: $archs"
    if ! grep -qw "$sys_arch" <<<"$archs"; then
      die "Architecture mismatch: Python ($sys_arch) vs libqcocoa ($archs). Use a matching Python."
    fi
    return 0
  fi

  # Fallback to 'file' output
  info="$(file "$lib_path" 2>/dev/null || true)"
  say "[check] system arch: $sys_arch | lib info: $info"
  if [[ "$info" == *"universal"* ]]; then
    # universal covers both, accept
    return 0
  fi
  if ! grep -qi "$sys_arch" <<<"$info"; then
    die "Architecture mismatch: Python ($sys_arch) vs lib info ($info)."
  fi
}

smoke_test() {
  # Return 0 on success, non-zero on failure. Writes detail into $LOG.
  "$PY" - <<'PY' >>"$LOG" 2>&1
import os, sys
from PySide6 import QtWidgets, QtCore
# Ensure Qt reads the env paths we set
QtCore.QCoreApplication.setLibraryPaths([os.environ.get("QT_PLUGIN_PATH","")])
app = QtWidgets.QApplication([])
# Create & immediately destroy a trivial widget to force plugin & backing surfaces
w = QtWidgets.QWidget(); w.setWindowTitle("smoke"); w.show(); w.hide(); w.deleteLater()
print("[smoke] QApplication constructed OK")
sys.exit(0)
PY
}

repair_pyside() {
  say "[repair] Reinstalling PySide6/shiboken6 ..."
  "$PIP" uninstall -y PySide6 PySide6-Addons PySide6-Essentials shiboken6 >>"$LOG" 2>&1 || true
  "$PIP" cache purge >>"$LOG" 2>&1 || true
  rm -rf "$VENV"/lib/python*/site-packages/PySide6* \
         "$VENV"/lib/python*/site-packages/shiboken6* 2>/dev/null || true
  "$PIP" install --upgrade --force-reinstall "$PYSIDE_SPEC" "$SHIBOKEN_SPEC" >>"$LOG" 2>&1

  # Clear macOS quarantine (Gatekeeper can silently block .dylibs)
  local qtroot
  qtroot="$("$PY" - <<'PY'
import PySide6, pathlib
print((pathlib.Path(PySide6.__file__).resolve().parent/"Qt").as_posix())
PY
)"
  xattr -r -d com.apple.quarantine "$qtroot" 2>>"$LOG" || true

  # Deep ad-hoc sign the Qt frameworks/plugins tree; prevents sporadic loader rejections
  codesign --force --deep --sign - "$qtroot" >>"$LOG" 2>&1 || true
}

rebuild_venv_if_enabled() {
  [[ "$REBUILD_VENV" == "1" ]] || return 0
  say "[rebuild] Nuking and recreating venv (REBUILD_VENV=1) ..."
  rm -rf "$VENV"
  /usr/bin/env python3.11 -m venv "$VENV"
  "$PIP" install -U pip wheel >>"$LOG" 2>&1
  # Minimal deps to launch GUI; add your own requirements as needed
  "$PIP" install -r requirements/base.txt requirements/torch/macos.txt>>"$LOG" 2>&1
}

launch_gui() {
  say "[run] Launching GUI ..."
  ( cd "$ROOT" && PYTHONPATH=src "$PY" -m system.View.gui ) >>"$LOG" 2>&1
}

# ---------- MAIN ----------
ensure_venv
clear_conflicting_env
qt_paths_env
arch_check

say "[info] Log -> $LOG"
say "[step] Smoke test 1"
if ! smoke_test; then
  say "[warn] Smoke test failed. Attempting repair."
  repair_pyside
  clear_conflicting_env; qt_paths_env
  say "[step] Smoke test 2 (after repair)"
  smoke_test || {
    say "[warn] Still failing after repair."
    rebuild_venv_if_enabled
    clear_conflicting_env; qt_paths_env
    say "[step] Smoke test 3 (after venv rebuild)"
    smoke_test || die "Cocoa still fails after repair/rebuild. See $LOG for details."
  }
fi

# If the smoke test passed, the same env launches the GUI.
# Run with QT_DEBUG_PLUGINS=1 for extra detail on failures.
export QT_DEBUG_PLUGINS="${QT_DEBUG_PLUGINS:-0}"
launch_gui || {
  say "[warn] GUI exited abnormally; retrying once after repair."
  repair_pyside
  clear_conflicting_env; qt_paths_env
  launch_gui || die "GUI launch failed again. See $LOG."
}

say "[ok] GUI exited normally."