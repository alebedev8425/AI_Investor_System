# src/system/Model/reporting/report_builder.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#111827",
        "axes.labelcolor": "#111827",
        "text.color": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "grid.color": "#e5e7eb",
        "axes.grid": True,
        "grid.alpha": 0.6,
    }
)


@dataclass
class RunInputs:
    daily_path: Path  # backtests/daily_returns.csv
    metrics_path: Path  # backtests/metrics.json
    manifest_path: Path  # manifest.json
    assets_dir: Path  # reports/assets


# ---------- IO helpers ----------
def _safe_read_metrics(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _safe_read_daily(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=["date", "ret"])
    df = pd.read_csv(p)
    # expected columns ['date','ticker','ret'] → aggregate to portfolio if needed
    if set(["date", "ticker", "ret"]).issubset(df.columns):
        df = df.groupby("date", as_index=False)["ret"].sum()
    return df


def _safe_read_manifest(p: Path) -> dict:
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


# ---------- small utils ----------
def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "—"


def _fmt(x: float | int | None, nd=3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _pretty_combo(manifest: dict) -> tuple[str, str, str]:
    """Return (features_str, model_str, alloc_str) in a minimal, human form."""
    cfg = (manifest.get("config") or {}) if isinstance(manifest.get("config"), dict) else {}

    # model: accept string or dict with 'type'/'key'
    def _pick_key(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("type") or x.get("key")
        return None

    model = _pick_key(manifest.get("model")) or _pick_key(cfg.get("model")) or "—"
    alloc = _pick_key(manifest.get("allocator")) or _pick_key(cfg.get("allocator")) or "—"

    # features: accept list like ["tech","sent",...] or map from pipeline booleans
    features_any = manifest.get("features") or cfg.get("features")
    pretty = []
    if isinstance(features_any, (list, tuple)):
        m = {
            "tech": "technical",
            "sent": "sentiment",
            "events": "events",
            "corr": "correlations",
            "graph": "graph",
        }
        pretty = [m.get(str(f).lower(), str(f)) for f in features_any]

    if not pretty and isinstance(cfg.get("pipelines"), dict):
        p = cfg["pipelines"]
        pairs = [
            ("features_technical", "technical"),
            ("features_sentiment", "sentiment"),
            ("features_events", "events"),
            ("features_correlation", "correlations"),
            ("features_graph", "graph"),
        ]
        for k, label in pairs:
            if bool(p.get(k)):
                pretty.append(label)

    if not pretty:
        # GUI keeps technical always-on; default to it if we can't infer.
        pretty = ["technical"]

    return (", ".join(pretty), str(model), str(alloc))


# ---------- plots ----------
def _plot_equity(df: pd.DataFrame, out_png: Path, label: str | None = None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 3))
    if df.empty:
        plt.title("No backtest data")
    else:
        ser = (1.0 + df["ret"].astype(float)).cumprod()
        plt.plot(pd.to_datetime(df["date"]), ser, label=label or "equity")
        if label:
            plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Equity (1 = start)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_drawdown(df: pd.DataFrame, out_png: Path, label: str | None = None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 3))
    if df.empty:
        plt.title("No drawdown data")
    else:
        equity = (1.0 + df["ret"].astype(float)).cumprod()
        dd = equity / equity.cummax() - 1.0
        plt.plot(pd.to_datetime(df["date"]), dd, label=label or "drawdown")
        if label:
            plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_hist(df: pd.DataFrame, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 3))
    if df.empty:
        plt.title("No return data")
    else:
        plt.hist(df["ret"].astype(float), bins=50)
    plt.xlabel("Daily return")
    plt.ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_monthly_heatmap(df: pd.DataFrame, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.7, 3.6))
    if df.empty:
        plt.title("No monthly data")
    else:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d["y"] = d["date"].dt.year
        d["m"] = d["date"].dt.month
        # monthly return = product(1+ret)-1 per (y,m)
        mon = (
            d.groupby(["y", "m"])["ret"]
            .apply(lambda s: (1 + s).prod() - 1)
            .unstack("m")
            .fillna(0.0)
        )
        # imshow basic heatmap
        im = plt.imshow(mon.values, aspect="auto", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.yticks(range(len(mon.index)), [str(int(y)) for y in mon.index])
        plt.xticks(
            range(12),
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            rotation=0,
        )
        plt.title("Monthly returns (heatmap)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------- single run ----------
def build_run_report(run: RunInputs, out_html: Path) -> None:
    df = _safe_read_daily(run.daily_path)
    metrics = _safe_read_metrics(run.metrics_path)
    manifest = _safe_read_manifest(run.manifest_path)
    # prediction-quality metrics (test set)
    pred_metrics_path = run.metrics_path.parent / "prediction_metrics.json"
    pred_metrics = _safe_read_metrics(pred_metrics_path)

    features_str, model_str, alloc_str = _pretty_combo(manifest)

    assets = run.assets_dir
    eq_png = assets / "equity.png"
    dd_png = assets / "drawdown.png"
    hist_png = assets / "hist.png"
    mon_png = assets / "monthly.png"

    _plot_equity(df, eq_png)
    _plot_drawdown(df, dd_png)
    _plot_hist(df, hist_png)
    _plot_monthly_heatmap(df, mon_png)

    # derive a couple of extras if not present
    if not df.empty and "n_days" not in metrics:
        metrics["n_days"] = int(len(df))
    if not df.empty and "cumulative_return" not in metrics:
        metrics["cumulative_return"] = float((1 + df["ret"].astype(float)).prod() - 1.0)

    css = """
    :root{ --paper:#ffffff; --ink:#111827; --muted:#374151; --line:#e5e7eb; }
    html,body{background:var(--paper)}
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;margin:24px;color:var(--ink)}
    /* clamp page so QTextBrowser never asks for huge width */
    .container{max-width:1060px;margin:0 auto}

    .grid{display:grid;gap:20px}
    /* critical: let tracks shrink below min-content of images/tables */
    .g2{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
    .g3{grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr)}
    .grid > *{min-width:0} /* allow children to shrink */

    .card{border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.04);background:#fff}
    table{border-collapse:collapse;width:100%;table-layout:fixed}
    th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;color:var(--ink);word-break:break-word;overflow-wrap:anywhere}
    .k{color:var(--ink);width:60%}
    img{display:block;width:100%;max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line)}
    code,pre{color:var(--ink);background:#f3f4f6;border:1px solid var(--line);border-radius:6px;padding:2px 6px;word-break:break-word;overflow-wrap:anywhere}
    a{color:#1f2937;text-decoration:underline}
    .card h3{display:block;margin:0 0 8px}
    figure.plot{margin:0}
    figure.plot figcaption{display:block;margin:0 0 8px;font-weight:600}
    """

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Run Report</title>
<style>{css}
.plot-table{{width:100%;border-collapse:collapse}}
.plot-table td{{padding:0;border:none}}
.card-title{{margin:0 0 8px;font-weight:600;font-size:16px}}
</style></head>
<body><div class="container">
  <h1>Run Report</h1>
  <div class="muted">run_id: {manifest.get("run_id", "?")} • created: {manifest.get("created_utc", "?")}</div>
  <div class="muted">model: <code>{model_str}</code> | alloc: <code>{alloc_str}</code> | features: <code>{features_str}</code></div>  <div class="grid g3" style="margin-top:16px">
    <div class="card">
      <h3 class="card-title">Headline</h3>
      <table>
        <tr><td class="k">Sharpe (like)</td><td>{_fmt(metrics.get("sharpe_like"), nd=3)}</td></tr>
        <tr><td class="k">Cumulative return</td><td>{_fmt_pct(metrics.get("cumulative_return"))}</td></tr>
        <tr><td class="k">Max drawdown</td><td>{_fmt_pct(metrics.get("max_drawdown"))}</td></tr>
        <tr><td class="k">Avg daily turnover</td><td>{_fmt_pct(metrics.get("avg_daily_turnover"))}</td></tr>
        <tr><td class="k">Days</td><td>{metrics.get("n_days", "—")}</td></tr>
      </table>
    </div>
    <div class="card">
      <h3 class="card-title">Prediction Metrics (test set)</h3>
      <table>
        <tr><td class="k">MAE</td><td>{_fmt(pred_metrics.get("mae"), nd=6)}</td></tr>
        <tr><td class="k">RMSE</td><td>{_fmt(pred_metrics.get("rmse"), nd=6)}</td></tr>
        <tr><td class="k">Hit rate (sign)</td><td>{_fmt_pct(pred_metrics.get("hit_rate"))}</td></tr>
        <tr><td class="k">IC (Spearman, mean)</td><td>{_fmt(pred_metrics.get("ic_mean"), nd=3)}</td></tr>
        <tr><td class="k">IC t-stat</td><td>{_fmt(pred_metrics.get("ic_tstat"), nd=3)}</td></tr>
        <tr><td class="k">Top–bottom decile spread</td><td>{_fmt_pct(pred_metrics.get("decile_long_short_spread"))}</td></tr>
      </table>
    </div>
    <div class="card" style="margin-top:16px">
      <h3 class="card-title">Artifacts</h3>
      <div class="muted">Daily portfolio CSV: <code>backtests/daily_returns.csv</code></div>
    </div>
  </div>

  <div class="grid g2">
    <div class="card" style="margin-top:20px">
      <table class="plot-table">
        <tr><td><div class="card-title">Equity Curve</div></td></tr>
        <tr><td><img src="assets/{eq_png.name}" alt="Equity curve"></td></tr>
      </table>
    </div>
    <div class="card" style="margin-top:20px">
      <table class="plot-table">
        <tr><td><div class="card-title">Drawdown</div></td></tr>
        <tr><td><img src="assets/{dd_png.name}" alt="Drawdown"></td></tr>
      </table>
    </div>
  </div>

  <div class="grid g2">
    <div class="card" style="margin-top:20px">
      <table class="plot-table">
        <tr><td><div class="card-title">Return Distribution</div></td></tr>
        <tr><td><img src="assets/{hist_png.name}" alt="Return distribution histogram"></td></tr>
      </table>
    </div>
    <div class="card" style="margin-top:20px">
      <table class="plot-table">
        <tr><td><div class="card-title">Monthly Returns</div></td></tr>
        <tr><td><img src="assets/{mon_png.name}" alt="Monthly returns heatmap"></td></tr>
      </table>
    </div>
  </div>
</div>
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


# ---------- compare ----------
def build_compare_report(
    a: RunInputs, b: RunInputs, out_html: Path, label_a: str, label_b: str
) -> None:
    dfa = _safe_read_daily(a.daily_path)
    dfb = _safe_read_daily(b.daily_path)
    ma = _safe_read_metrics(a.metrics_path)
    mb = _safe_read_metrics(b.metrics_path)

    assets = a.assets_dir  # shared compare assets dir is passed in
    eq_png = assets / "equity_compare.png"
    dd_png = assets / "drawdown_compare.png"

    # equity compare
    fig = plt.figure(figsize=(7, 3.2))
    if not dfa.empty:
        plt.plot(
            pd.to_datetime(dfa["date"]), (1 + dfa["ret"].astype(float)).cumprod(), label=label_a
        )
    if not dfb.empty:
        plt.plot(
            pd.to_datetime(dfb["date"]), (1 + dfb["ret"].astype(float)).cumprod(), label=label_b
        )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Equity")
    fig.tight_layout()
    fig.savefig(eq_png, bbox_inches="tight")
    plt.close(fig)

    # drawdown compare
    def _dd(df):
        if df.empty:
            return None
        eq = (1.0 + df["ret"].astype(float)).cumprod()
        return eq / eq.cummax() - 1.0

    dda = _dd(dfa)
    ddb = _dd(dfb)
    fig = plt.figure(figsize=(7, 3.2))
    if dda is not None:
        plt.plot(pd.to_datetime(dfa["date"]), dda, label=f"{label_a} DD")
    if ddb is not None:
        plt.plot(pd.to_datetime(dfb["date"]), ddb, label=f"{label_b} DD")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(dd_png, bbox_inches="tight")
    plt.close(fig)

    # deltas
    rows = [
        ("Sharpe (like)", _fmt(ma.get("sharpe_like")), _fmt(mb.get("sharpe_like"))),
        (
            "Cumulative return",
            _fmt_pct(ma.get("cumulative_return")),
            _fmt_pct(mb.get("cumulative_return")),
        ),
        ("Max drawdown", _fmt_pct(ma.get("max_drawdown")), _fmt_pct(mb.get("max_drawdown"))),
        (
            "Avg daily turnover",
            _fmt_pct(ma.get("avg_daily_turnover")),
            _fmt_pct(mb.get("avg_daily_turnover")),
        ),
        ("Days", str(ma.get("n_days", "—")), str(mb.get("n_days", "—"))),
    ]

    css = """
    :root{ --paper:#ffffff; --ink:#111827; --muted:#374151; --line:#e5e7eb; }
    html,body{background:var(--paper)}
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;margin:24px;color:var(--ink)}
    /* clamp page so QTextBrowser never asks for huge width */
    .container{max-width:1060px;margin:0 auto}

    .grid{display:grid;gap:20px}
    /* critical: let tracks shrink below min-content of images/tables */
    .g2{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
    .g3{grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr)}
    .grid > *{min-width:0} /* allow children to shrink */

    .card{border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.04);background:#fff}
    table{border-collapse:collapse;width:100%;table-layout:fixed}
    th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;color:var(--ink);word-break:break-word;overflow-wrap:anywhere}
    .k{color:var(--ink);width:60%}
    img{display:block;width:100%;max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line)}
    code,pre{color:var(--ink);background:#f3f4f6;border:1px solid var(--line);border-radius:6px;padding:2px 6px;word-break:break-word;overflow-wrap:anywhere}
    a{color:#1f2937;text-decoration:underline}
    .card h3{display:block;margin:0 0 8px}
    figure.plot{margin:0}
    figure.plot figcaption{display:block;margin:0 0 8px;font-weight:600}
    """

    trs = "\n".join(f"<tr><td>{k}</td><td>{va}</td><td>{vb}</td></tr>" for k, va, vb in rows)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Compare Runs</title>
<style>{css}
.plot-table{{width:100%;border-collapse:collapse}}
.plot-table td{{padding:0;border:none}}
.card-title{{margin:0 0 8px;font-weight:600;font-size:16px}}
</style></head><body><div class="container">
<h1>Comparison</h1>

<div class="grid g2">
  <div class="card" style="margin-top:20px">
    <table class="plot-table">
      <tr><td><div class="card-title">Equity</div></td></tr>
      <tr><td><img src="assets/{eq_png.name}" alt="Equity compare"></td></tr>
    </table>
  </div>
  <div class="card" style="margin-top:20px">
    <table class="plot-table">
      <tr><td><div class="card-title">Drawdown</div></td></tr>
      <tr><td><img src="assets/{dd_png.name}" alt="Drawdown compare"></td></tr>
    </table>
  </div>
</div>

<div class="card" style="margin-top:20px">
  <h3 class="card-title">Metrics</h3>
  <table>
    <tr><th>Metric</th><th>{label_a}</th><th>{label_b}</th></tr>
    {trs}
  </table>
</div>
</div></body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
