# src/system/Model/experiment_config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Union
import yaml
import datetime as dt


@dataclass
class BaselineModelConfig:
    type: str = "baseline"
    baseline_type: str = "past5d_mom"
    horizon: int = 5  # keeps it consistent with target_5d
    window: int = 60  # not strictly used, but keeps config shape similar


# setting up data for feature pipeline
@dataclass(frozen=True)
class PipelineConfig:
    ingest_prices: bool
    ingest_sentiment: bool
    ingest_events: bool
    features_technical: bool
    features_sentiment: bool
    features_events: bool
    features_correlation: bool
    features_graph: bool
    train_model: bool
    allocate: bool
    backtest: bool
    report: bool
    compare: bool
    features_stat_vol: bool = True  # stat vol features are default, not toggled by user


# ---------- Model configs (type-specific) ----------


@dataclass(frozen=True)
class LstmModelConfig:
    type: Literal["lstm"]
    window: int
    hidden: int
    epochs: int
    lr: float
    lambda_vol: float = 0.1  # weight between data and GARCH prior
    w_vol: float = 1.0  # overall vol-loss weight


@dataclass(frozen=True)
class TransformerModelConfig:
    type: Literal["transformer"]
    window: int
    d_model: int
    nhead: int
    num_layers: int
    epochs: int
    lr: float
    dim_feedforward: int
    dropout: float
    weight_decay: float
    patience: int
    pooling: str
    max_grad_norm: float


@dataclass(frozen=True)
class GnnModelConfig:
    type: Literal["gnn"]
    window: int  # lookback for GraphSequenceDataset
    hidden: int  # base width / temp+gnn hidden
    layers: int  # GNN layers
    epochs: int
    lr: float


ModelConfig = Union[LstmModelConfig, TransformerModelConfig, GnnModelConfig]


@dataclass(frozen=True)
class AllocatorConfig:
    type: Literal["softmax", "meanvar", "rl"]
    temperature: float | None = None
    weight_cap: float | None = None

    risk_aversion: float | None = None
    cov_lookback: int | None = None

    rl_lr: float | None = None
    rl_epochs: int | None = None
    rl_trade_cost_bps: float | None = None


# data for backtesting
@dataclass(frozen=True)
class BacktestConfig:
    rebalance: Literal["weekly"]
    transaction_cost_bps: int
    long_only: bool


# data for experiment seeds and reproducibility
@dataclass(frozen=True)
class Seeds:
    python: int
    numpy: int
    torch: int
    deterministic_torch: bool


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    artifacts_root: Path
    tickers: List[str]
    start: dt.date
    end: dt.date
    pipelines: PipelineConfig
    model: ModelConfig
    allocator: AllocatorConfig
    backtest: BacktestConfig
    seeds: Seeds

    # from_yaml static method to load config from a YAML file
    # ---- YAML loader ----
    @staticmethod
    def from_yaml(path: Path) -> "ExperimentConfig":
        data = yaml.safe_load(path.read_text())
        return ExperimentConfig.from_dict(data)

    # ---- dict loader (used by recipe) ----
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        def dte(x: str) -> dt.date:
            return dt.date.fromisoformat(x)

        # experiment
        exp = d.get("experiment", {})
        name = exp.get("name")
        if not name:
            raise ValueError("experiment.name is required")
        artifacts_root = Path(exp.get("artifacts_root", "artifacts"))

        # universe
        uni = d.get("universe", {})
        tickers_raw = uni.get("tickers") or []
        if not tickers_raw:
            raise ValueError("universe.tickers is required and cannot be empty")
        tickers = [str(t).upper() for t in tickers_raw]

        # dates
        dates = d.get("dates", {})
        start = dte(dates["start"])
        end = dte(dates["end"])

        # pipelines
        p_raw = d.get("pipelines", {})
        pipelines = PipelineConfig(**p_raw)

        # model (type-dispatched)
        m_raw = d.get("model", {})
        m_type = str(m_raw.get("type", "")).lower()
        if not m_type:
            raise ValueError("model.type is required")

        if m_type == "lstm":
            model = LstmModelConfig(
                type="lstm",
                window=int(m_raw["window"]),
                hidden=int(m_raw["hidden"]),
                epochs=int(m_raw["epochs"]),
                lr=float(m_raw["lr"]),
                lambda_vol=float(m_raw["lambda_vol"]),
                w_vol=float(m_raw["w_vol"]),
            )
        elif m_type == "transformer":
            model = TransformerModelConfig(
                type="transformer",
                window=int(m_raw["window"]),
                d_model=int(m_raw["d_model"]),
                nhead=int(m_raw["nhead"]),
                num_layers=int(m_raw["num_layers"]),
                epochs=int(m_raw["epochs"]),
                lr=float(m_raw["lr"]),
                dim_feedforward=int(m_raw["dim_feedforward"]),
                dropout=float(m_raw["dropout"]),
                weight_decay=float(m_raw["weight_decay"]),
                patience=int(m_raw["patience"]),
                pooling=str(m_raw["pooling"]),
                max_grad_norm=float(m_raw["max_grad_norm"]),
            )
        elif m_type == "gnn":
            model = GnnModelConfig(
                type="gnn",
                window=int(m_raw["window"]),
                hidden=int(m_raw["hidden"]),
                layers=int(m_raw["layers"]),
                epochs=int(m_raw["epochs"]),
                lr=float(m_raw["lr"]),
            )
        elif m_type == "baseline":
            model = BaselineModelConfig(
                type="baseline",
                baseline_type=str(m_raw["baseline_type"]),
                horizon=int(m_raw["horizon"]),
                window=int(m_raw["window"]),
            )
        else:
            raise ValueError(f"Unknown model.type: {m_type}")

        # ----- allocator -----
        a_raw = d.get("allocator", {})
        a_type = str(a_raw.get("type", "")).lower()
        if not a_type:
            raise ValueError("allocator.type is required")

        if a_type == "softmax":
            temperature = float(a_raw["temperature"])
            weight_cap = float(a_raw.get("weight_cap", 1.0))
            alloc = AllocatorConfig(
                type="softmax",
                temperature=temperature,
                weight_cap=weight_cap,
            )

        elif a_type in ("meanvar", "mean-variance"):
            risk_aversion = float(a_raw.get("risk_aversion", 5.0))
            cov_lookback = int(a_raw.get("cov_lookback", 60))
            weight_cap = a_raw.get("weight_cap")
            alloc = AllocatorConfig(
                type="meanvar",
                risk_aversion=risk_aversion,
                cov_lookback=cov_lookback,
                weight_cap=float(weight_cap) if weight_cap is not None else None,
            )

        elif a_type == "rl":
            rl_lr = float(a_raw.get("lr", 1e-2))
            rl_epochs = int(a_raw.get("epochs", 50))
            weight_cap = a_raw.get("weight_cap")
            trade_cost_bps = float(a_raw.get("trade_cost_bps", 0.0))
            alloc = AllocatorConfig(
                type="rl",
                rl_lr=rl_lr,
                rl_epochs=rl_epochs,
                weight_cap=float(weight_cap) if weight_cap is not None else None,
                rl_trade_cost_bps=trade_cost_bps,
            )

        else:
            raise ValueError(f"Unknown allocator.type: {a_type}")

        # backtest
        b_raw = d.get("backtest", {})
        backtest = BacktestConfig(
            rebalance=str(b_raw["rebalance"]),
            transaction_cost_bps=int(b_raw["transaction_cost_bps"]),
            long_only=bool(b_raw["long_only"]),
        )

        # seeds
        s_raw = d.get("seeds", {})
        seeds = Seeds(
            python=int(s_raw["python"]),
            numpy=int(s_raw["numpy"]),
            torch=int(s_raw["torch"]),
            deterministic_torch=bool(s_raw["deterministic_torch"]),
        )

        return ExperimentConfig(
            experiment_name=name,
            artifacts_root=artifacts_root,
            tickers=tickers,
            start=start,
            end=end,
            pipelines=pipelines,
            model=model,
            allocator=alloc,
            backtest=backtest,
            seeds=seeds,
        )
