from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
import yaml
import datetime as dt


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


# setting up data for model training
@dataclass(frozen=True)
class ModelConfig:
    type: Literal["lstm"]  # baseline only
    window: int
    hidden: int
    epochs: int
    lr: float


# data for allocation strategy
@dataclass(frozen=True)
class AllocatorConfig:
    type: Literal["softmax"]
    temperature: float
    weight_cap: float


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
    @staticmethod
    def from_yaml(path: Path):
        d = yaml.safe_load(path.read_text())

        def dte(x):
            return dt.date.fromisoformat(x)

        return ExperimentConfig(
            experiment_name=d["experiment"]["name"],
            artifacts_root=Path(d["experiment"]["artifacts_root"]),
            tickers=[t.upper() for t in d["universe"]["tickers"]],
            start=dte(d["dates"]["start"]),
            end=dte(d["dates"]["end"]),
            pipelines=PipelineConfig(**d["pipelines"]),
            model=ModelConfig(**d["model"]),
            allocator=AllocatorConfig(**d["allocator"]),
            backtest=BacktestConfig(**d["backtest"]),
            seeds=Seeds(**d["seeds"]),
        )
