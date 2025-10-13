# AI_Investor_System

## Overview

**AI_Investor_System** is a research-focused platform for offline experimentation in stock price prediction and portfolio optimization, based on academic research by Alexander Lebedev and Professor Ouda. It leverages advanced AI models (LSTM, Transformer, GNN) and portfolio allocators (softmax, mean–variance, RL) to evaluate strategies on the U.S. stock market (Nasdaq-100), combining historical prices, technical indicators, sentiment, and event-based features.

## Features

- **Data Intake:** Automated download and caching of OHLCV data (via yfinance), news sentiment (FinBERT), and company event flags (SEC EDGAR).
- **Feature Engineering:** Technical indicators, rolling correlations, sentiment/event integration, and graph features for GNNs.
- **Modeling:** Supports LSTM, Transformer, and GNN architectures for 5-day forward return prediction.
- **Portfolio Allocation:** Softmax, mean–variance, and RL-based allocators with realistic constraints (long-only, capped weights, transaction costs).
- **Backtesting:** Weekly rebalancing, performance metrics (Sharpe Ratio, Max Drawdown, Cumulative Returns), and reproducible results.
- **Experiment Management:** YAML-based configuration for flexible experiments and comparisons.
- **Reporting:** Automated generation of tables, charts, and PDF summaries.
- **CLI Interface:** Simple commands for running, managing, and comparing experiments.
- **Portability:** Docker support for easy deployment.

## Quick Start - may not be exact, depending on system setup

1. **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/your-org/AI_Investor_System.git
    cd AI_Investor_System
    pip install -r requirements.txt
    ```

2. **Run an experiment:**
    ```bash
    python train.py --config exp_transformer.yaml
    ```

3. **Generate a report:**
    ```bash
    python report.py --config exp_transformer.yaml
    ```

4. **Compare experiments:**
    ```bash
    python compare.py --configs exp_lstm.yaml exp_transformer.yaml
    ```

5. **Clean cache:**
    ```bash
    python clean.py
    ```

## Use Cases

- **Configure Experiment:** Select/edit YAML config for experiment setup.
- **Run Experiment:** Execute full pipeline (ingest, feature engineering, train, allocate, backtest).
- **Ingest Data:** Download/caches price, sentiment, and event data.
- **Feature Engineering:** Compute technical, sentiment, event, correlation, and graph features.
- **Model Training:** Train LSTM, Transformer, or GNN models.
- **Portfolio Allocation:** Assign weights using softmax, mean–variance, or RL.
- **Backtesting:** Simulate portfolio performance with constraints.
- **Reporting:** Generate PDF summaries of results.
- **Compare Experiments:** Visualize and compare multiple runs.
- **Clean/Reset Cache:** Remove cached data for fresh runs.

## System Architecture

- **Experiment Control:** `RunManager`, `ExperimentConfig`, `ArtifactStore`, `Logger`
- **Data:** `DataUniverse`, `TradingCalendar`, `DataSource` (Price/Sentiment/Event), `CacheManager`, `DataValidator`
- **Feature Engineering:** `FeaturePipeline`, `FeatureBuilder` (Technical/Correlation/Sentiment/Event/Graph)
- **Preprocessing:** `Splitter`, `Scaler`, `SequenceDataset`, `GraphDataset`
- **Modeling:** `ModelTrainer` (LSTM/Transformer/GNN), `Predictor`
- **Allocation:** `ConstraintSet`, `CovarianceEstimator`, `Allocator` (Softmax/MeanVariance/RLearn), `PortfolioEnvironment`, `RLearnPolicy`
- **Backtesting & Metrics:** `Backtester`, `PredictionMetrics`, `PortfolioMetrics`, `ReportGenerator`, `ExperimentComparator`, `CacheCleaner`

[Class Diagram (PDF)](https://drive.google.com/file/d/1sN1IIo4kAEVezkH-rMUJYDiV8g7l1q0a/view?usp=sharing)

## Performance & Constraints

- Handles up to 100 tickers over 10 years; full cycle runs in <3 hours on a laptop (Apple Silicon preferred).
- Long-only portfolios, capped allocations, 10bps transaction cost.
- Results are reproducible under fixed configurations.

## Dependencies

- Python 3.8+
- yfinance, FinBERT, PyTorch, PyTorch Geometric, Backtrader, Stable Baselines3
- Docker (for containerized deployment)

## References

- Lebedev, A., Ouda, A. "Leveraging AI for stock price prediction and Portfolio Optimization"
- Academic research on stock price prediction and portfolio optimization models

## License

Open-source, free for academic and research use.
