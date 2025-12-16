# Machine Learning Trading Strategy Project

This project explores the application of machine learning algorithms in quantitative trading, specifically focusing on **Tencent Holdings (00700.HK)** in the Hong Kong stock market. It implements a timing strategy based on **XGBoost** and validates it using the **Backtrader** framework with a highly realistic transaction cost model.

## 📂 Project Structure

```text
Machine_Learning_Project/
├── backtrader/                 # Core backtesting code
│   ├── backtest_with_backtrader.py  # Main entry point for backtesting
│   ├── basic_strategies.py          # Traditional technical strategies (MA, RSI, MACD, etc.)
│   └── xgboost_strategy.py          # XGBoost machine learning strategy implementation
├── data/                       # Data files and processing scripts
│   ├── add_indicators.py            # Script to add indicators to data
│   ├── data.txt                     # Raw data file
│   └── data_with_indicators.txt     # Preprocessed data with technical indicators
├── docs/                       # Documentation
│   ├── Report_EN.pdf                # Full Project Report (English PDF)
│   └── Report_EN.tex                # Full Project Report (English TeX Source)
├── requirements.txt            # Python dependencies
├── strategy_comparison.png     # Strategy comparison chart
└── README.md                   # Project documentation
```

## ✨ Key Features

*   **Professional Backtesting Framework**: Built on `Backtrader`, supporting event-driven backtesting with realistic order execution (Next-Day Open).
*   **Advanced Machine Learning Strategy (XGBoost)**:
    *   **Ensemble Learning**: Aggregates predictions from 50 independent XGBoost models to improve robustness.
    *   **Feature Engineering**: Includes Trend (MA, MACD), Momentum (RSI), Volatility (ATR, Bollinger Bands), and Lag Features.
    *   **Rolling Training**: Supports training models on historical data before backtesting.
*   **Realistic Hong Kong Market Simulation**:
    *   **Custom Commission Model**: Accurately simulates HK Stamp Duty (0.1%), Trading Fees, SFC Levies, and Platform Fees.
    *   **Lot Size Constraints**: Enforces the "100 shares per lot" rule, accounting for cash drag.
*   **Benchmark Comparison**: Includes classic strategies like Moving Average Crossover, RSI Mean Reversion, and MACD for performance comparison.
*   **Experiment Tracking**: Automatically logs hyperparameters and performance metrics (Sharpe, CAGR, Drawdown) to `xgboost_experiments.json`.

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python installed (recommended version 3.10+).

```bash
# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
*   `backtrader`: Backtesting engine
*   `xgboost`: Machine learning library
*   `pandas`, `numpy`: Data manipulation
*   `pandas_ta`: Technical analysis indicators

### 2. Running the Backtest

Use the main script `backtrader/backtest_with_backtrader.py` to run simulations.

**Basic Usage:**

```bash
python backtrader/backtest_with_backtrader.py
```

**Enable Realistic Fees:**

To apply the Hong Kong stock fee model (Stamp Duty, Commission, etc.):

```bash
python backtrader/backtest_with_backtrader.py --apply-fees
```

**Specify Date Range:**

```bash
python backtrader/backtest_with_backtrader.py --start 2024-01-01 --end 2025-04-24
```

**Run Specific Strategies:**

Available strategies: `xgboost`, `buy_hold`, `ma_cross`, `macd`, `rsi_reversion`, `boll_breakout`.

```bash
python backtrader/backtest_with_backtrader.py --strategies xgboost buy_hold --apply-fees
```

### 3. Viewing Results

The script prints a summary table of performance metrics to the console, including:
*   **Total Return**
*   **CAGR** (Compound Annual Growth Rate)
*   **Max Drawdown**
*   **Sharpe Ratio**
*   **Win Rate**

Detailed reports and analysis can be found in the `docs/` folder.

## 📊 Performance Highlight

*Based on backtest data from 2024-01-01 to 2025-04-24 (with fees applied):*

| Strategy | Total Return | Max Drawdown | Sharpe Ratio | Win Rate |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Enhanced** | **73.93%** | **7.99%** | **2.55** | **77.78%** |
| Buy & Hold | 60.64% | 23.22% | 1.59 | - |

*The XGBoost strategy significantly reduces drawdown while outperforming the benchmark.*

## 📝 Documentation

*   [**Full Report (English)**](docs/Report_EN.md): Detailed analysis of the strategy, methodology, and results.
*   [**Full Report (Chinese)**](docs/Report_CN.md): 中文版完整实训报告.
