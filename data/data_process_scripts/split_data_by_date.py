"""按时间分割带指标的数据文件（位于 data 目录）。

此脚本默认在与脚本同一目录下读取 `data_with_indicators.txt`（由 `add_indicators.py` 生成），
并输出两个文件：
- 训练集：`training_2018_2023.txt`（2018-01-01 至 2023-12-31）
- 投资期：`investment_2024_20250424.txt`（2024-01-01 至 2025-04-24）

使用：在项目根或任意位置运行此脚本，例如：
    python data\\split_data_by_date.py

实现要点：使用 `Path(__file__).parent` 确保文件路径相对于 `data/` 目录解析，
并使用 `index_col=0, parse_dates=True` 读取带索引的时间序列文件。
"""
from pathlib import Path
from typing import Tuple

import pandas as pd


def load_indicators_file(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    # 假设文件是由 add_indicators.py 使用 to_csv(..., sep='\t', index=True) 写出的
    df = pd.read_csv(p, sep='\t', index_col=0, parse_dates=True)
    # 确保索引为 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def split_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # pandas 支持基于索引的切片（闭区间），例如 df.loc['2018-01-01':'2023-12-31']
    return df.loc[start:end].copy()


def main(
    input_filename: str = "../data_with_indicators.txt",
    train_out: str = "../training_2018_2023.txt",
    invest_out: str = "../investment_2024_20250424.txt",
    train_start: str = "2018-01-01",
    train_end: str = "2023-12-31",
    invest_start: str = "2024-01-01",
    invest_end: str = "2025-04-24",
) -> Tuple[Path, Path]:
    base = Path(__file__).parent
    p_in = base / input_filename

    print(f"Loading indicators file: {p_in}")
    df = load_indicators_file(p_in)

    print(f"Splitting training period: {train_start} -> {train_end}")
    train_df = split_by_date(df, train_start, train_end)
    p_train = base / train_out
    train_df.to_csv(p_train, sep='\t', index=True)
    print(f"Saved training data to {p_train}  (rows: {len(train_df)})")

    print(f"Splitting investment period: {invest_start} -> {invest_end}")
    invest_df = split_by_date(df, invest_start, invest_end)
    p_invest = base / invest_out
    invest_df.to_csv(p_invest, sep='\t', index=True)
    print(f"Saved investment data to {p_invest}  (rows: {len(invest_df)})")

    return p_train, p_invest


if __name__ == "__main__":
    main()
