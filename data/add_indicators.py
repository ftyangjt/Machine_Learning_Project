from pathlib import Path

import pandas as pd
import pandas_ta as pta


def moving_average(series: pd.Series, window: int = 20) -> pd.Series:
    """简单移动平均线（SMA）—使用 pandas_ta。"""
    return pta.sma(series, length=window)


def exponential_moving_average(series: pd.Series, span: int = 20) -> pd.Series:
    """指数移动平均线（EMA）—使用 pandas_ta。"""
    return pta.ema(series, length=span)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指数（RSI）—使用 pandas_ta。"""
    return pta.rsi(series, length=period)


def macd(series: pd.Series,
         short_span: int = 12,
         long_span: int = 26,
         signal_span: int = 9) -> pd.DataFrame:
    """MACD 指标，返回包含 DIF, DEA, BAR 的 DataFrame（使用 pandas_ta）。"""
    macd_df = pta.macd(series, fast=short_span, slow=long_span, signal=signal_span)
    dif = macd_df.iloc[:, 0]
    dea = macd_df.iloc[:, -1]
    bar = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "bar": bar})


def bollinger_bands(series: pd.Series,
                     window: int = 20,
                     num_std: float = 2.0) -> pd.DataFrame:
    """布林带，返回中轨/上轨/下轨（使用 pandas_ta）。"""
    bb = pta.bbands(series, length=window, std={"mult": num_std})
    cols = list(bb.columns)
    mid_col = next((c for c in cols if "BBM" in c or "mb" in c.lower()), cols[0])
    upper_col = next((c for c in cols if "BBU" in c or "upper" in c.lower()), cols[1] if len(cols) > 1 else cols[0])
    lower_col = next((c for c in cols if "BBL" in c or "lower" in c.lower()), cols[2] if len(cols) > 2 else cols[0])
    return pd.DataFrame({"mid": bb[mid_col], "upper": bb[upper_col], "lower": bb[lower_col]})


def atr(high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14) -> pd.Series:
    """平均真实波幅（ATR）—使用 pandas_ta。"""
    return pta.atr(high, low, close, length=period)


def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在原始 OHLCV DataFrame 上添加常用技术指标列（使用 pandas_ta）。

    约定 df 至少包含列: "open", "high", "low", "close", "volume"。
    返回一个新的 DataFrame（不修改原 df）。
    """
    d = df.copy()
    close = d["close"]

    # 简单均线
    d["ma5"] = moving_average(close, 5)
    d["ma10"] = moving_average(close, 10)
    d["ma20"] = moving_average(close, 20)

    # 指数均线
    d["ema12"] = exponential_moving_average(close, 12)
    d["ema26"] = exponential_moving_average(close, 26)

    # RSI
    d["rsi14"] = rsi(close, 14)

    # MACD
    macd_df = macd(close)
    d["macd_dif"] = macd_df["dif"]
    d["macd_dea"] = macd_df["dea"]
    d["macd_bar"] = macd_df["bar"]

    # 布林带
    bb = bollinger_bands(close)
    d["boll_mid"] = bb["mid"]
    d["boll_upper"] = bb["upper"]
    d["boll_lower"] = bb["lower"]

    # ATR
    d["atr14"] = atr(d["high"], d["low"], close, 14)

    return d


def load_price_data(file_path: str) -> pd.DataFrame:
	df = pd.read_csv(
		file_path,
		sep=r"\s+",
		header=1,
		engine="python",
	)
	df.columns = ["date", "open", "high", "low", "close", "volume", "amount"]
	df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
	df = df.sort_values("date")
	df.set_index("date", inplace=True)
	return df


def main(input_path: str = "data/data.txt", output_path: str = "data/data_with_indicators.txt") -> None:
	p_in = Path(input_path)
	if not p_in.exists():
		raise FileNotFoundError(f"Input file not found: {p_in}")

	print(f"Reading {p_in} ...")
	df = load_price_data(str(p_in))

	print("Computing indicators (MA, EMA, RSI, MACD, Bollinger, ATR) ...")
	df_ind = add_common_indicators(df)

	p_out = Path(output_path)
	# save as tab-separated txt
	df_ind.to_csv(p_out, sep='\t', index=True)
	print(f"Saved indicators to {p_out}")


if __name__ == "__main__":
	main()
