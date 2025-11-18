from pathlib import Path

import pandas as pd

from indicators import add_common_indicators


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
