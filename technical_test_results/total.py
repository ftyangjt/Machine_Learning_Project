import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 将 data/data_process_scripts 加入模块搜索路径，便于导入 indicators.py
scripts_dir = Path(__file__).resolve().parent.parent / "data" / "data_process_scripts"
if str(scripts_dir) not in sys.path:
	sys.path.insert(0, str(scripts_dir))

from indicators import add_common_indicators


INITIAL_CAPITAL = 100000.0


def load_price_data(file_path: str = "data/data_with_indicators.txt") -> pd.DataFrame:
	"""读取并预处理带指标的数据（来自 data/data_with_indicators.txt），返回带日期索引的 DataFrame。"""
	p = Path(file_path)
	if not p.exists():
		# 若传入相对路径，尝试相对于仓库根目录的 data 文件夹查找
		p = Path(__file__).resolve().parent.parent / file_path

	# 文件为制表符分隔，第一行为列名
	df = pd.read_csv(p, sep='\t', engine='python')

	# 保证有 date 列并解析为 DatetimeIndex
	if 'date' not in df.columns:
		raise RuntimeError('数据文件缺少 date 列')
	df['date'] = pd.to_datetime(df['date'])
	df = df.sort_values('date')
	df.set_index('date', inplace=True)
	return df


def generate_rsi_signals(df: pd.DataFrame,
					   lower: float = 30.0,
					   upper: float = 70.0) -> pd.DataFrame:
	"""基于 RSI 生成买入/卖出信号。

	简单规则（可在报告中说明）：
	- 买入：RSI 从下穿越超卖阈值上来（前一日 rsi < lower，当日 rsi > lower），视为由超卖恢复，买入；
	- 卖出：RSI 从上穿越超买阈值下去（前一日 rsi > upper，当日 rsi < upper），视为由超买回落，卖出。
	"""
	d = df.copy()
	rsi = d["rsi14"]
	rsi_prev = rsi.shift(1)

	# 买入信号：rsi 从低于 lower 上穿
	d["buy_signal"] = (rsi_prev < lower) & (rsi > lower)
	# 卖出信号：rsi 从高于 upper 下穿
	d["sell_signal"] = (rsi_prev > upper) & (rsi < upper)

	return d


def generate_macd_signals(df: pd.DataFrame) -> pd.DataFrame:
	"""基于 MACD 生成金叉(买入)与死叉(卖出)信号。"""
	d = df.copy()
	# 假设 add_common_indicators 已经生成 macd_dif / macd_dea
	d["macd_diff"] = d["macd_dif"] - d["macd_dea"]
	d["macd_diff_prev"] = d["macd_diff"].shift(1)

	d["buy_signal_macd"] = (d["macd_diff_prev"] < 0) & (d["macd_diff"] > 0)
	d["sell_signal_macd"] = (d["macd_diff_prev"] > 0) & (d["macd_diff"] < 0)
	return d


def generate_bb_signals(df: pd.DataFrame) -> pd.DataFrame:
	"""基于布林带的简单均值回归信号：
	- 买入：当日收盘价跌破下轨
	- 卖出：当日收盘价回到中轨（均线）之上
	"""
	d = df.copy()
	# 需要 add_common_indicators 生成 boll_lower / boll_mid / boll_upper
	d["buy_signal_bb"] = d["close"] < d["boll_lower"]
	# 卖出当价格回到中轨或超过中轨
	d["sell_signal_bb"] = d["close"] > d["boll_mid"]
	return d


def backtest_rsi_strategy(df: pd.DataFrame,
						 initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
	"""使用 RSI 信号的简单全仓策略回测。

	假设：
	- 只能做多；
	- 买入信号当天收盘价买入（如果当前无持仓）；
	- 卖出信号当天收盘价卖出（如果当前有持仓）；
	- 不考虑手续费和滑点（可后续扩展）。
	"""
	d = df.copy()
	position = 0  # 持有股数
	cash = initial_capital
	portfolio_values = []

	for date, row in d.iterrows():
		price = row["close"]
		buy = bool(row.get("buy_signal", False))
		sell = bool(row.get("sell_signal", False))

		# 执行交易标志（记录是否实际成交）
		executed_buy = False
		executed_sell = False

		# 交易逻辑：优先处理卖出；如果当天既有卖出又有买入信号，
		# 为避免同日反复开平仓，采用“卖出优先且当天不再买入”的策略。
		if sell and position > 0:
			cash += position * price
			position = 0
			executed_sell = True

		# 仅当当天未发生卖出且当前无持仓时才考虑买入
		if (not executed_sell) and buy and position == 0:
			buy_shares = int(cash // price)
			if buy_shares > 0:
				position = buy_shares
				cash -= position * price
				executed_buy = True

		total_value = cash + position * price
		portfolio_values.append(
			{
				"date": date,
				"close": price,
				"position": position,
				"cash": cash,
				"total_value": total_value,
				"executed_buy": executed_buy,
				"executed_sell": executed_sell,
			}
		)

	result = pd.DataFrame(portfolio_values).set_index("date")
	return result


def backtest_macd_strategy(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
	"""使用 MACD 金叉/死叉进行全仓回测，记录是否实际成交。"""
	d = df.copy()
	position = 0
	cash = initial_capital
	portfolio_values = []

	for date, row in d.iterrows():
		price = row["close"]
		buy = bool(row.get("buy_signal_macd", False))
		sell = bool(row.get("sell_signal_macd", False))

		executed_buy = False
		executed_sell = False

		# 卖出优先，卖出后当天不再买入
		if sell and position > 0:
			cash += position * price
			position = 0
			executed_sell = True

		if (not executed_sell) and buy and position == 0:
			buy_shares = int(cash // price)
			if buy_shares > 0:
				position = buy_shares
				cash -= position * price
				executed_buy = True

		total_value = cash + position * price
		portfolio_values.append(
			{
				"date": date,
				"close": price,
				"position": position,
				"cash": cash,
				"total_value": total_value,
				"executed_buy": executed_buy,
				"executed_sell": executed_sell,
			}
		)

	result = pd.DataFrame(portfolio_values).set_index("date")
	return result


def backtest_bb_strategy(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
	"""对布林带策略进行全仓回测，规则见 generate_bb_signals。"""
	d = df.copy()
	position = 0
	cash = initial_capital
	portfolio_values = []

	for date, row in d.iterrows():
		price = row["close"]
		buy = bool(row.get("buy_signal_bb", False))
		sell = bool(row.get("sell_signal_bb", False))

		executed_buy = False
		executed_sell = False

		if sell and position > 0:
			cash += position * price
			position = 0
			executed_sell = True

		if (not executed_sell) and buy and position == 0:
			buy_shares = int(cash // price)
			if buy_shares > 0:
				position = buy_shares
				cash -= position * price
				executed_buy = True

		total_value = cash + position * price
		portfolio_values.append(
			{
				"date": date,
				"close": price,
				"position": position,
				"cash": cash,
				"total_value": total_value,
				"executed_buy": executed_buy,
				"executed_sell": executed_sell,
			}
		)

	result = pd.DataFrame(portfolio_values).set_index("date")
	return result


def buy_and_hold(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
	"""执行全仓买入并持有：在首个交易日收盘价买入尽可能多的整股，然后持有至最后一天。

	返回包含每日 total_value, position, cash, buy_price 的 DataFrame。
	"""
	d = df.copy()
	prices = d["close"]
	first_price = prices.iloc[0]
	# 可购买股数
	shares = int(initial_capital // first_price)
	cash = initial_capital - shares * first_price

	rows = []
	for date, price in prices.items():
		total = cash + shares * price
		rows.append({"date": date, "close": price, "position": shares, "cash": cash, "total_value": total, "buy_price": first_price})

	result = pd.DataFrame(rows).set_index("date")
	return result


def plot_price_signals_and_equity(price_df: pd.DataFrame,
        bt_macd: pd.DataFrame,
        bt_rsi: pd.DataFrame,
        bt_bb: pd.DataFrame,
        bah: pd.DataFrame,
        save_path: str = "comparison_all.png") -> None:
	"""在单图中绘制：Buy-and-Hold、MACD 策略、RSI 策略的资金曲线，并在右轴绘制价格。

	参数:
	    bt_result: （保留兼容）任意策略回测结果（未使用）
	    bt_macd: MACD 策略回测结果，需包含 `total_value` 列
	    bt_rsi: RSI 策略回测结果，需包含 `total_value` 列
	    bah: Buy-and-Hold 回测结果，需包含 `total_value` 列
	"""

	fig, ax_val = plt.subplots(figsize=(12, 6))

	# 绘制三/四条资金曲线
	ax_val.plot(bah.index, bah["total_value"], label="Buy and Hold", color="grey")
	ax_val.plot(bt_macd.index, bt_macd["total_value"], label="MACD Strategy", color="blue")
	ax_val.plot(bt_rsi.index, bt_rsi["total_value"], label="RSI Strategy", color="green")
	ax_val.plot(bt_bb.index, bt_bb["total_value"], label="Bollinger Strategy", color="purple")

	ax_val.set_ylabel("Total Value (CNY)")
	ax_val.grid(True)

	# 价格用右轴显示（缩放独立）
	ax_price = ax_val.twinx()
	# 使用 bah 的索引与原始价格（从 bah 的 close 列）
	ax_price.plot(bah.index, bah["close"], label="Close Price", color="black", alpha=0.3)
	ax_price.set_ylabel("Price (HKD)")

	# 合并图例
	lines, labels = ax_val.get_legend_handles_labels()
	lines2, labels2 = ax_price.get_legend_handles_labels()
	ax_val.legend(lines + lines2, labels + labels2, loc="upper left")

	plt.title("Strategy Equity Curves and Price")
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.show()


def main() -> None:
	# 1. 读取数据并计算 RSI 等指标
	price_df = load_price_data("data.txt")
	price_with_ind = add_common_indicators(price_df)

	# 2. 基于 RSI 生成交易信号并回测
	price_rsi = generate_rsi_signals(price_with_ind)
	bt_rsi = backtest_rsi_strategy(price_rsi, INITIAL_CAPITAL)

	# 3. 基于 MACD 生成交易信号并回测
	price_macd = generate_macd_signals(price_with_ind)
	bt_macd = backtest_macd_strategy(price_macd, INITIAL_CAPITAL)

	# 4. 基于 Bollinger 生成交易信号并回测
	price_bb = generate_bb_signals(price_with_ind)
	bt_bb = backtest_bb_strategy(price_bb, INITIAL_CAPITAL)

	# 4. Buy-and-Hold 回测
	bah_result = buy_and_hold(price_df, INITIAL_CAPITAL)

	# 5. 输出三种策略的收益情况
	def summarize(name: str, result: pd.DataFrame):
		final = result["total_value"].iloc[-1]
		profit = final - INITIAL_CAPITAL
		pct = profit / INITIAL_CAPITAL * 100
		print(f"{name}: 期末总资产 {final:.2f} 元, 总收益 {profit:.2f} 元 ({pct:.2f}%)")

	print(f"初始资金: {INITIAL_CAPITAL:.2f} 元\n")
	summarize("Buy and Hold", bah_result)
	summarize("MACD Strategy", bt_macd)
	summarize("RSI Strategy", bt_rsi)
	summarize("Bollinger Strategy", bt_bb)

	# 6. 在一张图上显示四条资金曲线和股价
	plot_price_signals_and_equity(price_with_ind, bt_macd, bt_rsi, bt_bb, bah_result)



if __name__ == "__main__":
	main()

