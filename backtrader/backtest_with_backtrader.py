from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import math
import backtrader as bt
import json
import datetime
import os

# Import Strategy Base and Concrete Strategies

from strategy_base import Strategy
from basic_strategies import MovingAverageCross, RSIReversion, BollingerBreakout, BuyAndHold, MACDStrategy
from ml_strategies import RandomForestStrategy
from LSTM import LSTMStrategy
from xgboost_strategy import XGBoostStrategy

# ============================= Data Loading =============================
def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', engine='python')
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' not in df.columns:
        raise ValueError("数据文件缺少 date 列")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少 {col} 列")
    return df

STRATEGY_MAP = {
    'buy_hold': BuyAndHold,
    # 'ma_cross': MovingAverageCross,
    # 'rsi_reversion': RSIReversion,
    # 'boll_breakout': BollingerBreakout,
    # 'macd': MACDStrategy,
    'xgboost': XGBoostStrategy,
    # 'random_forest': RandomForestStrategy,
    # 'lstm': LSTMStrategy,
}

# ============================= Backtrader Classes =============================

class SignalData(bt.feeds.PandasData):
    """
    扩展的 PandasData，包含 'signal' 列
    """
    lines = ('signal',)
    params = (
        ('signal', -1), # 自动匹配列名
    )

class HKCommission(bt.CommInfoBase):
    """
    港股手续费计算
    """
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0003), # 0.03%
    )

    def _getcommission(self, size, price, pseudoexec):
        amount = abs(size) * price
        # 1. 佣金: 0.03% * amount, 最低 3 HKD
        commission = max(3.0, amount * 0.0003)
        # 2. 平台使用费: 固定 15 HKD
        platform_fee = 15.0
        # 3. 系统使用费: 2023-01-01 起取消 (此处简化处理，假设当前均为2023后)
        sys_fee = 0.0 

        # 4. 印花税 (股票): 0.1% * amount, 向上取整, 最低 1 HKD
        stamp_duty = max(1.0, math.ceil(amount * 0.001))

        # 5. 其他杂费
        settlement_fee = amount * 0.000042
        trading_fee = max(0.01, amount * 0.0000565)
        sfc_levy = max(0.01, amount * 0.000027)
        frc_levy = amount * 0.0000015

        total = commission + platform_fee + sys_fee + stamp_duty + settlement_fee + trading_fee + sfc_levy + frc_levy
        return total

class VectorizedSignalStrategy(bt.Strategy):
    """
    接收预计算信号并执行的 Backtrader 策略
    """
    params = (
        ('max_position', 1.0),
    )

    def __init__(self):
        self.signal = self.data.signal
        self.total_commission = 0.0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            # 仅在 verbose 模式或调试时打印，这里为了回答用户问题默认打印关键交易
            # 为了避免输出过多，可以只打印第一笔和最后一笔，或者全部打印
            # 鉴于交易次数不多，全部打印
            dt = bt.num2date(order.executed.dt).date()
            print(f"[{dt}] {order.data._name} {'BUY' if order.isbuy() else 'SELL'} "
                  f"Size: {order.executed.size:.2f} @ {order.executed.price:.4f} "
                  f"Comm: {order.executed.comm:.2f} Cost: {order.executed.value:.2f}")
            self.total_commission += order.executed.comm

    def next(self):
        # 获取当前信号权重
        target_weight = self.signal[0]
        
        # 限制权重在 max_position 范围内
        target_weight = max(min(target_weight, self.params.max_position), -self.params.max_position)
        
        # 计算目标市值
        target_value = self.broker.getvalue() * target_weight
        
        # 预留手续费缓冲 (港股印花税0.1% + 佣金等，大资金时取整剩下的钱可能不够)
        # 如果是做多，预留 0.5% 的资金
        if target_weight > 0:
            target_value *= 0.995

        # 获取当前价格
        price = self.data.close[0]
        
        if price > 0:
            # 计算目标股数
            target_size = target_value / price
            
            # 向下取整到 100 的倍数 (整手交易)
            # int(x / 100) * 100 会将 199 -> 100, -199 -> -100
            target_size = int(target_size / 100) * 100
            
            # 执行调仓
            self.order_target_size(target=target_size)

# ============================= Backtest Engine =============================

def run_backtest(
    df: pd.DataFrame,
    positions: pd.Series,
    strategy_name: str,
    apply_fees: bool = False,
    capital: float = 100000.0,
    max_position: float = 1.0,
    strat_obj: Optional[Strategy] = None,
) -> Dict:
    
    # 将信号合并到 DataFrame
    data_df = df.copy()
    data_df['signal'] = positions
    
    cerebro = bt.Cerebro()
    
    data = SignalData(
        dataname=data_df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        signal='signal',
        openinterest=-1
    )
    cerebro.adddata(data)
    
    cerebro.addstrategy(VectorizedSignalStrategy, max_position=max_position)
    cerebro.broker.setcash(capital)
    
    if apply_fees:
        cerebro.broker.addcommissioninfo(HKCommission())
    else:
        # 无手续费，但必须显式设置为股票模式 (stocklike=True)
        # 否则 Backtrader 默认为期货模式，导致资产计算和下单逻辑差异
        cerebro.broker.setcommission(commission=0.0, stocklike=True)
        
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.04, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    
    results = cerebro.run()
    strat_result = results[0]
    
    metrics = {}
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / capital) - 1
    metrics['total_return'] = round(total_return, 6)
    
    ret_analyzer = strat_result.analyzers.returns.get_analysis()
    metrics['cagr'] = round(ret_analyzer.get('rnorm100', 0.0) / 100.0, 6) if 'rnorm100' in ret_analyzer else 0.0
    
    dd_analyzer = strat_result.analyzers.drawdown.get_analysis()
    metrics['max_drawdown'] = round(dd_analyzer['max']['drawdown'] / 100.0, 6)
    
    sharpe_analyzer = strat_result.analyzers.sharpe.get_analysis()
    metrics['sharpe'] = round(sharpe_analyzer.get('sharperatio', 0.0) or 0.0, 6)
    
    trade_analyzer = strat_result.analyzers.trades.get_analysis()
    total_trades = trade_analyzer.get('total', {}).get('total', 0)
    won_trades = trade_analyzer.get('won', {}).get('total', 0)
    metrics['trade_count'] = total_trades
    metrics['win_rate'] = round(won_trades / total_trades, 6) if total_trades > 0 else 0.0
    
    # 简单计算平均每笔交易收益 (近似值)
    metrics['avg_trade_return'] = 0.0 
    
    # 获取总手续费
    metrics['total_commission'] = round(strat_result.total_commission, 2)
    
    # 获取每日收益率序列
    timereturn_analyzer = strat_result.analyzers.timereturn.get_analysis()
    returns_series = pd.Series(timereturn_analyzer)
    # 记录 XGBoost 实验结果
    if strategy_name == 'xgboost' and strat_obj is not None:
        log_xgboost_experiment(strat_obj, metrics)

    return {
        'strategy': strategy_name,
        **metrics,
        'returns_series': returns_series
    }

def log_xgboost_experiment(strat_obj, metrics):
    """
    记录 XGBoost 实验参数和结果到 JSON 文件
    """
    log_file = 'xgboost_experiments.json'
    
    # 获取模型参数
    # 假设 strat_obj.models 是一个列表，我们取第一个模型的参数作为代表
    # 或者记录所有模型的参数配置（如果它们是一样的）
    params = {}
    if hasattr(strat_obj, 'models') and strat_obj.models:
        # 获取第一个模型的参数
        first_model = strat_obj.models[0]
        # XGBClassifier 的参数可以通过 get_params() 获取
        try:
            params = first_model.get_params()
            # 移除一些不必要的或者太长的参数
            keys_to_remove = [
                'n_jobs', 'missing', 'monotone_constraints', 'interaction_constraints', 'enable_categorical',
                'base_score', 'booster', 'callbacks', 'colsample_bylevel', 'colsample_bynode', 'device',
                'early_stopping_rounds', 'feature_types', 'feature_weights', 'gamma', 'grow_policy',
                'importance_type', 'max_bin', 'max_cat_threshold', 'max_cat_to_onehot', 'max_delta_step',
                'max_leaves', 'min_child_weight', 'multi_strategy', 'num_parallel_tree', 'reg_alpha',
                'reg_lambda', 'sampling_method', 'scale_pos_weight', 'tree_method', 'validate_parameters',
                'verbosity'
            ]
            for k in keys_to_remove:
                if k in params:
                    del params[k]
            
            # 记录集成模型的数量
            params['n_ensemble_models'] = len(strat_obj.models)
            
        except Exception as e:
            print(f"获取 XGBoost 参数失败: {e}")
            params = {"error": str(e)}
    
    # 构建日志条目
    entry = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': params,
        'metrics': metrics
    }
    
    # 读取现有日志
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            pass # 文件可能损坏或为空
            
    # 添加新条目
    logs.append(entry)
    
    # 写入文件
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        print(f"XGBoost 实验记录已保存至 {log_file}")
    except Exception as e:
        print(f"保存实验记录失败: {e}")

# ============================= CLI Runner =============================
def parse_args():
    p = argparse.ArgumentParser(description='Backtrader 回测框架')
    p.add_argument('--file', default='data/data_with_indicators.txt', help='数据文件路径 (含OHLC)')
    p.add_argument('--start', default='2024-01-01', help='起始日期 YYYY-MM-DD')
    p.add_argument('--end', default='2025-04-24', help='结束日期 YYYY-MM-DD')
    p.add_argument('--strategies', nargs='+', default=list(STRATEGY_MAP.keys()), help='策略名称列表')
    p.add_argument('--output', help='结果保存CSV文件名')
    p.add_argument('--apply-fees', action='store_true', help='启用港股手续费扣除')
    p.add_argument('--capital', type=float, default=100000.0, help='初始资金')
    p.add_argument('--max-position', type=float, default=1.0, help='持仓权重上限（绝对值）')
    return p.parse_args()

def main():
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f'数据文件不存在: {path}')

    df = load_price_data(path)
    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]
    if df.empty:
        raise ValueError('筛选后无数据')

    results = []
    equity_curves = {}

    for name in args.strategies:
        if name not in STRATEGY_MAP:
            print(f"跳过未知策略: {name}")
            continue
        
        strategy_cls = STRATEGY_MAP[name]
        # 实例化策略对象以调用 prepare
        strat_obj = strategy_cls()
        strat_obj.prepare(str(path))
        
        # 生成信号
        positions = strat_obj.generate_positions(df)
        
        res = run_backtest(
            df,
            positions,
            name,
            apply_fees=args.apply_fees,
            capital=args.capital,
            max_position=args.max_position,
            strat_obj=strat_obj
        )
        
        
        # 提取收益率序列用于绘图，并从结果中移除以免影响汇总表
        if 'returns_series' in res:
            equity_curves[name] = res.pop('returns_series')
            
        results.append(res)

    # 汇总输出
    summary_df = pd.DataFrame(results)
    print('\nBacktrader 回测结果汇总:')
    print(summary_df.to_string(index=False))

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f'结果已保存: {args.output}')
        
    # 绘制收益曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        for name, ret_series in equity_curves.items():
            # 将索引转换为 datetime 对象 (如果还不是)
            ret_series.index = pd.to_datetime(ret_series.index)
            
            # 计算累计收益 (净值曲线)
            # 初始资金 * (1 + 收益率).cumprod()
            equity_curve = args.capital * (1 + ret_series).cumprod()
            
            plt.plot(equity_curve.index, equity_curve.values, label=name)
            
        plt.title('Strategy Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity (Capital)')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plot_filename = 'strategy_comparison.png'
        if args.output:
            plot_filename = str(Path(args.output).with_suffix('.png'))
            
        plt.savefig(plot_filename)
        print(f"收益曲线图已保存至: {plot_filename}")
        
        # 尝试显示 (如果在支持 GUI 的环境中)
        # plt.show()
        
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
    except Exception as e:
        print(f"绘图时发生错误: {e}")

if __name__ == '__main__':
    main()
