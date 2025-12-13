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

# Import Concrete Strategies
from basic_strategies import MovingAverageCross, RSIReversion, BollingerBreakout, BuyAndHold, MACDStrategy
from xgboost_strategy import XGBoostStrategy

# ============================= 数据加载 =============================
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
}

# ============================= Backtrader 类定义 =============================

class MLData(bt.feeds.PandasData):
    """
    扩展的 PandasData，包含机器学习特征
    """
    lines = (
        'ma5', 'ma10', 'ma20',
        'rsi14',
        'macd_dif', 'macd_dea', 'macd_bar',
        'boll_upper', 'boll_lower',
        'atr14'
    )
    params = (
        ('ma5', -1),
        ('ma10', -1),
        ('ma20', -1),
        ('rsi14', -1),
        ('macd_dif', -1),
        ('macd_dea', -1),
        ('macd_bar', -1),
        ('boll_upper', -1),
        ('boll_lower', -1),
        ('atr14', -1),
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

# ============================= 回测引擎 =============================

def run_backtest(
    df: pd.DataFrame,
    strategy_cls: bt.Strategy,
    strategy_params: Dict = {},
    strategy_name: str = "",
    apply_fees: bool = False,
    capital: float = 100000.0,
) -> Dict:
    
    cerebro = bt.Cerebro()
    
    # 使用 MLData 以包含所有可能的列
    data = MLData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    cerebro.adddata(data)
    
    cerebro.addstrategy(strategy_cls, **strategy_params)
    cerebro.broker.setcash(capital)
    
    if apply_fees:
        cerebro.broker.addcommissioninfo(HKCommission())
    else:
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
    
    metrics['avg_trade_return'] = 0.0 
    metrics['total_commission'] = 0.0
    
    timereturn_analyzer = strat_result.analyzers.timereturn.get_analysis()
    returns_series = pd.Series(timereturn_analyzer)
    
    if strategy_name == 'xgboost':
        log_xgboost_experiment(strat_result, metrics)

    return {
        'strategy': strategy_name,
        **metrics,
        'returns_series': returns_series
    }

def log_xgboost_experiment(strat_obj, metrics):
    log_file = 'xgboost_experiments.json'
    params = {}
    if hasattr(strat_obj.params, 'models') and strat_obj.params.models:
        first_model = strat_obj.params.models[0]
        try:
            params = first_model.get_params()
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
            params['n_ensemble_models'] = len(strat_obj.params.models)
        except Exception as e:
            print(f"获取 XGBoost 参数失败: {e}")
            params = {"error": str(e)}
    
    entry = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': params,
        'metrics': metrics
    }
    
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            pass
            
    logs.append(entry)
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        print(f"XGBoost 实验记录已保存至 {log_file}")
    except Exception as e:
        print(f"保存实验记录失败: {e}")

# ============================= 命令行运行器 =============================
def parse_args():
    p = argparse.ArgumentParser(description='Backtrader 回测框架')
    p.add_argument('--file', default='data/data_with_indicators.txt', help='数据文件路径 (含OHLC)')
    p.add_argument('--start', default='2024-01-01', help='起始日期 YYYY-MM-DD')
    p.add_argument('--end', default='2025-04-24', help='结束日期 YYYY-MM-DD')
    p.add_argument('--strategies', nargs='+', default=list(STRATEGY_MAP.keys()), help='策略名称列表')
    p.add_argument('--output', help='结果保存CSV文件名')
    p.add_argument('--apply-fees', action='store_true', help='启用港股手续费扣除')
    p.add_argument('--capital', type=float, default=100000.0, help='初始资金')
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
        params = {}
        
        if name == 'xgboost':
            # 先训练模型
            models = XGBoostStrategy.train_model(str(path))
            if not models:
                print("XGBoost training failed or no models returned. Skipping.")
                continue
            params['models'] = models
        
        res = run_backtest(
            df,
            strategy_cls,
            strategy_params=params,
            strategy_name=name,
            apply_fees=args.apply_fees,
            capital=args.capital,
        )
        
        if 'returns_series' in res:
            equity_curves[name] = res.pop('returns_series')
            
        results.append(res)

    summary_df = pd.DataFrame(results)
    print('\nBacktrader 回测结果汇总:')
    print(summary_df.to_string(index=False))

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f'结果已保存: {args.output}')
        
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        for name, ret_series in equity_curves.items():
            ret_series.index = pd.to_datetime(ret_series.index)
            equity_curve = args.capital * (1 + ret_series).cumprod()
            plt.plot(equity_curve.index, equity_curve.values, label=name)
            
        plt.title('Strategy Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity (Capital)')
        plt.legend()
        plt.grid(True)
        
        plot_filename = 'strategy_comparison.png'
        if args.output:
            plot_filename = str(Path(args.output).with_suffix('.png'))
            
        plt.savefig(plot_filename)
        print(f"收益曲线图已保存至: {plot_filename}")
        
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
    except Exception as e:
        print(f"绘图时发生错误: {e}")

if __name__ == '__main__':
    main()
