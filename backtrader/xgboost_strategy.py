import backtrader as bt
import pandas as pd
import numpy as np

class XGBoostStrategy(bt.Strategy):
    params = (
        ('models', []),
        ('print_log', True),
    )

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # self.log('Order Canceled/Margin/Rejected')
            pass

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        if not self.params.models:
            return

        # 确保有足够的历史数据用于计算滞后特征 (5天)
        if len(self) < 6:
            return

        d = self.data
        
        # 构建特征向量，顺序必须与训练时完全一致！
        try:
            feats = [
                d.open[0], d.high[0], d.low[0], d.close[0], d.volume[0],
                d.ma5[0], d.ma10[0], d.ma20[0],
                d.rsi14[0],
                d.macd_dif[0], d.macd_dea[0], d.macd_bar[0],
                d.boll_upper[0], d.boll_lower[0],
                d.atr14[0]
            ]
        except AttributeError:
            # 如果缺少数据列，无法预测
            return

        # 添加滞后特征: close, volume, rsi14, macd_bar
        for src in [d.close, d.volume, d.rsi14, d.macd_bar]:
            for i in range(1, 6):
                feats.append(src[-i])
        
        # 添加收益率及其滞后特征
        curr_ret = (d.close[0] / d.close[-1]) - 1 if d.close[-1] != 0 else 0
        feats.append(curr_ret)
        
        for i in range(1, 6):
            r = (d.close[-i] / d.close[-i-1]) - 1 if d.close[-i-1] != 0 else 0
            feats.append(r)
            
        # 预测
        X = np.array([feats])
        
        avg_prob = 0.0
        for model in self.params.models:
            avg_prob += model.predict_proba(X)[0, 1]
        avg_prob /= len(self.params.models)
        
        # 简单的阈值策略 (Simple Threshold Strategy)
        
        target_pct = None
        
        if avg_prob > 0.52:
            target_pct = 0.99
        elif avg_prob < 0.40:
            target_pct = 0.0
            
        if target_pct is not None:
            # 强制整手交易 (100股/手)
            # 计算目标市值
            value = self.broker.get_value()
            target_value = value * target_pct
            price = self.data.close[0]
            
            if price > 0:
                # 向下取整到 100 的倍数
                target_size = int(target_value / price // 100) * 100
            else:
                target_size = 0
                
            self.order_target_size(target=target_size)

    @classmethod
    def train_model(cls, data_path):
        try:
            import xgboost as xgb
        except ImportError:
            print("Error: xgboost not installed. Please run 'pip install xgboost'.")
            return []

        print(f"[XGBoost] Loading data for training from {data_path}...")
        df = pd.read_csv(data_path, sep='\t', parse_dates=['date'])
        df.set_index('date', inplace=True)

        # 特征工程
        # 必须使用与 next() 中一致的特征
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20',
            'rsi14',
            'macd_dif', 'macd_dea', 'macd_bar',
            'boll_upper', 'boll_lower',
            'atr14'
        ]
        
        # 检查缺失列
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"Warning: Missing columns in data: {missing}")
        
        feature_list = features.copy()
        
        cols_to_lag = ['close', 'volume', 'rsi14', 'macd_bar']
        lags = 5
        
        # 构建滞后特征
        for col in cols_to_lag:
            if col not in df.columns: continue
            for lag in range(1, lags + 1):
                feat_name = f'{col}_lag_{lag}'
                df[feat_name] = df[col].shift(lag)
                feature_list.append(feat_name)

        # 构建收益率特征
        df['return'] = df['close'].pct_change()
        feature_list.append('return')
        
        for lag in range(1, lags + 1):
            feat_name = f'return_lag_{lag}'
            df[feat_name] = df['return'].shift(lag)
            feature_list.append(feat_name)
            
        train_df = df.copy()
        train_df.dropna(inplace=True)

        # 目标：次日收盘价上涨
        train_df['target'] = (train_df['close'].shift(-1) > train_df['close']).astype(int)
        train_df.dropna(inplace=True)

        # 训练集划分 (2024年前)
        train_data = train_df[train_df.index < '2024-01-01']
        
        if train_data.empty:
            print(f"[XGBoost] Warning: No training data found before 2024-01-01.")
            return []

        X_train = train_data[feature_list]
        y_train = train_data['target']

        print(f"[XGBoost] Training XGBoost model on {len(X_train)} samples...")
        
        n_models = 50
        models = []
        
        for i in range(n_models):
            seed = 42 + i
            model = xgb.XGBClassifier(
                n_estimators=90,
                learning_rate=0.02,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.80,
                random_state=seed,
                eval_metric='logloss',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            models.append(model)
            
        print(f"[XGBoost] Training complete. Ensemble of {n_models} models created.")
        return models
