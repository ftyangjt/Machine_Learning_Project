import pandas as pd
import numpy as np
try:
    from .strategy_base import Strategy
except ImportError:
    from strategy_base import Strategy


class XGBoostStrategy(Strategy):
    name = "xgboost"

    def __init__(self):
        self.models = [] # Changed from self.model to self.models list
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20',
            'rsi14',
            'macd_dif', 'macd_dea', 'macd_bar',
            'boll_upper', 'boll_lower',
            'atr14'
        ]
        self.lags = 5
        self.cols_to_lag = ['close', 'volume', 'rsi14', 'macd_bar']
        self.full_df = None
        self.feature_names = []

    def prepare(self, data_path: str):
        try:
            import xgboost as xgb
        except ImportError:
            print("Error: xgboost not installed. Please run 'pip install xgboost'.")
            return

        print(f"[{self.name}] Loading data for training from {data_path}...")
        df = pd.read_csv(data_path, sep='\t', parse_dates=['date'])
        df.set_index('date', inplace=True)
        self.full_df = df.copy() # Keep full data for feature engineering context

        # Feature Engineering
        train_df = self._engineer_features(df.copy())
        train_df.dropna(inplace=True)

        # Target: Next day close > Today close
        train_df['target'] = (train_df['close'].shift(-1) > train_df['close']).astype(int)
        train_df.dropna(inplace=True)

        # Split for training (Fixed split as per 1.py logic)
        # Train on 2018-2023
        train_data = train_df[train_df.index < '2024-01-01']
        
        if train_data.empty:
            print(f"[{self.name}] Warning: No training data found before 2024-01-01.")
            return

        X_train = train_data[self.feature_names]
        y_train = train_data['target']

        print(f"[{self.name}] Training XGBoost model on {len(X_train)} samples...")
        
        # 为了评估模型的平均表现，我们训练多个不同随机种子的模型并取平均预测
        n_models = 50
        self.models = []
        
        for i in range(n_models):
            seed = 42 + i
            model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.02,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                eval_metric='logloss',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models.append(model)
            
        print(f"[{self.name}] Training complete. Ensemble of {n_models} models created.")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Base features must exist
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            # Try to compute missing indicators if possible, or warn
            # For now assume they exist as per data_with_indicators.txt
            pass
        
        feature_list = self.features.copy()
        
        # Lag features
        for col in self.cols_to_lag:
            if col not in df.columns: continue
            for lag in range(1, self.lags + 1):
                feat_name = f'{col}_lag_{lag}'
                df[feat_name] = df[col].shift(lag)
                feature_list.append(feat_name)

        # Return features
        df['return'] = df['close'].pct_change()
        for lag in range(1, self.lags + 1):
            feat_name = f'return_lag_{lag}'
            df[feat_name] = df['return'].shift(lag)
            feature_list.append(feat_name)
            
        self.feature_names = feature_list
        return df

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        if not hasattr(self, 'models') or not self.models:
            return pd.Series(0, index=df.index)

        # We need context for lag features. 
        # If self.full_df is available, use it to generate features for the requested dates.
        if self.full_df is not None:
            # Use full_df to generate features, then slice
            full_feat = self._engineer_features(self.full_df.copy())
            # Align with requested df
            # We need to predict for the dates in df
            # But be careful: we need features at time T to predict T+1 return (signal for T)
            
            # Filter for the relevant dates
            # We need to ensure we have data for the requested index
            valid_indices = df.index.intersection(full_feat.index)
            X_test = full_feat.loc[valid_indices, self.feature_names]
            
            # Predict using Ensemble
            # Average the probabilities from all models
            avg_probs = np.zeros(len(X_test))
            for model in self.models:
                avg_probs += model.predict_proba(X_test)[:, 1]
            avg_probs /= len(self.models)
            
            # 改进策略：引入滞后阈值 (Hysteresis) 防止过早止盈
            # 逻辑：
            # 1. 入场/加仓：信心较高 (Prob > 0.55)
            # 2. 离场/止损：信心较低 (Prob < 0.45)
            # 3. 持仓观望：信心一般 (0.45 <= Prob <= 0.55)，保持现有仓位
            
            signals = []
            current_pos = 0 # 初始假设空仓
            
            for p in avg_probs:
                if p > 0.55:
                    current_pos = 1
                elif p < 0.45:
                    current_pos = 0
                # else: 保持 current_pos 不变 (Hold)
                
                signals.append(current_pos)
            
            return pd.Series(signals, index=valid_indices).reindex(df.index).fillna(0)
        else:
            # Fallback if prepare wasn't called (shouldn't happen if framework is updated)
            return pd.Series(0, index=df.index)