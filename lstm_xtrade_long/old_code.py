from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
import pandas as pd
from jesse import research
import joblib
import numpy as np
import jesse.helpers as jh
# from jesse.research import backtest 
import json
from indicators import feature_indicators

import torch

from .lstm_model import BiLSTMClassifierWithAttention




coin = 'SOL'
selected_indices = [5, 6, 8, 12, 141, 14, 15, 16, 38, 41, 43, 46, 55, 56, 58, 60, 74, 76, 103, 107, 109, 121]
timestep = 35
hidden_dim = 32
num_layers = 2
dropout_rate = 0.1
input_dim = 14 + len(selected_indices)
output_dim = 3


def get_indicator_key_value(feature_indicators, index):
    key = list(feature_indicators.keys())[index]
    value = list(feature_indicators.values())[index]
    return key, value 


class lstm_xtrade_long(Strategy):
    def __init__(self):
        super().__init__()
        self.model = BiLSTMClassifierWithAttention(input_dim, hidden_dim, num_layers, 3, dropout_rate)
        model_filename = '/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/strategies/lstm_xtrade_long/SOL_scaler_params_index_[5, 6, 8, 12, 141, 14, 15, 16, 38, 41, 43, 46, 55, 56, 58, 60, 74, 76, 103, 107, 109, 121].joblib'
        if model_filename:
            self.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
            self.model.eval()
        else:
            raise ValueError("Model filename not provided")
        scaler_params_filename = '/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/strategies/lstm_xtrade_long/SOL_TOP_trained_model_lstm_index_[5, 6, 8, 12, 141, 14, 15, 16, 38, 41, 43, 46, 55, 56, 58, 60, 74, 76, 103, 107, 109, 121].pth'
        self.scaler_params = joblib.load(scaler_params_filename)  

    def generate_features(self):
        if len(self.candles) < 40:
            return None
        
        np_candles = np.array(self.candles)
        
        opens = np_candles[:, 1].astype(np.float32)
        highs = np_candles[:, 3].astype(np.float32)
        lows = np_candles[:, 4].astype(np.float32)
        closes = np_candles[:, 2].astype(np.float32)
        volumes = np_candles[:, 5].astype(np.float32)

        # Your existing feature calculations
        ret1 = np.diff(closes) / closes[:-1]
        ret1 = np.insert(ret1, 0, np.nan)

        def rolling_feature(values, window, func):
            return np.array([func(values[i-window:i]) if i >= window else np.nan for i in range(1, len(values)+1)])
        
        ret5 = rolling_feature(ret1, 5, np.sum)
        ret10 = rolling_feature(ret1, 10, np.sum)
        ret20 = rolling_feature(ret1, 20, np.sum)
        ret40 = rolling_feature(ret1, 40, np.sum)
        
        std5 = rolling_feature(ret1, 5, np.std)
        std10 = rolling_feature(ret1, 10, np.std)
        std20 = rolling_feature(ret1, 20, np.std)
        std40 = rolling_feature(ret1, 40, np.std)

        # Base feature list
        base_features = [opens, highs, lows, closes, volumes, ret1, ret5, ret10, ret20, ret40, std5, std10, std20, std40]  
        
        # Dynamically compute selected indicators
        selected_features = []
        for index in selected_indices:
            key, indicator_func = get_indicator_key_value(feature_indicators, index)
            indicator_value = indicator_func(np_candles)
            selected_features.append(indicator_value)
            
        # Combine base features and selected indicators
        all_features = base_features + selected_features
        
        # Ensure all arrays are correctly shaped for stacking
        all_features = [f if isinstance(f, np.ndarray) else np.array(f) for f in all_features]
        
        features = np.column_stack(all_features)
        features = features[~np.isnan(features).any(axis=1)]

        if features.shape[0] < timestep:
            return None

        recent_features = features[-timestep:]
        scaled_features = (recent_features - self.scaler_params['mean']) / self.scaler_params['scale']
        feature_tensor = torch.tensor(scaled_features, dtype=torch.float).unsqueeze(0)

        return feature_tensor


    
    def should_long(self) -> bool:
        features = self.generate_features()
        with torch.no_grad():
            prediction = self.model(features)
            _, predicted_class = torch.max(prediction, 1)
        return predicted_class.item() == 2

    def should_short(self) -> bool:
        return False
    
    def should_cancel_entry(self) -> bool:
        return False
    
    def go_long(self):
        qty = utils.size_to_qty(2000, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(2000, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price

        
    def update_position(self):
        features = self.generate_features()
        with torch.no_grad():
            prediction = self.model(features)
            _, predicted_class = torch.max(prediction, 1)
        if self.is_long and predicted_class.item() == 0:
            self.liquidate()
