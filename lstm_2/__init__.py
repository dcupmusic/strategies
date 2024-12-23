from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
import pandas as pd
from jesse import research
import joblib
import numpy as np
import jesse.helpers as jh
# from jesse.research import backtest 

import torch
import torch.nn as nn
import torch.nn.functional as F


# exchange = 'Bybit USDT Perpetual'
# symbol = 'SOL-USDT'
# start_date = '2024-02-01'
# end_date = '2024-03-15'
# timeframe = '1h'
# sol_candles = research.get_candles(exchange, symbol, timeframe, start_date, end_date)
# backtest_candles = research.get_candles(exchange, symbol, '1m', start_date, end_date)



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        weights = torch.tanh(self.linear(lstm_output))
        weights = F.softmax(weights, dim=1)
        
        context = weights * lstm_output
        context = torch.sum(context, dim=1)
        return context, weights

class BiLSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(BiLSTMClassifierWithAttention, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)        
        self.bn_conv1 = nn.BatchNorm1d(hidden_dim)        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)        
        self.attention = Attention(hidden_dim * 2)        
        self.dropout = nn.Dropout(dropout_rate)        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)        
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)        
        self.fc2 = nn.Linear(hidden_dim, output_dim)        
        self.dropout_fc = nn.Dropout(dropout_rate / 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)        
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)        
        x = x.permute(0, 2, 1)        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)        
        out, _ = self.lstm(x, (h0, c0))        
        context, _ = self.attention(out)        
        out = self.fc1(context)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out








coin = 'SOL'
timestep = 20
hidden_dim = 32
num_layers = 2
dropout_rate = 0.1
input_dim = 11
output_dim = 3



class lstm(Strategy):
    def __init__(self):
        super().__init__()
        self.model = BiLSTMClassifierWithAttention(input_dim, hidden_dim, num_layers, 3, dropout_rate)
        model_filename = f'model.pth'
        if model_filename:
            self.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
            self.model.eval()
        else:
            raise ValueError("Model filename not provided")
        scaler_params_filename = f"{coin}_scaler_params.joblib"
        self.scaler_params = joblib.load(scaler_params_filename)
        

    def generate_features(self):
        if len(self.candles) < 40:  # Need at least 40 candles for ret40 and std40
            return None
        np_candles = np.array(self.candles)
        closes = np_candles[:, 2].astype(np.float32)  # Close prices
        volumes = np_candles[:, 5].astype(np.float32)  # Volume

        # Calculate returns
        ret1 = np.diff(closes) / closes[:-1]
        
        # Prepend with nan to maintain length after np.diff
        ret1 = np.insert(ret1, 0, np.nan)

        # Calculate rolling features
        def rolling_feature(values, window, func):
            # Use np.lib.stride_tricks.sliding_window_view in newer numpy versions for efficiency
            return np.array([func(values[i-window:i]) if i >= window else np.nan for i in range(1, len(values)+1)])
        
        ret5 = rolling_feature(ret1, 5, np.sum)
        ret10 = rolling_feature(ret1, 10, np.sum)
        ret20 = rolling_feature(ret1, 20, np.sum)
        ret40 = rolling_feature(ret1, 40, np.sum)
        
        std5 = rolling_feature(ret1, 5, np.std)
        std10 = rolling_feature(ret1, 10, np.std)
        std20 = rolling_feature(ret1, 20, np.std)
        std40 = rolling_feature(ret1, 40, np.std)

        # Combine all features
        features = np.column_stack((closes, volumes, ret1, ret5, ret10, ret20, ret40, std5, std10, std20, std40))

        # Remove any rows with nan values (due to rolling calculations)
        features = features[~np.isnan(features).any(axis=1)]

        # Ensure we have enough rows for the timestep
        if features.shape[0] < timestep:
            return None

        # Select the most recent 'timestep' rows for input
        recent_features = features[-timestep:]

        # Scale features according to scaler parameters
        scaled_features = (recent_features - self.scaler_params['mean']) / self.scaler_params['scale']

        # Convert to tensor
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
        qty = utils.size_to_qty(self.balance, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(self.balance, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price

        
    def update_position(self):
        features = self.generate_features()
        with torch.no_grad():
            prediction = self.model(features)
            _, predicted_class = torch.max(prediction, 1)
        if self.is_long and predicted_class.item() == 0:
            self.liquidate()


# def run_backtest(model_filename):
    
#     # os.environ['MODEL_FILENAME'] = model_filename
    
#     exchange_name = 'Bybit USDT Perpetual'
#     symbol = 'SOL-USDT'
#     timeframe = '1h'
#     config = {
#         'starting_balance': 10_000,
#         'fee': 0,
#         'type': 'futures',
#         'futures_leverage': 10,
#         'futures_leverage_mode': 'cross',
#         'exchange': exchange_name,
#         'warm_up_candles': 300
#     }
#     routes = [
#         {'exchange': exchange_name, 
#         'strategy': lstm, 
#         'symbol': symbol, 
#         'timeframe': timeframe}
#     ]
#     extra_routes = []
#     candles = {
#         jh.key(exchange_name, symbol): {
#             'exchange': exchange_name,
#             'symbol': symbol,
#             'candles': backtest_candles,
#         },
#     }
#     result = backtest(
#         config,
#         routes,
#         extra_routes,
#         candles, 
#         generate_charts=True
#     )

#     # del os.environ['MODEL_FILENAME']
    




# if __name__ == "__main__":
    
#     run_backtest('models/SOL_trained_model_lstm_100.pth')