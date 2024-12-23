from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
from jesse import research
import jesse.helpers as jh
from jesse.research import get_candles

from functions import process_data, save_model
from indicators import feature_indicators

import numpy as np
import pandas as pd 
import random
import optuna
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

import warnings
warnings.simplefilter("ignore", UserWarning)

import vectorbtpro as vbt
vbt.settings.set_theme('dark')
vbt.settings['plotting']['layout']['width'] = 1600
vbt.settings['plotting']['layout']['height'] = 600


import plotly.graph_objects as go

import joblib
import sys


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)



# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

exchange = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
train_start_date = '2024-02-01'
train_end_date = '2024-07-06'
timeframe = '1h'
coin = 'SOL'
# Convert date strings to timestamps
train_start_date_timestamp = jh.date_to_timestamp(train_start_date)
train_end_date_timestamp = jh.date_to_timestamp(train_end_date)

# Fetch candle data
candle_data = get_candles(exchange, symbol, timeframe, train_start_date_timestamp, train_end_date_timestamp)


backtest_candles = candle_data[1]



# def plot_target():
#     signal = y
#     entries = signal == 2
#     exits = signal == 0
#     pf = vbt.Portfolio.from_signals(
#         close=X.close, 
#         long_entries=entries, 
#         short_entries=exits,
#         size=100,
#         size_type='value',
#         # accumulate=True,
#         init_cash='auto'
#     )
#     pf.plot({"orders"}).show()
# plot_target()





class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_length, hidden_dim]
        weights = torch.tanh(self.linear(lstm_output))
        weights = F.softmax(weights, dim=1)
        
        # Context vector with weighted sum
        context = weights * lstm_output
        context = torch.sum(context, dim=1)
        return context, weights

class BiLSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(BiLSTMClassifierWithAttention, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization Layer for Conv1d
        self.bn_conv1 = nn.BatchNorm1d(hidden_dim)
        
        # LSTM Layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Attention Layer
        self.attention = Attention(hidden_dim * 2)  # For bidirectional LSTM
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # Adjusted for attention context vector
        
        # Batch Normalization Layer for FC1
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        
        # Additional Dropout for the fully connected layer
        self.dropout_fc = nn.Dropout(dropout_rate / 2)

    def forward(self, x):
        # Reshape x for Conv1d
        x = x.permute(0, 2, 1)
        
        # Convolutional layer
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        
        # Reshape back for LSTM
        x = x.permute(0, 2, 1)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Applying attention mechanism to LSTM outputs
        context, _ = self.attention(out)
        
        # Fully connected layers using the context vector from attention
        out = self.fc1(context)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out







num_trials = len(feature_indicators.keys())

highest_calmar = 0
highest_indicies = []



def objective(trial):
    
    global highest_calmar
    global highest_indicies

    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    
    timestep = 35
    window_size = 10

    df_candles = pd.DataFrame(backtest_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) 
    df_candles['timestamp'] = pd.to_datetime(df_candles['timestamp'], unit='ms')
    df_candles.set_index('timestamp', inplace=True)


    df = process_data(df_candles, window_size, coin)

    df['ret1'] = df.close.pct_change()
    df['ret5'] = df.ret1.rolling(5).sum()
    df['ret10'] = df.ret1.rolling(10).sum()
    df['ret20'] = df.ret1.rolling(20).sum()
    df['ret40'] = df.ret1.rolling(40).sum()


    df['std5'] = df.ret1.rolling(5).std()
    df['std10'] = df.ret1.rolling(10).std()
    df['std20'] = df.ret1.rolling(20).std()
    df['std40'] = df.ret1.rolling(40).std()
    



    index = trial.number
    
    def get_indicator_key_value(feature_indicators, index):
        key = list(feature_indicators.keys())[index]
        value = list(feature_indicators.values())[index]
        return key, value
        
        
    if trial.number in main_indicies:
        raise optuna.exceptions.TrialPruned()
    
    selected_indices = main_indicies + [index]
    selected_indices = list(set(selected_indices))
    
    print(f"Selected indicies: {selected_indices}")
    
    
    for i in selected_indices:
        key, value = get_indicator_key_value(feature_indicators, i)
        df[key] = value(backtest_candles)



    data = vbt.Data.from_data(df)

    data.data['symbol'] = data.data['symbol'].dropna(axis=1, how='all')
    data.data['symbol'] = data.data['symbol'].dropna()


    predictor_list = data.data['symbol'].drop('signal', axis=1).columns.tolist()
    X = data.data['symbol'][predictor_list]
    y = data.data['symbol']['signal']




    
    
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.3, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # scaler_params = {'mean': scaler.mean_, 'scale': scaler.scale_}
    # joblib.dump(scaler_params, f"/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/models/{coin}_scaler_params_index_{selected_indices}.joblib")
    
    def validate_financials(model, X_test_selected_gpu):
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_selected_gpu)
            probabilities = torch.softmax(y_test_pred, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)
            predicted_labels_numpy = predicted_labels.cpu().numpy()
        adjusted_indices_test = indices_test[timestep:] 
        df_split = data.data['symbol'].iloc[adjusted_indices_test].copy()
        df_split.loc[:, "signal"] = predicted_labels_numpy
        signal = df_split['signal']
        entries = signal == 2
        exits = signal == 0
        pf = vbt.Portfolio.from_signals(
            close=df_split.close, 
            long_entries=entries, 
            long_exits=exits,
            size=100,
            size_type='value',
            init_cash='auto'
        )
        stats = pf.stats()
        plot = pf.plot({"orders", "cum_returns"}, title=f"Index {index}")
        total_return = stats['Total Return [%]']
        orders = stats['Total Orders']
        calmar = stats['Calmar Ratio']
        calmer_returns = (total_return + calmar)
        # print(f"Total Return: {total_return}%")
        # print(f"Orders: {orders}")
        model.train()
        return {
            "orders": orders,
            "calmer_returns": (calmer_returns),
            "plot": plot
        }


    def create_sequences(input_data, timestep):
        sequences = []
        data_len = len(input_data)
        for i in range(data_len - timestep):
            seq = input_data[i:(i + timestep)]
            sequences.append(seq)
        return np.array(sequences)


    X_train_list = create_sequences(X_train_scaled, timestep)
    X_test_list = create_sequences(X_test_scaled, timestep)
    y_train_seq_ar = y_train[timestep:]
    y_test_seq_ar = y_test[timestep:]
    


    x_train_ar = np.array(X_train_list)
    y_train_seq = np.array(y_train_seq_ar).astype(int)
    x_test_ar = np.array(X_test_list)  
    y_test_seq = np.array(y_test_seq_ar).astype(int)


    X_train_tensor = torch.tensor(x_train_ar, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test_ar, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.long)
    
    if isinstance(y_train_seq, torch.Tensor):
        y_train_seq_np = y_train_seq.cpu().numpy()
    else:
        y_train_seq_np = y_train_seq 
        

    unique_labels = np.unique(y_train_seq_np)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_train_seq_np)
    

    
    if 0 not in unique_labels or 1 not in unique_labels or 2 not in unique_labels:
        print(f"Not all classes are present in the training data. Skipping trial.")
        raise optuna.exceptions.TrialPruned()

    index_of_class_short = np.where(unique_labels == 0)[0][0]
    index_of_class_none = np.where(unique_labels == 1)[0][0]
    index_of_class_long = np.where(unique_labels == 2)[0][0]

    class_weights[index_of_class_short] *= 1.9
    class_weights[index_of_class_none] *= 0
    class_weights[index_of_class_long] *= 0.9
    
    # class_weights[index_of_class_short] *= trial.suggest_float('short weight', 0.5, 0.7, step=0.1)
    # class_weights[index_of_class_none] *= trial.suggest_float('none weight', 0, 0.5, step=0.1)
    # class_weights[index_of_class_long] *= trial.suggest_float('long weight', 1.1, 1.5, step=0.1)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights_tensor = class_weights_tensor.to(device) 


    num_epochs = 90
    hidden_dim = 32
    num_layers = 2
    learning_rate= 0.1
    gamma=0.9
    step_size=10
    dropout_rate=0.1




    ''' multi feature selection'''
    # num_epochs = trial.suggest_int('num_epochs', 100, 300, step=50)
    # hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64])
    # num_layers = trial.suggest_int('num_layers', 1, 3)
    # learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # gamma = trial.suggest_float('gamma', 0.85, 0.99)
    # step_size = trial.suggest_int('step_size', 10, 100)
    # dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)

    

    X_train_selected = X_train_tensor
    X_test_selected = X_test_tensor
    

    X_train_selected_gpu = X_train_selected.float().to(device)
    X_test_selected_gpu = X_test_selected.float().to(device)
    y_train_tensor_gpu = y_train_tensor.long().to(device)
    y_test_tensor_gpu = y_test_tensor.long().to(device)
    
    out_dims = len(np.unique(y_train_tensor.cpu().numpy()))

    

    model = BiLSTMClassifierWithAttention(input_dim=X_train_selected.shape[-1], hidden_dim=hidden_dim, num_layers=num_layers, output_dim=out_dims, dropout_rate=dropout_rate).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


    calmar = 0
   
    model.train()
    for epoch in range(1, num_epochs+1):
        optimiser.zero_grad()   
        output = model(X_train_selected_gpu)
        output = torch.squeeze(output) 
        loss = criterion(output, y_train_tensor_gpu)  
        loss.backward()
        optimiser.step()
        scheduler.step()
        
        if epoch % 2 == 0:
            financial_results = validate_financials(model, X_test_selected_gpu)
            calmar_res = round(financial_results['calmer_returns'], 2)
            # print(f"Calmar: {calmar_res}, Epoch: {epoch}")
            if (calmar_res > calmar) and financial_results['orders'] > 10:
                calmar = calmar_res
                if calmar_res > 80 and calmar_res > highest_calmar:
                    highest_calmar = calmar_res
                    highest_indicies = selected_indices
                    # save_model(model, selected_indices, 'ADA')
                    print(f"calmar: {calmar_res}")
                    print("\033[93m Plotting.... \033[0m")
                    financial_results['plot'].show()


    


    if (financial_results['orders'] > 4):
        del financial_results, X_train_selected_gpu, X_test_selected_gpu, y_train_tensor_gpu, y_test_tensor_gpu, 
        del X_train_list,X_test_list,y_train_seq_ar,y_test_seq_ar,y_train_seq,y_test_seq,x_train_ar,x_test_ar,
        del X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor, X_train, X_test, y_train, y_test, indices_train, indices_test, model
        del scaler, X_train_scaled, X_test_scaled, X_train_selected, X_test_selected
        del class_weights_tensor, class_weights, unique_labels, y_train_seq_np
        del criterion, optimiser, scheduler, data, df_candles, df, X, y
        torch.cuda.empty_cache()  
        gc.collect()  # Explicitly invoke garbage collector
        return calmar
    else:
        print(f"Results were no good. Orders: {financial_results['orders']}, calmer_return: {financial_results['calmer_returns']}")
        del financial_results, X_train_selected_gpu, X_test_selected_gpu, y_train_tensor_gpu, y_test_tensor_gpu, 
        del X_train_list,X_test_list,y_train_seq_ar,y_test_seq_ar,y_train_seq,y_test_seq,x_train_ar,x_test_ar,
        del X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor, X_train, X_test, y_train, y_test, indices_train, indices_test, model
        del scaler, X_train_scaled, X_test_scaled, X_train_selected, X_test_selected
        del class_weights_tensor, class_weights, unique_labels, y_train_seq_np
        del criterion, optimiser, scheduler, data, df_candles, df, X, y
        torch.cuda.empty_cache()
        gc.collect()  # Explicitly invoke garbage collector  
        return 0



# main_indicies = [5, 6, 8, 12, 141, 14, 15, 16, 38, 41, 43, 46, 55, 56, 58, 60, 74, 76, 103, 107, 109, 121]
# main_indicies = [5, 6, 103, 8, 41, 74, 107, 12, 109, 46, 14, 15, 16, 76, 55, 56, 121, 58, 60, 141, 43]
main_indicies = [5, 6, 8, 12, 141, 14, 15, 16, 38, 41, 43, 46, 55, 56, 58, 60, 74, 76, 103, 107, 109, 121]
# main_indicies = [5, 6, 8, 12, 141, 14, 15, 16, 32, 38, 41, 43, 46, 55, 56, 58, 60, 69, 74, 76, 88, 103, 107, 109, 121]


import gc

for i in range(1):
    print(f"Study number: {i+1}")
    print(f"Main indicies: {main_indicies}")
    # num_trials = 10
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_trials)
    print('Best trial:', study.best_trial.params)
    print('Best trial number:', study.best_trial.number)
    print('Best indicator:', list(feature_indicators.keys())[study.best_trial.number])
    
    best_trial_value = study.best_trial.value
    print(f"Best trial value: {best_trial_value}")

    if best_trial_value > highest_calmar:
        highest_calmar = best_trial_value
        highest_indicies = main_indicies

    all_trials = study.get_trials()

    sorted_trials = sorted((trial for trial in all_trials if trial.value is not None), key=lambda trial: trial.value, reverse=True)

    for trial in sorted_trials:
        print(f'Trial number: {trial.number}, Score: {trial.value}, Params: {trial.params}')
    
    if study.best_trial.number not in main_indicies:
        main_indicies.append(study.best_trial.number)
    else:
        for trial in sorted_trials:
            if trial.number not in main_indicies:
                main_indicies.append(trial.number)
                break
    
    print(f"Highest clamar: {highest_calmar}")
    print(f"Highest indicies: {highest_indicies}")
    
    torch.cuda.empty_cache()
    del study  # Free up memory by deleting the study object
    gc.collect()  # Explicitly invoke garbage collector




