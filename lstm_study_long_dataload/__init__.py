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
train_start_date = '2023-08-01'
train_end_date = '2024-04-17'
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

import torch
from torch.utils.data import Dataset, DataLoader

class FinancialDataset(Dataset):
    def __init__(self, X, y, timestep):
        self.X = X
        self.y = y
        self.timestep = timestep
        self.sequences = self.create_sequences(X, y)

    def create_sequences(self, input_data, targets):
        sequences = []
        data_len = len(input_data)
        for i in range(data_len - self.timestep):
            seq = input_data[i:(i + self.timestep)]
            label = targets[i + self.timestep]
            sequences.append((seq, label))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]



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



from sklearn.utils.class_weight import compute_class_weight



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
    
    print(f"Selected indices: {selected_indices}")
    
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
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)  # Ensure float32 type
    X_test_scaled = scaler.transform(X_test).astype(np.float32)  # Ensure float32 type

    # Check for NaN or infinite values
    assert not np.any(np.isnan(X_train_scaled)), "Training data contains NaN values."
    assert not np.any(np.isinf(X_train_scaled)), "Training data contains infinite values."
    assert not np.any(np.isnan(X_test_scaled)), "Test data contains NaN values."
    assert not np.any(np.isinf(X_test_scaled)), "Test data contains infinite values."

    unique_labels = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_train)
    
    if 0 not in unique_labels or 1 not in unique_labels or 2 not in unique_labels:
        print(f"Not all classes are present in the training data. Skipping trial.")
        raise optuna.exceptions.TrialPruned()

    index_of_class_short = np.where(unique_labels == 0)[0][0]
    index_of_class_none = np.where(unique_labels == 1)[0][0]
    index_of_class_long = np.where(unique_labels == 2)[0][0]

    class_weights[index_of_class_short] *= 1.9
    class_weights[index_of_class_none] *= 0
    class_weights[index_of_class_long] *= 0.9

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    train_dataset = FinancialDataset(X_train_scaled, y_train, timestep)
    test_dataset = FinancialDataset(X_test_scaled, y_test, timestep)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def validate_financials(model, data_loader):
        model.eval()
        all_predicted_labels = []
        all_indices_test = []

        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                probabilities = torch.softmax(y_test_pred, dim=1)
                _, predicted_labels = torch.max(probabilities, 1)
                all_predicted_labels.extend(predicted_labels.cpu().numpy())
                all_indices_test.extend(range(len(predicted_labels)))

        adjusted_indices_test = np.array(all_indices_test) + timestep
        df_split = data.data['symbol'].iloc[adjusted_indices_test].copy()
        df_split.loc[:, "signal"] = all_predicted_labels
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
        model.train()
        return {
            "orders": orders,
            "calmer_returns": (calmer_returns),
            "plot": plot
        }

    num_epochs = 90
    hidden_dim = 32
    num_layers = 2
    learning_rate = 0.01  # Reduced learning rate
    gamma = 0.9
    step_size = 10
    dropout_rate = 0.1

    model = BiLSTMClassifierWithAttention(input_dim=train_dataset[0][0].shape[1], hidden_dim=hidden_dim, num_layers=num_layers, output_dim=len(np.unique(y_train)), dropout_rate=dropout_rate).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    calmar = 0

    model.train()
    for epoch in range(1, num_epochs + 1):
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Debugging: Print the shape and a few examples of the batches
            if epoch == 1 and i < 2:  # Print only for the first two batches of the first epoch
                print(f"Batch {i} - X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")
                print(f"X_batch sample: {X_batch[0]}, y_batch sample: {y_batch[0]}")
            
            optimiser.zero_grad()
            output = model(X_batch)
            output = torch.squeeze(output)
            loss = criterion(output, y_batch)
            
            # Debugging: Print the loss value
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimiser.step()
        scheduler.step()
        
        if epoch % 2 == 0:
            print(f"Epoch: {epoch}")
            financial_results = validate_financials(model, test_loader)
            calmar_res = round(financial_results['calmer_returns'], 2)
            if (calmar_res > calmar) and financial_results['orders'] > 10:
                calmar = calmar_res
                if calmar_res > 80 and calmar_res > highest_calmar:
                    highest_calmar = calmar_res
                    highest_indicies = selected_indices
                    print(f"calmar: {calmar_res}")
                    print("\033[93m Plotting.... \033[0m")
                    financial_results['plot'].show()

    if financial_results['orders'] > 4:
        torch.cuda.empty_cache()
        return calmar
    else:
        print(f"Results were no good. Orders: {financial_results['orders']}, calmer_return: {financial_results['calmer_returns']}")
        torch.cuda.empty_cache()
        return 0







main_indicies = [96, 128, 101, 102, 104, 11, 45, 49, 18, 51, 83, 93, 89, 63, 57]


import gc

for i in range(5):
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




