import numpy as np # type: ignore
import pandas as pd  # type: ignore
import random
import optuna # type: ignore
import math

from jesse.strategies import Strategy, cached # type: ignore
import jesse.indicators as ta # type: ignore
from jesse import utils # type: ignore
from jesse import research # type: ignore
import jesse.helpers as jh # type: ignore
from jesse.research import backtest, get_candles # type: ignore

from functions import process_data, save_model
from indicators import feature_indicators


import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore


from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore
from collections import defaultdict

import warnings
warnings.simplefilter("ignore", UserWarning)


import joblib # type: ignore
import sys

from strategies.neural_nets.lstm_model import BiLSTMClassifierWithAttention

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)



# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

exchange = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
train_start_date = '2024-08-01'
train_end_date = '2024-09-08'
timeframe = '1m'
coin = 'SOL'
timestep = 35
window_size = 10


train_start_date_timestamp = jh.date_to_timestamp(train_start_date)
train_end_date_timestamp = jh.date_to_timestamp(train_end_date)


candle_data = get_candles(exchange, symbol, timeframe, train_start_date_timestamp, train_end_date_timestamp)
backtest_candles = candle_data[1]

total_candles = len(candle_data)
split_index = int(total_candles * 0.7)
testing_candles = backtest_candles[split_index:]

# main_indicies = [0, 98, 4, 5, 6, 75, 45, 46, 111, 14, 18, 84, 87, 55, 126]
main_indicies = [0, 1, 98, 4, 5, 6, 137, 75, 45, 46, 111, 14, 18, 84, 55, 87, 124, 126]



class lstm_xtrade_long_term_bt(Strategy):
    def __init__(self):
        super().__init__()
        self.model = BiLSTMClassifierWithAttention(input_dim, hidden_dim, num_layers, 3, dropout_rate)
        model_filename = '/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/models/SOL_TOP_trained_model_lstm_index_1.pth'
        if model_filename:
            self.model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
            self.model.eval()
        else:
            raise ValueError("Model filename not provided")
        scaler_params_filename = '/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/models/SOL_scaler_params_index_1.joblib'
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


        base_features = [opens, highs, lows, closes, volumes, ret1, ret5, ret10, ret20, ret40, std5, std10, std20, std40]

        selected_features = []


        for index in selected_indices:
            key, indicator_func = get_indicator_key_value(feature_indicators, index)
            indicator_value = indicator_func(np_candles)
            selected_features.append(indicator_value)



        all_features = base_features + selected_features

        all_features = [f if isinstance(f, np.ndarray) else np.array(f) for f in all_features]

        features = np.column_stack(all_features)
        features = features[~np.isnan(features).any(axis=1)]

        if features.shape[0] < timestep:
            return None

        recent_features = features[-timestep:]
        

        mean = np.array(self.scaler_params['mean'])
        scale = np.array(self.scaler_params['scale'])

        # Reshape if necessary
        if mean.shape[0] != recent_features.shape[1]:
            raise ValueError(f"Scaler mean shape {mean.shape} does not match feature shape {recent_features.shape[1]}")

        # Scale features
        scaled_features = (recent_features - mean) / scale
        feature_tensor = torch.tensor(scaled_features, dtype=torch.float).unsqueeze(0)

        return feature_tensor



    
    def should_long(self) -> bool:
        features = self.generate_features()
        # Assuming features is the input to your model
        # print(f"Features shape: {features.shape if features is not None else 'None'}")
        
        if features is None:
            # print("Error: Features are None, cannot proceed with the model prediction.")
            return False  # Prevent further execution
        
        try:
            prediction = self.model(features)
        except Exception as e:
            # print(f"Error during model prediction: {e}")
            return False  # Handle the exception gracefully
        
        
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




# num_trials = len(feature_indicators.keys())
num_trials = 1

highest_calmar = 0
highest_indicies = []

def get_indicator_key_value(feature_indicators, index):
    key = list(feature_indicators.keys())[index]
    value = list(feature_indicators.values())[index]
    return key, value

def objective(trial):
    
    global highest_calmar
    global highest_indicies

    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    

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
    
        
        
    # if trial.number in main_indicies:
    #     raise optuna.exceptions.TrialPruned()
    
    global selected_indices
    # selected_indices = main_indicies + [index]
    selected_indices = main_indicies
    selected_indices = list(set(selected_indices))
    
    print(f"Selected indicies: {selected_indices}")

    
    
    original_feature_map = {}
    for i in selected_indices:
        key, value = get_indicator_key_value(feature_indicators, i)
        df[key] = value(backtest_candles)
        original_feature_map[key] = i 


    data = df.copy()





    data = data.dropna(axis=1, how='all')
    data = data.dropna()


    remaining_columns = set(data.columns)


    updated_selected_indices = [
        original_feature_map[col] for col in remaining_columns if col in original_feature_map
    ]


    selected_indices = updated_selected_indices


    predictor_list = data.drop('signal', axis=1).columns.tolist()
    print(f"Predictor list: {predictor_list}")
    X = data[predictor_list]
    y = data['signal']


    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    scaler_params = {'mean': scaler.mean_, 'scale': scaler.scale_}
    joblib.dump(scaler_params, f"/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/models/{coin}_scaler_params_index_1.joblib")



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
    global hidden_dim
    hidden_dim = 32
    global num_layers
    num_layers = 2
    learning_rate= 0.1
    gamma=0.9
    step_size=10
    global dropout_rate
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

    global input_dim
    input_dim=X_train_selected.shape[-1]

    model = BiLSTMClassifierWithAttention(input_dim=X_train_selected.shape[-1], hidden_dim=hidden_dim, num_layers=num_layers, output_dim=out_dims, dropout_rate=dropout_rate).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


    
    
    # Backtest configuration
    def btest():
        config = {
            'starting_balance': 10_000,
            'fee': 0,
            'type': 'futures',
            'futures_leverage': 2,
            'futures_leverage_mode': 'cross',
            'exchange': exchange,
            'warm_up_candles': 2000
        }

        routes = [
            {'exchange': exchange, 'strategy': lstm_xtrade_long_term_bt, 'symbol': symbol, 'timeframe': '1h'}
        ]

        extra_routes = []

        candles = {
            jh.key(exchange, symbol): {
                'exchange': exchange,
                'symbol': symbol,
                'candles': testing_candles,
            },
        }

        # Execute backtest
        result = backtest(
            config,
            routes,
            extra_routes,
            candles
        )
        return result['metrics']
    
    
    calmar = 0
    highest_trades = 0
    
    model.train()
    print("model training...")
    for epoch in range(1, num_epochs+1):
        optimiser.zero_grad()   
        output = model(X_train_selected_gpu)
        output = torch.squeeze(output) 
        loss = criterion(output, y_train_tensor_gpu)  
        loss.backward()
        optimiser.step()
        scheduler.step()
        
        if epoch % 2 == 0:
            save_model(model, 1, coin)
            financial_results = btest()
            trades = financial_results['total']
            net_profit_percentage = financial_results['net_profit_percentage']
            calmar_res = round(financial_results.get('calmar_ratio', 0), 2)
            print(f"Calmar: {calmar_res}, Profit: {round(net_profit_percentage, 2)}%, Trades: {trades}, Epoch: {epoch}")
            if math.isinf(calmar_res):
                calmar_res = 0
            if (calmar_res > calmar) and trades > 5:
                calmar = calmar_res
                if calmar_res > 10 and calmar_res > highest_calmar:
                    highest_calmar = calmar_res
                    highest_indicies = selected_indices
                    print(f"\033[93m highest calmar: {calmar_res}\033[0m")
                if trades > highest_trades:
                    highest_trades = trades




    def cleanup_resources(variables):
        """Helper function to delete variables and clear cache."""
        for var in variables:
            if var in globals():
                del globals()[var]
            elif var in locals():
                del locals()[var]
        torch.cuda.empty_cache()
        gc.collect()

    cleanup_vars = [
        'financial_results', 'X_train_selected_gpu', 'X_test_selected_gpu', 'y_train_tensor_gpu', 'y_test_tensor_gpu',
        'X_train_list', 'X_test_list', 'y_train_seq_ar', 'y_test_seq_ar', 'y_train_seq', 'y_test_seq', 'x_train_ar', 'x_test_ar',
        'X_train_tensor', 'X_test_tensor', 'y_train_tensor', 'y_test_tensor', 'X_train', 'X_test', 'y_train', 'y_test', 'model',
        'scaler', 'X_train_scaled', 'X_test_scaled', 'X_train_selected', 'X_test_selected',
        'class_weights_tensor', 'class_weights', 'unique_labels', 'y_train_seq_np',
        'criterion', 'optimiser', 'scheduler', 'data', 'df_candles', 'df', 'X', 'y'
    ]

    #fix this
    if highest_trades > 4:
        cleanup_resources(cleanup_vars)
        return calmar
    else:
        print(f"Results were no good. Orders: {trades}, calmar_ratio: {calmar_res}")
        cleanup_resources(cleanup_vars)
        return 0










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
    print(f"Main indicies: {main_indicies}")
    
    torch.cuda.empty_cache()
    del study  # Free up memory by deleting the study object
    gc.collect()  # Explicitly invoke garbage collector




