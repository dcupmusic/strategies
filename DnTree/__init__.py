from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
import pandas as pd
from jesse import research
import jesse.indicators as ta
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import jesse.helpers as jh
from jesse.research import backtest
import optuna
import os
from functions import process_data

sol_candles = research.get_candles('Bybit USDT Perpetual', 'SOL-USDT', '1m', '2023-06-01', '2024-03-19')


def objective(trial):
    # max_depth = trial.suggest_int('max_depth', 2, 32)
    rsi_exit = 65 # trial.suggest_int('rsi_exit', 65, 85)
    exchange = 'Bybit USDT Perpetual'
    symbol = 'SOL-USDT'
    timeframe = '1h'
    start_date = '2023-06-01'
    end_date = '2024-03-19'
   
    candles = research.get_candles(exchange, symbol, timeframe, start_date, end_date)  
    
    feature_indicators = {
        'RSI': ta.rsi(candles, 8, sequential=True),
        'EMA': ta.ema(candles, period=10, sequential=True),
        'STDDEV': ta.stddev(candles, 10, 1, sequential=True)
    }

    feature_names = list(feature_indicators.keys())
    
    included_features_flags = [trial.suggest_categorical(f'include_feature_{i}', [True, False]) for i in range(len(feature_names))]
    included_features = [feature_names[i] for i, flag in enumerate(included_features_flags) if flag]

    print(included_features)
    if len(included_features) < 1:
        print(f"No features were selected for trial {trial.number}. Skipping...")
        return 0
    
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])   
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')   
    
    for feature in included_features:
        df[feature] = feature_indicators[feature]
        

    df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
    df = process_data(df, 10, 'SOL')
    df = df.dropna()
    
    
    X = df[['close'] + included_features]
    y = df['signal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    class DnTree(Strategy):
        def __init__(self):
            super().__init__()
            model_filename = os.getenv('MODEL_FILENAME')
            if model_filename:
                self.model = joblib.load(model_filename)
            else:
                raise ValueError("Model filename not provided")
            
        @property
        def rsi(self):
            return ta.rsi(self.candles, 8)
        @property
        def ema(self):
            return ta.ema(self.candles, period=10)
        @property
        def stddev(self):
            return ta.stddev(self.candles, period=10, nbdev=1)
        
        def generate_features(self):
            last_candle = self.candles[-1]
            close_price = last_candle[2]
            gend_features = [close_price]
            feature_methods = {
            'RSI': self.rsi,
            'EMA': self.ema,
            'STDDEV': self.stddev,
            }
            for feature in included_features:
                if feature in feature_methods:
                    gend_features.append(feature_methods[feature])
            
            return gend_features
        
        def should_long(self) -> bool:
            features = self.generate_features()
            signal = self.model.predict([features])
            return signal == 1 

        def should_short(self) -> bool:
            pass
        
        def should_cancel_entry(self) -> bool:
            return False
        
        def go_long(self):
            qty = utils.size_to_qty(self.balance, self.price, fee_rate=self.fee_rate)
            self.buy = qty, self.price

        def go_short(self):
            pass
            
        def update_position(self):
            features = self.generate_features()
            signal = self.model.predict([features]) 
            if self.is_long and signal == 0:
                self.liquidate()


    def run_backtest(model_filename):
        
        os.environ['MODEL_FILENAME'] = model_filename
        
        exchange_name = 'Bybit USDT Perpetual'
        symbol = 'SOL-USDT'
        timeframe = '1h'
        config = {
            'starting_balance': 10_000,
            'fee': 0,
            'type': 'futures',
            'futures_leverage': 75,
            'futures_leverage_mode': 'cross',
            'exchange': exchange_name,
            'warm_up_candles': 210
        }
        routes = [
            {'exchange': exchange_name, 
            'strategy': DnTree, 
            'symbol': symbol, 
            'timeframe': timeframe}
        ]
        extra_routes = []
        candles = {
            jh.key(exchange_name, symbol): {
                'exchange': exchange_name,
                'symbol': symbol,
                'candles': sol_candles,
            },
        }
        result = backtest(
            config,
            routes,
            extra_routes,
            candles, 
            generate_charts=True
        )

        del os.environ['MODEL_FILENAME']
        
        result['charts']
        result['logs']
        return round(result['metrics']['calmar_ratio'], 2)




    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)
    model_filename = f'model_temp_{trial.number}.joblib'
    joblib.dump(model, model_filename)    
    calmar_ratio = run_backtest(model_filename)    
    return calmar_ratio


if __name__ == "__main__":
    
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=4)

    print('Best trial:', study.best_trial.params)



