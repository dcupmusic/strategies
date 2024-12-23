# backtest_lstm_xtrade.py

import jesse.helpers as jh
from jesse.research import backtest, get_candles
from strategies.lstm_xtrade_long_term_bt import lstm_xtrade_long_term_bt

# Backtesting configuration
exchange_name = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
timeframe = '1m'
train_start_date = '2024-08-11'
train_end_date = '2024-08-19'

# Convert date strings to timestamps
train_start_date_timestamp = jh.date_to_timestamp(train_start_date)
train_end_date_timestamp = jh.date_to_timestamp(train_end_date)

# Prepare candle data
candle_data = get_candles(exchange_name, symbol, timeframe, train_start_date_timestamp, train_end_date_timestamp)   
backtest_candles = candle_data[1]

# Backtest configuration
config = {
    'starting_balance': 10_000,
    'fee': 0,
    'type': 'futures',
    'futures_leverage': 2,
    'futures_leverage_mode': 'cross',
    'exchange': exchange_name,
    'warm_up_candles': 2000
}

routes = [
    {'exchange': exchange_name, 'strategy': lstm_xtrade_long_term_bt, 'symbol': symbol, 'timeframe': '1h'}
]

extra_routes = []

candles = {
    jh.key(exchange_name, symbol): {
        'exchange': exchange_name,
        'symbol': symbol,
        'candles': backtest_candles,
    },
}

# Execute backtest
result = backtest(
    config,
    routes,
    extra_routes,
    candles,
)

# Print the metrics
print(result['metrics'])

