from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils
from jesse.research import backtest, get_candles
import jesse.helpers as jh

# generate fake candles
exchange_name = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
timeframe = '1m'
train_start_date = '2024-06-01'
train_end_date = '2024-08-19'

train_start_date_timestamp = jh.date_to_timestamp(train_start_date)
train_end_date_timestamp = jh.date_to_timestamp(train_end_date)

candle_data = get_candles(exchange_name, symbol, timeframe, train_start_date_timestamp, train_end_date_timestamp)   
backtest_candles = candle_data[1]

# strategy
class backtest_demo(Strategy):
    def should_long(self):
        return True

    def should_short(self):
        return False

    def should_cancel_entry(self):
        return False

    def go_long(self):
        entry_price = self.price
        qty = utils.size_to_qty(self.balance * 0.5, entry_price)
        self.buy = qty, entry_price

    def go_short(self):
        pass

# prepare inputs
exchange_name = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
timeframe = '1h'
config = {
    'starting_balance': 10_000,
    'fee': 0,
    'type': 'futures',
    'futures_leverage': 2,
    'futures_leverage_mode': 'cross',
    'exchange': exchange_name,
    'warm_up_candles': 0
}
routes = [
    {'exchange': exchange_name, 'strategy': backtest_demo, 'symbol': symbol, 'timeframe': timeframe}
]
extra_routes = []
candles = {
    jh.key(exchange_name, symbol): {
        'exchange': exchange_name,
        'symbol': symbol,
        'candles': backtest_candles,
    },
}

# execute backtest
result = backtest(
    config,
    routes,
    extra_routes,
    candles,
    generate_charts=True
)
# access the metrics dict:
result['metrics']
# access the charts string (path of the generated file):
result['charts']
# access the logs list:
result['logs']
