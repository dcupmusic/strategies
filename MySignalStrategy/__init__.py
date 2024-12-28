from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils
import pandas as pd
from jesse.research import backtest, get_candles
import jesse.helpers as jh


df = pd.read_csv('/Users/duncanmaclennan/Documents/Jesse/trade/my-bot/df_target.csv')
print(f"length of df: {len(df)}")


class MySignalStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.current_index = 0

    def should_long(self):
        if df['signal'][self.current_index] == 2:
            # print(f"long signal at index: {self.current_index}")
            return True

    def should_short(self):
        if df['signal'][self.current_index] == 0:
            # print(f"short signal at index: {self.current_index}")
            return True


    def go_long(self):
        qty = utils.size_to_qty(2000, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(2000, self.price, fee_rate=self.fee_rate)
        self.sell = qty, self.price

    def update_position(self):
        if self.is_long and df['signal'][self.current_index] == 0:
            self.liquidate()
        if self.is_short and df['signal'][self.current_index] == 2:
            self.liquidate()

    def after(self):
        self.current_index += 1


# exchange = 'Bybit USDT Perpetual'
# symbol = 'SOL-USDT'
# train_start_date = '2024-08-01'
# train_end_date = '2024-09-08'
# timeframe = '1m'
# coin = 'SOL'

# train_start_date_timestamp = jh.date_to_timestamp(train_start_date)
# train_end_date_timestamp = jh.date_to_timestamp(train_end_date)

# candle_data = get_candles(exchange, symbol, timeframe, train_start_date_timestamp, train_end_date_timestamp)

# backtest_candles = candle_data[1]

# exchange_name = 'Bybit USDT Perpetual'

# config = {
#     'starting_balance': 10_000,
#     'fee': 0,
#     'type': 'futures',
#     'futures_leverage': 1,
#     'futures_leverage_mode': 'cross',
#     'exchange': exchange_name,
#     'warm_up_candles': 0
# }
# routes = [
#     {'exchange': exchange_name, 'strategy': MySignalStrategy, 'symbol': symbol, 'timeframe': '1h'}
# ]
# extra_routes = []
# candles = {
#     jh.key(exchange_name, symbol): {
#         'exchange': exchange_name,
#         'symbol': symbol,
#         'candles': backtest_candles,
#     },
# }

# result = backtest(
#     config,
#     routes,
#     extra_routes,
#     candles
# )

# print(result['metrics'])

# result['logs']
