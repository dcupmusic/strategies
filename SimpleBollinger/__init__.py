from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils
from jesse.research import backtest, get_candles
import jesse.helpers as jh


exchange_name = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
timeframe = '1m'
train_start_date = '2024-06-01'
train_end_date = '2024-08-19'

train_start_date_timestamp = jh.date_to_timestamp(train_start_date)
train_end_date_timestamp = jh.date_to_timestamp(train_end_date)

candle_data = get_candles(exchange_name, symbol, timeframe, train_start_date_timestamp, train_end_date_timestamp)   
backtest_candles = candle_data[1]

class SimpleBollinger(Strategy):
    @property
    def bb(self):
        return ta.bollinger_bands(self.candles, source_type="hl2")

    @property
    def ichimoku(self):
        return ta.ichimoku_cloud(self.candles)

    def filter_trend(self):
        return self.close > self.ichimoku.span_a and self.close > self.ichimoku.span_b

    def filters(self):
        return [self.filter_trend]

    def should_long(self) -> bool:
        return self.close > self.bb[0]

    def should_short(self) -> bool:
        return False

    def should_cancel_entry(self) -> bool:
        return True

    def go_long(self):
        qty = utils.size_to_qty(self.balance, self.price, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        pass

    def update_position(self):
        if self.close < self.bb[1]:
            self.liquidate()
            
            
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
    {'exchange': exchange_name, 'strategy': SimpleBollinger, 'symbol': symbol, 'timeframe': '1h'}
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
# Print the metrics
print(f"Calmar Ratio: ", result['metrics']['calmar_ratio'])
# Print the path of the generated chart
print(result['charts'])