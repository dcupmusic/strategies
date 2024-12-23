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


class RSI_Trend(Strategy):
    def hyperparameters(self):
        return [
                {'name':'rsi', 'type': int, 'min': 10, 'max':30, 'default': 5},
                {'name':'stop_loss', 'type': float, 'min': .5, 'max': .99, 'default': .95},
                {'name':'take_profit', 'type': float, 'min': 1.1, 'max': 1.2, 'default': 1.1},
                {'name':'xparam', 'type':int, 'min': 60, 'max': 90, 'default': 75}
        ]

    @property
    def rsi(self):
        return ta.rsi(self.candles, self.hp['rsi'], sequential=True)

    def should_long(self):
        qty = utils.size_to_qty(self.balance, self.price, 3, fee_rate=self.fee_rate) 

        if utils.crossed(self.rsi, 35, direction="above") and qty > 0 and self.available_margin > (qty * self.price):
            return True

    def should_short(self):
        return False

    def should_cancel_entry(self):
        return False

    def go_long(self):
        qty = utils.size_to_qty(self.balance, self.price, 3, fee_rate=self.fee_rate) 
        self.buy = qty, self.price
        self.stop_loss = qty, (self.price * self.hp['stop_loss'])        # Willing to lose 5%
        self.take_profit = qty, (self.price * self.hp['take_profit'])     # Take profits at 10%

    def go_short(self):
        pass

    def update_position(self):
        if utils.crossed(self.rsi, self.hp['xparam'], direction="below") or utils.crossed(self.rsi, 10, direction="below"):
            self.liquidate()
            



config = {
    'starting_balance': 10_000,
    'fee': 0,
    'type': 'futures',
    'futures_leverage': 2,
    'futures_leverage_mode': 'cross',
    'exchange': exchange_name,
    'warm_up_candles': 500000
}
routes = [
    {'exchange': exchange_name, 'strategy': RSI_Trend, 'symbol': symbol, 'timeframe': '1h'}
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
print(result['metrics'])
# Print the path of the generated chart
print(result['charts'])
