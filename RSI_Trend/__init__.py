from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils
from jesse.research import backtest, get_candles
import jesse.helpers as jh
import numpy as np

exchange_name = 'Bybit USDT Perpetual'
symbol = 'SOL-USDT'
timeframe = '1m'
train_start_date = '2024-06-01'
train_end_date = '2024-08-19'


class RSI_Trend(Strategy):
    def hyperparameters(self):
        return [
            {'name': 'rsi', 'type': int, 'min': 10, 'max': 30, 'default': 5},
            {'name': 'stop_loss', 'type': float, 'min': .5, 'max': .99, 'default': .95},
            {'name': 'take_profit', 'type': float, 'min': 1.1, 'max': 1.2, 'default': 1.1},
            {'name': 'xparam', 'type': int, 'min': 60, 'max': 90, 'default': 75}
        ]

    @property
    def rsi(self):
        return ta.rsi(self.candles, self.hp['rsi'], sequential=True)

    def should_long(self):
        qty = utils.size_to_qty(self.balance, self.price, 3, fee_rate=self.fee_rate)

        if utils.crossed(self.rsi, 35, direction="above") and qty > 0 and self.available_margin > (qty * self.price):
            return True

    def should_short(self):
        qty = utils.size_to_qty(self.balance, self.price, 3, fee_rate=self.fee_rate)

        if utils.crossed(self.rsi, 65, direction="below") and qty > 0 and self.available_margin > (qty * self.price):
            return True

    def should_cancel_entry(self):
        return False

    def go_long(self):
        qty = utils.size_to_qty(self.balance, self.price, 3, fee_rate=self.fee_rate)
        self.buy = qty, self.price

    def go_short(self):
        qty = utils.size_to_qty(self.balance, self.price, 3, fee_rate=self.fee_rate)
        self.sell = qty, self.price

    def update_position(self):
        if utils.crossed(self.rsi, self.hp['xparam'], direction="below") or utils.crossed(self.rsi, 10, direction="below"):
            self.liquidate()
        elif utils.crossed(self.rsi, 65, direction="below") and self.is_long:
            self.liquidate()
        elif utils.crossed(self.rsi, 35, direction="above") and self.is_short:
            self.liquidate()

    def after(self):
        self.add_extra_line_chart(chart_name='RSI', title='RSI', value=ta.rsi(self.candles))
