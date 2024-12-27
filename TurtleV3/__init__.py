from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils


class TurtleV3(Strategy):
    last_closing_index = 0

    @property
    def donchian(self):
        return ta.donchian(self.candles[:-1], period=self.hp['donchian_period'])

    def should_long(self) -> bool:
        return self.price > self.donchian.upperband and self.chop and self.passed_time and self.adx and self.long_term_ma == 1

    def go_short(self):
        entry = self.price
        stop = self.price + ta.atr(self.candles) * self.hp['stop_loss']
        qty = utils.risk_to_qty(self.available_margin, 3, entry, stop, fee_rate=self.fee_rate) * 2.2
        self.sell = qty, self.price

    def should_short(self) -> bool:
        return self.price < self.donchian.lowerband and self.chop and self.passed_time and self.adx and self.long_term_ma == -1

    def go_long(self):
        entry = self.price
        stop = self.price - ta.atr(self.candles) * self.hp['stop_loss']
        qty = utils.risk_to_qty(self.available_margin, 3, entry, stop, fee_rate=self.fee_rate) * 2.2
        self.buy = qty, self.price

    def should_cancel_entry(self) -> bool:
        return True

    def on_open_position(self, order) -> None:
        if self.is_long:
            self.stop_loss = self.position.qty, self.price - ta.atr(self.candles) * self.hp['stop_loss']
        elif self.is_short:
            self.stop_loss = self.position.qty, self.price + ta.atr(self.candles) * self.hp['stop_loss']

    def update_position(self) -> None:
        if self.is_long:
            self.stop_loss = self.position.qty, max(self.average_stop_loss, self.price - ta.atr(self.candles) * self.hp['stop_loss'])
        elif self.is_short:
            self.stop_loss = self.position.qty, min(self.average_stop_loss, self.price + ta.atr(self.candles) * self.hp['stop_loss'])

    @property
    def passed_time(self):
        return self.index - self.last_closing_index > 0

    @property
    def long_term_candles(self):
        return self.get_candles(self.exchange, self.symbol, '4h')

    @property
    def long_term_ma(self):
        if self.price > ta.ema(self.long_term_candles, self.hp['long_term_ma_period']):
            return 1
        else:
            return -1

    @property
    def adx(self):
        return ta.adx(self.candles) > self.hp['adx_threshold']

    @property
    def chop(self):
        return ta.chop(self.candles) < self.hp['chop_threshold']

    def on_close_position(self, order) -> None:
        self.last_closing_index = self.index

    def after(self) -> None:
        self.add_line_to_candle_chart('upperband', self.donchian.upperband)
        self.add_line_to_candle_chart('lowerband', self.donchian.lowerband)

    def hyperparameters(self) -> list:
        return [
            {'name': 'adx_threshold', 'type': int, 'min': 20, 'max': 60, 'default': 30},
            {'name': 'chop_threshold', 'type': int, 'min': 20, 'max': 60, 'default': 40},
            {'name': 'stop_loss', 'type': float, 'min': 1, 'max': 4, 'default': 2.5},
            {'name': 'long_term_ma_period', 'type': int, 'min': 100, 'max': 200, 'default': 200},
            {'name': 'donchian_period', 'type': int, 'min': 10, 'max': 40, 'default': 20},
        ]

    def dna(self) -> str:
        return ')R0[t'