from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils

class ExampleStrategy(Strategy):
    def should_long(self) -> bool:
        return False

    def should_short(self) -> bool:
        return False

    def should_cancel_entry(self) -> bool:
        return False

    def go_long(self):
        pass

    def go_short(self):
        pass
    
    @property
    def short_trend(self):
        return ta.ema(self.candles, period=8)
