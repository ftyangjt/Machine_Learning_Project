import backtrader as bt
import pandas as pd
import numpy as np

class MovingAverageCross(bt.Strategy):
    params = (
        ('fast', 5),
        ('slow', 20),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow)

    def next(self):
        if not self.position:
            if self.fast_ma[0] > self.slow_ma[0]:
                self.order_target_percent(target=0.99)
        elif self.fast_ma[0] < self.slow_ma[0]:
            self.close()


class RSIReversion(bt.Strategy):
    params = (
        ('period', 14),
        ('low', 30.0),
        ('high', 70.0),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.period)

    def next(self):
        if not self.position:
            if self.rsi[0] < self.params.low:
                self.order_target_percent(target=0.99)
        else:
            if self.rsi[0] > self.params.high:
                self.close()


class BollingerBreakout(bt.Strategy):
    params = (
        ('period', 20),
        ('devfactor', 2.0),
    )

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(self.data.close, period=self.params.period, devfactor=self.params.devfactor)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.boll.lines.top[0]:
                self.order_target_percent(target=0.99)
        else:
            if self.data.close[0] < self.boll.lines.bot[0]:
                self.close()


class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.order_target_percent(target=0.99)


class MACDStrategy(bt.Strategy):
    params = (
        ('fast', 12),
        ('slow', 26),
        ('signal', 9),
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.fast,
            period_me2=self.params.slow,
            period_signal=self.params.signal
        )

    def next(self):
        if not self.position:
            if self.macd.macd[0] > self.macd.signal[0]:
                self.order_target_percent(target=0.99)
        else:
            if self.macd.macd[0] < self.macd.signal[0]:
                self.close()
