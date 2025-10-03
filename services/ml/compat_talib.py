"""
TA-Lib compatibility shim implemented with the 'ta' pandas library.

This module exposes a subset of TA-Lib functions used by our ML service:
- RSI, EMA, SMA, MACD, STOCH, ROC
- BBANDS, ATR
- OBV, AD
- ADX, AROON
- MFI, CCI

Each function returns numpy arrays consistent with TA-Lib's return shapes
so existing code paths can continue to work without modification.

Implementation notes:
- Inputs are numpy arrays (float64); we wrap into pandas Series to use 'ta'.
- We align default parameters to TA-Lib defaults where applicable.
- We ensure outputs are numpy arrays and length-match inputs, padding with NaN
  where the indicator requires warmup periods, similar to TA-Lib behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

# Helpers
def _series(x: np.ndarray) -> pd.Series:
    return pd.Series(x, dtype="float64")

def _align(series: pd.Series, length: int) -> np.ndarray:
    s = series.reindex(range(length))
    return s.to_numpy(dtype=float)


# Momentum
def RSI(prices: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    s = _series(prices)
    r = ta.momentum.RSIIndicator(s, window=timeperiod).rsi()
    return _align(r, len(s))


def ROC(prices: np.ndarray, timeperiod: int = 10) -> np.ndarray:
    s = _series(prices)
    r = ta.momentum.ROCIndicator(s, window=timeperiod).roc()
    return _align(r, len(s))


def STOCH(high: np.ndarray, low: np.ndarray, close: np.ndarray,
          fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0,
          slowd_period: int = 3, slowd_matype: int = 0):
    hs, ls, cs = _series(high), _series(low), _series(close)
    stoch = ta.momentum.StochasticOscillator(
        high=hs, low=ls, close=cs,
        window=fastk_period, smooth_window=slowk_period
    )
    k = _align(stoch.stoch(), len(cs))
    d = _align(stoch.stoch_signal(), len(cs))
    return k, d


# Trend
def EMA(prices: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    s = _series(prices)
    r = ta.trend.ema_indicator(s, window=timeperiod)
    return _align(r, len(s))


def SMA(prices: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    s = _series(prices)
    r = ta.trend.sma_indicator(s, window=timeperiod)
    return _align(r, len(s))


def MACD(prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
    s = _series(prices)
    macd = ta.trend.MACD(s, window_fast=fastperiod, window_slow=slowperiod, window_sign=signalperiod)
    macd_line = _align(macd.macd(), len(s))
    macd_signal = _align(macd.macd_signal(), len(s))
    macd_hist = _align(macd.macd_diff(), len(s))
    return macd_line, macd_signal, macd_hist


def ADX(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    hs, ls, cs = _series(high), _series(low), _series(close)
    r = ta.trend.ADXIndicator(hs, ls, cs, window=timeperiod).adx()
    return _align(r, len(cs))


def AROON(high: np.ndarray, low: np.ndarray, timeperiod: int = 25):
    hs, ls = _series(high), _series(low)
    a = ta.trend.AroonIndicator(hs, ls, window=timeperiod)
    up = _align(a.aroon_up(), len(hs))
    down = _align(a.aroon_down(), len(hs))
    return up, down


# Volatility
def BBANDS(prices: np.ndarray, timeperiod: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0, matype: int = 0):
    s = _series(prices)
    bb = ta.volatility.BollingerBands(s, window=timeperiod, window_dev=nbdevup)
    upper = _align(bb.bollinger_hband(), len(s))
    middle = _align(bb.bollinger_mavg(), len(s))
    lower = _align(bb.bollinger_lband(), len(s))
    return upper, middle, lower


def ATR(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    hs, ls, cs = _series(high), _series(low), _series(close)
    r = ta.volatility.AverageTrueRange(hs, ls, cs, window=timeperiod).average_true_range()
    return _align(r, len(cs))


# Volume
def OBV(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    cs, vs = _series(close), _series(volume)
    r = ta.volume.OnBalanceVolumeIndicator(cs, vs).on_balance_volume()
    return _align(r, len(cs))


def AD(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    hs, ls, cs, vs = _series(high), _series(low), _series(close), _series(volume)
    r = ta.volume.AccDistIndexIndicator(hs, ls, cs, vs).acc_dist_index()
    return _align(r, len(cs))


# Others
def MFI(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    hs, ls, cs, vs = _series(high), _series(low), _series(close), _series(volume)
    r = ta.volume.MFIIndicator(hs, ls, cs, vs, window=timeperiod).money_flow_index()
    return _align(r, len(cs))


def CCI(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 20) -> np.ndarray:
    hs, ls, cs = _series(high), _series(low), _series(close)
    r = ta.trend.CCIIndicator(hs, ls, cs, window=timeperiod).cci()
    return _align(r, len(cs))
