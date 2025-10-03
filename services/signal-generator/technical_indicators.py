from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

@dataclass
class IndicatorResult:
    """Result from indicator calculation."""
    value: float
    signal: str
    confidence: float
    metadata: Dict = None

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_macd(prices: pd.Series) -> Dict:
    """Calculate MACD."""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return {
        'macd': macd.iloc[-1],
        'signal': signal.iloc[-1],
        'histogram': (macd - signal).iloc[-1]
    }