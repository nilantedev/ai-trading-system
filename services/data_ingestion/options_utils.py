"""Options utilities for constructing provider-specific tickers and parsing.

Currently supports Polygon "O:" option tickers. Example:
  underlying: SPY, expiry: 2025-09-19, type: C, strike: 450 -> O:SPY250919C00450000

Strike encoding: dollars*1000 padded to 8 digits per Polygon convention.
"""
from __future__ import annotations

from datetime import datetime


def polygon_option_ticker(underlying: str, expiry: datetime, right: str, strike: float) -> str:
    u = underlying.upper()
    yy = expiry.strftime('%y')
    mm = expiry.strftime('%m')
    dd = expiry.strftime('%d')
    r = right.upper()[0]  # C or P
    # strike: encode as int of price * 1000, left pad to 8
    strike_milli = int(round(strike * 1000))
    strike_str = str(strike_milli).rjust(8, '0')
    return f"O:{u}{yy}{mm}{dd}{r}{strike_str}"


__all__ = ["polygon_option_ticker"]
