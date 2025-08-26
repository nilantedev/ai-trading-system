#!/usr/bin/env python3
"""
Production-ready technical indicators using battle-tested ta library.
All approximations replaced with accurate implementations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import ta
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators using the 'ta' library for accuracy."""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given dataframe.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        
        Returns:
            DataFrame with original data plus all calculated indicators
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        result = df.copy()
        
        try:
            # Add all ta indicators at once (most efficient)
            result = ta.add_all_ta_features(
                result,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=True  # Handle NaN values
            )
        except Exception as e:
            logger.error(f"Error calculating indicators with ta library: {e}")
            # Fallback to individual calculations
            result = TechnicalIndicators._calculate_individual_indicators(result)
        
        return result
    
    @staticmethod
    def _calculate_individual_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators individually for more control."""
        
        # Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # ROC
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        
        # Volatility Indicators
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_percent'] = bollinger.bollinger_pband()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = keltner.keltner_channel_hband()
        df['kc_lower'] = keltner.keltner_channel_lband()
        df['kc_middle'] = keltner.keltner_channel_mband()
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_upper'] = donchian.donchian_channel_hband()
        df['dc_lower'] = donchian.donchian_channel_lband()
        df['dc_middle'] = donchian.donchian_channel_mband()
        
        # Volume Indicators
        
        # On Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()
        
        # Force Index
        df['fi'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()
        
        # Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()
        
        # VWAP
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            df['high'], df['low'], df['close'], df['volume']
        ).volume_weighted_average_price()
        
        # Additional Custom Indicators
        df = TechnicalIndicators._add_custom_indicators(df)
        
        return df
    
    @staticmethod
    def _add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators not in ta library."""
        
        # Price action patterns
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot_point'] - df['low']
        df['support_1'] = 2 * df['pivot_point'] - df['high']
        df['resistance_2'] = df['pivot_point'] + (df['high'] - df['low'])
        df['support_2'] = df['pivot_point'] - (df['high'] - df['low'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Market microstructure
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['close'] * 100
        
        # Efficiency Ratio
        df['efficiency_ratio'] = TechnicalIndicators._calculate_efficiency_ratio(df['close'])
        
        # Z-Score
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        return df
    
    @staticmethod
    def _calculate_efficiency_ratio(prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio."""
        change = abs(prices.diff(period))
        volatility = prices.diff().abs().rolling(period).sum()
        return change / volatility
    
    @staticmethod
    def calculate_indicator(df: pd.DataFrame, indicator_name: str, **params) -> pd.Series:
        """
        Calculate a specific indicator with custom parameters.
        
        Args:
            df: DataFrame with OHLCV data
            indicator_name: Name of the indicator to calculate
            **params: Parameters specific to the indicator
        
        Returns:
            Series with the calculated indicator
        """
        indicator_map = {
            'sma': lambda: ta.trend.sma_indicator(df['close'], window=params.get('period', 20)),
            'ema': lambda: ta.trend.ema_indicator(df['close'], window=params.get('period', 20)),
            'rsi': lambda: ta.momentum.RSIIndicator(df['close'], window=params.get('period', 14)).rsi(),
            'macd': lambda: ta.trend.MACD(df['close'], 
                                          window_slow=params.get('slow', 26),
                                          window_fast=params.get('fast', 12),
                                          window_sign=params.get('signal', 9)).macd(),
            'bb_upper': lambda: ta.volatility.BollingerBands(
                df['close'], 
                window=params.get('period', 20),
                window_dev=params.get('std', 2)
            ).bollinger_hband(),
            'bb_lower': lambda: ta.volatility.BollingerBands(
                df['close'],
                window=params.get('period', 20),
                window_dev=params.get('std', 2)
            ).bollinger_lband(),
            'atr': lambda: ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'],
                window=params.get('period', 14)
            ).average_true_range(),
            'adx': lambda: ta.trend.ADXIndicator(
                df['high'], df['low'], df['close'],
                window=params.get('period', 14)
            ).adx(),
            'obv': lambda: ta.volume.OnBalanceVolumeIndicator(
                df['close'], df['volume']
            ).on_balance_volume(),
            'mfi': lambda: ta.volume.MFIIndicator(
                df['high'], df['low'], df['close'], df['volume'],
                window=params.get('period', 14)
            ).money_flow_index(),
        }
        
        if indicator_name in indicator_map:
            return indicator_map[indicator_name]()
        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """
        Validate that dataframe has required columns for indicator calculation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns. Required: {required_columns}, Got: {df.columns.tolist()}")
            return False
        
        # Check for sufficient data
        if len(df) < 200:  # Need at least 200 bars for SMA 200
            logger.warning(f"Insufficient data for all indicators. Have {len(df)} rows, recommend at least 200")
        
        # Check for NaN values in critical columns
        if df[required_columns].isnull().any().any():
            logger.warning("DataFrame contains NaN values in required columns")
        
        return True
    
    @staticmethod
    def get_indicator_signals(df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate trading signals based on calculated indicators.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Dictionary of signals for each indicator
        """
        signals = {}
        
        try:
            # RSI Signal
            if 'rsi' in df.columns:
                last_rsi = df['rsi'].iloc[-1]
                if last_rsi < 30:
                    signals['rsi'] = 'oversold'
                elif last_rsi > 70:
                    signals['rsi'] = 'overbought'
                else:
                    signals['rsi'] = 'neutral'
            
            # MACD Signal
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                last_macd = df['macd'].iloc[-1]
                last_signal = df['macd_signal'].iloc[-1]
                if last_macd > last_signal:
                    signals['macd'] = 'bullish'
                else:
                    signals['macd'] = 'bearish'
            
            # Bollinger Bands Signal
            if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                last_close = df['close'].iloc[-1]
                last_upper = df['bb_upper'].iloc[-1]
                last_lower = df['bb_lower'].iloc[-1]
                if last_close > last_upper:
                    signals['bollinger'] = 'overbought'
                elif last_close < last_lower:
                    signals['bollinger'] = 'oversold'
                else:
                    signals['bollinger'] = 'neutral'
            
            # ADX Trend Strength
            if 'adx' in df.columns:
                last_adx = df['adx'].iloc[-1]
                if last_adx > 50:
                    signals['trend_strength'] = 'very_strong'
                elif last_adx > 25:
                    signals['trend_strength'] = 'strong'
                else:
                    signals['trend_strength'] = 'weak'
            
            # Volume Signal
            if 'volume_ratio' in df.columns:
                last_ratio = df['volume_ratio'].iloc[-1]
                if last_ratio > 2:
                    signals['volume'] = 'high'
                elif last_ratio < 0.5:
                    signals['volume'] = 'low'
                else:
                    signals['volume'] = 'normal'
                    
        except Exception as e:
            logger.error(f"Error generating indicator signals: {e}")
        
        return signals