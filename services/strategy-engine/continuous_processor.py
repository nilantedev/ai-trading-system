#!/usr/bin/env python3
"""
Continuous Processing Orchestrator
Coordinates end-to-end analysis of all watchlist symbols
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

import asyncio
import requests
from typing import List, Dict, Set
from datetime import datetime, timedelta
from trading_common import get_logger
from watchlist_manager import WatchlistManager
from portfolio_manager import PortfolioManager, PortfolioConfig

logger = get_logger(__name__)


class ContinuousProcessor:
    """
    Orchestrates continuous trading pipeline:
    1. Load watchlist
    2. For each symbol:
       - Get market data
       - Get ML predictions
       - Generate signals
       - Evaluate strategies
       - Check risk
       - Execute trades (if approved)
    3. Update positions
    4. Repeat
    """
    
    def __init__(self, 
                 watchlist_manager: WatchlistManager,
                 portfolio_manager: PortfolioManager,
                 update_interval: int = 60):
        self.watchlist_mgr = watchlist_manager
        self.portfolio_mgr = portfolio_manager
        self.update_interval = update_interval
        self.running = False
        
        # Service endpoints
        self.services = {
            'ml': 'http://localhost:8002',
            'signals': 'http://localhost:8003',
            'strategy': 'http://localhost:8006',
            'execution': 'http://localhost:8004',
            'risk': 'http://localhost:8005'
        }
        
        self.stats = {
            'symbols_processed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'last_run': None
        }
    
    def get_market_data(self, symbol: str, lookback_days: int = 90) -> Dict:
        """
        Get market data for symbol from QuestDB
        
        Args:
            symbol: Trading symbol
            lookback_days: Days of historical data
        
        Returns:
            Dict with OHLCV data
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            query = f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM market_data 
                WHERE symbol = '{symbol}' 
                  AND timestamp >= '{start_date.strftime('%Y-%m-%d')}' 
                  AND timestamp < '{end_date.strftime('%Y-%m-%d')}'
                ORDER BY timestamp DESC
                LIMIT 500
            """
            
            response = requests.get(
                'http://localhost:9000/exec',
                params={'query': query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                dataset = data.get('dataset', [])
                
                if not dataset:
                    return None
                
                # Convert to format expected by strategies
                return {
                    'symbol': symbol,
                    'close': [row[4] for row in reversed(dataset)],
                    'open': [row[1] for row in reversed(dataset)],
                    'high': [row[2] for row in reversed(dataset)],
                    'low': [row[3] for row in reversed(dataset)],
                    'volume': [row[5] for row in reversed(dataset)],
                    'timestamp': [row[0] for row in reversed(dataset)],
                    'bars': len(dataset)
                }
            else:
                logger.warning(f"Failed to get data for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def get_ml_prediction(self, symbol: str, market_data: Dict) -> Dict:
        """Get ML prediction for symbol"""
        try:
            # In production, this would call ML service with features
            # For now, return placeholder
            return {
                'symbol': symbol,
                'prediction': 0.0,  # -1 to 1
                'confidence': 0.5,
                'model': 'placeholder'
            }
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return None
    
    async def evaluate_strategy(self, strategy_name: str, symbol: str, 
                                market_data: Dict, ml_prediction: Dict = None) -> Dict:
        """
        Evaluate a strategy for a symbol
        
        Args:
            strategy_name: Strategy to evaluate
            symbol: Trading symbol
            market_data: Market data dict
            ml_prediction: ML prediction (optional)
        
        Returns:
            Strategy signal dict
        """
        try:
            # Get strategy instance endpoint
            url = f"{self.services['strategy']}/strategies/{strategy_name}/evaluate"
            
            payload = {
                'symbol': symbol,
                'data': market_data
            }
            
            if ml_prediction:
                payload['ml_prediction'] = ml_prediction
            
            # Note: In production, strategies are evaluated internally
            # For now, we'll use the strategy manager's evaluate method
            # This is a placeholder for the full implementation
            
            return {
                'strategy': strategy_name,
                'symbol': symbol,
                'signal_type': 'HOLD',  # Placeholder
                'confidence': 0.0,
                'reason': 'Placeholder - strategy evaluation needed'
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {strategy_name} for {symbol}: {e}")
            return None
    
    def check_risk_approval(self, symbol: str, signal: Dict, 
                           position_size: Dict) -> Dict:
        """
        Check if trade is approved by risk manager
        
        Args:
            symbol: Trading symbol
            signal: Strategy signal
            position_size: Calculated position size
        
        Returns:
            Dict with approval status
        """
        try:
            # Check portfolio limits
            limits = self.portfolio_mgr.check_risk_limits()
            
            if not limits['ok']:
                return {
                    'approved': False,
                    'reason': f"Risk limits breached: {', '.join(limits['breaches'])}"
                }
            
            # Check position size
            if position_size['shares'] == 0:
                return {
                    'approved': False,
                    'reason': position_size['reason']
                }
            
            # Check signal confidence
            if signal['confidence'] < 0.5:
                return {
                    'approved': False,
                    'reason': f"Low confidence: {signal['confidence']:.2f}"
                }
            
            return {
                'approved': True,
                'reason': 'All checks passed'
            }
            
        except Exception as e:
            logger.error(f"Error in risk approval: {e}")
            return {
                'approved': False,
                'reason': f'Error: {str(e)}'
            }
    
    async def process_symbol(self, symbol: str, strategies: List[str]) -> Dict:
        """
        Process a single symbol through the entire pipeline
        
        Args:
            symbol: Trading symbol
            strategies: List of strategies to evaluate
        
        Returns:
            Dict with processing results
        """
        try:
            # 1. Get market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                return {
                    'symbol': symbol,
                    'success': False,
                    'reason': 'No market data'
                }
            
            # 2. Get ML prediction
            ml_prediction = self.get_ml_prediction(symbol, market_data)
            
            # 3. Evaluate strategies
            best_signal = None
            best_confidence = 0.0
            
            for strategy in strategies:
                signal = await self.evaluate_strategy(strategy, symbol, market_data, ml_prediction)
                
                if signal and signal['confidence'] > best_confidence:
                    best_signal = signal
                    best_confidence = signal['confidence']
            
            if not best_signal or best_signal['signal_type'] == 'HOLD':
                return {
                    'symbol': symbol,
                    'success': True,
                    'action': 'HOLD',
                    'reason': 'No strong signals'
                }
            
            # 4. Calculate position size
            current_price = market_data['close'][-1]
            position_size = self.portfolio_mgr.calculate_position_size(
                symbol, 
                current_price, 
                confidence=best_signal['confidence']
            )
            
            # 5. Check risk approval
            approval = self.check_risk_approval(symbol, best_signal, position_size)
            
            if not approval['approved']:
                return {
                    'symbol': symbol,
                    'success': True,
                    'action': 'REJECTED',
                    'reason': approval['reason']
                }
            
            # 6. Execute trade (paper trading mode)
            if best_signal['signal_type'] == 'BUY':
                success = self.portfolio_mgr.add_position(
                    symbol, 
                    position_size['shares'], 
                    current_price
                )
                
                if success:
                    self.stats['trades_executed'] += 1
                    return {
                        'symbol': symbol,
                        'success': True,
                        'action': 'BUY',
                        'shares': position_size['shares'],
                        'price': current_price,
                        'value': position_size['value']
                    }
            
            return {
                'symbol': symbol,
                'success': True,
                'action': 'EVALUATED',
                'signal': best_signal['signal_type']
            }
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self.stats['errors'] += 1
            return {
                'symbol': symbol,
                'success': False,
                'reason': str(e)
            }
    
    async def run_cycle(self, strategies: List[str] = None) -> Dict:
        """
        Run one complete processing cycle
        
        Args:
            strategies: List of strategies to use (default: momentum)
        
        Returns:
            Dict with cycle results
        """
        if strategies is None:
            strategies = ['momentum']
        
        logger.info("Starting processing cycle...")
        start_time = datetime.utcnow()
        
        # Get watchlist
        watchlist = self.watchlist_mgr.get_watchlist()
        
        if not watchlist:
            logger.warning("Empty watchlist, syncing from QuestDB...")
            sync_result = self.watchlist_mgr.sync_from_questdb(min_bars=100, max_symbols=50)
            watchlist = self.watchlist_mgr.get_watchlist()
        
        logger.info(f"Processing {len(watchlist)} symbols...")
        
        # Process symbols
        results = []
        for symbol in list(watchlist)[:20]:  # Limit to 20 for testing
            result = await self.process_symbol(symbol, strategies)
            results.append(result)
            self.stats['symbols_processed'] += 1
            
            if result['success']:
                self.stats['signals_generated'] += 1
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        self.stats['last_run'] = datetime.utcnow().isoformat()
        
        # Generate summary
        actions = {}
        for r in results:
            action = r.get('action', 'ERROR')
            actions[action] = actions.get(action, 0) + 1
        
        summary = {
            'cycle_complete': True,
            'duration_seconds': duration,
            'symbols_processed': len(results),
            'actions': actions,
            'trades_executed': self.stats['trades_executed'],
            'portfolio': self.portfolio_mgr.get_portfolio_summary(),
            'timestamp': self.stats['last_run']
        }
        
        logger.info(f"Cycle complete: {len(results)} symbols in {duration:.1f}s")
        logger.info(f"Actions: {actions}")
        
        return summary
    
    async def start(self, strategies: List[str] = None):
        """Start continuous processing loop"""
        self.running = True
        logger.info("Starting continuous processing...")
        
        while self.running:
            try:
                summary = await self.run_cycle(strategies)
                logger.info(f"Waiting {self.update_interval}s until next cycle...")
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def stop(self):
        """Stop continuous processing"""
        self.running = False
        logger.info("Stopping continuous processing...")


async def main():
    """Main entry point for testing"""
    print("=== Continuous Processing Orchestrator ===\n")
    
    # Initialize components
    watchlist_mgr = WatchlistManager()
    portfolio_config = PortfolioConfig(
        total_capital=100000.0,
        max_position_pct=0.10,
        max_positions=10
    )
    portfolio_mgr = PortfolioManager(portfolio_config)
    
    # Create orchestrator
    processor = ContinuousProcessor(
        watchlist_mgr,
        portfolio_mgr,
        update_interval=60
    )
    
    # Run one cycle
    print("Running test cycle...")
    summary = await processor.run_cycle(strategies=['momentum'])
    
    print(f"\nCycle Summary:")
    print(f"  Duration: {summary['duration_seconds']:.1f}s")
    print(f"  Symbols: {summary['symbols_processed']}")
    print(f"  Actions: {summary['actions']}")
    print(f"  Portfolio: {summary['portfolio']['position_count']} positions")
    print(f"  Capital Used: ${summary['portfolio']['cash_used']:,.2f}")


if __name__ == "__main__":
    asyncio.run(main())
