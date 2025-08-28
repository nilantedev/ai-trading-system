#!/usr/bin/env python3
"""
Safe Trading System Startup Script
Performs all safety checks before allowing trading to begin
"""

import asyncio
import sys
import os
import yaml
import logging
from datetime import datetime
from typing import Dict, Tuple, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.risk_monitor.trading_governor import TradingGovernor, TradingMode
from services.data_ingestion.data_validator import MarketDataValidator
from shared.python_common.trading_common.risk_limits import HardLimits, RiskLevel
from shared.python_common.trading_common.audit_trail import get_audit_logger, EventType


class SafeTradingStartup:
    """Ensures safe startup of trading system"""
    
    def __init__(self, config_path: str = "config/safe_trading_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.startup_checks = []
        self.check_results = {}
        
    def _load_config(self) -> Dict:
        """Load safety configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run_startup_checks(self) -> Tuple[bool, List[str]]:
        """Run all startup safety checks"""
        logger.info("=" * 60)
        logger.info("STARTING SAFE TRADING SYSTEM")
        logger.info("=" * 60)
        
        failures = []
        
        # Check 1: Verify we're in paper trading mode
        logger.info("✓ Checking trading mode...")
        if self.config['trading_mode'] != 'paper':
            if self.config['trading_mode'] == 'live':
                failures.append("CRITICAL: System set to LIVE trading - switching to paper for safety")
                self.config['trading_mode'] = 'paper'
        logger.info(f"  Mode: {self.config['trading_mode'].upper()}")
        
        # Check 2: Initialize Governor with safe defaults
        logger.info("✓ Initializing Trading Governor...")
        governor = TradingGovernor()
        await governor.initialize_default_settings()
        await governor.apply_trading_mode(TradingMode.PAPER)
        logger.info("  Governor initialized with PAPER mode")
        
        # Check 3: Verify kill switch is ready
        logger.info("✓ Testing kill switch...")
        if not self.config['safety']['kill_switch_enabled']:
            failures.append("Kill switch disabled - enabling for safety")
            self.config['safety']['kill_switch_enabled'] = True
        
        # Test kill switch
        await governor.emergency_stop("Startup test")
        is_active = await governor.check_kill_switch()
        if not is_active:
            failures.append("Kill switch test failed!")
        else:
            # Clear after test
            await governor.clear_kill_switch("test_clear")
            logger.info("  Kill switch tested successfully")
        
        # Check 4: Verify risk limits are conservative
        logger.info("✓ Checking risk limits...")
        risk_config = self.config['risk_limits']
        if risk_config['max_position_size_usd'] > 5000:
            failures.append(f"Position size ${risk_config['max_position_size_usd']} too large for initial deployment")
        if risk_config['max_daily_loss_usd'] > 500:
            failures.append(f"Daily loss limit ${risk_config['max_daily_loss_usd']} too high")
        logger.info(f"  Max position: ${risk_config['max_position_size_usd']}")
        logger.info(f"  Max daily loss: ${risk_config['max_daily_loss_usd']}")
        
        # Check 5: Test data validation
        logger.info("✓ Testing data validation...")
        validator = MarketDataValidator()
        
        # Test with bad data
        bad_tick = {
            'symbol': 'TEST',
            'price': -100,
            'timestamp': datetime.now()
        }
        result = validator.validate_tick(bad_tick)
        if result.is_valid:
            failures.append("Data validator failed to catch invalid price!")
        else:
            logger.info("  Data validator working correctly")
        
        # Check 6: Initialize audit trail
        logger.info("✓ Initializing audit trail...")
        audit_log = get_audit_logger()
        await audit_log.add_event(
            EventType.SYSTEM_ERROR,
            {
                'event': 'startup',
                'mode': self.config['trading_mode'],
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Verify chain
        is_valid, _ = audit_log.verify_chain()
        if not is_valid:
            failures.append("Audit trail integrity check failed!")
        else:
            logger.info("  Audit trail initialized and verified")
        
        # Check 7: Verify auto-trading is disabled
        logger.info("✓ Verifying auto-trading disabled...")
        if self.config['safety']['auto_trade_enabled']:
            logger.warning("  Auto-trading was enabled - disabling for safety")
            await governor.update_setting("auto_trade_enabled", False)
        else:
            logger.info("  Auto-trading correctly disabled")
        
        # Check 8: Test database connections
        logger.info("✓ Testing connections...")
        try:
            import redis.asyncio as redis
            r = redis.Redis(host='localhost', port=6379)
            await r.ping()
            logger.info("  Redis connection: OK")
        except Exception as e:
            failures.append(f"Redis connection failed: {e}")
        
        # Check 9: Verify we have test API keys only
        logger.info("✓ Checking API configuration...")
        alpaca_url = os.getenv('ALPACA_BASE_URL', '')
        if 'paper' not in alpaca_url:
            failures.append(f"CRITICAL: Not using paper trading API! URL: {alpaca_url}")
        else:
            logger.info(f"  Using paper trading API: {alpaca_url}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        if failures:
            logger.error("STARTUP CHECKS FAILED:")
            for i, failure in enumerate(failures, 1):
                logger.error(f"  {i}. {failure}")
            logger.error("\nFIX THESE ISSUES BEFORE PROCEEDING!")
            return False, failures
        else:
            logger.info("✅ ALL SAFETY CHECKS PASSED")
            logger.info("\nSYSTEM READY FOR PAPER TRADING")
            logger.info("\nIMPORTANT REMINDERS:")
            logger.info("  1. This is PAPER TRADING mode only")
            logger.info("  2. Auto-trading is DISABLED - enable manually when ready")
            logger.info("  3. Kill switch is ACTIVE and tested")
            logger.info("  4. All safety limits are ENFORCED")
            logger.info("  5. Monitor the system closely for the first 24 hours")
            logger.info("\nTo enable auto-trading:")
            logger.info("  curl -X POST http://localhost:8000/api/v1/governor/setting")
            logger.info('  -H "Content-Type: application/json"')
            logger.info('  -d \'{"key": "auto_trade_enabled", "value": true}\'')
            
            return True, []
    
    async def display_current_settings(self):
        """Display current safety settings"""
        logger.info("\nCURRENT SAFETY SETTINGS:")
        logger.info("-" * 40)
        
        governor = TradingGovernor()
        state = await governor.get_current_state()
        
        logger.info(f"Trading Mode: {state.get('mode', 'unknown').upper()}")
        logger.info(f"Auto-Trading: {state.get('auto_trading', False)}")
        logger.info(f"Current Positions: {state.get('current_positions', 0)}")
        logger.info(f"Daily P&L: ${state.get('daily_pnl', 0):.2f}")
        
        settings = state.get('settings', {})
        if settings:
            logger.info(f"Max Position Size: ${settings.get('max_position_size', 0)}")
            logger.info(f"Max Daily Loss: ${settings.get('max_daily_loss', 0)}")
            logger.info(f"Max Open Positions: {settings.get('max_open_positions', 0)}")


async def main():
    """Main startup routine"""
    startup = SafeTradingStartup()
    
    # Run safety checks
    success, failures = await startup.run_startup_checks()
    
    if not success:
        logger.error("\n⛔ SYSTEM NOT STARTED DUE TO SAFETY CHECK FAILURES")
        sys.exit(1)
    
    # Display current settings
    await startup.display_current_settings()
    
    logger.info("\n" + "=" * 60)
    logger.info("🚀 TRADING SYSTEM STARTED IN SAFE MODE")
    logger.info("=" * 60)
    
    # Log successful startup
    audit_log = get_audit_logger()
    await audit_log.add_event(
        EventType.USER_ACTION,
        {
            'action': 'system_startup',
            'mode': 'paper',
            'safety_checks': 'passed',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    logger.info("\nSystem is ready. Start the API server with:")
    logger.info("  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    logger.info("\nMonitor the system at:")
    logger.info("  http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(main())