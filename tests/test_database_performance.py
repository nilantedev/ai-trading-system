#!/usr/bin/env python3
"""Database performance tests for Phase 2."""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import List

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python-common"))

from trading_common.database import get_database_manager, QuestDBOperations
from trading_common.cache import get_trading_cache
from trading_common.models import MarketData, TradingSignal, SignalType, TimeFrame
from trading_common.logging import get_logger

logger = get_logger(__name__)


class DatabasePerformanceTests:
    """Test database performance and optimization."""
    
    def __init__(self):
        self.db_manager = None
        self.questdb_ops = None
        self.cache = None
    
    async def initialize(self):
        """Initialize test components."""
        try:
            self.db_manager = await get_database_manager()
            self.questdb_ops = QuestDBOperations(self.db_manager)
            self.cache = await get_trading_cache()
            logger.info("Performance test components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tests: {e}")
            raise
    
    def generate_test_market_data(self, symbol: str, count: int) -> List[MarketData]:
        """Generate test market data."""
        data_points = []
        base_price = 100.0 + random.uniform(-50, 50)
        base_time = datetime.utcnow() - timedelta(hours=count)
        
        for i in range(count):
            price_change = random.uniform(-2, 2)
            current_price = max(0.01, base_price + price_change)
            
            data_points.append(MarketData(
                symbol=symbol,
                timestamp=base_time + timedelta(minutes=i),
                open=current_price + random.uniform(-0.5, 0.5),
                high=current_price + random.uniform(0, 1),
                low=current_price - random.uniform(0, 1),
                close=current_price,
                volume=random.randint(1000, 50000),
                vwap=current_price + random.uniform(-0.1, 0.1),
                trade_count=random.randint(10, 500),
                data_source="test_provider"
            ))
            
            base_price = current_price
        
        return data_points
    
    async def test_redis_performance(self) -> dict:
        """Test Redis cache performance."""
        logger.info("Testing Redis cache performance...")
        
        test_data = self.generate_test_market_data("TEST_SYMBOL", 1000)
        
        # Test cache writes
        write_start = time.time()
        for data in test_data[:100]:  # Test with 100 records
            await self.cache.cache_market_data(data)
        write_time = (time.time() - write_start) * 1000
        
        # Test cache reads
        read_start = time.time()
        for _ in range(100):
            await self.cache.get_latest_market_data("TEST_SYMBOL")
        read_time = (time.time() - read_start) * 1000
        
        # Test cache stats
        stats = await self.cache.get_cache_stats()
        
        return {
            "redis_write_time_ms": write_time,
            "redis_read_time_ms": read_time,
            "redis_write_throughput": 100 / (write_time / 1000),
            "redis_read_throughput": 100 / (read_time / 1000),
            "cache_stats": stats
        }
    
    async def test_questdb_performance(self) -> dict:
        """Test QuestDB time-series performance."""
        logger.info("Testing QuestDB performance...")
        
        try:
            # Test table creation
            create_start = time.time()
            await self.questdb_ops.create_tables()
            create_time = (time.time() - create_start) * 1000
            
            # Test batch inserts
            test_data = self.generate_test_market_data("PERF_TEST", 1000)
            
            insert_start = time.time()
            inserted_count = await self.questdb_ops.insert_market_data(test_data)
            insert_time = (time.time() - insert_start) * 1000
            
            # Test queries
            query_start = time.time()
            latest_data = await self.questdb_ops.get_latest_market_data("PERF_TEST", 100)
            query_time = (time.time() - query_start) * 1000
            
            return {
                "questdb_table_creation_ms": create_time,
                "questdb_batch_insert_ms": insert_time,
                "questdb_query_ms": query_time,
                "questdb_insert_throughput": inserted_count / (insert_time / 1000),
                "questdb_records_inserted": inserted_count,
                "questdb_records_queried": len(latest_data)
            }
            
        except Exception as e:
            logger.warning(f"QuestDB test failed (may not be ready): {e}")
            return {
                "questdb_status": "unavailable",
                "error": str(e)
            }
    
    async def test_trading_signal_performance(self) -> dict:
        """Test trading signal storage performance."""
        logger.info("Testing trading signal performance...")
        
        signals = []
        for i in range(100):
            signal = TradingSignal(
                id=f"test_signal_{i}",
                timestamp=datetime.utcnow(),
                symbol=f"TEST{i % 10}",
                signal_type=SignalType.BUY,
                confidence=random.uniform(0.5, 1.0),
                target_price=random.uniform(90, 110),
                timeframe=TimeFrame.ONE_HOUR,
                strategy_name="test_strategy",
                agent_id="test_agent"
            )
            signals.append(signal)
        
        # Test cache performance
        cache_start = time.time()
        for signal in signals:
            await self.cache.cache_trading_signal(signal)
        cache_time = (time.time() - cache_start) * 1000
        
        # Test retrieval
        retrieval_start = time.time()
        active_signals = await self.cache.get_active_signals()
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        try:
            # Test database storage
            db_start = time.time()
            for signal in signals:
                await self.questdb_ops.insert_trading_signal(signal)
            db_time = (time.time() - db_start) * 1000
            
            return {
                "signal_cache_time_ms": cache_time,
                "signal_retrieval_time_ms": retrieval_time,
                "signal_db_time_ms": db_time,
                "signals_cached": len(signals),
                "signals_retrieved": len(active_signals)
            }
        except Exception as e:
            return {
                "signal_cache_time_ms": cache_time,
                "signal_retrieval_time_ms": retrieval_time,
                "signal_db_error": str(e),
                "signals_cached": len(signals),
                "signals_retrieved": len(active_signals)
            }
    
    async def test_concurrent_operations(self) -> dict:
        """Test concurrent database operations."""
        logger.info("Testing concurrent operations...")
        
        async def concurrent_cache_operations():
            tasks = []
            for i in range(50):
                data = MarketData(
                    symbol=f"CONCURRENT_{i % 5}",
                    timestamp=datetime.utcnow(),
                    open=100.0, high=101.0, low=99.0, close=100.5,
                    volume=10000, data_source="test"
                )
                tasks.append(self.cache.cache_market_data(data))
            
            await asyncio.gather(*tasks)
        
        concurrent_start = time.time()
        await concurrent_cache_operations()
        concurrent_time = (time.time() - concurrent_start) * 1000
        
        return {
            "concurrent_cache_operations_ms": concurrent_time,
            "concurrent_operations_count": 50,
            "concurrent_throughput": 50 / (concurrent_time / 1000)
        }
    
    async def run_all_tests(self) -> dict:
        """Run all performance tests."""
        logger.info("Starting comprehensive performance tests...")
        
        results = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "test_duration_ms": 0
        }
        
        overall_start = time.time()
        
        # Run individual tests
        try:
            results["redis_tests"] = await self.test_redis_performance()
        except Exception as e:
            results["redis_tests"] = {"error": str(e)}
        
        try:
            results["questdb_tests"] = await self.test_questdb_performance()
        except Exception as e:
            results["questdb_tests"] = {"error": str(e)}
        
        try:
            results["signal_tests"] = await self.test_trading_signal_performance()
        except Exception as e:
            results["signal_tests"] = {"error": str(e)}
        
        try:
            results["concurrent_tests"] = await self.test_concurrent_operations()
        except Exception as e:
            results["concurrent_tests"] = {"error": str(e)}
        
        results["test_duration_ms"] = (time.time() - overall_start) * 1000
        
        return results
    
    async def cleanup(self):
        """Clean up test data."""
        try:
            # Clear test cache data
            await self.cache.clear_cache_pattern("TEST_*")
            await self.cache.clear_cache_pattern("PERF_*")
            await self.cache.clear_cache_pattern("CONCURRENT_*")
            logger.info("Cleaned up test data")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


async def main():
    """Run performance tests."""
    logger.info("Starting database performance tests")
    
    tester = DatabasePerformanceTests()
    
    try:
        await tester.initialize()
        results = await tester.run_all_tests()
        
        # Print results
        print("\n" + "="*60)
        print("DATABASE PERFORMANCE TEST RESULTS")
        print("="*60)
        
        print(f"Test completed in: {results['test_duration_ms']:.2f}ms")
        
        # Redis results
        redis_tests = results.get("redis_tests", {})
        if "error" not in redis_tests:
            print(f"\nRedis Cache Performance:")
            print(f"  Write throughput: {redis_tests.get('redis_write_throughput', 0):.1f} ops/sec")
            print(f"  Read throughput: {redis_tests.get('redis_read_throughput', 0):.1f} ops/sec")
            print(f"  Cache hit rate: {redis_tests.get('cache_stats', {}).get('hit_rate_percent', 0)}%")
        else:
            print(f"\nRedis Error: {redis_tests['error']}")
        
        # QuestDB results
        questdb_tests = results.get("questdb_tests", {})
        if "error" not in questdb_tests and questdb_tests.get("questdb_status") != "unavailable":
            print(f"\nQuestDB Time-Series Performance:")
            print(f"  Insert throughput: {questdb_tests.get('questdb_insert_throughput', 0):.1f} records/sec")
            print(f"  Query time: {questdb_tests.get('questdb_query_ms', 0):.2f}ms")
            print(f"  Records inserted: {questdb_tests.get('questdb_records_inserted', 0)}")
        else:
            print(f"\nQuestDB Status: {questdb_tests.get('error', 'unavailable')}")
        
        # Signal tests
        signal_tests = results.get("signal_tests", {})
        if "signal_db_error" not in signal_tests:
            print(f"\nTrading Signal Performance:")
            print(f"  Cache time: {signal_tests.get('signal_cache_time_ms', 0):.2f}ms")
            print(f"  Retrieval time: {signal_tests.get('signal_retrieval_time_ms', 0):.2f}ms")
            if "signal_db_time_ms" in signal_tests:
                print(f"  Database time: {signal_tests.get('signal_db_time_ms', 0):.2f}ms")
        
        # Concurrent tests
        concurrent_tests = results.get("concurrent_tests", {})
        if "error" not in concurrent_tests:
            print(f"\nConcurrent Operations:")
            print(f"  Throughput: {concurrent_tests.get('concurrent_throughput', 0):.1f} ops/sec")
            print(f"  50 operations in: {concurrent_tests.get('concurrent_cache_operations_ms', 0):.2f}ms")
        
        print("\n" + "="*60)
        
        # Cleanup
        await tester.cleanup()
        
    except Exception as e:
        logger.error(f"Performance tests failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)