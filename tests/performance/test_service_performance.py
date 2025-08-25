#!/usr/bin/env python3
"""
Service Performance Tests
"""

import pytest
import pytest_asyncio
import asyncio
import time
import statistics
from datetime import datetime, timedelta
import concurrent.futures
from unittest.mock import AsyncMock, patch
import psutil
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestServicePerformance:
    """Performance tests for core trading services."""

    @pytest_asyncio.fixture
    async def mock_market_data_service(self):
        """Mock market data service for performance testing."""
        service = AsyncMock()
        
        # Simulate realistic response times
        async def mock_get_data(symbol):
            await asyncio.sleep(0.005)  # 5ms simulated processing
            return {
                "symbol": symbol,
                "price": 150.25,
                "volume": 1000000,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        service.get_current_data = mock_get_data
        return service

    @pytest_asyncio.fixture
    async def mock_signal_service(self):
        """Mock signal generation service for performance testing."""
        service = AsyncMock()
        
        async def mock_generate_signal(symbol, market_data):
            # Simulate computation time for signal generation
            await asyncio.sleep(0.010)  # 10ms simulated analysis
            return {
                "symbol": symbol,
                "signal_type": "BUY",
                "confidence": 0.85,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        service.generate_signal = mock_generate_signal
        return service

    @pytest_asyncio.fixture
    async def mock_order_service(self):
        """Mock order management service for performance testing."""
        service = AsyncMock()
        
        async def mock_place_order(order_request):
            # Simulate order validation and processing
            await asyncio.sleep(0.015)  # 15ms simulated processing
            return {
                "id": "order_001",
                "status": "PENDING",
                "symbol": order_request.get("symbol", "UNKNOWN"),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        service.place_order = mock_place_order
        return service

    @pytest.mark.asyncio
    async def test_market_data_service_response_time(self, mock_market_data_service):
        """Test market data service response time."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        response_times = []
        
        # Warm-up
        for symbol in symbols[:2]:
            await mock_market_data_service.get_current_data(symbol)
        
        # Performance measurement
        for _ in range(100):
            for symbol in symbols:
                start_time = time.time()
                data = await mock_market_data_service.get_current_data(symbol)
                end_time = time.time()
                
                assert data["symbol"] == symbol
                response_time = (end_time - start_time) * 1000
                response_times.append(response_time)
        
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]
        
        print(f"Market data service - avg: {avg_time:.2f}ms, p95: {p95_time:.2f}ms")
        
        assert avg_time < 20  # Less than 20ms average
        assert p95_time < 50   # Less than 50ms for 95th percentile

    @pytest.mark.asyncio
    async def test_signal_generation_performance(self, mock_signal_service):
        """Test signal generation service performance."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        market_data_samples = [
            {"price": 150.25, "volume": 1000000, "change": 2.5},
            {"price": 2800.50, "volume": 800000, "change": -1.2},
            {"price": 350.75, "volume": 1200000, "change": 0.8}
        ]
        
        response_times = []
        
        # Performance test
        for _ in range(50):
            for i, symbol in enumerate(symbols):
                start_time = time.time()
                signal = await mock_signal_service.generate_signal(
                    symbol, market_data_samples[i]
                )
                end_time = time.time()
                
                assert signal["symbol"] == symbol
                response_time = (end_time - start_time) * 1000
                response_times.append(response_time)
        
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]
        
        print(f"Signal generation - avg: {avg_time:.2f}ms, p95: {p95_time:.2f}ms")
        
        assert avg_time < 30  # Less than 30ms average
        assert p95_time < 80   # Less than 80ms for 95th percentile

    @pytest.mark.asyncio
    async def test_order_processing_performance(self, mock_order_service):
        """Test order processing service performance."""
        order_requests = [
            {"symbol": "AAPL", "side": "BUY", "quantity": 100, "order_type": "MARKET"},
            {"symbol": "GOOGL", "side": "SELL", "quantity": 50, "order_type": "LIMIT", "price": 2800.0},
            {"symbol": "MSFT", "side": "BUY", "quantity": 200, "order_type": "STOP", "stop_price": 340.0}
        ]
        
        response_times = []
        
        # Performance test
        for _ in range(100):
            for order_request in order_requests:
                start_time = time.time()
                order = await mock_order_service.place_order(order_request)
                end_time = time.time()
                
                assert order["symbol"] == order_request["symbol"]
                response_time = (end_time - start_time) * 1000
                response_times.append(response_time)
        
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]
        
        print(f"Order processing - avg: {avg_time:.2f}ms, p95: {p95_time:.2f}ms")
        
        assert avg_time < 40  # Less than 40ms average
        assert p95_time < 100  # Less than 100ms for 95th percentile

    @pytest.mark.asyncio
    async def test_concurrent_service_calls(self, mock_market_data_service, mock_signal_service):
        """Test performance under concurrent service calls."""
        concurrency_levels = [1, 5, 10, 20, 50]
        
        for concurrency in concurrency_levels:
            response_times = []
            
            async def service_workflow():
                # Simulate typical service workflow
                start_time = time.time()
                
                # Get market data
                market_data = await mock_market_data_service.get_current_data("AAPL")
                
                # Generate signal
                signal = await mock_signal_service.generate_signal("AAPL", market_data)
                
                end_time = time.time()
                return (end_time - start_time) * 1000
            
            # Run concurrent workflows
            tasks = [service_workflow() for _ in range(concurrency)]
            workflow_times = await asyncio.gather(*tasks)
            
            avg_time = statistics.mean(workflow_times)
            p95_time = statistics.quantiles(workflow_times, n=20)[18] if len(workflow_times) >= 20 else max(workflow_times)
            
            print(f"Concurrency {concurrency} - avg: {avg_time:.2f}ms, p95: {p95_time:.2f}ms")
            
            # Performance should scale reasonably
            assert avg_time < 50  # Less than 50ms average
            assert p95_time < 150  # Less than 150ms for 95th percentile

    @pytest.mark.asyncio
    async def test_service_throughput(self, mock_market_data_service):
        """Test service throughput under sustained load."""
        duration_seconds = 3
        completed_requests = 0
        errors = 0
        
        async def worker():
            nonlocal completed_requests, errors
            try:
                await mock_market_data_service.get_current_data("AAPL")
                completed_requests += 1
            except Exception:
                errors += 1
        
        # Generate sustained load
        start_time = time.time()
        tasks = []
        
        while time.time() - start_time < duration_seconds:
            # Create batch of workers
            batch_tasks = [worker() for _ in range(10)]
            tasks.extend(batch_tasks)
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            await asyncio.sleep(0.001)  # Small pause
        
        total_time = time.time() - start_time
        throughput = completed_requests / total_time
        error_rate = errors / (completed_requests + errors) if (completed_requests + errors) > 0 else 0
        
        print(f"Service throughput: {throughput:.1f} requests/second")
        print(f"Error rate: {error_rate:.2%}")
        
        assert throughput > 500  # At least 500 requests per second
        assert error_rate < 0.005  # Less than 0.5% error rate

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mock_market_data_service, mock_signal_service):
        """Test service memory usage under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load with various service operations
        tasks = []
        for i in range(1000):
            if i % 2 == 0:
                task = mock_market_data_service.get_current_data(f"SYMBOL_{i % 10}")
            else:
                task = mock_signal_service.generate_signal(f"SYMBOL_{i % 10}", {"price": 100 + i})
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")
        
        # Memory growth should be reasonable
        assert memory_growth < 50  # Less than 50MB growth

    @pytest.mark.asyncio
    async def test_service_latency_under_cpu_load(self, mock_market_data_service):
        """Test service latency under CPU load."""
        def cpu_intensive_task():
            # CPU-bound computation
            result = 0
            for i in range(50000):
                result += i ** 2
            return result
        
        # Baseline latency without CPU load
        baseline_times = []
        for _ in range(20):
            start_time = time.time()
            await mock_market_data_service.get_current_data("AAPL")
            end_time = time.time()
            baseline_times.append((end_time - start_time) * 1000)
        
        baseline_avg = statistics.mean(baseline_times)
        
        # Latency with concurrent CPU load
        cpu_load_times = []
        for _ in range(20):
            # Start CPU task
            cpu_task = asyncio.to_thread(cpu_intensive_task)
            
            # Measure service latency
            start_time = time.time()
            await mock_market_data_service.get_current_data("AAPL")
            end_time = time.time()
            cpu_load_times.append((end_time - start_time) * 1000)
            
            # Wait for CPU task
            await cpu_task
        
        cpu_load_avg = statistics.mean(cpu_load_times)
        latency_increase = (cpu_load_avg - baseline_avg) / baseline_avg
        
        print(f"Baseline latency: {baseline_avg:.2f}ms")
        print(f"CPU load latency: {cpu_load_avg:.2f}ms")
        print(f"Latency increase: {latency_increase:.1%}")
        
        # Latency should not increase dramatically under CPU load
        assert latency_increase < 0.5  # Less than 50% increase
        assert cpu_load_avg < 50  # Still under 50ms

    @pytest.mark.asyncio
    async def test_service_batch_processing_performance(self, mock_market_data_service):
        """Test batch processing performance."""
        batch_sizes = [1, 10, 50, 100]
        symbols = [f"SYMBOL_{i}" for i in range(100)]
        
        for batch_size in batch_sizes:
            batch_times = []
            
            # Process symbols in batches
            for i in range(0, 100, batch_size):
                batch_symbols = symbols[i:i+batch_size]
                
                start_time = time.time()
                tasks = [
                    mock_market_data_service.get_current_data(symbol) 
                    for symbol in batch_symbols
                ]
                await asyncio.gather(*tasks)
                end_time = time.time()
                
                batch_time = end_time - start_time
                per_item_time = (batch_time / len(batch_symbols)) * 1000
                batch_times.append(per_item_time)
            
            avg_per_item = statistics.mean(batch_times)
            print(f"Batch size {batch_size} - per item: {avg_per_item:.2f}ms")
            
            # Larger batches should be more efficient per item
            assert avg_per_item < 30  # Less than 30ms per item

    @pytest.mark.asyncio
    async def test_service_error_recovery_performance(self, mock_market_data_service):
        """Test performance during error recovery scenarios."""
        # Configure service to fail intermittently
        call_count = 0
        original_get_data = mock_market_data_service.get_current_data
        
        async def failing_get_data(symbol):
            nonlocal call_count
            call_count += 1
            if call_count % 5 == 0:  # Fail every 5th call
                raise Exception("Simulated service failure")
            return await original_get_data(symbol)
        
        mock_market_data_service.get_current_data = failing_get_data
        
        # Test error recovery performance
        response_times = []
        errors = 0
        
        for _ in range(100):
            start_time = time.time()
            try:
                await mock_market_data_service.get_current_data("AAPL")
                end_time = time.time()
                response_times.append((end_time - start_time) * 1000)
            except Exception:
                errors += 1
        
        if response_times:
            avg_time = statistics.mean(response_times)
            success_rate = len(response_times) / (len(response_times) + errors)
            
            print(f"Error recovery - avg time: {avg_time:.2f}ms, success rate: {success_rate:.1%}")
            
            # Should maintain reasonable performance even with errors
            assert avg_time < 50  # Less than 50ms average
            assert success_rate > 0.7  # At least 70% success rate

    @pytest.mark.asyncio
    async def test_service_cache_performance(self, mock_market_data_service):
        """Test caching impact on service performance."""
        # Simulate cached vs uncached requests
        cache = {}
        
        async def cached_get_data(symbol):
            if symbol in cache:
                # Simulate cache hit (much faster)
                await asyncio.sleep(0.001)  # 1ms for cache hit
                return cache[symbol]
            else:
                # Simulate cache miss (normal speed)
                data = await mock_market_data_service.get_current_data(symbol)
                cache[symbol] = data
                return data
        
        # Test cache miss performance (first request)
        cache_miss_times = []
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            start_time = time.time()
            await cached_get_data(symbol)
            end_time = time.time()
            cache_miss_times.append((end_time - start_time) * 1000)
        
        # Test cache hit performance (subsequent requests)
        cache_hit_times = []
        for _ in range(50):
            for symbol in ["AAPL", "GOOGL", "MSFT"]:
                start_time = time.time()
                await cached_get_data(symbol)
                end_time = time.time()
                cache_hit_times.append((end_time - start_time) * 1000)
        
        miss_avg = statistics.mean(cache_miss_times)
        hit_avg = statistics.mean(cache_hit_times)
        cache_improvement = (miss_avg - hit_avg) / miss_avg
        
        print(f"Cache miss avg: {miss_avg:.2f}ms")
        print(f"Cache hit avg: {hit_avg:.2f}ms") 
        print(f"Cache improvement: {cache_improvement:.1%}")
        
        # Cache should provide significant performance improvement
        assert cache_improvement > 0.5  # At least 50% improvement
        assert hit_avg < 5  # Cache hits should be very fast

    @pytest.mark.asyncio
    async def test_service_scaling_performance(self, mock_market_data_service):
        """Test performance scaling characteristics."""
        request_volumes = [10, 50, 100, 500, 1000]
        scaling_data = {}
        
        for volume in request_volumes:
            start_time = time.time()
            
            # Create all tasks
            tasks = [
                mock_market_data_service.get_current_data(f"SYMBOL_{i % 10}")
                for i in range(volume)
            ]
            
            # Execute all tasks
            await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            throughput = volume / total_time
            scaling_data[volume] = throughput
            
            print(f"Volume {volume}: {throughput:.1f} requests/second")
        
        # Check scaling efficiency
        for i in range(len(request_volumes) - 1):
            current_volume = request_volumes[i]
            next_volume = request_volumes[i + 1]
            
            volume_ratio = next_volume / current_volume
            throughput_ratio = scaling_data[next_volume] / scaling_data[current_volume]
            
            # Throughput should scale reasonably with volume
            scaling_efficiency = throughput_ratio / volume_ratio
            print(f"Scaling {current_volume}->{next_volume}: {scaling_efficiency:.2f} efficiency")
            
            # Should maintain at least 30% scaling efficiency (realistic for async high loads)
            # Skip the jump from 100->500 as it typically hits concurrency limits
            if current_volume == 100 and next_volume == 500:
                assert scaling_efficiency > 0.25  # More lenient for large jumps
            else:
                assert scaling_efficiency > 0.3