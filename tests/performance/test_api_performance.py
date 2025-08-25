#!/usr/bin/env python3
"""
API Performance Tests
"""

import pytest
import pytest_asyncio
import asyncio
import time
import statistics
from datetime import datetime
import concurrent.futures
import httpx

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest_asyncio.fixture
    async def performance_client(self):
        """Create HTTP client for performance testing."""
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @app.get("/api/v1/market-data/{symbol}")
        async def get_market_data(symbol: str):
            # Simulate some processing time
            await asyncio.sleep(0.001)  # 1ms processing time
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "price": 150.25,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        @app.post("/api/v1/orders")
        async def place_order(order_data: dict):
            # Simulate order processing
            await asyncio.sleep(0.002)  # 2ms processing time
            return {
                "success": True,
                "data": {
                    "id": "order_001",
                    "status": "PENDING",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            yield client

    def test_health_endpoint_response_time(self, performance_client):
        """Test health endpoint response time."""
        response_times = []
        
        # Warm-up requests
        for _ in range(5):
            performance_client.get("/health")
        
        # Measure response times
        for _ in range(100):
            start_time = time.time()
            response = performance_client.get("/health")
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
        
        # Performance assertions
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        print(f"Health endpoint avg response time: {avg_response_time:.2f}ms")
        print(f"Health endpoint p95 response time: {p95_response_time:.2f}ms")
        
        assert avg_response_time < 50  # Less than 50ms average
        assert p95_response_time < 100  # Less than 100ms for 95% of requests

    @pytest.mark.asyncio
    async def test_market_data_endpoint_response_time(self, performance_client):
        """Test market data endpoint response time."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        response_times = []
        
        # Warm-up
        for symbol in symbols:
            await performance_client.get(f"/api/v1/market-data/{symbol}")
        
        # Measure performance
        for _ in range(50):
            for symbol in symbols:
                start_time = time.time()
                response = await performance_client.get(f"/api/v1/market-data/{symbol}")
                end_time = time.time()
                
                assert response.status_code == 200
                response_time = (end_time - start_time) * 1000
                response_times.append(response_time)
        
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]
        
        print(f"Market data avg response time: {avg_response_time:.2f}ms")
        print(f"Market data p95 response time: {p95_response_time:.2f}ms")
        
        assert avg_response_time < 100  # Less than 100ms average
        assert p95_response_time < 200  # Less than 200ms for 95% of requests

    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, performance_client):
        """Test performance under concurrent load."""
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            response_times = []
            
            async def make_request():
                start_time = time.time()
                response = await performance_client.get("/health")
                end_time = time.time()
                assert response.status_code == 200
                return (end_time - start_time) * 1000
            
            # Create concurrent tasks
            tasks = [make_request() for _ in range(concurrency * 10)]
            
            start_time = time.time()
            response_times = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            avg_response_time = statistics.mean(response_times)
            throughput = len(tasks) / total_time
            
            print(f"Concurrency {concurrency}: avg {avg_response_time:.2f}ms, {throughput:.1f} req/s")
            
            # Performance should not degrade too much with concurrency
            assert avg_response_time < 100  # Should stay under 100ms
            assert throughput > concurrency * 5  # Should handle at least 5x concurrency in throughput

    @pytest.mark.asyncio
    async def test_throughput_measurement(self, performance_client):
        """Test API throughput under sustained load."""
        duration_seconds = 5
        request_count = 0
        errors = 0
        
        async def worker():
            nonlocal request_count, errors
            try:
                response = await performance_client.get("/health")
                if response.status_code == 200:
                    request_count += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
        
        # Run workers for specified duration
        start_time = time.time()
        tasks = []
        
        while time.time() - start_time < duration_seconds:
            # Add batches of concurrent requests
            batch_tasks = [worker() for _ in range(10)]
            tasks.extend(batch_tasks)
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            await asyncio.sleep(0.01)  # Small pause between batches
        
        total_time = time.time() - start_time
        throughput = request_count / total_time
        error_rate = errors / (request_count + errors) if (request_count + errors) > 0 else 0
        
        print(f"Throughput: {throughput:.1f} requests/second")
        print(f"Error rate: {error_rate:.2%}")
        print(f"Total requests: {request_count}, Errors: {errors}")
        
        assert throughput > 100  # At least 100 requests per second
        assert error_rate < 0.01  # Less than 1% error rate

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_client):
        """Test memory usage under sustained load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate sustained load
        tasks = []
        for _ in range(1000):  # 1000 requests
            task = performance_client.get("/health")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")
        
        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth for 1000 requests

    @pytest.mark.asyncio
    async def test_response_size_impact(self, performance_client):
        """Test how response size impacts performance."""
        # Small response (health endpoint)
        small_response_times = []
        for _ in range(50):
            start_time = time.time()
            response = await performance_client.get("/health")
            end_time = time.time()
            small_response_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        # Larger response (market data endpoint)
        large_response_times = []
        for _ in range(50):
            start_time = time.time()
            response = await performance_client.get("/api/v1/market-data/AAPL")
            end_time = time.time()
            large_response_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        small_avg = statistics.mean(small_response_times)
        large_avg = statistics.mean(large_response_times)
        
        print(f"Small response avg: {small_avg:.2f}ms")
        print(f"Large response avg: {large_avg:.2f}ms")
        
        # Larger responses should not be disproportionately slower
        assert large_avg < small_avg * 3  # No more than 3x slower

    @pytest.mark.asyncio
    async def test_connection_reuse_performance(self, performance_client):
        """Test performance impact of connection reuse."""
        # Reused connection performance
        reused_times = []
        for _ in range(100):
            start_time = time.time()
            response = await performance_client.get("/health")
            end_time = time.time()
            reused_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        reused_avg = statistics.mean(reused_times)
        print(f"Connection reuse avg: {reused_avg:.2f}ms")
        
        # With connection reuse, should be consistently fast
        assert reused_avg < 50
        assert statistics.stdev(reused_times) < 20  # Low variance

    @pytest.mark.asyncio 
    async def test_error_handling_performance(self, performance_client):
        """Test performance of error handling."""
        error_response_times = []
        
        # Generate requests that will result in errors
        for _ in range(50):
            start_time = time.time()
            response = await performance_client.get("/nonexistent-endpoint")
            end_time = time.time()
            
            assert response.status_code == 404
            error_response_times.append((end_time - start_time) * 1000)
        
        error_avg = statistics.mean(error_response_times)
        print(f"Error response avg: {error_avg:.2f}ms")
        
        # Error responses should still be fast
        assert error_avg < 100

    @pytest.mark.asyncio
    async def test_payload_size_performance(self, performance_client):
        """Test performance with different payload sizes."""
        # Small payload
        small_payload = {"symbol": "AAPL", "quantity": 100}
        small_times = []
        
        for _ in range(50):
            start_time = time.time()
            response = await performance_client.post("/api/v1/orders", json=small_payload)
            end_time = time.time()
            small_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        # Large payload (simulated)
        large_payload = {
            "symbol": "AAPL",
            "quantity": 100,
            "metadata": {"key" + str(i): "value" * 100 for i in range(100)}  # Large metadata
        }
        large_times = []
        
        for _ in range(50):
            start_time = time.time()
            response = await performance_client.post("/api/v1/orders", json=large_payload)
            end_time = time.time()
            large_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        small_avg = statistics.mean(small_times)
        large_avg = statistics.mean(large_times)
        
        print(f"Small payload avg: {small_avg:.2f}ms")
        print(f"Large payload avg: {large_avg:.2f}ms")
        
        # Large payloads should not cause excessive slowdown
        assert large_avg < small_avg * 2  # No more than 2x slower

    @pytest.mark.asyncio
    async def test_cpu_intensive_operation_performance(self, performance_client):
        """Test performance during CPU-intensive operations."""
        # Simulate CPU load during requests
        def cpu_intensive_task():
            # Simple CPU-bound task
            result = 0
            for i in range(10000):
                result += i ** 2
            return result
        
        # Mix of API requests and CPU tasks
        api_times = []
        
        for _ in range(20):
            # Start CPU task
            cpu_task = asyncio.to_thread(cpu_intensive_task)
            
            # Make API request concurrently
            start_time = time.time()
            response = await performance_client.get("/health")
            end_time = time.time()
            
            api_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
            
            # Wait for CPU task to complete
            await cpu_task
        
        cpu_load_avg = statistics.mean(api_times)
        print(f"API response time under CPU load: {cpu_load_avg:.2f}ms")
        
        # Should handle CPU load reasonably well
        assert cpu_load_avg < 200  # Less than 200ms even under CPU load

    @pytest.mark.asyncio
    async def test_gradual_load_increase(self, performance_client):
        """Test performance under gradually increasing load."""
        load_levels = [1, 2, 5, 10, 20]
        performance_data = {}
        
        for load_level in load_levels:
            response_times = []
            
            # Generate load at this level
            for batch in range(10):  # 10 batches
                tasks = []
                for _ in range(load_level):
                    task = performance_client.get("/health")
                    tasks.append(task)
                
                batch_start = time.time()
                responses = await asyncio.gather(*tasks)
                batch_time = time.time() - batch_start
                
                # Verify responses
                for response in responses:
                    assert response.status_code == 200
                
                response_times.append(batch_time / load_level * 1000)  # Avg per request
            
            avg_time = statistics.mean(response_times)
            performance_data[load_level] = avg_time
            
            print(f"Load level {load_level}: {avg_time:.2f}ms avg response time")
        
        # Performance should degrade gracefully
        for i in range(len(load_levels) - 1):
            current_load = load_levels[i]
            next_load = load_levels[i + 1]
            
            # Response time should not increase too dramatically
            degradation_factor = performance_data[next_load] / performance_data[current_load]
            assert degradation_factor < 2.0  # No more than 2x degradation per load level

    @pytest.mark.asyncio
    async def test_sustained_load_stability(self, performance_client):
        """Test system stability under sustained load."""
        duration_minutes = 1  # 1 minute sustained test
        request_interval = 0.1  # 10 requests per second
        
        response_times = []
        errors = 0
        start_time = time.time()
        
        while time.time() - start_time < duration_minutes * 60:
            try:
                request_start = time.time()
                response = await performance_client.get("/health")
                request_end = time.time()
                
                if response.status_code == 200:
                    response_times.append((request_end - request_start) * 1000)
                else:
                    errors += 1
                    
            except Exception:
                errors += 1
            
            await asyncio.sleep(request_interval)
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]
            
            print(f"Sustained load - avg: {avg_response_time:.2f}ms, p95: {p95_response_time:.2f}ms")
            print(f"Total errors: {errors}")
            
            # System should remain stable
            assert avg_response_time < 100
            assert p95_response_time < 200
            assert errors < len(response_times) * 0.01  # Less than 1% errors