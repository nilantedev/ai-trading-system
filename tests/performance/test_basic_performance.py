#!/usr/bin/env python3
"""
Basic Performance Tests
"""

import pytest
import time
import statistics
import asyncio
from datetime import datetime


class TestBasicPerformance:
    """Basic performance tests to verify framework."""

    def test_simple_computation_performance(self):
        """Test simple computation performance."""
        response_times = []
        
        def simple_computation():
            """Simple computation for timing."""
            result = 0
            for i in range(1000):
                result += i ** 2
            return result
        
        # Measure computation times
        for _ in range(100):
            start_time = time.time()
            result = simple_computation()
            end_time = time.time()
            
            assert result > 0
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
        
        # Performance metrics
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        print(f"Simple computation - avg: {avg_time:.3f}ms, p95: {p95_time:.3f}ms")
        
        assert avg_time < 10  # Should be very fast
        assert p95_time < 20

    @pytest.mark.asyncio
    async def test_async_operation_performance(self):
        """Test async operation performance."""
        response_times = []
        
        async def async_operation():
            """Simple async operation."""
            await asyncio.sleep(0.001)  # 1ms delay
            return {"status": "completed", "timestamp": datetime.utcnow().isoformat()}
        
        # Measure async operation times
        for _ in range(50):
            start_time = time.time()
            result = await async_operation()
            end_time = time.time()
            
            assert result["status"] == "completed"
            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)
        
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]
        
        print(f"Async operation - avg: {avg_time:.2f}ms, p95: {p95_time:.2f}ms")
        
        assert avg_time < 10  # Should be fast even with sleep
        assert p95_time < 20

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self):
        """Test concurrent operations performance."""
        async def worker_task(worker_id):
            """Simple worker task."""
            await asyncio.sleep(0.001)  # 1ms work
            return f"worker_{worker_id}_completed"
        
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = [worker_task(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            assert len(results) == concurrency
            assert all("completed" in result for result in results)
            
            print(f"Concurrency {concurrency}: {total_time:.2f}ms total")
            
            # Should scale well with concurrency
            assert total_time < 100  # Less than 100ms even for 20 concurrent tasks

    def test_data_processing_performance(self):
        """Test data processing performance."""
        # Generate test data
        test_data = [
            {
                "symbol": f"STOCK_{i}",
                "price": 100.0 + (i * 0.5),
                "volume": 1000000 + (i * 1000),
                "timestamp": datetime.utcnow().isoformat()
            }
            for i in range(1000)
        ]
        
        def process_data(data):
            """Process market data."""
            processed = []
            for item in data:
                processed_item = {
                    "symbol": item["symbol"],
                    "price": item["price"],
                    "value": item["price"] * item["volume"],
                    "processed_at": datetime.utcnow().isoformat()
                }
                processed.append(processed_item)
            return processed
        
        # Measure processing time
        start_time = time.time()
        processed_data = process_data(test_data)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        
        assert len(processed_data) == len(test_data)
        assert all("value" in item for item in processed_data)
        
        print(f"Data processing (1000 items): {processing_time:.2f}ms")
        
        # Should process 1000 items quickly
        assert processing_time < 100  # Less than 100ms

    def test_memory_usage_simulation(self):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_data_sets = []
        for i in range(10):
            data_set = [j for j in range(10000)]  # 10K integers
            large_data_sets.append(data_set)
        
        # Process the data
        processed_results = []
        for data_set in large_data_sets:
            result = sum(data_set)
            processed_results.append(result)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")
        
        assert len(processed_results) == 10
        # Memory growth should be reasonable for this operation
        assert memory_growth < 100  # Less than 100MB growth

    def test_throughput_calculation(self):
        """Test throughput calculation accuracy."""
        operations_per_batch = 100
        num_batches = 10
        
        def batch_operation():
            """Simulate batch of operations."""
            results = []
            for i in range(operations_per_batch):
                result = i ** 2
                results.append(result)
            return results
        
        start_time = time.time()
        
        total_operations = 0
        for batch in range(num_batches):
            batch_results = batch_operation()
            total_operations += len(batch_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = total_operations / total_time
        
        print(f"Throughput: {throughput:.1f} operations/second")
        print(f"Total operations: {total_operations}, Time: {total_time:.3f}s")
        
        assert total_operations == operations_per_batch * num_batches
        assert throughput > 10000  # Should achieve good throughput

    @pytest.mark.parametrize("data_size", [100, 500, 1000, 5000])
    def test_scaling_performance(self, data_size):
        """Test performance scaling with different data sizes."""
        # Generate data of specified size
        test_data = list(range(data_size))
        
        def process_list(data):
            """Process list of numbers."""
            return [x * 2 + 1 for x in data]
        
        start_time = time.time()
        result = process_list(test_data)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        per_item_time = processing_time / data_size
        
        print(f"Data size {data_size}: {processing_time:.2f}ms total, {per_item_time:.4f}ms per item")
        
        assert len(result) == data_size
        # Per-item time should remain relatively constant (linear scaling)
        assert per_item_time < 0.1  # Less than 0.1ms per item

    def test_error_handling_performance(self):
        """Test performance of error handling."""
        success_times = []
        error_times = []
        
        def operation_with_errors(should_fail=False):
            """Operation that may fail."""
            if should_fail:
                raise ValueError("Simulated error")
            return "success"
        
        # Measure successful operations
        for _ in range(50):
            start_time = time.time()
            try:
                result = operation_with_errors(False)
                end_time = time.time()
                success_times.append((end_time - start_time) * 1000)
            except:
                pass
        
        # Measure error handling
        for _ in range(50):
            start_time = time.time()
            try:
                result = operation_with_errors(True)
            except ValueError:
                end_time = time.time()
                error_times.append((end_time - start_time) * 1000)
        
        success_avg = statistics.mean(success_times)
        error_avg = statistics.mean(error_times)
        
        print(f"Success path avg: {success_avg:.3f}ms")
        print(f"Error path avg: {error_avg:.3f}ms")
        
        # Error handling should not be significantly slower
        assert error_avg < success_avg * 2  # No more than 2x slower
        assert error_avg < 1  # Should still be very fast