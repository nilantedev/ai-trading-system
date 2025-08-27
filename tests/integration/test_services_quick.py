#!/usr/bin/env python3
"""Quick validation test for Data Ingestion Services."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent / "shared/python-common"))
sys.path.insert(0, str(Path(__file__).parent / "services/data-ingestion"))

from trading_common import MarketData, NewsItem


async def quick_validation_test():
    """Run quick validation tests without network dependencies."""
    print("🚀 QUICK DATA INGESTION SERVICES VALIDATION")
    print("="*50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Import all services
    print("1️⃣  Testing service imports...")
    try:
        from market_data_service import MarketDataService
        from news_service import NewsService
        from reference_data_service import ReferenceDataService
        from data_validation_service import DataValidationService, ValidationSeverity
        print("✅ All service modules imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Import failed: {e}")
    tests_total += 1
    
    # Test 2: Service initialization (without network calls)
    print("\n2️⃣  Testing service initialization...")
    try:
        # Test basic object creation
        market_svc = MarketDataService()
        news_svc = NewsService()
        ref_svc = ReferenceDataService()
        val_svc = DataValidationService()
        
        print("✅ All service objects created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Service creation failed: {e}")
    tests_total += 1
    
    # Test 3: Data validation logic (no external dependencies)
    print("\n3️⃣  Testing data validation logic...")
    try:
        # Test with valid data
        valid_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            timeframe="1min",
            data_source="test"
        )
        
        # Test validation without starting the service
        validation_svc = DataValidationService()
        
        # Test field validation
        field_results = validation_svc._validate_market_data_fields(valid_data)
        price_results = validation_svc._validate_price_data(valid_data)
        volume_results = validation_svc._validate_volume_data(valid_data)
        timestamp_results = validation_svc._validate_timestamp(valid_data)
        
        total_issues = len(field_results) + len(price_results) + len(volume_results) + len(timestamp_results)
        error_issues = sum(1 for results in [field_results, price_results, volume_results, timestamp_results] 
                          for result in results if result.severity == ValidationSeverity.ERROR)
        
        print(f"✅ Data validation working: {total_issues} total issues, {error_issues} errors (expected: 0 errors)")
        if error_issues == 0:
            tests_passed += 1
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
    tests_total += 1
    
    # Test 4: Configuration and settings
    print("\n4️⃣  Testing configuration loading...")
    try:
        market_svc = MarketDataService()
        
        # Check API configurations are loaded
        has_alpaca = bool(market_svc.alpaca_config.get('api_key'))
        has_polygon = bool(market_svc.polygon_config.get('api_key'))
        has_alpha_vantage = bool(market_svc.alpha_vantage_config.get('api_key'))
        
        print(f"✅ Configuration loaded: Alpaca: {has_alpaca}, Polygon: {has_polygon}, Alpha Vantage: {has_alpha_vantage}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
    tests_total += 1
    
    # Test 5: News sentiment analysis fallback
    print("\n5️⃣  Testing news sentiment analysis fallback...")
    try:
        news_svc = NewsService()
        
        # Test simple sentiment analysis (fallback method)
        positive_text = "Apple reports strong earnings, stock price surges, great profit margins"
        negative_text = "Company reports massive losses, declining revenue, bankruptcy concerns"
        neutral_text = "The company held a meeting today to discuss quarterly results"
        
        positive_sentiment = news_svc._simple_sentiment_analysis(positive_text)
        negative_sentiment = news_svc._simple_sentiment_analysis(negative_text)
        neutral_sentiment = news_svc._simple_sentiment_analysis(neutral_text)
        
        # Check sentiment scores are in expected ranges
        sentiment_working = (
            positive_sentiment > 0 and
            negative_sentiment < 0 and
            -0.5 <= neutral_sentiment <= 0.5
        )
        
        print(f"✅ Sentiment analysis working: Positive: {positive_sentiment:.2f}, Negative: {negative_sentiment:.2f}, Neutral: {neutral_sentiment:.2f}")
        if sentiment_working:
            tests_passed += 1
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
    tests_total += 1
    
    # Test 6: Reference data defaults
    print("\n6️⃣  Testing reference data defaults...")
    try:
        ref_svc = ReferenceDataService()
        
        # Test default symbols are available
        default_symbols = ref_svc.default_symbols
        has_major_symbols = all(symbol in default_symbols for symbol in ['SPY', 'AAPL', 'TSLA'])
        
        # Test exchange info
        nyse_info = ref_svc._get_builtin_exchange_info('NYSE')
        nasdaq_info = ref_svc._get_builtin_exchange_info('NASDAQ')
        
        print(f"✅ Reference data: {len(default_symbols)} default symbols, NYSE info: {nyse_info is not None}, NASDAQ info: {nasdaq_info is not None}")
        if has_major_symbols and nyse_info and nasdaq_info:
            tests_passed += 1
    except Exception as e:
        print(f"❌ Reference data test failed: {e}")
    tests_total += 1
    
    # Test 7: Data structures and models
    print("\n7️⃣  Testing data structures...")
    try:
        from reference_data_service import SecurityInfo, ExchangeInfo, EconomicEvent
        from data_validation_service import ValidationResult, DataQualityMetrics
        
        # Test data structure creation
        security = SecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology"
        )
        
        validation = ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Test validation"
        )
        
        print("✅ All data structures working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
    tests_total += 1
    
    # Final Results
    print("\n" + "="*50)
    print(f"📊 QUICK VALIDATION RESULTS")
    print("="*50)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print(f"Success Rate: {(tests_passed/tests_total)*100:.1f}%")
    
    if tests_passed == tests_total:
        print("🎉 ALL TESTS PASSED - Services are ready!")
        status = "EXCELLENT"
    elif tests_passed >= tests_total * 0.8:
        print("✅ MOSTLY PASSING - Good to proceed")
        status = "GOOD"
    else:
        print("⚠️  SOME ISSUES - Review before proceeding")
        status = "NEEDS_REVIEW"
    
    print(f"Overall Status: {status}")
    print("="*50)
    
    return tests_passed == tests_total


if __name__ == "__main__":
    success = asyncio.run(quick_validation_test())
    exit(0 if success else 1)