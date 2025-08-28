#!/usr/bin/env python3
"""
Comprehensive System Health Check and Intelligence Assessment
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def check_system_health() -> Dict:
    """Perform comprehensive health check"""
    
    print("🔍 AI Trading System - Comprehensive Health Check")
    print("=" * 50)
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "HEALTHY",
        "intelligence_score": 0,
        "components": {},
        "issues": [],
        "recommendations": []
    }
    
    # Check core components
    print("\n📦 Checking Core Components...")
    
    # 1. API Health
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                results["components"]["api"] = "✅ HEALTHY"
                results["intelligence_score"] += 10
            else:
                results["components"]["api"] = "⚠️ DEGRADED"
                results["issues"].append("API responding but not fully healthy")
    except Exception as e:
        results["components"]["api"] = "❌ DOWN"
        results["issues"].append(f"API not responding: {str(e)}")
        results["overall_status"] = "UNHEALTHY"
    
    # 2. Database Connection
    try:
        from trading_common.database_manager import DatabaseManager
        db = DatabaseManager()
        await db.initialize()
        results["components"]["database"] = "✅ CONNECTED"
        results["intelligence_score"] += 10
    except Exception as e:
        results["components"]["database"] = "❌ DISCONNECTED"
        results["issues"].append(f"Database connection failed: {str(e)}")
    
    # 3. Redis Cache
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url('redis://localhost:6379')
        await redis_client.ping()
        results["components"]["redis"] = "✅ CONNECTED"
        results["intelligence_score"] += 10
        await redis_client.aclose()
    except Exception as e:
        results["components"]["redis"] = "⚠️ DISCONNECTED"
        results["issues"].append(f"Redis not available: {str(e)}")
    
    # 4. Check AI Intelligence
    print("\n🧠 Checking AI Intelligence...")
    
    # Lightweight AI
    try:
        from services.ml.lightweight_intelligence import get_trading_intelligence
        ai = await get_trading_intelligence()
        ai_status = await ai.get_market_intelligence()
        results["components"]["lightweight_ai"] = "✅ ACTIVE"
        results["intelligence_score"] += 30
        results["ai_capabilities"] = ai_status
    except Exception as e:
        results["components"]["lightweight_ai"] = "❌ UNAVAILABLE"
        results["issues"].append(f"Lightweight AI not available: {str(e)}")
    
    # Check for advanced models
    try:
        import ollama
        models = ollama.list()
        if models:
            results["components"]["ollama"] = f"✅ {len(models['models'])} models"
            results["intelligence_score"] += 20
        else:
            results["components"]["ollama"] = "⚠️ No models downloaded"
            results["recommendations"].append("Run ./upgrade_ai_intelligence.sh to download AI models")
    except Exception:
        results["components"]["ollama"] = "❌ Not installed"
        results["recommendations"].append("Install Ollama for advanced AI capabilities")
    
    # 5. Check Trading Services
    print("\n📈 Checking Trading Services...")
    
    services_to_check = [
        ("signal_generator", "services.signal_generator.signal_generation_service"),
        ("order_management", "services.execution.order_management_system"),
        ("risk_monitor", "services.risk_monitor.risk_monitoring_service"),
        ("backtester", "services.ml.intelligent_backtesting")
    ]
    
    for service_name, module_path in services_to_check:
        try:
            module = __import__(module_path, fromlist=[''])
            results["components"][service_name] = "✅ LOADED"
            results["intelligence_score"] += 5
        except Exception as e:
            results["components"][service_name] = "⚠️ NOT LOADED"
            results["issues"].append(f"{service_name} not available: {str(e)}")
    
    # 6. Security Check
    print("\n🔐 Checking Security...")
    
    # Check for environment variables
    critical_vars = ["SECRET_KEY", "JWT_SECRET", "DB_PASSWORD"]
    secure_config = True
    for var in critical_vars:
        if not os.getenv(var):
            secure_config = False
            results["issues"].append(f"Missing critical environment variable: {var}")
    
    if secure_config:
        results["components"]["security"] = "✅ CONFIGURED"
        results["intelligence_score"] += 10
    else:
        results["components"]["security"] = "⚠️ NEEDS CONFIGURATION"
        results["recommendations"].append("Configure all security environment variables")
    
    # Calculate final intelligence score
    max_score = 100
    results["intelligence_score"] = min(results["intelligence_score"], max_score)
    
    # Determine intelligence level
    score = results["intelligence_score"]
    if score >= 90:
        results["intelligence_level"] = "🧠 GENIUS - Fully optimized AI trading system"
    elif score >= 70:
        results["intelligence_level"] = "🎯 ADVANCED - Strong capabilities, some optimization needed"
    elif score >= 50:
        results["intelligence_level"] = "📊 ENHANCED - Good foundation, significant improvements possible"
    elif score >= 30:
        results["intelligence_level"] = "📈 BASIC - Core functionality working, major enhancements needed"
    else:
        results["intelligence_level"] = "⚠️ MINIMAL - Critical components missing"
    
    # Generate recommendations based on score
    if score < 90:
        if "ollama" not in results["components"] or "❌" in results["components"].get("ollama", ""):
            results["recommendations"].append("Install and configure Ollama for advanced AI models")
        if score < 70:
            results["recommendations"].append("Deploy to server for full computational resources")
            results["recommendations"].append("Implement comprehensive monitoring and logging")
        if score < 50:
            results["recommendations"].append("Configure all API keys and external services")
            results["recommendations"].append("Complete security configuration")
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 HEALTH CHECK RESULTS")
    print("=" * 50)
    
    print(f"\nOverall Status: {results['overall_status']}")
    print(f"Intelligence Score: {results['intelligence_score']}/100")
    print(f"Intelligence Level: {results['intelligence_level']}")
    
    print("\n🔧 Component Status:")
    for component, status in results["components"].items():
        print(f"  {component}: {status}")
    
    if results["issues"]:
        print("\n⚠️ Issues Found:")
        for issue in results["issues"]:
            print(f"  • {issue}")
    
    if results["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"  • {rec}")
    
    print("\n" + "=" * 50)
    
    # Save results to file
    with open("health_check_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📁 Results saved to health_check_results.json")
    
    return results


async def main():
    """Run health check"""
    try:
        results = await check_system_health()
        
        # Exit code based on status
        if results["overall_status"] == "UNHEALTHY":
            sys.exit(1)
        elif results["intelligence_score"] < 30:
            sys.exit(2)  # Critical components missing
        else:
            sys.exit(0)  # Success
            
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())