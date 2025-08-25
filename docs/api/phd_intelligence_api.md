# PhD-Level Intelligence API Documentation

## Overview

The PhD-Level Intelligence API provides access to cutting-edge machine learning models for advanced market analysis and trading signal generation. This system represents a revolutionary 10x leap in sophistication from traditional technical analysis.

## Base URL
```
http://localhost:8000/api/v1/intelligence
```

## Authentication
All endpoints require bearer token authentication:
```http
Authorization: Bearer <token>
```

---

## 🧠 Graph Neural Network API

### GET /gnn/market-structure
Get current market structure analysis from Graph Neural Network.

**Response:**
```json
{
  "timestamp": "2025-08-25T10:30:00Z",
  "market_structure": {
    "graph_density": 0.73,
    "clustering_coefficient": 0.42,
    "network_centrality": {
      "AAPL": 0.95,
      "MSFT": 0.89,
      "GOOGL": 0.87
    },
    "community_detection": {
      "tech_cluster": ["AAPL", "MSFT", "GOOGL"],
      "finance_cluster": ["JPM", "BAC", "GS"]
    }
  },
  "predictions": {
    "AAPL": {
      "signal": 0.73,
      "confidence": 0.86,
      "network_influence": 0.91
    }
  }
}
```

### POST /gnn/analyze-symbol
Analyze specific symbol using Graph Neural Network.

**Request:**
```json
{
  "symbol": "AAPL",
  "include_network_effects": true,
  "time_horizon": "1d"
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "gnn_signal": 0.73,
  "confidence": 0.86,
  "network_position": {
    "centrality": 0.95,
    "clustering": 0.67,
    "influence_score": 0.91
  },
  "connected_assets": [
    {"symbol": "MSFT", "correlation": 0.78, "causality": 0.34},
    {"symbol": "GOOGL", "correlation": 0.71, "causality": 0.29}
  ]
}
```

---

## 📊 Advanced Factor Models API

### GET /factors/current-exposures
Get current factor exposures and loadings.

**Response:**
```json
{
  "timestamp": "2025-08-25T10:30:00Z",
  "factors": {
    "market": {
      "value": 0.012,
      "t_stat": 2.34,
      "p_value": 0.019
    },
    "size": {
      "value": -0.008,
      "t_stat": -1.89,
      "p_value": 0.059
    },
    "value": {
      "value": 0.015,
      "t_stat": 3.12,
      "p_value": 0.002
    },
    "profitability": {
      "value": 0.009,
      "t_stat": 2.78,
      "p_value": 0.005
    },
    "investment": {
      "value": -0.006,
      "t_stat": -1.45,
      "p_value": 0.146
    },
    "momentum": {
      "value": 0.018,
      "t_stat": 4.23,
      "p_value": 0.000
    }
  },
  "portfolio_alpha": 0.023,
  "alpha_t_stat": 2.89,
  "r_squared": 0.78
}
```

### POST /factors/analyze-symbol
Get factor analysis for specific symbol.

**Request:**
```json
{
  "symbol": "AAPL",
  "lookback_period": 252,
  "include_factor_timing": true
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "factor_loadings": {
    "market": 1.12,
    "size": -0.23,
    "value": -0.45,
    "profitability": 0.67,
    "investment": -0.12,
    "momentum": 0.89
  },
  "expected_return": 0.085,
  "factor_signal": 0.42,
  "alpha": 0.019,
  "risk_attribution": {
    "systematic_risk": 0.78,
    "idiosyncratic_risk": 0.22
  }
}
```

---

## 🔗 Transfer Entropy Analysis API

### GET /causality/network
Get current causality network between assets.

**Response:**
```json
{
  "timestamp": "2025-08-25T10:30:00Z",
  "causality_matrix": {
    "AAPL": {
      "MSFT": 0.034,
      "GOOGL": 0.029,
      "SPY": 0.012
    },
    "MSFT": {
      "AAPL": 0.028,
      "GOOGL": 0.031
    }
  },
  "information_flow": {
    "leading_indicators": [
      {"symbol": "SPY", "influence_score": 0.89},
      {"symbol": "VIX", "influence_score": 0.73}
    ],
    "lagging_indicators": [
      {"symbol": "QQQ", "lag": 2, "strength": 0.45}
    ]
  }
}
```

### POST /causality/predict-flow
Predict information flow effects for trading decision.

**Request:**
```json
{
  "target_symbol": "AAPL",
  "leading_symbols": ["SPY", "VIX", "MSFT"],
  "prediction_horizon": 5
}
```

**Response:**
```json
{
  "target_symbol": "AAPL",
  "causality_signal": 0.67,
  "confidence": 0.82,
  "predicted_movements": [
    {
      "lag": 1,
      "probability": 0.73,
      "direction": "up",
      "magnitude": 0.015
    },
    {
      "lag": 2,
      "probability": 0.68,
      "direction": "up",
      "magnitude": 0.008
    }
  ],
  "information_sources": [
    {"symbol": "SPY", "contribution": 0.45},
    {"symbol": "VIX", "contribution": 0.32},
    {"symbol": "MSFT", "contribution": 0.23}
  ]
}
```

---

## 📈 Stochastic Volatility Models API

### GET /volatility/surface
Get current volatility surface from stochastic models.

**Response:**
```json
{
  "timestamp": "2025-08-25T10:30:00Z",
  "models": {
    "heston": {
      "current_vol": 0.23,
      "long_term_vol": 0.21,
      "vol_of_vol": 0.42,
      "mean_reversion": 2.34,
      "correlation": -0.67
    },
    "sabr": {
      "alpha": 0.18,
      "beta": 0.85,
      "rho": -0.32,
      "nu": 0.47
    }
  },
  "volatility_forecast": {
    "1d": 0.231,
    "5d": 0.225,
    "10d": 0.219,
    "30d": 0.214
  },
  "regime": "medium_volatility",
  "regime_probability": 0.78
}
```

### POST /volatility/option-pricing
Price options using stochastic volatility models.

**Request:**
```json
{
  "symbol": "AAPL",
  "strike": 150,
  "expiry": "2025-09-15",
  "option_type": "call",
  "model": "heston"
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "option_price": 12.45,
  "greeks": {
    "delta": 0.67,
    "gamma": 0.023,
    "theta": -0.045,
    "vega": 0.234,
    "rho": 0.089
  },
  "implied_volatility": 0.228,
  "model_confidence": 0.91,
  "volatility_surface": {
    "atm_vol": 0.23,
    "skew": -0.15,
    "term_structure": 0.08
  }
}
```

---

## 🎯 Advanced Intelligence Coordinator API

### GET /coordinator/ensemble-signal
Get coordinated signal from all PhD-level models.

**Response:**
```json
{
  "timestamp": "2025-08-25T10:30:00Z",
  "market_regime": {
    "volatility_regime": "medium",
    "factor_regime": "bull",
    "network_regime": "connected",
    "causality_regime": "trending",
    "overall_regime": "favorable",
    "regime_confidence": 0.84
  },
  "ensemble_signals": {
    "AAPL": {
      "gnn_signal": 0.73,
      "factor_signal": 0.42,
      "causality_signal": 0.67,
      "volatility_signal": 0.31,
      "ensemble_signal": 0.58,
      "confidence": 0.79,
      "optimal_holding_period": 3,
      "risk_adjusted_signal": 0.74
    }
  },
  "portfolio_optimization": {
    "recommended_positions": {
      "AAPL": 0.15,
      "MSFT": 0.12,
      "GOOGL": 0.10
    },
    "expected_return": 0.089,
    "expected_volatility": 0.167,
    "sharpe_ratio": 0.532
  }
}
```

### POST /coordinator/analyze-portfolio
Get comprehensive analysis for portfolio optimization.

**Request:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"],
  "current_positions": {
    "AAPL": 0.25,
    "MSFT": 0.20,
    "GOOGL": 0.15,
    "TSLA": 0.10
  },
  "risk_budget": 0.20
}
```

**Response:**
```json
{
  "portfolio_analysis": {
    "current_risk": 0.187,
    "expected_return": 0.092,
    "factor_exposures": {
      "market": 1.05,
      "size": -0.12,
      "value": -0.23
    },
    "diversification_ratio": 0.78
  },
  "recommendations": {
    "rebalancing_needed": true,
    "suggested_positions": {
      "AAPL": 0.22,
      "MSFT": 0.18,
      "GOOGL": 0.17,
      "TSLA": 0.08
    },
    "expected_improvement": {
      "return_increase": 0.007,
      "risk_reduction": 0.015,
      "sharpe_improvement": 0.089
    }
  },
  "risk_metrics": {
    "var_95": -0.032,
    "cvar_95": -0.045,
    "maximum_drawdown": 0.087
  }
}
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": {
    "code": "MODEL_ERROR",
    "message": "Graph neural network model is currently retraining",
    "details": {
      "retry_after": 300,
      "fallback_available": true
    }
  }
}
```

### Error Codes
- `MODEL_ERROR`: Model is unavailable or retraining
- `INSUFFICIENT_DATA`: Not enough historical data for analysis
- `INVALID_SYMBOL`: Symbol not supported or invalid
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `AUTHENTICATION_ERROR`: Invalid or expired token

---

## Rate Limits

- **Standard endpoints**: 1000 requests/hour
- **Heavy computation endpoints**: 100 requests/hour
- **Real-time endpoints**: 10 requests/minute

## WebSocket Real-Time Intelligence

For real-time updates, connect to:
```
ws://localhost:8000/ws/intelligence
```

Subscribe to channels:
- `gnn_signals` - Real-time graph neural network signals
- `factor_updates` - Factor model updates
- `causality_events` - Information flow events  
- `volatility_regime` - Volatility regime changes
- `ensemble_signals` - Coordinated ensemble signals