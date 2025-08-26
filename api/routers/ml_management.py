#!/usr/bin/env python3
"""ML Management Endpoints: backtest & promotion.

Endpoints:
 - POST /api/v1/ml/{model_name}/backtest
 - POST /api/v1/ml/{model_name}/promote

Promotion uses simple policy thresholds (will be externalized via config in later task).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from trading_common.ml_pipeline import get_ml_pipeline, TrainingConfig, ModelType
from trading_common.model_registry import get_model_registry, ModelState, PromotionError
from trading_common.promotion_policy import get_policy_for_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml-management"])


async def auth_dependency():  # placeholder
    return True


class BacktestRequest(BaseModel):
    version: Optional[str] = None
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001


class BacktestResponse(BaseModel):
    model_name: str
    version: str
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    total_trades: int
    win_rate: float


@router.post("/{model_name}/backtest", response_model=BacktestResponse)
async def backtest(model_name: str, req: BacktestRequest, _: bool = Depends(auth_dependency)):
    pipeline = await get_ml_pipeline()
    registry = await get_model_registry()
    entry = await registry.get_entry(model_name, req.version) if req.version else await registry.get_entry(model_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")
    model = await pipeline.get_model(model_name, entry.version)
    if not model:
        raise HTTPException(status_code=400, detail="Model artifact not loadable")
    result = await pipeline.backtest_model(model, req.start_date, req.end_date, req.initial_capital, req.transaction_cost)
    return BacktestResponse(
        model_name=model_name,
        version=entry.version,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        total_return=result.total_return,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
    )


class PromotionRequest(BaseModel):
    version: str
    target_state: ModelState = Field(..., description="Target state e.g. STAGING or PRODUCTION")
    reason: str = Field(..., description="Business or experimental justification")
    # Temporary inline policy until externalized
    min_sharpe: float | None = None
    max_drawdown: float | None = None  # negative value e.g. -0.2


class PromotionResponse(BaseModel):
    model_name: str
    version: str
    new_state: str
    promotion_history: Any


@router.post("/{model_name}/promote", response_model=PromotionResponse)
async def promote(model_name: str, req: PromotionRequest, _: bool = Depends(auth_dependency)):
    registry = await get_model_registry()
    # Merge request overrides with policy file values
    base_policy = get_policy_for_model(model_name)
    policy = {
        "min_sharpe": req.min_sharpe if req.min_sharpe is not None else base_policy.get("min_sharpe"),
        "max_drawdown": req.max_drawdown if req.max_drawdown is not None else base_policy.get("max_drawdown"),
    }
    try:
        entry = await registry.promote(model_name, req.version, req.target_state, req.reason, policy)
    except PromotionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PromotionResponse(
        model_name=model_name,
        version=req.version,
        new_state=entry.state.value,
        promotion_history=entry.promotion_history,
    )

__all__ = ["router"]
