"""
Trading Governor API Router - Control panel endpoints for safe unattended operation
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.risk_monitor.trading_governor import TradingGovernor, TradingMode

router = APIRouter(prefix="/api/v1/governor", tags=["governor"])

# Initialize governor
governor = TradingGovernor()

class ModeRequest(BaseModel):
    mode: str

class SettingRequest(BaseModel):
    key: str
    value: Any

class EmergencyStopRequest(BaseModel):
    reason: str

@router.post("/emergency-stop")
async def emergency_stop(request: EmergencyStopRequest):
    """EMERGENCY STOP - Halt all trading immediately"""
    result = await governor.emergency_stop(request.reason)
    return result

@router.post("/mode")
async def set_trading_mode(request: ModeRequest):
    """Set trading mode (stopped/paper/conservative/normal/aggressive)"""
    try:
        mode = TradingMode(request.mode)
        await governor.apply_trading_mode(mode)
        return {"status": "success", "mode": mode.value}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

@router.post("/setting")
async def update_setting(request: SettingRequest):
    """Update a specific setting"""
    await governor.update_setting(request.key, request.value)
    return {"status": "success", "setting": request.key, "value": request.value}

@router.get("/state")
async def get_system_state():
    """Get current system state for control panel"""
    state = await governor.get_current_state()
    return state

@router.get("/settings")
async def get_all_settings():
    """Get all current settings"""
    state = await governor.get_current_state()
    return state.get("settings", {})

@router.post("/validate-trade")
async def validate_trade(symbol: str, amount: float):
    """Check if a trade would be allowed under current settings"""
    allowed, reason = await governor.can_trade(symbol, amount)
    return {
        "allowed": allowed,
        "reason": reason,
        "symbol": symbol,
        "amount": amount
    }

@router.get("/health")
async def governor_health():
    """Check governor service health"""
    health = await governor._system_health_check()
    return health