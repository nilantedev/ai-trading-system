#!/usr/bin/env python3
"""ML Model Inference & Metadata Endpoints.

Endpoints:
 - POST /api/v1/models/{model_name}/predict
 - GET  /api/v1/models/{model_name}/status
 - GET  /api/v1/models/{model_name}/versions

Notes:
 - Uses feature store to assemble latest feature vector(s) for provided entity_ids.
 - Validates feature graph hash against registry stored hash; blocks prediction if mismatch.
 - Simple auth dependency placeholder (reuse existing auth if available).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import pandas as pd

from trading_common.ml_pipeline import get_ml_pipeline
from trading_common.model_registry import get_model_registry, ModelState
from trading_common.feature_graph import validate_feature_graph
from trading_common.feature_store import get_feature_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["ml-models"])


class PredictionRequest(BaseModel):
    entity_ids: List[str] = Field(..., description="Symbols or entity IDs")
    features: Optional[List[str]] = Field(None, description="Override feature list (default = model config)")
    as_of: Optional[datetime] = Field(None, description="Timestamp for feature snapshot (default now)")


class PredictionResponse(BaseModel):
    model_name: str
    version: str
    predictions: Dict[str, float]
    as_of: datetime
    feature_graph_valid: bool


async def auth_dependency():  # placeholder for real auth
    return True


@router.post("/{model_name}/predict", response_model=PredictionResponse)
async def predict(model_name: str, request: PredictionRequest, _: bool = Depends(auth_dependency)):
    pipeline = await get_ml_pipeline()
    registry = await get_model_registry()
    entry = await registry.get_entry(model_name)
    if not entry or entry.state not in (ModelState.PRODUCTION, ModelState.STAGING):
        raise HTTPException(status_code=404, detail="Model not found or not deployable")
    model = await pipeline.get_model(model_name, entry.version)
    if not model or not model.is_trained:
        raise HTTPException(status_code=400, detail="Model not loaded or untrained")
    feature_store = await get_feature_store()
    feature_names = request.features or model.config.feature_names
    # Validate feature graph
    fg_valid = await validate_feature_graph(feature_store, entry.feature_graph_hash, feature_names)
    if not fg_valid:
        raise HTTPException(status_code=409, detail="Feature graph mismatch; retrain model or update registry")
    as_of = request.as_of or datetime.utcnow()
    rows = []
    for eid in request.entity_ids:
        vec = await feature_store.get_feature_vector(eid, as_of, feature_names)
        if not vec:
            continue
        row = {**vec.get_feature_dict()}
        row['entity_id'] = eid
        rows.append(row)
    if not rows:
        raise HTTPException(status_code=400, detail="No feature vectors available for requested entities")
    df = pd.DataFrame(rows)
    if 'entity_id' in df.columns:
        feature_df = df[feature_names].copy()
    else:
        feature_df = df
    try:
        preds = await model.predict(feature_df)
    except Exception as e:
        logger.exception("Prediction failure")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    prediction_map = {}
    for eid, val in zip(df['entity_id'], preds):
        prediction_map[eid] = float(val)
    return PredictionResponse(
        model_name=model_name,
        version=entry.version,
        predictions=prediction_map,
        as_of=as_of,
        feature_graph_valid=fg_valid,
    )


class ModelStatus(BaseModel):
    model_name: str
    version: str
    state: str
    metrics: Dict[str, Any]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    feature_graph_hash: str


@router.get("/{model_name}/status", response_model=ModelStatus)
async def status(model_name: str, _: bool = Depends(auth_dependency)):
    registry = await get_model_registry()
    entry = await registry.get_entry(model_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelStatus(
        model_name=model_name,
        version=entry.version,
        state=entry.state.value,
        metrics=entry.metrics,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        feature_graph_hash=entry.feature_graph_hash,
    )


class VersionInfo(BaseModel):
    version: str
    state: str
    created_at: Optional[datetime]


@router.get("/{model_name}/versions", response_model=List[VersionInfo])
async def versions(model_name: str, _: bool = Depends(auth_dependency)):
    registry = await get_model_registry()
    # Fetch all versions (simple query)
    rows = await registry.db.fetch_all("SELECT version, state, created_at FROM model_registry WHERE model_name=%s ORDER BY created_at DESC", [model_name])
    if not rows:
        raise HTTPException(status_code=404, detail="Model not found")
    return [VersionInfo(version=r['version'], state=r['state'], created_at=r.get('created_at')) for r in rows]

__all__ = ["router"]
