#!/usr/bin/env python3
"""Promotion Policy Loader.

Loads YAML-based promotion policy thresholds and exposes a helper to fetch
policy for a given model (with strategy-specific overrides by tag or name).
"""
from __future__ import annotations

import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

DEFAULT_PATHS = [
    Path("config/ml_promotion_policy.yaml"),
    Path("../config/ml_promotion_policy.yaml"),
]


@lru_cache(maxsize=1)
def _load_policy_raw() -> Dict[str, Any]:
    for p in DEFAULT_PATHS:
        if p.exists():
            try:
                with open(p, 'r') as f:
                    data = yaml.safe_load(f) or {}
                return data
            except Exception as e:
                logger.error("Failed to load promotion policy from %s: %s", p, e)
                return {}
    logger.warning("Promotion policy file not found; using empty policy")
    return {}


def get_policy_for_model(model_name: str) -> Dict[str, Any]:
    raw = _load_policy_raw()
    default = raw.get('default', {})
    overrides = raw.get('overrides', {})
    # Strategy mapping: look for override key contained in model_name
    for key, ov in overrides.items():
        if key in model_name:
            merged = {**default, **ov}
            logger.debug("Applied promotion policy override '%s' for model '%s'", key, model_name)
            return merged
    return default

__all__ = ["get_policy_for_model"]
