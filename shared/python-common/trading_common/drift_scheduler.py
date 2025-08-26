#!/usr/bin/env python3
"""Drift Scheduler

Replaces earlier prototype. Periodically runs `run_model_drift_scan` for registered
models using the existing drift_detection API signature.

Design choices:
- Explicit model registration (vs auto-scanning registry) keeps control surface simple.
- Jittered initial delay prevents sync across replicas.
- Provides start/stop helpers wrapping a singleton monitor.

Non-goals: distributed coordination, auto-demotion logic, alert routing.
"""
from __future__ import annotations
import asyncio
import logging
import random
from datetime import datetime
from typing import List, Optional, Tuple

from .drift_detection import run_model_drift_scan

logger = logging.getLogger(__name__)

class DriftMonitor:
    def __init__(self, interval_seconds: int = 3600):
        self.interval = interval_seconds
        self.models: List[Tuple[str, Optional[str]]] = []  # (model_name, version)
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.scans_run = 0
        self.last_run: Optional[datetime] = None
        self.failures = 0

    def register_model(self, model_name: str, version: Optional[str] = None):
        pair = (model_name, version)
        if pair not in self.models:
            self.models.append(pair)
            logger.info("Registered model for drift monitoring: %s%s", model_name, f"@{version}" if version else "")

    async def start(self):
        if self._task and not self._task.done():  # already running
            return
        jitter = random.uniform(0, min(60, self.interval * 0.1))
        logger.info("Starting drift monitor (interval=%ss, jitter=%.2fs) models=%d", self.interval, jitter, len(self.models))
        await asyncio.sleep(jitter)
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="drift-monitor-loop")

    async def stop(self):
        logger.info("Stopping drift monitor")
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        while not self._stop_event.is_set():
            start = datetime.utcnow()
            if not self.models:
                logger.debug("Drift monitor tick: no models registered")
            for model_name, version in list(self.models):
                try:
                    result = await run_model_drift_scan(model_name, version)
                    self.scans_run += 1
                    self.last_run = datetime.utcnow()
                    logger.info("Drift scan complete model=%s version=%s worst=%s features=%s status=%s", result['model_name'], result['version'], result.get('worst_severity'), result.get('features_evaluated'), result.get('status','ok'))
                except Exception as e:  # pragma: no cover
                    self.failures += 1
                    logger.warning("Drift scan failed for %s%s: %s", model_name, f"@{version}" if version else "", e)
            # Sleep respecting stop event
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                continue

_MONITOR: Optional[DriftMonitor] = None

async def start_drift_monitor(interval_seconds: int = 3600, models: Optional[List[Tuple[str, Optional[str]]]] = None) -> DriftMonitor:
    """Create or reuse singleton DriftMonitor and start loop."""
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = DriftMonitor(interval_seconds=interval_seconds)
        if models:
            for m, v in models:
                _MONITOR.register_model(m, v)
        await _MONITOR.start()
    return _MONITOR

async def stop_drift_monitor(monitor: Optional[DriftMonitor] = None):
    global _MONITOR
    mon = monitor or _MONITOR
    if mon:
        await mon.stop()
    if mon is _MONITOR:
        _MONITOR = None

__all__ = ["DriftMonitor", "start_drift_monitor", "stop_drift_monitor"]
