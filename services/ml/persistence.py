from __future__ import annotations
"""Persistence helpers for ML orchestration audit logging.

This module defines SQLAlchemy models and async helper to persist promotion audit
records. Designed to be best-effort: failures do not block orchestrator flow.
"""
from datetime import datetime
from typing import Any, Dict
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from sqlalchemy import String, DateTime, JSON, Integer

Base = declarative_base()

class PromotionAuditEntry(Base):
    __tablename__ = 'ml_promotion_audit'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=True)
    model_type: Mapped[str] = mapped_column(String(64), index=True, nullable=True)
    decision: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow, index=True, nullable=False)
    details: Mapped[Dict[str, Any] | None] = mapped_column(JSON, nullable=True)


_engine = None
_Session = None

async def get_engine():
    global _engine, _Session
    if _engine is None:
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            # Fallback to individual env vars (compose style)
            user = os.getenv('DB_USER', 'trading_user')
            pwd = os.getenv('DB_PASSWORD', 'trading_pass')
            host = os.getenv('DB_HOST', 'trading-postgres')
            port = os.getenv('DB_PORT', '5432')
            name = os.getenv('DB_NAME', 'trading_db')
            db_url = f"postgresql+asyncpg://{user}:{pwd}@{host}:{port}/{name}"
        _engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
        _Session = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine

async def get_session() -> AsyncSession:
    await get_engine()
    return _Session()  # type: ignore

async def persist_promotion_audit(entry: Dict[str, Any]):
    """Persist a promotion audit entry (non-blocking best-effort)."""
    try:
        session = await get_session()
        async with session.begin():
            rec = PromotionAuditEntry(
                model_id=entry.get('model_id','unknown'),
                symbol=entry.get('symbol'),
                model_type=entry.get('model_type'),
                decision=entry.get('decision','unknown'),
                details=entry.get('details')
            )
            session.add(rec)
    except Exception:
        # Silent fail (logged upstream in orchestrator if needed)
        pass
