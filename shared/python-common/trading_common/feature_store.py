#!/usr/bin/env python3
"""
Feature Store for Trading System
Production-grade feature management with lineage tracking, versioning, and real-time serving.
Supports both batch and streaming feature computation with automatic data quality monitoring.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import uuid

from .cache import get_trading_cache
from .database import get_database
from .models import MarketData

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Types of features in the store."""
    MARKET = "market"         # Price, volume, volatility
    TECHNICAL = "technical"   # Technical indicators  
    FUNDAMENTAL = "fundamental"  # Financial metrics
    SENTIMENT = "sentiment"   # News, social sentiment
    MACRO = "macro"          # Economic indicators
    DERIVED = "derived"      # Engineered features
    ENSEMBLE = "ensemble"    # Multi-model features


class ComputeMode(str, Enum):
    """Feature computation modes."""
    BATCH = "batch"          # Offline batch computation
    STREAMING = "streaming"  # Real-time streaming
    ON_DEMAND = "on_demand"  # Computed when requested


@dataclass 
class FeatureDefinition:
    """Definition of a feature with metadata and lineage."""
    name: str
    feature_type: FeatureType
    description: str
    data_type: str  # "float64", "int64", "string", etc.
    compute_mode: ComputeMode
    dependencies: List[str] = field(default_factory=list)  # Feature dependencies
    source_tables: List[str] = field(default_factory=list)  # Data source tables
    transformation_logic: Optional[str] = None  # SQL or Python code
    update_frequency: Optional[str] = None  # "1min", "1h", "1d", etc.
    lookback_window: Optional[str] = None  # "30d", "1y", etc.
    
    # Metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: Set[str] = field(default_factory=set)
    
    # Data quality
    expected_range: Optional[Tuple[float, float]] = None
    null_tolerance: float = 0.1  # Max allowed null ratio
    
    def get_feature_id(self) -> str:
        """Generate unique feature ID based on name and version."""
        return f"{self.name}:{self.version}"
    
    def get_hash(self) -> str:
        """Get hash of feature definition for change tracking."""
        content = f"{self.name}{self.transformation_logic}{self.dependencies}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class FeatureValue:
    """A computed feature value with metadata."""
    feature_name: str
    entity_id: str  # e.g., symbol "AAPL"
    timestamp: datetime
    value: Any
    confidence: Optional[float] = None
    data_quality_score: Optional[float] = None
    computation_time: Optional[datetime] = None
    feature_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feature_name": self.feature_name,
            "entity_id": self.entity_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "confidence": self.confidence,
            "data_quality_score": self.data_quality_score,
            "computation_time": self.computation_time.isoformat() if self.computation_time else None,
            "feature_version": self.feature_version
        }


@dataclass
class FeatureVector:
    """Collection of features for a single entity at a point in time."""
    entity_id: str
    timestamp: datetime
    features: Dict[str, FeatureValue]
    vector_id: Optional[str] = None
    
    def __post_init__(self):
        if self.vector_id is None:
            self.vector_id = f"{self.entity_id}_{int(self.timestamp.timestamp())}"
    
    def get_feature_array(self, feature_names: List[str]) -> np.ndarray:
        """Get numpy array of feature values in specified order."""
        values = []
        for name in feature_names:
            if name in self.features:
                values.append(self.features[name].value)
            else:
                values.append(np.nan)
        return np.array(values)
    
    def get_feature_dict(self) -> Dict[str, Any]:
        """Get dictionary of feature name -> value."""
        return {name: fv.value for name, fv in self.features.items()}


class BaseFeatureComputer(ABC):
    """Base class for feature computation engines."""
    
    def __init__(self, feature_def: FeatureDefinition):
        self.feature_def = feature_def
        
    @abstractmethod
    async def compute_features(self, entity_ids: List[str], 
                             start_time: datetime, 
                             end_time: datetime) -> List[FeatureValue]:
        """Compute features for given entities and time range."""
        pass
    
    @abstractmethod 
    async def compute_streaming_feature(self, entity_id: str, 
                                      market_data: MarketData) -> Optional[FeatureValue]:
        """Compute feature value from streaming market data."""
        pass


class TechnicalIndicatorComputer(BaseFeatureComputer):
    """Computes technical indicator features."""
    
    async def compute_features(self, entity_ids: List[str],
                             start_time: datetime,
                             end_time: datetime) -> List[FeatureValue]:
        """Compute technical indicators for symbols."""
        features = []
        
        # Get market data for computation
        db = await get_database()
        
        for entity_id in entity_ids:
            # Fetch historical data
            query = """
            SELECT timestamp, close, high, low, volume 
            FROM market_data 
            WHERE symbol = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            
            rows = await db.fetch_all(query, [entity_id, start_time, end_time])
            if len(rows) < 20:  # Need minimum data for indicators
                continue
            
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Compute various technical indicators
            indicator_values = await self._compute_indicators(df, entity_id)
            features.extend(indicator_values)
        
        return features
    
    async def _compute_indicators(self, df: pd.DataFrame, symbol: str) -> List[FeatureValue]:
        """Compute technical indicators from price data."""
        features = []
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            sma = df['close'].rolling(window=period).mean()
            for timestamp, value in sma.dropna().items():
                features.append(FeatureValue(
                    feature_name=f"sma_{period}",
                    entity_id=symbol,
                    timestamp=timestamp,
                    value=float(value),
                    computation_time=datetime.utcnow(),
                    feature_version=self.feature_def.version
                ))
        
        # RSI
        rsi = self._calculate_rsi(df['close'], 14)
        for timestamp, value in rsi.dropna().items():
            features.append(FeatureValue(
                feature_name="rsi_14",
                entity_id=symbol,
                timestamp=timestamp,
                value=float(value),
                computation_time=datetime.utcnow(),
                feature_version=self.feature_def.version
            ))
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], 20, 2)
        for timestamp in bb_upper.dropna().index:
            features.extend([
                FeatureValue("bb_upper_20_2", symbol, timestamp, float(bb_upper[timestamp]), 
                           computation_time=datetime.utcnow(), feature_version=self.feature_def.version),
                FeatureValue("bb_middle_20_2", symbol, timestamp, float(bb_middle[timestamp]),
                           computation_time=datetime.utcnow(), feature_version=self.feature_def.version),
                FeatureValue("bb_lower_20_2", symbol, timestamp, float(bb_lower[timestamp]),
                           computation_time=datetime.utcnow(), feature_version=self.feature_def.version),
            ])
        
        return features
    
    async def compute_streaming_feature(self, entity_id: str, 
                                      market_data: MarketData) -> Optional[FeatureValue]:
        """Compute real-time technical indicators."""
        # For streaming, we need historical context
        # This is a simplified implementation - would need more context in practice
        return FeatureValue(
            feature_name="price",
            entity_id=entity_id,
            timestamp=market_data.timestamp,
            value=market_data.close,
            computation_time=datetime.utcnow(),
            feature_version=self.feature_def.version
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower


class FeatureStore:
    """Production-grade feature store with lineage tracking and monitoring."""
    
    def __init__(self):
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_computers: Dict[str, BaseFeatureComputer] = {}
        self.cache = None
        self.db = None
        
    async def initialize(self):
        """Initialize feature store connections."""
        self.cache = await get_trading_cache()
        self.db = await get_database()
        
        # Create feature store tables
        await self._create_tables()
        
        # Register default feature computers
        await self._register_default_features()
        
        logger.info("Feature store initialized")
    
    async def _create_tables(self):
        """Create feature store database tables."""
        # Feature definitions table
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS feature_definitions (
            feature_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            feature_type VARCHAR(50) NOT NULL,
            description TEXT,
            data_type VARCHAR(50),
            compute_mode VARCHAR(50),
            dependencies JSON,
            source_tables JSON,
            transformation_logic TEXT,
            version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
        """)
        
        # Feature values table (partitioned by date)
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS feature_values (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            feature_name VARCHAR(255) NOT NULL,
            entity_id VARCHAR(100) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            value JSONB,
            confidence FLOAT,
            data_quality_score FLOAT,
            computation_time TIMESTAMP,
            feature_version VARCHAR(50),
            UNIQUE(feature_name, entity_id, timestamp, feature_version)
        )
        """)
        
        # Create indexes for performance
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_feature_values_entity_time 
        ON feature_values(entity_id, timestamp DESC)
        """)
        
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_feature_values_name_time
        ON feature_values(feature_name, timestamp DESC)
        """)

        # Feature views metadata
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS feature_views (
            view_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL DEFAULT '1',
            feature_names JSONB NOT NULL,
            description TEXT,
            entities JSONB,
            tags JSONB,
            transformation_logic TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(view_name, version)
        )
        """)

        # Materialized feature view snapshots (denormalized for fast serving)
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS feature_view_materializations (
            view_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            entity_id VARCHAR(100) NOT NULL,
            as_of TIMESTAMP NOT NULL,
            features JSONB NOT NULL,
            vector_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(view_name, version, entity_id, as_of)
        )
        """)

        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_feature_view_materializations_lookup
        ON feature_view_materializations(view_name, version, entity_id, as_of DESC)
        """)
    
    async def _register_default_features(self):
        """Register built-in feature definitions."""
        # Technical indicators
        tech_features = [
            FeatureDefinition(
                name="sma_20",
                feature_type=FeatureType.TECHNICAL,
                description="20-period Simple Moving Average",
                data_type="float64",
                compute_mode=ComputeMode.BATCH,
                dependencies=["market_data.close"],
                source_tables=["market_data"],
                update_frequency="1min",
                lookback_window="30d"
            ),
            FeatureDefinition(
                name="rsi_14", 
                feature_type=FeatureType.TECHNICAL,
                description="14-period Relative Strength Index",
                data_type="float64",
                compute_mode=ComputeMode.BATCH,
                dependencies=["market_data.close"],
                source_tables=["market_data"],
                update_frequency="1min",
                expected_range=(0.0, 100.0)
            ),
            FeatureDefinition(
                name="bb_position",
                feature_type=FeatureType.TECHNICAL, 
                description="Position within Bollinger Bands (0-1)",
                data_type="float64",
                compute_mode=ComputeMode.STREAMING,
                dependencies=["bb_upper_20_2", "bb_lower_20_2", "market_data.close"],
                expected_range=(0.0, 1.0)
            )
        ]
        
        for feature_def in tech_features:
            await self.register_feature(feature_def)
    
    async def register_feature(self, feature_def: FeatureDefinition):
        """Register a new feature definition."""
        feature_id = feature_def.get_feature_id()
        
        # Store in memory
        self.feature_definitions[feature_id] = feature_def
        
        # Persist to database
        await self.db.execute("""
        INSERT INTO feature_definitions 
        (feature_id, name, feature_type, description, data_type, compute_mode,
         dependencies, source_tables, transformation_logic, version, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (feature_id) DO UPDATE SET
            description = EXCLUDED.description,
            transformation_logic = EXCLUDED.transformation_logic,
            metadata = EXCLUDED.metadata
        """, [
            feature_id,
            feature_def.name,
            feature_def.feature_type.value,
            feature_def.description,
            feature_def.data_type,
            feature_def.compute_mode.value,
            json.dumps(feature_def.dependencies),
            json.dumps(feature_def.source_tables),
            feature_def.transformation_logic,
            feature_def.version,
            json.dumps({
                "tags": list(feature_def.tags),
                "created_by": feature_def.created_by,
                "update_frequency": feature_def.update_frequency,
                "lookback_window": feature_def.lookback_window,
                "expected_range": feature_def.expected_range,
                "null_tolerance": feature_def.null_tolerance
            })
        ])
        
        # Register feature computer if applicable
        if feature_def.feature_type == FeatureType.TECHNICAL:
            self.feature_computers[feature_id] = TechnicalIndicatorComputer(feature_def)
        
        logger.info(f"Registered feature: {feature_id}")
    
    async def compute_features(self, feature_names: List[str], 
                             entity_ids: List[str],
                             start_time: datetime,
                             end_time: datetime,
                             force_recompute: bool = False) -> List[FeatureValue]:
        """Compute features for given entities and time range."""
        all_features = []
        
        for feature_name in feature_names:
            # Find feature definition
            feature_def = None
            for fid, fdef in self.feature_definitions.items():
                if fdef.name == feature_name:
                    feature_def = fdef
                    break
            
            if not feature_def:
                logger.warning(f"Feature definition not found: {feature_name}")
                continue
            
            feature_id = feature_def.get_feature_id()
            
            if not force_recompute:
                # Check if features already exist
                existing = await self._get_existing_features(
                    feature_name, entity_ids, start_time, end_time
                )
                if existing:
                    all_features.extend(existing)
                    continue
            
            # Compute new features
            if feature_id in self.feature_computers:
                computer = self.feature_computers[feature_id]
                computed_features = await computer.compute_features(
                    entity_ids, start_time, end_time
                )
                
                # Store computed features
                await self._store_features(computed_features)
                all_features.extend(computed_features)
        
        return all_features
    
    async def get_feature_vector(self, entity_id: str, 
                               timestamp: datetime,
                               feature_names: Optional[List[str]] = None) -> Optional[FeatureVector]:
        """Get feature vector for entity at specific timestamp."""
        if feature_names is None:
            feature_names = [fdef.name for fdef in self.feature_definitions.values()]
        
        # Query features from database
        placeholders = ','.join(['%s'] * len(feature_names))
        query = f"""
        SELECT feature_name, value, confidence, data_quality_score, feature_version
        FROM feature_values
        WHERE entity_id = %s 
        AND feature_name IN ({placeholders})
        AND timestamp <= %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        rows = await self.db.fetch_all(
            query, [entity_id] + feature_names + [timestamp, len(feature_names)]
        )
        
        if not rows:
            return None
        
        # Build feature vector
        features = {}
        for row in rows:
            if row['feature_name'] not in features:  # Take most recent value
                features[row['feature_name']] = FeatureValue(
                    feature_name=row['feature_name'],
                    entity_id=entity_id,
                    timestamp=timestamp,
                    value=row['value'],
                    confidence=row['confidence'],
                    data_quality_score=row['data_quality_score'],
                    feature_version=row['feature_version']
                )
        
        return FeatureVector(
            entity_id=entity_id,
            timestamp=timestamp,
            features=features
        )
    
    async def get_feature_matrix(self, entity_ids: List[str],
                               feature_names: List[str],
                               start_time: datetime,
                               end_time: datetime) -> pd.DataFrame:
        """Get feature matrix as pandas DataFrame for ML training."""
        all_vectors = []
        
        # Generate time points (e.g., hourly)
        current = start_time
        while current <= end_time:
            for entity_id in entity_ids:
                vector = await self.get_feature_vector(entity_id, current, feature_names)
                if vector:
                    row_data = {
                        'entity_id': entity_id,
                        'timestamp': current,
                        **vector.get_feature_dict()
                    }
                    all_vectors.append(row_data)
            current += timedelta(hours=1)
        
        return pd.DataFrame(all_vectors)
    
    async def _get_existing_features(self, feature_name: str,
                                   entity_ids: List[str],
                                   start_time: datetime,
                                   end_time: datetime) -> List[FeatureValue]:
        """Get existing feature values from storage."""
        placeholders = ','.join(['%s'] * len(entity_ids))
        query = f"""
        SELECT feature_name, entity_id, timestamp, value, confidence, 
               data_quality_score, computation_time, feature_version
        FROM feature_values
        WHERE feature_name = %s
        AND entity_id IN ({placeholders})
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp
        """
        
        rows = await self.db.fetch_all(
            query, [feature_name] + entity_ids + [start_time, end_time]
        )
        
        features = []
        for row in rows:
            features.append(FeatureValue(
                feature_name=row['feature_name'],
                entity_id=row['entity_id'], 
                timestamp=row['timestamp'],
                value=row['value'],
                confidence=row['confidence'],
                data_quality_score=row['data_quality_score'],
                computation_time=row['computation_time'],
                feature_version=row['feature_version']
            ))
        
        return features
    
    async def _store_features(self, features: List[FeatureValue]):
        """Store computed features to database."""
        if not features:
            return
        
        # Batch insert for performance
        values = []
        for feature in features:
            values.append([
                feature.feature_name,
                feature.entity_id,
                feature.timestamp,
                json.dumps(feature.value),
                feature.confidence,
                feature.data_quality_score,
                feature.computation_time,
                feature.feature_version
            ])
        
        await self.db.executemany("""
        INSERT INTO feature_values 
        (feature_name, entity_id, timestamp, value, confidence, 
         data_quality_score, computation_time, feature_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (feature_name, entity_id, timestamp, feature_version) 
        DO UPDATE SET
            value = EXCLUDED.value,
            confidence = EXCLUDED.confidence,
            data_quality_score = EXCLUDED.data_quality_score,
            computation_time = EXCLUDED.computation_time
        """, values)
    
    async def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get feature lineage and dependencies."""
        feature_def = None
        for fdef in self.feature_definitions.values():
            if fdef.name == feature_name:
                feature_def = fdef
                break
        
        if not feature_def:
            return {}
        
        return {
            "feature_name": feature_name,
            "dependencies": feature_def.dependencies,
            "source_tables": feature_def.source_tables,
            "transformation_logic": feature_def.transformation_logic,
            "version": feature_def.version,
            "created_at": feature_def.created_at.isoformat(),
            "created_by": feature_def.created_by
        }
    
    async def validate_data_quality(self, features: List[FeatureValue]) -> Dict[str, Any]:
        """Validate data quality of computed features."""
        quality_report = {
            "total_features": len(features),
            "null_count": 0,
            "out_of_range_count": 0,
            "quality_issues": []
        }
        
        for feature in features:
            # Check for nulls
            if feature.value is None or (isinstance(feature.value, float) and np.isnan(feature.value)):
                quality_report["null_count"] += 1
            
            # Check expected ranges
            feature_def = None
            for fdef in self.feature_definitions.values():
                if fdef.name == feature.feature_name:
                    feature_def = fdef
                    break
            
            if feature_def and feature_def.expected_range:
                min_val, max_val = feature_def.expected_range
                if isinstance(feature.value, (int, float)):
                    if feature.value < min_val or feature.value > max_val:
                        quality_report["out_of_range_count"] += 1
                        quality_report["quality_issues"].append({
                            "feature": feature.feature_name,
                            "entity": feature.entity_id,
                            "issue": "out_of_range",
                            "value": feature.value,
                            "expected_range": [min_val, max_val]
                        })
        
        # Calculate overall quality score
        if quality_report["total_features"] > 0:
            error_rate = (quality_report["null_count"] + quality_report["out_of_range_count"]) / quality_report["total_features"]
            quality_report["quality_score"] = max(0.0, 1.0 - error_rate)
        else:
            quality_report["quality_score"] = 0.0
        
        return quality_report

    # ---------------- Feature Views -----------------
    async def register_feature_view(self, view_name: str, feature_names: List[str], *, version: str = "1", description: Optional[str] = None, entities: Optional[List[str]] = None, tags: Optional[Dict[str, Any]] = None, transformation_logic: Optional[str] = None):
        """Register a logical feature view (collection of feature names)."""
        await self.db.execute("""
        INSERT INTO feature_views(view_name, version, feature_names, description, entities, tags, transformation_logic)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (view_name, version) DO UPDATE SET description=EXCLUDED.description, feature_names=EXCLUDED.feature_names, tags=EXCLUDED.tags, transformation_logic=EXCLUDED.transformation_logic
        """, [view_name, version, json.dumps(feature_names), description, json.dumps(entities or []), json.dumps(tags or {}), transformation_logic])
        logger.info("Registered feature view %s v%s (%d features)", view_name, version, len(feature_names))

    async def materialize_feature_view(self, view_name: str, *, version: str = "1", entity_ids: List[str], as_of: datetime, backfill_hours: int = 0) -> int:
        """Materialize a snapshot for the given entities at a timestamp.

        Grabs most recent values <= as_of for each feature in view and stores consolidated row.
        backfill_hours optionally computes additional hourly snapshots (as_of - n .. as_of).
        Returns number of rows materialized.
        """
        # Load view metadata
        row = await self.db.fetch_one("SELECT feature_names FROM feature_views WHERE view_name=%s AND version=%s", [view_name, version])
        if not row:
            raise ValueError(f"Feature view {view_name} v{version} not found")
        feature_names = json.loads(row['feature_names'])
        timestamps = [as_of]
        if backfill_hours > 0:
            base = as_of
            timestamps.extend([base - timedelta(hours=i) for i in range(1, backfill_hours + 1)])
        inserted = 0
        for ts in timestamps:
            for entity_id in entity_ids:
                vector = await self.get_feature_vector(entity_id, ts, feature_names)
                if not vector:
                    continue
                features_dict = vector.get_feature_dict()
                await self.db.execute("""
                INSERT INTO feature_view_materializations(view_name, version, entity_id, as_of, features, vector_id)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (view_name, version, entity_id, as_of) DO UPDATE SET features=EXCLUDED.features, vector_id=EXCLUDED.vector_id, created_at=CURRENT_TIMESTAMP
                """, [view_name, version, entity_id, ts, json.dumps(features_dict), vector.vector_id])
                inserted += 1
        logger.info("Materialized feature view %s v%s rows=%d", view_name, version, inserted)
        return inserted

    async def get_feature_view_snapshot(self, view_name: str, *, version: str = "1", entity_id: str, as_of: datetime) -> Optional[Dict[str, Any]]:
        row = await self.db.fetch_one("""
        SELECT features, vector_id FROM feature_view_materializations
        WHERE view_name=%s AND version=%s AND entity_id=%s AND as_of <= %s
        ORDER BY as_of DESC LIMIT 1
        """, [view_name, version, entity_id, as_of])
        if not row:
            return None
        return {"features": row['features'], "vector_id": row.get('vector_id')}


# Global feature store instance
_feature_store: Optional[FeatureStore] = None


async def get_feature_store() -> FeatureStore:
    """Get global feature store instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
        await _feature_store.initialize()
    return _feature_store


# Convenience functions
async def compute_technical_features(symbols: List[str], 
                                   start_date: datetime,
                                   end_date: datetime) -> pd.DataFrame:
    """Compute technical features for symbols and return as DataFrame."""
    store = await get_feature_store()
    
    technical_features = [
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "rsi_14", "bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2"
    ]
    
    await store.compute_features(technical_features, symbols, start_date, end_date)
    return await store.get_feature_matrix(symbols, technical_features, start_date, end_date)


async def get_latest_features(symbol: str, feature_names: List[str]) -> Optional[FeatureVector]:
    """Get latest feature vector for a symbol."""
    store = await get_feature_store()
    return await store.get_feature_vector(symbol, datetime.utcnow(), feature_names)