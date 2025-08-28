#!/usr/bin/env python3
"""
Immutable Audit Trail - Cryptographically secured audit logging
Ensures all trading events are recorded and tamper-proof
"""

import hashlib
import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events to audit"""
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_LIMIT_HIT = "risk_limit_hit"
    EMERGENCY_STOP = "emergency_stop"
    CONFIG_CHANGE = "config_change"
    MODEL_PREDICTION = "model_prediction"
    DATA_ANOMALY = "data_anomaly"
    SYSTEM_ERROR = "system_error"
    USER_ACTION = "user_action"


@dataclass
class AuditEvent:
    """Immutable audit event"""
    event_type: EventType
    event_data: Dict[str, Any]
    timestamp: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    index: int
    prev_hash: str
    hash: str = ""


class ImmutableAuditLog:
    """
    Immutable, cryptographically secured audit log.
    Each event is chained to the previous one via hash.
    """
    
    def __init__(self, log_dir: str = "/var/log/trading-system"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Audit log file (append-only)
        self.audit_file = self.log_dir / "audit.log"
        self.chain_file = self.log_dir / "audit_chain.json"
        
        # In-memory chain for current session
        self.chain: List[AuditEvent] = []
        self.current_index = 0
        
        # Load existing chain if available
        self._load_chain()
        
        # Lock for thread-safe writes
        self.write_lock = asyncio.Lock()
    
    def _load_chain(self):
        """Load existing audit chain from disk"""
        if self.chain_file.exists():
            try:
                with open(self.chain_file, 'r') as f:
                    chain_data = json.load(f)
                    self.current_index = chain_data.get("last_index", 0)
                    logger.info(f"Loaded audit chain with {self.current_index} events")
            except Exception as e:
                logger.error(f"Failed to load audit chain: {e}")
                self.current_index = 0
    
    def _calculate_hash(self, event_dict: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of event"""
        # Sort keys for consistent hashing
        event_str = json.dumps(event_dict, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    async def add_event(self, 
                       event_type: EventType,
                       event_data: Dict[str, Any],
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None,
                       ip_address: Optional[str] = None) -> AuditEvent:
        """
        Add a new event to the immutable audit log
        """
        async with self.write_lock:
            # Create event
            timestamp = datetime.utcnow().isoformat()
            index = self.current_index
            
            # Get previous hash
            if self.chain:
                prev_hash = self.chain[-1].hash
            else:
                prev_hash = "0" * 64  # Genesis block
            
            # Create event object
            event = AuditEvent(
                event_type=event_type,
                event_data=event_data,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                index=index,
                prev_hash=prev_hash
            )
            
            # Calculate hash (excluding the hash field itself)
            event_dict = asdict(event)
            del event_dict['hash']
            event.hash = self._calculate_hash(event_dict)
            
            # Add to chain
            self.chain.append(event)
            self.current_index += 1
            
            # Persist to disk
            await self._persist_event(event)
            
            # Log critical events
            if event_type in [EventType.EMERGENCY_STOP, EventType.RISK_LIMIT_HIT]:
                logger.warning(f"CRITICAL EVENT: {event_type.value} - {event_data}")
            
            return event
    
    async def _persist_event(self, event: AuditEvent):
        """Persist event to append-only log file"""
        try:
            # Convert enum to string for JSON serialization
            event_dict = asdict(event)
            event_dict['event_type'] = event.event_type.value
            
            # Write to append-only log
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
            
            # Update chain file
            with open(self.chain_file, 'w') as f:
                json.dump({
                    "last_index": self.current_index,
                    "last_hash": event.hash,
                    "last_timestamp": event.timestamp
                }, f)
        
        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}")
            # In production, this should trigger an alert
            raise
    
    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """
        Verify the integrity of the audit chain
        Returns (is_valid, first_invalid_index)
        """
        if not self.chain:
            return True, None
        
        # Check genesis block
        first_event = self.chain[0]
        if first_event.prev_hash != "0" * 64:
            return False, 0
        
        # Verify each event's hash and chain
        for i, event in enumerate(self.chain):
            # Recalculate hash
            event_dict = asdict(event)
            stored_hash = event_dict['hash']
            del event_dict['hash']
            calculated_hash = self._calculate_hash(event_dict)
            
            # Check hash integrity
            if calculated_hash != stored_hash:
                logger.error(f"Hash mismatch at index {i}")
                return False, i
            
            # Check chain integrity (except for first event)
            if i > 0:
                if event.prev_hash != self.chain[i-1].hash:
                    logger.error(f"Chain broken at index {i}")
                    return False, i
        
        return True, None
    
    async def log_order(self, order: Dict[str, Any], action: str, user_id: Optional[str] = None):
        """Convenience method for logging order events"""
        event_type_map = {
            "placed": EventType.ORDER_PLACED,
            "cancelled": EventType.ORDER_CANCELLED,
            "filled": EventType.ORDER_FILLED
        }
        
        event_type = event_type_map.get(action, EventType.ORDER_PLACED)
        
        await self.add_event(
            event_type=event_type,
            event_data={
                "order_id": order.get("order_id"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "quantity": order.get("quantity"),
                "price": order.get("price"),
                "order_type": order.get("order_type"),
                "action": action
            },
            user_id=user_id
        )
    
    async def log_risk_event(self, risk_type: str, details: Dict[str, Any]):
        """Log risk-related events"""
        await self.add_event(
            event_type=EventType.RISK_LIMIT_HIT,
            event_data={
                "risk_type": risk_type,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def log_config_change(self, setting: str, old_value: Any, new_value: Any, changed_by: str):
        """Log configuration changes"""
        await self.add_event(
            event_type=EventType.CONFIG_CHANGE,
            event_data={
                "setting": setting,
                "old_value": old_value,
                "new_value": new_value,
                "changed_at": datetime.utcnow().isoformat()
            },
            user_id=changed_by
        )
    
    def get_recent_events(self, count: int = 100, event_type: Optional[EventType] = None) -> List[Dict]:
        """Get recent events from the audit log"""
        events = self.chain[-count:] if count < len(self.chain) else self.chain
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return [asdict(e) for e in events]
    
    def export_audit_log(self, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> List[Dict]:
        """Export audit log for specific date range"""
        events = []
        
        try:
            with open(self.audit_file, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    event_time = datetime.fromisoformat(event['timestamp'])
                    
                    if start_date and event_time < start_date:
                        continue
                    if end_date and event_time > end_date:
                        continue
                    
                    events.append(event)
        
        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")
        
        return events
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report from audit trail"""
        report = {
            "total_events": len(self.chain),
            "chain_valid": self.verify_chain()[0],
            "event_counts": {},
            "risk_events": [],
            "config_changes": [],
            "emergency_stops": []
        }
        
        # Count events by type
        for event in self.chain:
            event_type = event.event_type.value
            report["event_counts"][event_type] = report["event_counts"].get(event_type, 0) + 1
            
            # Collect specific events for compliance
            if event.event_type == EventType.RISK_LIMIT_HIT:
                report["risk_events"].append({
                    "timestamp": event.timestamp,
                    "data": event.event_data
                })
            elif event.event_type == EventType.CONFIG_CHANGE:
                report["config_changes"].append({
                    "timestamp": event.timestamp,
                    "data": event.event_data
                })
            elif event.event_type == EventType.EMERGENCY_STOP:
                report["emergency_stops"].append({
                    "timestamp": event.timestamp,
                    "data": event.event_data
                })
        
        return report


# Global audit logger instance
_audit_logger = None

def get_audit_logger() -> ImmutableAuditLog:
    """Get or create global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = ImmutableAuditLog()
    return _audit_logger