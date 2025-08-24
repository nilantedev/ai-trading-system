"""Custom exceptions for the trading system."""

from typing import Optional, Dict, Any


class TradingError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        return " ".join(parts)


class ValidationError(TradingError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None) -> None:
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context=context,
        )


class ConfigError(TradingError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, setting: Optional[str] = None) -> None:
        context = {}
        if setting:
            context["setting"] = setting
        
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            context=context,
        )


class DatabaseError(TradingError):
    """Raised when database operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
    ) -> None:
        context = {}
        if operation:
            context["operation"] = operation
        if table:
            context["table"] = table
        
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            context=context,
        )


class MessageQueueError(TradingError):
    """Raised when message queue operations fail."""
    
    def __init__(
        self,
        message: str,
        topic: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> None:
        context = {}
        if topic:
            context["topic"] = topic
        if operation:
            context["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="MESSAGE_QUEUE_ERROR", 
            context=context,
        )


class AuthenticationError(TradingError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(
            message=message,
            error_code="AUTH_ERROR",
        )


class AuthorizationError(TradingError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", required_permission: Optional[str] = None) -> None:
        context = {}
        if required_permission:
            context["required_permission"] = required_permission
        
        super().__init__(
            message=message,
            error_code="AUTHZ_ERROR",
            context=context,
        )


class TradingDecisionError(TradingError):
    """Raised when trading decision logic fails."""
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        decision_type: Optional[str] = None,
    ) -> None:
        context = {}
        if symbol:
            context["symbol"] = symbol
        if decision_type:
            context["decision_type"] = decision_type
        
        super().__init__(
            message=message,
            error_code="TRADING_DECISION_ERROR",
            context=context,
        )


class RiskManagementError(TradingError):
    """Raised when risk management rules are violated."""
    
    def __init__(
        self,
        message: str,
        risk_type: Optional[str] = None,
        limit_exceeded: Optional[float] = None,
        current_value: Optional[float] = None,
    ) -> None:
        context = {}
        if risk_type:
            context["risk_type"] = risk_type
        if limit_exceeded:
            context["limit_exceeded"] = limit_exceeded
        if current_value:
            context["current_value"] = current_value
        
        super().__init__(
            message=message,
            error_code="RISK_MANAGEMENT_ERROR",
            context=context,
        )


class MarketDataError(TradingError):
    """Raised when market data operations fail."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        symbol: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> None:
        context = {}
        if provider:
            context["provider"] = provider
        if symbol:
            context["symbol"] = symbol
        if data_type:
            context["data_type"] = data_type
        
        super().__init__(
            message=message,
            error_code="MARKET_DATA_ERROR",
            context=context,
        )


class AIModelError(TradingError):
    """Raised when AI model operations fail."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> None:
        context = {}
        if model_name:
            context["model_name"] = model_name
        if operation:
            context["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="AI_MODEL_ERROR",
            context=context,
        )


class CircuitBreakerError(TradingError):
    """Raised when circuit breaker is triggered."""
    
    def __init__(
        self,
        message: str = "Circuit breaker activated",
        service: Optional[str] = None,
        failure_count: Optional[int] = None,
    ) -> None:
        context = {}
        if service:
            context["service"] = service
        if failure_count:
            context["failure_count"] = failure_count
        
        super().__init__(
            message=message,
            error_code="CIRCUIT_BREAKER_ERROR",
            context=context,
        )