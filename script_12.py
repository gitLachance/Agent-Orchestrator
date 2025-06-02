# Create logging utility
logging_content = '''"""
Structured logging configuration for the Legal Agent Orchestrator.
Provides secure, compliant logging with privacy protection for sensitive legal data.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
import structlog
from rich.console import Console
from rich.logging import RichHandler


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from log messages."""
    
    SENSITIVE_PATTERNS = [
        "api_key", "password", "token", "secret", "client_id", "private_key",
        "ssn", "social_security", "credit_card", "bank_account", "routing_number"
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive data from log records."""
        try:
            # Check message for sensitive patterns
            message = str(record.getMessage()).lower()
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in message:
                    # Replace with redacted version
                    record.msg = "[REDACTED - SENSITIVE DATA]"
                    record.args = ()
                    break
            
            # Check record attributes
            if hasattr(record, 'args') and record.args:
                safe_args = []
                for arg in record.args:
                    if isinstance(arg, (str, dict)):
                        arg_str = str(arg).lower()
                        if any(pattern in arg_str for pattern in self.SENSITIVE_PATTERNS):
                            safe_args.append("[REDACTED]")
                        else:
                            safe_args.append(arg)
                    else:
                        safe_args.append(arg)
                record.args = tuple(safe_args)
            
            return True
            
        except Exception:
            # If filtering fails, allow the log through
            return True


class LegalComplianceFormatter(logging.Formatter):
    """Formatter that ensures legal compliance for log messages."""
    
    def __init__(self):
        super().__init__()
        self.sensitive_filter = SensitiveDataFilter()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with compliance considerations."""
        # Apply sensitive data filtering
        self.sensitive_filter.filter(record)
        
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add thread and process info for debugging
        if hasattr(record, 'process') and hasattr(record, 'thread'):
            log_entry["process"] = record.process
            log_entry["thread"] = record.thread
        
        return json.dumps(log_entry, ensure_ascii=False)


class AgentActivityLogger:
    """Specialized logger for agent activities and interactions."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"agent.{agent_id}")
    
    def log_interaction(
        self, 
        interaction_type: str, 
        details: Dict[str, Any],
        sensitive: bool = False
    ) -> None:
        """Log agent interactions with privacy protection."""
        log_data = {
            "agent_id": self.agent_id,
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "sensitive": sensitive
        }
        
        # Redact sensitive details
        if sensitive:
            safe_details = {}
            for key, value in details.items():
                if key in ["prompt", "response", "content", "message"]:
                    safe_details[key] = f"[REDACTED - {len(str(value))} chars]"
                else:
                    safe_details[key] = value
            log_data["details"] = safe_details
        else:
            log_data["details"] = details
        
        self.logger.info("Agent interaction", extra={"structured_data": log_data})
    
    def log_reasoning_step(
        self, 
        step_type: str, 
        content: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log reasoning steps for transparency."""
        log_data = {
            "agent_id": self.agent_id,
            "step_type": step_type,
            "content_length": len(content),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug("Reasoning step", extra={"structured_data": log_data})
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log errors with context."""
        log_data = {
            "agent_id": self.agent_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.error("Agent error", extra={"structured_data": log_data}, exc_info=True)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    enable_rich_console: bool = True,
    enable_json_formatting: bool = False
) -> None:
    """Set up logging configuration for the application."""
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    handlers = []
    
    # Console handler with Rich formatting (for development)
    if enable_rich_console:
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setLevel(log_level)
        handlers.append(rich_handler)
    else:
        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if enable_json_formatting:
            console_handler.setFormatter(LegalComplianceFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
        
        handlers.append(console_handler)
    
    # File handler (for production)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(LegalComplianceFormatter())
        handlers.append(file_handler)
    
    # Add sensitive data filter to all handlers
    sensitive_filter = SensitiveDataFilter()
    for handler in handlers:
        handler.addFilter(sensitive_filter)
        root_logger.addHandler(handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FUNC_NAME]
            ),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration."""
    return logging.getLogger(name)


def get_agent_logger(agent_id: str) -> AgentActivityLogger:
    """Get a specialized logger for agent activities."""
    return AgentActivityLogger(agent_id)


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, **context):
        self.context = context
        self.logger = structlog.get_logger()
    
    def __enter__(self):
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.bound_logger.error(
                "Exception in log context",
                exc_type=exc_type.__name__,
                exc_value=str(exc_val)
            )


def log_performance(func_name: str, duration: float, **kwargs) -> None:
    """Log performance metrics."""
    logger = get_logger("performance")
    logger.info(
        f"Performance: {func_name}",
        extra={
            "structured_data": {
                "function": func_name,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
        }
    )


def log_security_event(
    event_type: str,
    severity: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None
) -> None:
    """Log security-related events."""
    logger = get_logger("security")
    
    log_data = {
        "event_type": event_type,
        "severity": severity,
        "user_id": user_id,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    log_level = getattr(logging, severity.upper(), logging.INFO)
    logger.log(log_level, f"Security event: {event_type}", extra={"structured_data": log_data})


def log_compliance_event(
    action: str,
    data_type: str,
    user_id: Optional[str] = None,
    details: Dict[str, Any] = None
) -> None:
    """Log compliance-related events for audit trails."""
    logger = get_logger("compliance")
    
    log_data = {
        "action": action,
        "data_type": data_type,
        "user_id": user_id,
        "details": details or {},
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Compliance: {action}", extra={"structured_data": log_data})


# Performance monitoring decorator
def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            log_performance(func.__name__, duration, success=True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_performance(func.__name__, duration, success=False, error=str(e))
            raise
    
    return wrapper


def log_execution_time_async(func):
    """Async decorator to log function execution time."""
    import time
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            log_performance(func.__name__, duration, success=True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_performance(func.__name__, duration, success=False, error=str(e))
            raise
    
    return wrapper
'''

with open("legal_agent_orchestrator/utils/logging.py", "w") as f:
    f.write(logging_content)

print("Logging utility created!")