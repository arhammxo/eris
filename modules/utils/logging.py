"""
Logging configuration for Web Summarizer.

This module provides standardized logging setup with structured logging
capabilities to improve observability and diagnostics.
"""
import logging
import logging.handlers
import os
import json
import time
import traceback
from typing import Any, Dict, Optional, Union

# Log levels
DEFAULT_LEVEL = logging.INFO

class StructuredLogFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 
                           'filename', 'funcName', 'id', 'levelname', 'levelno', 
                           'lineno', 'module', 'msecs', 'message', 'msg', 'name', 
                           'pathname', 'process', 'processName', 'relativeCreated', 
                           'stack_info', 'thread', 'threadName']:
                try:
                    # Ensure the value is JSON serializable
                    json.dumps({key: value})
                    log_data[key] = value
                except (TypeError, OverflowError):
                    log_data[key] = str(value)
        
        return json.dumps(log_data)


class ContextAdapter(logging.LoggerAdapter):
    """
    Adapter that adds context to log records.
    
    This allows passing additional context with each log message.
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context from the adapter to the kwargs."""
        kwargs["extra"] = {**self.extra, **(kwargs.get("extra", {}))}
        return msg, kwargs


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> Union[logging.Logger, ContextAdapter]:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        context: Optional dictionary of context values to include in logs
        
    Returns:
        Logger or LoggerAdapter if context is provided
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextAdapter(logger, context)
    
    return logger


def setup_logging(
    level: int = DEFAULT_LEVEL, 
    structured: bool = False,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> None:
    """
    Configure application-wide logging.
    
    Args:
        level: Log level (e.g., logging.INFO)
        structured: Whether to use JSON structured logging
        log_file: Path to log file (if None, file logging is disabled)
        log_to_console: Whether to log to console
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    if structured:
        formatter = StructuredLogFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if path is provided
    if log_file:
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Use a rotating file handler to manage log size
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def log_function_call(logger: logging.Logger):
    """
    Decorator that logs function calls with arguments and results.
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log function call
            arg_str = ', '.join([repr(arg) for arg in args])
            kwarg_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            all_args = f"{arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str}"
            logger.debug(f"Calling {func.__name__}({all_args})")
            
            # Call the function and log the result
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def log_async_function_call(logger: logging.Logger):
    """
    Decorator that logs async function calls with arguments and results.
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorated async function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Log function call
            arg_str = ', '.join([repr(arg) for arg in args])
            kwarg_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            all_args = f"{arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str}"
            logger.debug(f"Calling async {func.__name__}({all_args})")
            
            # Call the function and log the result
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"Async {func.__name__} completed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Async {func.__name__} failed after {elapsed:.4f}s: {str(e)}")
                raise
        
        return wrapper
    
    return decorator