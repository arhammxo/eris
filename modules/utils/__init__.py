"""
Utility modules for the Web Summarizer application.

This package provides common utilities used throughout the application:

- Errors and exception handling (errors.py)
- Logging configuration (logging.py)
- Type definitions (types.py)
"""

from modules.utils.errors import (
    WebSummarizerError,
    SearchError, SearchAPIError, NoResultsError,
    CrawlerError, HTTPError, BrowserError, RobotsExclusionError,
    ProcessorError, ContentExtractionError, TextAnalysisError,
    SummarizerError, ModelAPIError, PromptGenerationError,
    CacheError, CacheAccessError,
    handle_exceptions, handle_async_exceptions, create_error_response
)

from modules.utils.logging import (
    setup_logging,
    get_logger,
    log_function_call,
    log_async_function_call
)

# Import type definitions but don't expose them directly
# to avoid cluttering the namespace
import modules.utils.types as types

__all__ = [
    # Error handling
    'WebSummarizerError',
    'SearchError', 'SearchAPIError', 'NoResultsError',
    'CrawlerError', 'HTTPError', 'BrowserError', 'RobotsExclusionError',
    'ProcessorError', 'ContentExtractionError', 'TextAnalysisError',
    'SummarizerError', 'ModelAPIError', 'PromptGenerationError',
    'CacheError', 'CacheAccessError',
    'handle_exceptions', 'handle_async_exceptions', 'create_error_response',
    
    # Logging
    'setup_logging',
    'get_logger',
    'log_function_call',
    'log_async_function_call',
    
    # Types
    'types'
]