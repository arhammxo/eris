"""
Custom exception hierarchy and error handling utilities for Web Summarizer.

This module provides a structured approach to error handling across the application,
ensuring consistent error reporting and recovery strategies.
"""
from typing import Any, Dict, Optional, Type, Union
import traceback
import logging
import functools
import asyncio

logger = logging.getLogger(__name__)

class WebSummarizerError(Exception):
    """Base exception class for all Web Summarizer errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error with a message and optional details.
        
        Args:
            message: Human-readable error description
            details: Additional contextual information about the error
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


# Search-related exceptions
class SearchError(WebSummarizerError):
    """Base class for search-related errors."""
    pass

class SearchAPIError(SearchError):
    """Error when interacting with search API (e.g., SerpAPI)."""
    pass

class NoResultsError(SearchError):
    """Error when no search results are found."""
    pass


# Crawler-related exceptions
class CrawlerError(WebSummarizerError):
    """Base class for crawler-related errors."""
    pass

class HTTPError(CrawlerError):
    """Error when HTTP request fails."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, url: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the HTTP error.
        
        Args:
            message: Error description
            status_code: HTTP status code if available
            url: The URL that was being accessed
            details: Additional error details
        """
        self.status_code = status_code
        self.url = url
        details = details or {}
        if status_code:
            details['status_code'] = status_code
        if url:
            details['url'] = url
        super().__init__(message, details)

class BrowserError(CrawlerError):
    """Error when browser-based crawling fails."""
    pass

class RobotsExclusionError(CrawlerError):
    """Error when crawling is forbidden by robots.txt."""
    pass


# Processor-related exceptions
class ProcessorError(WebSummarizerError):
    """Base class for processor-related errors."""
    pass

class ContentExtractionError(ProcessorError):
    """Error extracting content from HTML."""
    pass

class TextAnalysisError(ProcessorError):
    """Error during text analysis."""
    pass


# Summarizer-related exceptions
class SummarizerError(WebSummarizerError):
    """Base class for summarizer-related errors."""
    pass

class ModelAPIError(SummarizerError):
    """Error when interacting with the LLM API (e.g., OpenAI)."""
    pass

class PromptGenerationError(SummarizerError):
    """Error generating prompts for the LLM."""
    pass


# Cache-related exceptions
class CacheError(WebSummarizerError):
    """Base class for cache-related errors."""
    pass

class CacheAccessError(CacheError):
    """Error accessing the cache."""
    pass


# Utility functions for error handling
def handle_exceptions(error_type: Type[WebSummarizerError] = WebSummarizerError, 
                     default_message: str = "An unexpected error occurred"):
    """
    Decorator for handling exceptions in functions.
    
    Args:
        error_type: Type of exception to raise if an error occurs
        default_message: Default error message to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except WebSummarizerError:
                # Re-raise application exceptions as-is
                raise
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise error_type(f"{default_message}: {str(e)}")
        return wrapper
    return decorator

def handle_async_exceptions(error_type: Type[WebSummarizerError] = WebSummarizerError, 
                           default_message: str = "An unexpected error occurred"):
    """
    Decorator for handling exceptions in async functions.
    
    Args:
        error_type: Type of exception to raise if an error occurs
        default_message: Default error message to use
        
    Returns:
        Decorated async function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except WebSummarizerError:
                # Re-raise application exceptions as-is
                raise
            except Exception as e:
                logger.error(f"Error in async {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise error_type(f"{default_message}: {str(e)}")
        return wrapper
    return decorator

def create_error_response(error: Union[WebSummarizerError, Exception]) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints.
    
    Args:
        error: The exception that occurred
        
    Returns:
        Dictionary with error details
    """
    if isinstance(error, WebSummarizerError):
        response = {
            'error': error.__class__.__name__,
            'message': error.message,
            'status': 'error'
        }
        if error.details:
            response['details'] = error.details
        return response
    else:
        return {
            'error': 'UnexpectedError',
            'message': str(error),
            'status': 'error'
        }
    
# File-related exceptions
class FileError(WebSummarizerError):
    """Base class for file-related errors."""
    pass

class FileExtractionError(FileError):
    """Error during file content extraction."""
    pass

class FileAccessError(FileError):
    """Error accessing a file."""
    pass