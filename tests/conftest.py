"""
Test configuration and fixtures for the Web Summarizer application.
"""
import os
import sys
import pytest
import asyncio
from typing import Dict, Any, Generator, AsyncGenerator
import logging
from quart import Quart

try:  # Quart >= 0.19 renamed TestClient to QuartClient
    from quart.testing import QuartClient as TestClient
except ImportError:  # pragma: no cover - older Quart releases
    from quart.testing import TestClient

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as quart_app
from modules.utils.logging import setup_logging

# Configure logging for tests
setup_logging(level=logging.INFO, structured=False)

@pytest.fixture
def app() -> Generator[Quart, None, None]:
    """
    Create a Quart application for testing.
    
    Returns:
        Quart application instance
    """
    # Configure the app for testing
    quart_app.config['TESTING'] = True
    quart_app.config['DEBUG'] = False
    
    # Use in-memory cache for testing
    quart_app.config['CACHE_TYPE'] = 'SimpleCache'
    
    # Disable browser crawler for tests by default
    quart_app.config['USE_BROWSER_CRAWLER'] = False
    
    # Mock API keys if not present in environment
    if not quart_app.config.get('OPENAI_API_KEY'):
        quart_app.config['OPENAI_API_KEY'] = 'test_openai_key'
    if not quart_app.config.get('SERPAPI_API_KEY'):
        quart_app.config['SERPAPI_API_KEY'] = 'test_serpapi_key'
    
    yield quart_app

@pytest.fixture
def test_client(app) -> Generator[TestClient, None, None]:
    """
    Create a test client for the Quart application.
    
    Args:
        app: Quart application fixture
        
    Returns:
        Test client
    """
    return app.test_client()

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """
    Create a test configuration.
    
    Returns:
        Configuration dictionary
    """
    return {
        'SERPAPI_API_KEY': 'test_serpapi_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'MAX_URLS_TO_CRAWL': 3,
        'CRAWL_TIMEOUT': 5,
        'USE_BROWSER_CRAWLER': False,
        'MAX_CONCURRENT_REQUESTS': 5,
        'MAX_CONCURRENT_PER_DOMAIN': 2,
        'CACHE_TYPE': 'SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': 60,
        'CHUNK_SIZE': 1000,
        'QUALITY_THRESHOLD': 0.3
    }

@pytest.fixture
def mock_source() -> Dict[str, Any]:
    """
    Create a mock source for testing.
    
    Returns:
        Mock source dictionary
    """
    return {
        'url': 'https://example.com/test',
        'title': 'Test Page',
        'domain': 'example.com',
        'content': """
        This is a test page with some content.
        It contains multiple paragraphs and sentences.
        
        Here is another paragraph with more information.
        This should be enough for testing purposes.
        
        We add a third paragraph to make it more realistic.
        Testing is important for ensuring code quality.
        """,
        'crawled_at': 1633027200.0,
        'original_length': 300
    }

@pytest.fixture
async def async_mock_source() -> AsyncGenerator[Dict[str, Any], None]:
    """
    Create a mock source for async testing.
    
    Returns:
        Mock source dictionary
    """
    source = {
        'url': 'https://example.com/test',
        'title': 'Test Page',
        'domain': 'example.com',
        'content': """
        This is a test page with some content.
        It contains multiple paragraphs and sentences.
        
        Here is another paragraph with more information.
        This should be enough for testing purposes.
        
        We add a third paragraph to make it more realistic.
        Testing is important for ensuring code quality.
        """,
        'crawled_at': 1633027200.0,
        'original_length': 300
    }
    
    yield source

@pytest.fixture
def mock_processed_source() -> Dict[str, Any]:
    """
    Create a mock processed source for testing.
    
    Returns:
        Mock processed source dictionary
    """
    return {
        'url': 'https://example.com/test',
        'title': 'Test Page',
        'domain': 'example.com',
        'cleaned_text': """
        this is a test page with some content
        it contains multiple paragraphs and sentences
        
        here is another paragraph with more information
        this should be enough for testing purposes
        
        we add a third paragraph to make it more realistic
        testing is important for ensuring code quality
        """,
        'important_sentences': [
            "This is a test page with some content.",
            "Testing is important for ensuring code quality."
        ],
        'chunks': [
            """this is a test page with some content
            it contains multiple paragraphs and sentences
            
            here is another paragraph with more information
            this should be enough for testing purposes"""
        ],
        'original_length': 300,
        'processed_length': 280,
        'quality_score': 0.75
    }

# Automatically use the event loop
@pytest.fixture(scope="session")
def event_loop():
    """
    Create an asyncio event loop for testing.
    
    Returns:
        Asyncio event loop
    """
    try:
        loop = asyncio.get_event_loop_policy().new_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    
    yield loop
    
    # Clean up
    loop.close()