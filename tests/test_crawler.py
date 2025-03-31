"""
Tests for the crawler modules.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from bs4 import BeautifulSoup

from modules.crawlers.http import HTTPCrawler
from modules.crawlers.browser import BrowserCrawler
from modules.crawlers.integrated import IntegratedCrawler
from modules.utils.errors import HTTPError, BrowserError

@pytest.mark.asyncio
async def test_http_crawler_init():
    """Test HTTP crawler initialization."""
    crawler = HTTPCrawler(
        max_concurrent_total=5,
        max_concurrent_per_domain=2,
        timeout=10,
        respect_robots_txt=True,
        max_retries=3
    )
    
    assert crawler.max_concurrent_total == 5
    assert crawler.max_concurrent_per_domain == 2
    assert crawler.timeout == 10
    assert crawler.respect_robots_txt == True
    assert crawler.max_retries == 3
    assert isinstance(crawler.total_semaphore, asyncio.Semaphore)

@pytest.mark.asyncio
async def test_extract_main_content():
    """Test content extraction from HTML."""
    html = """
    <html>
      <head><title>Test Page</title></head>
      <body>
        <header>Header content</header>
        <nav>Navigation</nav>
        <article>
          <h1>Main Article</h1>
          <p>This is the first paragraph.</p>
          <p>This is the second paragraph.</p>
        </article>
        <footer>Footer content</footer>
      </body>
    </html>
    """
    
    soup = BeautifulSoup(html, 'lxml')
    crawler = HTTPCrawler()
    
    content = crawler._extract_main_content(soup)
    
    # The content should contain the paragraphs but not header/footer
    assert "first paragraph" in content
    assert "second paragraph" in content
    assert "Header content" not in content
    assert "Footer content" not in content

@pytest.mark.asyncio
async def test_clean_content():
    """Test cleaning extracted text."""
    raw_text = """
    This is some content with extra  whitespace.
    
    It contains some noise phrases like subscribe to our newsletter and accept cookies.
    
    It also has a URL https://example.com and an email@example.com.
    """
    
    crawler = HTTPCrawler()
    cleaned = crawler._clean_content(raw_text)
    
    # Check noise removal
    assert "subscribe to our newsletter" not in cleaned
    assert "accept cookies" not in cleaned
    assert "https://example.com" not in cleaned
    assert "email@example.com" not in cleaned
    
    # Check whitespace normalization
    assert "  " not in cleaned

@pytest.mark.asyncio
@patch('modules.crawlers.http.HTTPCrawler.crawl_url')
async def test_crawl_urls(mock_crawl_url):
    """Test crawling multiple URLs."""
    # Setup mock
    mock_responses = [
        {
            'url': 'https://example.com/1',
            'title': 'Page 1',
            'domain': 'example.com',
            'content': 'Content 1',
            'crawled_at': 1633027200.0,
            'original_length': 9
        },
        {
            'url': 'https://example.com/2',
            'title': 'Page 2',
            'domain': 'example.com',
            'content': 'Content 2',
            'crawled_at': 1633027200.0,
            'original_length': 9
        },
        None,  # Simulate a failed crawl
        {
            'url': 'https://example.com/4',
            'title': 'Page 4',
            'domain': 'example.com',
            'content': 'Content 4',
            'crawled_at': 1633027200.0,
            'original_length': 9
        }
    ]
    
    # Make the mock return different values for different URLs
    async def side_effect(url):
        index = int(url.split('/')[-1]) - 1
        if index < len(mock_responses):
            return mock_responses[index]
        return None
    
    mock_crawl_url.side_effect = side_effect
    
    # Test crawling with target count
    crawler = HTTPCrawler()
    urls = [f'https://example.com/{i}' for i in range(1, 5)]
    results = await crawler.crawl_urls(urls, target_count=2)
    
    # Should return 2 valid results
    assert len(results) == 2
    assert results[0]['url'] == 'https://example.com/1'
    assert results[1]['url'] == 'https://example.com/2'
    
    # Check that crawl_url was called for each URL
    assert mock_crawl_url.call_count == 3  # Stops after getting 2 results

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.get')
@patch('modules.crawlers.http.HTTPCrawler._check_robots_txt')
async def test_crawl_url(mock_check_robots, mock_get):
    """Test crawling a single URL."""
    # Setup mocks
    mock_check_robots.return_value = True
    
    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text.return_value = """
    <html>
      <head><title>Test Page</title></head>
      <body>
        <article>
          <p>Test content paragraph 1.</p>
          <p>Test content paragraph 2.</p>
        </article>
      </body>
    </html>
    """
    mock_response.closed = False
    mock_get.return_value.__aenter__.return_value = mock_response
    
    # Test crawling
    crawler = HTTPCrawler()
    result = await crawler.crawl_url('https://example.com/test')
    
    # Check result
    assert result is not None
    assert result['url'] == 'https://example.com/test'
    assert result['title'] == 'Test Page'
    assert result['domain'] == 'example.com'
    assert 'Test content paragraph 1.' in result['content']
    assert 'Test content paragraph 2.' in result['content']
    assert 'crawled_at' in result
    assert 'original_length' in result

@pytest.mark.asyncio
@patch('modules.crawlers.browser.BrowserCrawler.crawl_url')
@patch('modules.crawlers.http.HTTPCrawler.crawl_urls')
async def test_integrated_crawler(mock_http_crawl, mock_browser_crawl):
    """Test the integrated crawler."""
    # Setup mocks
    mock_http_results = [
        {
            'url': 'https://example.com/1',
            'title': 'Page 1',
            'domain': 'example.com',
            'content': 'Content 1',
            'crawled_at': 1633027200.0,
            'original_length': 9
        }
    ]
    mock_http_crawl.return_value = mock_http_results
    
    mock_browser_results = {
        'url': 'https://example.com/2',
        'title': 'Page 2',
        'domain': 'example.com',
        'content': 'Content 2',
        'crawled_at': 1633027200.0,
        'rendered_with_browser': True,
        'original_length': 9
    }
    mock_browser_crawl.return_value = mock_browser_results
    
    # Test integrated crawling with browser fallback
    crawler = IntegratedCrawler(use_browser=True)
    urls = ['https://example.com/1', 'https://example.com/2']
    results = await crawler.crawl_urls(urls, target_count=2)
    
    # Check that both crawlers were used
    mock_http_crawl.assert_called_once()
    
    # Should contain results from both crawlers
    assert len(results) == 2
    assert results[0]['url'] == 'https://example.com/1'
    assert 'rendered_with_browser' not in results[0]
    assert results[1]['url'] == 'https://example.com/2'
    assert results[1]['rendered_with_browser'] == True

@pytest.mark.asyncio
async def test_should_use_browser():
    """Test browser necessity detection."""
    # HTML with React
    html_with_react = """
    <html>
      <head>
        <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
      </head>
      <body>
        <div id="root"></div>
      </body>
    </html>
    """
    
    # HTML without JavaScript frameworks
    html_without_js = """
    <html>
      <head><title>Regular Page</title></head>
      <body>
        <article>
          <p>Regular content.</p>
        </article>
      </body>
    </html>
    """
    
    # Test detection
    assert await BrowserCrawler.should_use_browser('https://twitter.com/example') == True
    assert await BrowserCrawler.should_use_browser('https://example.com', html_with_react) == True
    assert await BrowserCrawler.should_use_browser('https://example.com', html_without_js) == False