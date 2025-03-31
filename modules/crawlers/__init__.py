"""
Web crawler modules for content extraction.

This package provides different crawling strategies for extracting content from web pages:

- HTTP Crawler: Standard HTTP-based crawling for regular websites
- Browser Crawler: Browser-based crawling for JavaScript-heavy websites
- Integrated Crawler: Smart combination of HTTP and browser crawling
"""

from modules.crawlers.http import HTTPCrawler
from modules.crawlers.browser import BrowserCrawler
from modules.crawlers.integrated import (
    IntegratedCrawler, 
    crawl_urls,
    cleanup_crawlers
)

__all__ = [
    'HTTPCrawler',
    'BrowserCrawler',
    'IntegratedCrawler',
    'crawl_urls',
    'cleanup_crawlers'
]