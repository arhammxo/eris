"""
Browser-based crawler for JavaScript-heavy websites.

This module provides browser-based crawling capabilities for sites that require
JavaScript rendering, using Playwright for headless browser automation.
"""
import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Set, Any
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser, Page, Playwright
from bs4 import BeautifulSoup
import re

from modules.utils.types import URL, Source
from modules.utils.errors import BrowserError, handle_async_exceptions
from modules.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)

# Browser instance management
_playwright: Optional[Playwright] = None
_browser: Optional[Browser] = None
_browser_semaphore = asyncio.Semaphore(1)  # Semaphore for browser instance creation

# List of patterns that indicate JavaScript-heavy sites
JS_HEAVY_INDICATORS = [
    r'react', r'vue', r'angular', r'svelte', r'spa',
    r'single-page-application', r'javascript-required'
]

# List of domains known to require JavaScript
JS_REQUIRED_DOMAINS = [
    'twitter.com', 'x.com', 'instagram.com', 'facebook.com',
    'medium.com', 'reddit.com', 'nytimes.com', 'wsj.com',
    'washingtonpost.com', 'bloomberg.com', 'cnbc.com'
]

class BrowserCrawler:
    """
    Browser-based crawler for JavaScript-rendered websites.
    
    This crawler uses Playwright to render pages in a headless browser,
    allowing extraction of content from JavaScript-heavy sites.
    """
    
    def __init__(
        self,
        max_browser_instances: int = 2,
        timeout: int = 30,
        headless: bool = True
    ):
        """
        Initialize the browser crawler.
        
        Args:
            max_browser_instances: Maximum concurrent browser instances
            timeout: Page load timeout in seconds
            headless: Whether to run the browser in headless mode
        """
        self.max_browser_instances = max_browser_instances
        self.timeout = timeout
        self.headless = headless
        
        # Create a semaphore to control browser instance usage
        global _browser_semaphore
        _browser_semaphore = asyncio.Semaphore(max_browser_instances)
    
    @staticmethod
    async def should_use_browser(url: URL, html_content: Optional[str] = None) -> bool:
        """
        Determine if a URL should be processed with headless browser.
        
        Args:
            url: URL to check
            html_content: HTML content if already fetched
            
        Returns:
            True if browser should be used, False otherwise
        """
        # Check domain against known list
        domain = urlparse(url).netloc.lower()
        if any(js_domain in domain for js_domain in JS_REQUIRED_DOMAINS):
            return True
        
        # If we have HTML content, check for SPA indicators
        if html_content:
            # Look for SPA frameworks
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Check for React/Vue/Angular indicators
            scripts = soup.find_all('script')
            for script in scripts:
                src = script.get('src', '')
                content = script.string or ''
                
                # Check script src and content for framework indicators
                if any(re.search(pattern, src.lower()) for pattern in JS_HEAVY_INDICATORS):
                    return True
                if any(re.search(pattern, content.lower()) for pattern in JS_HEAVY_INDICATORS):
                    return True
            
            # Check for empty content body but with scripts
            main_content = soup.find('main') or soup.find('article') or soup.find('div', id=['content', 'main'])
            if scripts and (not main_content or len(str(main_content)) < 500):
                return True
        
        return False
    
    async def get_browser(self) -> Browser:
        """
        Get or create a browser instance.
        
        Returns:
            Playwright browser instance
        
        Raises:
            BrowserError: If browser initialization fails
        """
        global _playwright, _browser, _browser_semaphore
        
        async with _browser_semaphore:
            if _browser is None or not _browser.is_connected():
                try:
                    logger.info("Starting new Playwright browser instance")
                    if _playwright is None:
                        _playwright = await async_playwright().start()
                    
                    _browser = await _playwright.chromium.launch(
                        headless=self.headless,
                        args=[
                            '--disable-gpu',
                            '--disable-dev-shm-usage',
                            '--disable-setuid-sandbox',
                            '--no-sandbox'
                        ]
                    )
                except Exception as e:
                    logger.error(f"Browser initialization failed: {str(e)}")
                    raise BrowserError(f"Failed to initialize browser: {str(e)}")
        
        return _browser
    
    @staticmethod
    async def close_browsers() -> None:
        """Close all browser instances."""
        global _playwright, _browser
        
        try:
            if _browser is not None:
                logger.info("Closing Playwright browser instance")
                await _browser.close()
                _browser = None
            
            if _playwright is not None:
                logger.info("Stopping Playwright")
                await _playwright.stop()
                _playwright = None
                
        except Exception as e:
            logger.error(f"Error closing browser resources: {str(e)}")
    
    @handle_async_exceptions(BrowserError, "Browser crawling failed")
    async def crawl_url(self, url: URL) -> Optional[Source]:
        """
        Crawl a URL using a headless browser.
        
        Args:
            url: URL to crawl
            
        Returns:
            Source with extracted content or None if extraction failed
        """
        browser = await self.get_browser()
        
        try:
            # Create a new context for isolation
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            # Create a new page
            page = await context.new_page()
            
            # Random wait for a more human-like behavior
            await asyncio.sleep(random.uniform(1, 3))
            
            # Navigate to URL with timeout
            logger.info(f"Browser navigating to: {url}")
            
            try:
                await page.goto(
                    url, 
                    timeout=self.timeout * 1000, 
                    wait_until='networkidle'
                )
            except Exception as e:
                logger.warning(f"Navigation error for {url}: {str(e)}")
                # If navigation fails, try a more lenient approach
                await page.goto(
                    url,
                    timeout=self.timeout * 1000,
                    wait_until='domcontentloaded'
                )
            
            # Wait for content to load
            await self._wait_for_content(page)
            
            # Scroll to load lazy content
            await self._scroll_page(page)
            
            # Get the page content
            html_content = await page.content()
            
            # Get page title
            title = await page.title()
            
            # Close context to free resources
            await context.close()
            
            # Extract content
            content = self._extract_content(html_content, url)
            
            if not content:
                logger.warning(f"No content extracted from browser rendering of URL: {url}")
                return None
            
            # Get domain from URL
            domain = self._extract_domain(url)
            
            return {
                'url': url,
                'title': title,
                'domain': domain,
                'content': content,
                'crawled_at': time.time(),
                'rendered_with_browser': True,
                'original_length': len(content)
            }
            
        except Exception as e:
            logger.error(f"Error during browser crawling of {url}: {str(e)}")
            raise BrowserError(f"Browser crawling failed for {url}: {str(e)}")
    
    async def _wait_for_content(self, page: Page) -> None:
        """
        Wait for content to load on the page.
        
        Args:
            page: Playwright page object
        """
        # First try to find common content selectors
        content_selectors = [
            'main', 'article', '#content', '.content', '#main-content',
            '.main-content', '#main', '.main', '.post', '.entry-content'
        ]
        
        # Wait for at least one content element to appear
        for selector in content_selectors:
            try:
                await page.wait_for_selector(selector, timeout=5000)
                logger.info(f"Found content with selector: {selector}")
                break
            except:
                pass
        
        # Additional delay to ensure dynamic content loads
        await asyncio.sleep(2)
    
    async def _scroll_page(self, page: Page) -> None:
        """
        Scroll the page to load lazy content.
        
        Args:
            page: Playwright page object
        """
        # Scroll to middle
        await page.evaluate("""
            window.scrollTo(0, document.body.scrollHeight / 2);
        """)
        await asyncio.sleep(1)
        
        # Scroll to bottom
        await page.evaluate("""
            window.scrollTo(0, document.body.scrollHeight);
        """)
        await asyncio.sleep(1)
    
    def _extract_content(self, html_content: str, url: str) -> str:
        """
        Extract main content from HTML.
        
        Args:
            html_content: HTML content
            url: URL of the page
            
        Returns:
            Extracted content
        """
        from modules.crawlers.http import HTTPCrawler
        
        # Reuse content extraction from HTTP crawler
        soup = BeautifulSoup(html_content, 'lxml')
        http_crawler = HTTPCrawler()
        content = http_crawler._extract_main_content(soup)
        
        # If content extraction failed, try a more aggressive approach
        if not content:
            # Get all text from the page
            body = soup.find('body')
            if body:
                content = http_crawler._clean_content(body.get_text())
        
        return content
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL
            
        Returns:
            Domain name
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain