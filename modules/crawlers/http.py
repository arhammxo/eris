"""
HTTP-based crawler for web content extraction.

This module provides asynchronous HTTP crawling capabilities with robust
error handling, rate limiting, and content extraction.
"""
import aiohttp
import asyncio
import logging
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import urllib.robotparser
from typing import Dict, List, Optional, Set, Any, Tuple

from modules.utils.types import URL, Source
from modules.utils.errors import (
    CrawlerError, HTTPError, RobotsExclusionError, handle_async_exceptions
)
from modules.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)

# Global session and semaphores
_session: Optional[aiohttp.ClientSession] = None
_domain_semaphores: Dict[str, asyncio.Semaphore] = {}
_robots_parsers: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}

# List of common browsers user agents
BROWSER_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
]

async def get_session() -> aiohttp.ClientSession:
    """Get or create the global aiohttp session."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session

async def close_session() -> None:
    """Close the global session if it exists."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
        _session = None
        logger.info("Closed global HTTP session")

class HTTPCrawler:
    """
    Asynchronous HTTP-based web crawler.
    
    This crawler fetches content from URLs using HTTP requests, handles rate limiting,
    respects robots.txt, and extracts the main content from HTML.
    """
    
    def __init__(
        self,
        max_concurrent_total: int = 10,
        max_concurrent_per_domain: int = 3,
        timeout: int = 10,
        respect_robots_txt: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize the HTTP crawler.
        
        Args:
            max_concurrent_total: Maximum concurrent requests overall
            max_concurrent_per_domain: Maximum concurrent requests per domain
            timeout: Request timeout in seconds
            respect_robots_txt: Whether to respect robots.txt rules
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.max_concurrent_total = max_concurrent_total
        self.max_concurrent_per_domain = max_concurrent_per_domain
        self.timeout = timeout
        self.respect_robots_txt = respect_robots_txt
        self.max_retries = max_retries
        self.total_semaphore = asyncio.Semaphore(max_concurrent_total)
        
    async def crawl_urls(self, urls: List[URL], target_count: int = 1) -> List[Source]:
        """
        Crawl multiple URLs and extract content.
        
        Args:
            urls: List of URLs to crawl
            target_count: Number of successful sources to collect
            
        Returns:
            List of sources with extracted content
        """
        sources: List[Source] = []
        attempted_urls: Set[URL] = set()
        
        # Process URLs asynchronously until we reach the target count
        url_index = 0
        while len(sources) < target_count and url_index < len(urls):
            # Determine how many more tasks to launch
            tasks_needed = min(
                target_count - len(sources) + 5,  # Add buffer for potential failures
                len(urls) - url_index
            )
            
            # Launch new tasks
            new_tasks = []
            for _ in range(tasks_needed):
                if url_index < len(urls):
                    url = urls[url_index]
                    if url not in attempted_urls:
                        new_tasks.append(asyncio.create_task(self.crawl_url(url)))
                        attempted_urls.add(url)
                    url_index += 1
            
            if not new_tasks:
                break
                
            # Wait for tasks to complete with overall timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*new_tasks, return_exceptions=True),
                    timeout=self.timeout * 2  # Double the normal timeout for gather
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout while gathering crawl results for batch of {len(new_tasks)} URLs")
                # Cancel any pending tasks
                for task in new_tasks:
                    if not task.done():
                        task.cancel()
                # Just use whatever results we got so far
                results = [None] * len(new_tasks)
            
            # Process results
            for result in results:
                if result is not None and not isinstance(result, Exception):
                    sources.append(result)
                    if len(sources) >= target_count:
                        break
        
        # Log crawl statistics
        logger.info(f"Crawl completed: {len(sources)}/{target_count} sources collected")
        logger.info(f"Attempted {len(attempted_urls)} URLs in total")
        
        # Return only up to the target count
        return sources[:target_count]
    
    @handle_async_exceptions(CrawlerError, "Failed to crawl URL")
    async def crawl_url(self, url: URL) -> Optional[Source]:
        """
        Crawl a single URL and extract content with timeout.
        
        Args:
            url: URL to crawl
            
        Returns:
            Source with extracted content or None if extraction failed
        """
        # Add an overall timeout to prevent hanging
        try:
            return await asyncio.wait_for(
                self._crawl_url_internal(url), 
                timeout=self.timeout * 3  # Triple the normal timeout as a safety net
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout when crawling URL: {url}")
            raise HTTPError(f"Timeout when crawling URL: {url}")
    
    async def _crawl_url_internal(self, url: URL) -> Optional[Source]:
        """
        Internal implementation of crawl_url.
        
        Args:
            url: URL to crawl
            
        Returns:
            Source with extracted content or None if extraction failed
        """
        if not url or not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL format: {url}")
            return None
        
        # Parse URL components
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        domain = parsed_url.netloc
        
        # Set up domain-based rate limiting
        if domain not in _domain_semaphores:
            _domain_semaphores[domain] = asyncio.Semaphore(self.max_concurrent_per_domain)
        domain_semaphore = _domain_semaphores[domain]
        
        # Check robots.txt
        if self.respect_robots_txt:
            allowed = await self._check_robots_txt(base_url, url)
            if not allowed:
                raise RobotsExclusionError(f"Robots.txt disallows crawling of URL: {url}")
        
        # Get a user agent
        user_agent = random.choice(BROWSER_USER_AGENTS)
        
        # Set up headers
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Add a delay to respect rate limits
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        # Make the request with retries
        retry_delay = 1
        session = await get_session()
        response = None
        
        # Use semaphores to control concurrency
        async with self.total_semaphore, domain_semaphore:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Request attempt {attempt+1} for URL: {url}")
                    
                    # Add timeout for each request
                    response = await asyncio.wait_for(
                        session.get(
                            url, 
                            headers=headers,
                            allow_redirects=True
                        ),
                        timeout=self.timeout
                    )
                    
                    if response.status == 200:
                        break
                    
                    # Handle status codes
                    if response.status == 403:
                        # Try a different user agent
                        headers['User-Agent'] = random.choice(BROWSER_USER_AGENTS)
                        logger.info(f"Switching user agent after 403")
                    
                    # Close the response
                    response.close()
                    response = None
                    
                    # Exponential backoff
                    retry_delay *= 2
                    await asyncio.sleep(retry_delay)
                    
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Request failed (attempt {attempt+1}): {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    
                    if attempt == self.max_retries - 1:
                        raise HTTPError(f"HTTP request failed after {self.max_retries} attempts: {str(e)}", url=url)
        
        # Check if request was successful
        if not response or response.status != 200:
            status_code = response.status if response else None
            if response:
                response.close()
            raise HTTPError(
                f"Failed to fetch URL: {url}", 
                status_code=status_code, 
                url=url
            )
        
        try:
            # Parse the HTML content with timeout
            html_content_future = response.text()
            try:
                html_content = await asyncio.wait_for(html_content_future, timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading response content from {url}")
                raise HTTPError(f"Timeout reading response from {url}")
                
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract content
            content = self._extract_main_content(soup)
            
            if not content:
                logger.warning(f"No content extracted from URL: {url}")
                return None
                
            # Get metadata
            title = self._extract_title(soup)
            domain_name = self._extract_domain(url)
            
            # Return the result
            return {
                'url': url,
                'title': title,
                'domain': domain_name,
                'content': content,
                'crawled_at': time.time(),
                'original_length': len(content)
            }
            
        finally:
            # Always close the response
            if response and not response.closed:
                response.close()
    
    async def _check_robots_txt(self, base_url: str, url: str) -> bool:
        """
        Check if a URL is allowed by robots.txt.
        
        Args:
            base_url: Base URL (scheme + domain)
            url: Full URL to check
            
        Returns:
            True if crawling is allowed, False otherwise
        """
        if not self.respect_robots_txt:
            return True
            
        # Check if we already parsed robots.txt for this domain
        if base_url not in _robots_parsers:
            robots_url = urljoin(base_url, "/robots.txt")
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            
            try:
                # For simplicity, we'll handle robots.txt synchronously
                # In a production app, this should also be async
                rp.read()
                _robots_parsers[base_url] = rp
            except Exception as e:
                logger.warning(f"Could not read robots.txt for {base_url}: {e}")
                # If we can't read robots.txt, assume we're allowed
                _robots_parsers[base_url] = None
        
        # Check if we're allowed to crawl
        rp = _robots_parsers[base_url]
        if rp:
            user_agent = random.choice(BROWSER_USER_AGENTS)
            return rp.can_fetch(user_agent, url)
        
        return True
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main textual content from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted content
        """
        # Remove unwanted elements
        for element in soup.select('nav, footer, header, aside, script, style, iframe, .sidebar, .menu, .navigation, .footer, .header, .ads, .banner'):
            if element:
                element.decompose()
        
        # Try to find content by common article container IDs and classes
        content_indicators = [
            '#content', '.content', '#main-content', '.main-content',
            '#main', '.main', 'article', '.post', '.post-content',
            '#article', '.article', '.story', '.entry', '.entry-content'
        ]
        
        # Check for potential main content containers
        for indicator in content_indicators:
            content_area = soup.select_one(indicator)
            if content_area:
                # Extract paragraphs from the content area
                paragraphs = content_area.find_all('p')
                if paragraphs and len(paragraphs) > 3:
                    main_content = " ".join([p.get_text() for p in paragraphs])
                    if len(main_content) > 200:
                        return self._clean_content(main_content)
        
        # Plan B: Look for the largest block of paragraphs
        main_tags = soup.find_all(['main', 'article', 'div', 'section'])
        
        best_tag = None
        max_text_length = 0
        
        for tag in main_tags:
            # Skip if it's likely a navigation, sidebar, or footer
            if any(cls in str(tag.get('class', [])).lower() for cls in ['nav', 'menu', 'footer', 'header', 'sidebar']):
                continue
                
            # Get all text in this tag
            paragraphs = tag.find_all('p')
            if paragraphs:
                text = " ".join([p.get_text() for p in paragraphs])
                if len(text) > max_text_length:
                    max_text_length = len(text)
                    best_tag = tag
        
        # If we found a good tag with substantial text
        if best_tag and max_text_length > 200:
            main_content = " ".join([p.get_text() for p in best_tag.find_all('p')])
            return self._clean_content(main_content)
        
        # Plan C: Get all paragraphs from the page
        paragraphs = soup.find_all('p')
        if paragraphs:
            main_content = " ".join([p.get_text() for p in paragraphs])
            return self._clean_content(main_content)
        
        # Last resort: get all text from the body
        body = soup.find('body')
        if body:
            return self._clean_content(body.get_text())
        
        return ""
    
    def _clean_content(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common scripts and ads text
        noise_phrases = [
            "accept cookies", "cookie policy", "privacy policy", 
            "all rights reserved", "terms of service", "copyright",
            "subscribe to our newsletter", "sign up for our newsletter",
            "subscribe now", "subscribe to", "sign up to", 
            "advertisement", "advertisements", "advertise with us",
            "skip to content", "skip to main content", 
            "share this", "share on", "follow us", "connect with us",
            "read more", "click here", "learn more", "sign in", "log in",
            "create account", "your account", "my account", "register now",
            "related articles", "related stories", "recommended for you",
            "see also", "you might also like", "popular posts",
            "comments", "leave a comment", "post a comment",
            "this site uses cookies", "we use cookies"
        ]
        
        # Case-insensitive removal of noise phrases
        for phrase in noise_phrases:
            text = re.sub(r'(?i)' + re.escape(phrase), '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra spaces after cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try h1 if no title tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "Untitled Page"
    
    def _extract_domain(self, url: str) -> str:
        """Extract the domain name from a URL."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain