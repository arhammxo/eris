"""
Integrated crawler that combines HTTP and browser-based crawling.

This module provides a unified interface for content extraction,
intelligently switching between HTTP and browser crawling as needed.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from quart import current_app

from modules.utils.types import URL, Source
from modules.utils.errors import CrawlerError, handle_async_exceptions
from modules.utils.logging import get_logger
from modules.crawlers.http import HTTPCrawler
from modules.crawlers.browser import BrowserCrawler
from modules.crawlers.file import FileCrawler  # Add this line

# Create a logger for this module
logger = get_logger(__name__)

class IntegratedCrawler:
    """
    Integrated crawler that combines HTTP and browser-based crawling.
    
    This crawler first attempts regular HTTP crawling, and if that fails or detects
    a JavaScript-heavy site, it falls back to browser-based crawling.
    """
    
    def __init__(
        self,
        use_browser: bool = True,
        use_file_search: bool = True,
        max_concurrent_requests: int = 10,
        max_concurrent_per_domain: int = 3,
        http_timeout: int = 10,
        browser_timeout: int = 30,
        max_browser_instances: int = 2,
        base_directories: List[str] = None,
        respect_robots_txt: bool = True
    ):
        """
        Initialize the integrated crawler.
        
        Args:
            use_browser: Whether to use browser crawling when needed
            max_concurrent_requests: Maximum concurrent HTTP requests
            max_concurrent_per_domain: Maximum concurrent requests per domain
            http_timeout: HTTP request timeout in seconds
            browser_timeout: Browser request timeout in seconds
            max_browser_instances: Maximum concurrent browser instances
            respect_robots_txt: Whether to respect robots.txt
        """
        self.use_browser = use_browser
        self.http_crawler = HTTPCrawler(
            max_concurrent_total=max_concurrent_requests,
            max_concurrent_per_domain=max_concurrent_per_domain,
            timeout=http_timeout,
            respect_robots_txt=respect_robots_txt
        )
        
        self.use_file_search = use_file_search
        
        if use_file_search:
            self.file_crawler = FileCrawler(
                base_directories=base_directories,
                max_concurrent_extractions=max_concurrent_requests
            )
        else:
            self.file_crawler = None
        
        if use_browser:
            self.browser_crawler = BrowserCrawler(
                max_browser_instances=max_browser_instances,
                timeout=browser_timeout
            )
        else:
            self.browser_crawler = None

    # Add a new method for handling file search
    async def crawl_local_files(self, query: str, target_count: int = 1) -> List[Source]:
        """
        Crawl local files matching the query.
        
        Args:
            query: Search query
            target_count: Number of matching files to collect
            
        Returns:
            List of sources with extracted content
        """
        if not self.use_file_search or not self.file_crawler:
            return []
            
        return await self.file_crawler.crawl_files(query, target_count=target_count)
    
    # Modify crawl_urls to include file search results
    @handle_async_exceptions(CrawlerError, "Integrated crawling failed")
    async def crawl_urls(self, urls: List[URL], query: str = "", target_count: int = 1, 
                        include_files: bool = False) -> List[Source]:
        """
        Crawl multiple URLs using an integrated approach.
        
        Args:
            urls: List of URLs to crawl
            query: Original search query (used for file search)
            target_count: Number of successful sources to collect
            include_files: Whether to include file search results
            
        Returns:
            List of sources with extracted content
        """
        logger.info(f"Starting integrated crawling for {len(urls)} URLs, target count: {target_count}")
        
        sources: List[Source] = []
        
        # First attempt: Try regular crawling
        regular_sources = await self.http_crawler.crawl_urls(
            urls[:target_count * 2],  # Request more URLs than needed
            target_count
        )
        
        sources.extend(regular_sources)
        
        # Add browser crawling (existing code)...
        
        # Add file search if enabled
        if include_files and self.use_file_search and self.file_crawler and query:
            remaining = target_count - len(sources)
            if remaining > 0:
                file_sources = await self.file_crawler.crawl_files(query, target_count=remaining)
                sources.extend(file_sources)
        
        # Return results, respecting target count
        return sources[:target_count]
    
    async def _browser_crawl_batch(self, urls: List[URL], browser_crawled: Set[URL]) -> List[Source]:
        """
        Crawl a batch of URLs using browser crawling.
        
        Args:
            urls: URLs to crawl
            browser_crawled: Set of already browser-crawled URLs
            
        Returns:
            List of successfully crawled sources
        """
        if not self.use_browser or not self.browser_crawler:
            return []
            
        browser_sources: List[Source] = []
        
        # Create tasks for browser crawling
        tasks = []
        for url in urls:
            if url not in browser_crawled:
                browser_crawled.add(url)
                task = asyncio.create_task(self._browser_crawl_task(url))
                tasks.append(task)
        
        if not tasks:
            return []
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if result is not None and not isinstance(result, Exception):
                browser_sources.append(result)
        
        return browser_sources
    
    async def _browser_crawl_task(self, url: URL) -> Optional[Source]:
        """
        Task for browser-based crawling of a URL.
        
        Args:
            url: URL to crawl
            
        Returns:
            Source with extracted content or None if extraction failed
        """
        if not self.browser_crawler:
            return None
            
        try:
            logger.info(f"Attempting browser crawling for: {url}")
            
            # Use browser crawler
            result = await self.browser_crawler.crawl_url(url)
            
            if result:
                logger.info(f"Successfully crawled with browser: {url}")
                return result
            else:
                logger.warning(f"Browser crawling failed for: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error in browser crawling task for {url}: {str(e)}")
            return None
    
    @staticmethod
    async def cleanup() -> None:
        """Clean up all crawler resources."""
        tasks = []
        
        # Close HTTP session
        tasks.append(HTTPCrawler.close_session())
        
        # Close browser instances
        tasks.append(BrowserCrawler.close_browsers())
        
        # Wait for all cleanup tasks to complete
        await asyncio.gather(*tasks)
        logger.info("All crawler resources cleaned up")


# Convenience functions
async def create_crawler_from_config() -> IntegratedCrawler:
    """
    Create an integrated crawler from application configuration.
    
    Returns:
        Configured IntegratedCrawler instance
    """
    # Get configuration from current_app
    use_browser = current_app.config.get('USE_BROWSER_CRAWLER', True)
    max_concurrent_requests = current_app.config.get('MAX_CONCURRENT_REQUESTS', 10)
    max_concurrent_per_domain = current_app.config.get('MAX_CONCURRENT_PER_DOMAIN', 3)
    http_timeout = current_app.config.get('CRAWL_TIMEOUT', 10)
    browser_timeout = current_app.config.get('BROWSER_TIMEOUT', 30)
    max_browser_instances = current_app.config.get('BROWSER_INSTANCES', 2)
    respect_robots_txt = current_app.config.get('RESPECT_ROBOTS_TXT', True)
    
    return IntegratedCrawler(
        use_browser=use_browser,
        max_concurrent_requests=max_concurrent_requests,
        max_concurrent_per_domain=max_concurrent_per_domain,
        http_timeout=http_timeout,
        browser_timeout=browser_timeout,
        max_browser_instances=max_browser_instances,
        respect_robots_txt=respect_robots_txt
    )

async def crawl_urls(urls: List[URL], target_count: int = 1) -> List[Source]:
    """
    Crawl URLs using the integrated crawler.
    
    Args:
        urls: List of URLs to crawl
        target_count: Number of sources to collect
        
    Returns:
        List of sources with extracted content
    """
    crawler = await create_crawler_from_config()
    return await crawler.crawl_urls(urls, target_count)

async def cleanup_crawlers() -> None:
    """Clean up all crawler resources."""
    await IntegratedCrawler.cleanup()