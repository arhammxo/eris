import aiohttp
import asyncio
import logging
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from typing import List, Dict, Any, Optional, Set, Union
from quart import current_app
from abc import ABC, abstractmethod
from playwright.async_api import async_playwright, Browser, Page

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global session for HTTP crawler
http_session: Optional[aiohttp.ClientSession] = None

# Global browser instances for browser crawler
browser_instances: Dict[str, Any] = {}

# Global semaphores for controlling concurrency
domain_semaphores: Dict[str, asyncio.Semaphore] = {}
browser_semaphore: Optional[asyncio.Semaphore] = None

# List of domains known to require JavaScript
JS_REQUIRED_DOMAINS = [
    'twitter.com', 'x.com', 'instagram.com', 'facebook.com',
    'medium.com', 'reddit.com', 'nytimes.com', 'wsj.com',
    'washingtonpost.com', 'bloomberg.com', 'cnbc.com'
]

# List of patterns that indicate JavaScript-heavy sites
JS_HEAVY_INDICATORS = [
    r'react', r'vue', r'angular', r'svelte', r'spa',
    r'single-page-application', r'javascript-required'
]


class BaseCrawler(ABC):
    """Abstract base class for all crawler implementations."""
    
    @abstractmethod
    async def crawl_urls(self, urls: List[str], target_count: int) -> List[Dict[str, Any]]:
        """
        Crawl a list of URLs and extract relevant content.
        
        Args:
            urls: List of URLs to crawl
            target_count: Number of sources to collect
            
        Returns:
            List of dictionaries containing extracted content and metadata
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources used by the crawler."""
        pass


class HTTPCrawler(BaseCrawler):
    """Crawler implementation using HTTP requests."""
    
    def __init__(self):
        """Initialize the HTTP crawler."""
        self.attempted_urls: Set[str] = set()
        
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create the global aiohttp session."""
        global http_session
        if http_session is None:
            http_session = aiohttp.ClientSession()
        return http_session
    
    async def close_session(self) -> None:
        """Close the global HTTP session."""
        global http_session
        if http_session is not None:
            await http_session.close()
            http_session = None
    
    async def crawl_urls(self, urls: List[str], target_count: int = 1) -> List[Dict[str, Any]]:
        """
        Crawl a list of URLs using HTTP requests.
        
        Args:
            urls: List of URLs to crawl
            target_count: Number of sources to collect
            
        Returns:
            List of dictionaries containing extracted content and metadata
        """
        sources: List[Dict[str, Any]] = []
        
        # Get configuration values
        timeout = current_app.config.get('CRAWL_TIMEOUT', 10)
        max_concurrent_total = current_app.config.get('MAX_CONCURRENT_REQUESTS', 10)
        max_concurrent_per_domain = current_app.config.get('MAX_CONCURRENT_PER_DOMAIN', 3)
        
        # Common browser user agents
        browser_user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        ]
        
        # Create overall concurrency semaphore
        total_semaphore = asyncio.Semaphore(max_concurrent_total)
        
        # Process URLs asynchronously
        async def process_url(url: str) -> Optional[Dict[str, Any]]:
            """Process a single URL and return content if successful."""
            if url in self.attempted_urls:
                return None
                
            self.attempted_urls.add(url)
            
            try:
                # Check URL validity
                if not url or not url.startswith(('http://', 'https://')):
                    logger.warning(f"Invalid URL format: {url}")
                    return None
                
                # Parse the URL
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                
                # Get/create domain semaphore
                if domain not in domain_semaphores:
                    domain_semaphores[domain] = asyncio.Semaphore(max_concurrent_per_domain)
                domain_semaphore = domain_semaphores[domain]
                
                # Add delay to respect rate limits
                delay = random.uniform(1.0, 3.0)
                await asyncio.sleep(delay)
                
                # Get HTTP session
                session = await self.get_session()
                
                # Select a random user agent
                user_agent = random.choice(browser_user_agents)
                
                # Define headers
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
                
                # Make the request with timeout and retries
                max_retries = 3
                retry_delay = 1
                response = None
                
                # Acquire semaphores to control concurrency
                async with total_semaphore, domain_semaphore:
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"HTTP Request attempt {attempt+1} for URL: {url}")
                            
                            # Make the request
                            response = await session.get(
                                url, 
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout),
                                allow_redirects=True
                            )
                            
                            # Check if successful
                            if response.status == 200:
                                break
                            
                            # Try a different user agent on 403
                            if response.status == 403:
                                new_user_agent = random.choice(browser_user_agents)
                                headers['User-Agent'] = new_user_agent
                                logger.info(f"Switching user agent for retry")
                            
                            # Exponential backoff
                            retry_delay *= 2
                            await asyncio.sleep(retry_delay)
                            
                            # Close failed response
                            response.close()
                        
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            logger.warning(f"HTTP request failed (attempt {attempt+1}): {str(e)}")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                
                # Skip if all retries failed
                if not response or response.status != 200:
                    status_code = response.status if response else 'unknown'
                    logger.warning(f"Received status code {status_code} for URL: {url}")
                    if response:
                        response.close()
                    return None
                
                # Get HTML content from response
                html_content = await response.text()
                
                # Check if this is a JS-heavy site
                if await should_use_browser(url, html_content):
                    logger.info(f"Detected JS-heavy site, suggesting browser crawler: {url}")
                    # Close the response
                    response.close()
                    return None
                
                # Parse HTML
                soup = BeautifulSoup(html_content, 'lxml')
                
                # Close the response
                response.close()
                
                # Extract main content
                content = extract_main_content(soup)
                
                if not content:
                    logger.warning(f"No content extracted from URL: {url}")
                    return None
                
                # Get metadata
                title = extract_title(soup)
                domain = extract_domain(url)
                
                # Return result
                return {
                    'url': url,
                    'title': title,
                    'domain': domain,
                    'content': content,
                    'crawled_at': time.time(),
                    'crawler_type': 'http'
                }
                
            except Exception as e:
                logger.error(f"Error crawling URL {url}: {str(e)}")
                return None
        
        # Use task batching to process URLs until we reach target
        results: List[Dict[str, Any]] = []
        url_index = 0
        
        while len(results) < target_count and url_index < len(urls):
            # Calculate how many more tasks to launch
            tasks_needed = min(
                target_count - len(results) + 5,  # Add buffer for failures
                len(urls) - url_index
            )
            
            # Create tasks
            tasks = []
            for i in range(tasks_needed):
                if url_index < len(urls):
                    tasks.append(asyncio.create_task(process_url(urls[url_index])))
                    url_index += 1
            
            if not tasks:
                break
                
            # Wait for tasks to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if result is not None and not isinstance(result, Exception):
                    results.append(result)
                    if len(results) >= target_count:
                        break
        
        # Log stats
        logger.info(f"HTTP Crawl completed: {len(results)}/{target_count} sources collected")
        logger.info(f"Attempted {len(self.attempted_urls)} URLs in total")
        
        return results[:target_count]
    
    async def cleanup(self) -> None:
        """Clean up resources used by the HTTP crawler."""
        await self.close_session()


class BrowserCrawler(BaseCrawler):
    """Crawler implementation using headless browser."""
    
    def __init__(self):
        """Initialize the browser crawler."""
        self.attempted_urls: Set[str] = set()
        global browser_semaphore
        
        max_browser_instances = current_app.config.get('BROWSER_INSTANCES', 2)
        if browser_semaphore is None:
            browser_semaphore = asyncio.Semaphore(max_browser_instances)
    
    async def get_browser(self) -> Browser:
        """Get or create a browser instance."""
        global browser_instances, browser_semaphore
        
        async with browser_semaphore:
            if 'browser' not in browser_instances:
                logger.info("Starting new Playwright browser instance")
                playwright = await async_playwright().start()
                browser_instances['playwright'] = playwright
                browser_instances['browser'] = await playwright.chromium.launch(
                    headless=True,
                    args=['--disable-gpu', '--disable-dev-shm-usage', '--disable-setuid-sandbox', '--no-sandbox']
                )
            return browser_instances['browser']
    
    async def close_browsers(self) -> None:
        """Close all browser instances."""
        global browser_instances
        
        if 'browser' in browser_instances:
            logger.info("Closing Playwright browser instance")
            await browser_instances['browser'].close()
            await browser_instances['playwright'].stop()
            browser_instances.clear()
    
    async def crawl_urls(self, urls: List[str], target_count: int = 1) -> List[Dict[str, Any]]:
        """
        Crawl a list of URLs using a headless browser.
        
        Args:
            urls: List of URLs to crawl
            target_count: Number of sources to collect
            
        Returns:
            List of dictionaries containing extracted content and metadata
        """
        sources: List[Dict[str, Any]] = []
        
        # Get configuration values
        timeout = current_app.config.get('BROWSER_TIMEOUT', 30)
        
        async def crawl_with_browser(url: str) -> Optional[Dict[str, Any]]:
            """Crawl a URL using headless browser."""
            if url in self.attempted_urls:
                return None
                
            self.attempted_urls.add(url)
            
            try:
                # Get browser instance
                browser = await self.get_browser()
                
                # Create context for isolation
                context = await browser.new_context(
                    viewport={'width': 1280, 'height': 800},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                
                # Create page
                page = await context.new_page()
                
                # Random wait
                await asyncio.sleep(random.uniform(1, 3))
                
                # Navigate to URL
                logger.info(f"Browser navigating to: {url}")
                await page.goto(url, timeout=timeout * 1000, wait_until='networkidle')
                
                # Try to find content selectors
                content_selectors = [
                    'main', 'article', '#content', '.content', '#main-content',
                    '.main-content', '#main', '.main', '.post', '.entry-content'
                ]
                
                # Wait for content
                for selector in content_selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=5000)
                        logger.info(f"Found content with selector: {selector}")
                        break
                    except:
                        pass
                
                # Wait for dynamic content
                await asyncio.sleep(2)
                
                # Scroll to load lazy content
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2);")
                await asyncio.sleep(1)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(1)
                
                # Get page content
                html_content = await page.content()
                
                # Get page title
                title = await page.title()
                
                # Parse and extract content
                soup = BeautifulSoup(html_content, 'lxml')
                
                # Extract content
                content = extract_main_content(soup)
                domain = extract_domain(url)
                
                # Close context
                await context.close()
                
                if not content:
                    logger.warning(f"No content extracted from browser rendering of URL: {url}")
                    return None
                
                return {
                    'url': url,
                    'title': title,
                    'domain': domain,
                    'content': content,
                    'crawled_at': time.time(),
                    'crawler_type': 'browser'
                }
                
            except Exception as e:
                logger.error(f"Error during browser crawling of {url}: {str(e)}")
                return None
        
        # Process URLs with the browser crawler
        tasks = []
        url_index = 0
        
        while len(sources) < target_count and url_index < len(urls):
            # Calculate how many more URLs to process
            remaining = min(target_count - len(sources), len(urls) - url_index)
            
            # Create tasks
            batch_tasks = []
            for _ in range(remaining):
                if url_index < len(urls):
                    batch_tasks.append(asyncio.create_task(
                        crawl_with_browser(urls[url_index])
                    ))
                    url_index += 1
            
            if not batch_tasks:
                break
                
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if result is not None and not isinstance(result, Exception):
                    sources.append(result)
                    if len(sources) >= target_count:
                        break
        
        # Log stats
        logger.info(f"Browser Crawl completed: {len(sources)}/{target_count} sources collected")
        
        return sources[:target_count]
    
    async def cleanup(self) -> None:
        """Clean up resources used by the browser crawler."""
        await self.close_browsers()


class IntegratedCrawler(BaseCrawler):
    """
    Crawler that combines HTTP and browser-based crawling strategies.
    """
    
    def __init__(self):
        """Initialize the integrated crawler."""
        self.http_crawler = HTTPCrawler()
        self.browser_crawler = BrowserCrawler()
        self.attempted_urls: Set[str] = set()
    
    async def crawl_urls(self, urls: List[str], target_count: int = 1) -> List[Dict[str, Any]]:
        """
        Crawl a list of URLs using the most appropriate crawler.
        
        Args:
            urls: List of URLs to crawl
            target_count: Number of sources to collect
            
        Returns:
            List of dictionaries containing extracted content and metadata
        """
        sources: List[Dict[str, Any]] = []
        
        # Step 1: First attempt with HTTP crawler for all URLs
        logger.info(f"Starting regular HTTP crawling for {len(urls)} URLs")
        
        # Request more URLs than needed to account for failures
        buffer_factor = 2
        initial_urls = urls[:target_count * buffer_factor]
        
        # Try HTTP crawling first
        http_sources = await self.http_crawler.crawl_urls(initial_urls, target_count)
        
        # Track attempted URLs
        self.attempted_urls.update([source['url'] for source in http_sources])
        
        # Add successful sources
        sources.extend(http_sources)
        
        # Check if we need more sources
        remaining = target_count - len(sources)
        
        if remaining > 0:
            logger.info(f"HTTP crawling got {len(sources)}/{target_count} sources. Need {remaining} more.")
            
            # Get URLs we haven't tried yet
            remaining_urls = [url for url in urls if url not in self.attempted_urls]
            
            # If we have remaining URLs, try browser crawling
            if remaining_urls:
                browser_sources = await self.browser_crawler.crawl_urls(
                    remaining_urls[:remaining * 2],  # Add buffer
                    remaining
                )
                
                # Add browser results
                sources.extend(browser_sources)
        
        # Sort by URL order to maintain relevance
        url_order = {url: i for i, url in enumerate(urls)}
        sources.sort(key=lambda s: url_order.get(s['url'], float('inf')))
        
        # Return only up to target count
        return sources[:target_count]
    
    async def cleanup(self) -> None:
        """Clean up resources used by the integrated crawler."""
        await self.http_crawler.cleanup()
        await self.browser_crawler.cleanup()


class CrawlerFactory:
    """Factory class for creating crawler instances."""
    
    def create_crawler(self) -> BaseCrawler:
        """
        Create an appropriate crawler instance based on configuration.
        
        Returns:
            BaseCrawler: An instance of a crawler
        """
        # Check if browser crawler is enabled
        use_browser = current_app.config.get('USE_BROWSER_CRAWLER', True)
        
        if use_browser:
            return IntegratedCrawler()
        else:
            return HTTPCrawler()
    
    async def cleanup_resources(self) -> None:
        """Clean up all crawler resources."""
        http_crawler = HTTPCrawler()
        browser_crawler = BrowserCrawler()
        
        await http_crawler.cleanup()
        await browser_crawler.cleanup()


# Utility functions for content extraction

async def should_use_browser(url: str, html_content: Optional[str] = None) -> bool:
    """
    Determine if a URL should be processed with headless browser.
    
    Args:
        url: The URL to check
        html_content: HTML content if already fetched
        
    Returns:
        bool: True if browser should be used, False otherwise
    """
    # Check domain against known list
    domain = urlparse(url).netloc.lower()
    if any(js_domain in domain for js_domain in JS_REQUIRED_DOMAINS):
        return True
    
    # If we have HTML content, check for SPA indicators
    if html_content:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Check for framework indicators
        scripts = soup.find_all('script')
        for script in scripts:
            src = script.get('src', '')
            content = script.string or ''
            
            # Check for framework indicators
            if any(re.search(pattern, src.lower()) for pattern in JS_HEAVY_INDICATORS):
                return True
            if any(re.search(pattern, content.lower()) for pattern in JS_HEAVY_INDICATORS):
                return True
        
        # Check for empty content with scripts
        main_content = soup.find('main') or soup.find('article') or soup.find('div', id=['content', 'main'])
        if scripts and (not main_content or len(str(main_content)) < 500):
            return True
    
    return False

def extract_main_content(soup: BeautifulSoup) -> str:
    """
    Extract the main textual content from a webpage.
    
    Args:
        soup: BeautifulSoup object of the page
        
    Returns:
        str: Extracted main content
    """
    # Remove unwanted elements
    for element in soup.select('nav, footer, header, aside, script, style, iframe, .sidebar, .menu, .navigation, .footer, .header, .ads, .banner'):
        element.decompose()
    
    # Try to find content by common containers
    content_indicators = [
        '#content', '.content', '#main-content', '.main-content',
        '#main', '.main', 'article', '.post', '.post-content',
        '#article', '.article', '.story', '.entry', '.entry-content'
    ]
    
    # Check potential containers
    for indicator in content_indicators:
        content_area = soup.select_one(indicator)
        if content_area:
            paragraphs = content_area.find_all('p')
            if paragraphs and len(paragraphs) > 3:
                main_content = " ".join([p.get_text() for p in paragraphs])
                if len(main_content) > 200:
                    return clean_content(main_content)
    
    # Find largest paragraph block
    main_tags = soup.find_all(['main', 'article', 'div', 'section'])
    
    best_tag = None
    max_text_length = 0
    
    for tag in main_tags:
        # Skip navigation, etc.
        if any(cls in str(tag.get('class', [])).lower() for cls in ['nav', 'menu', 'footer', 'header', 'sidebar']):
            continue
            
        # Get tag text
        paragraphs = tag.find_all('p')
        if paragraphs:
            text = " ".join([p.get_text() for p in paragraphs])
            if len(text) > max_text_length:
                max_text_length = len(text)
                best_tag = tag
    
    # Use best tag if found
    if best_tag and max_text_length > 200:
        main_content = " ".join([p.get_text() for p in best_tag.find_all('p')])
        return clean_content(main_content)
    
    # Get all paragraphs as fallback
    paragraphs = soup.find_all('p')
    if paragraphs:
        main_content = " ".join([p.get_text() for p in paragraphs])
        return clean_content(main_content)
    
    # Last resort: all body text
    body = soup.find('body')
    if body:
        return clean_content(body.get_text())
    
    return ""

def clean_content(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and common noise.
    
    Args:
        text: Raw text content
        
    Returns:
        str: Cleaned text
    """
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Remove common noise phrases
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
    
    # Remove noise phrases (case insensitive)
    for phrase in noise_phrases:
        text = re.sub(r'(?i)' + re.escape(phrase), '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_title(soup: BeautifulSoup) -> str:
    """
    Extract the page title.
    
    Args:
        soup: BeautifulSoup object of the page
        
    Returns:
        str: Page title
    """
    # Try standard title tag
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    
    # Try h1 as fallback
    h1_tag = soup.find('h1')
    if h1_tag:
        return h1_tag.get_text().strip()
    
    return "Untitled Page"

def extract_domain(url: str) -> str:
    """
    Extract the domain name from a URL.
    
    Args:
        url: Full URL
        
    Returns:
        str: Domain name
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Remove www prefix
    if domain.startswith('www.'):
        domain = domain[4:]
        
    return domain