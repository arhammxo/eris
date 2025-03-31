import asyncio
import logging
import time
import random
from playwright.async_api import async_playwright
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to track playwright browser instances
browser_instances = {}
max_browser_instances = 2  # Limit number of concurrent browser instances

# List of patterns that indicate JavaScript-heavy sites
js_heavy_indicators = [
    r'react', r'vue', r'angular', r'svelte', r'spa',
    r'single-page-application', r'javascript-required'
]

# List of domains known to require JavaScript
js_required_domains = [
    'twitter.com', 'x.com', 'instagram.com', 'facebook.com',
    'medium.com', 'reddit.com', 'nytimes.com', 'wsj.com',
    'washingtonpost.com', 'bloomberg.com', 'cnbc.com'
]

# Semaphore to control browser instance creation
browser_semaphore = asyncio.Semaphore(max_browser_instances)

async def should_use_browser(url, html_content=None):
    """
    Determine if a URL should be processed with headless browser.
    
    Args:
        url (str): The URL to check
        html_content (str, optional): HTML content if already fetched
        
    Returns:
        bool: True if browser should be used, False otherwise
    """
    # Check domain against known list
    domain = urlparse(url).netloc.lower()
    if any(js_domain in domain for js_domain in js_required_domains):
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
            if any(re.search(pattern, src.lower()) for pattern in js_heavy_indicators):
                return True
            if any(re.search(pattern, content.lower()) for pattern in js_heavy_indicators):
                return True
        
        # Check for empty content body but with scripts
        main_content = soup.find('main') or soup.find('article') or soup.find('div', id=['content', 'main'])
        if scripts and (not main_content or len(str(main_content)) < 500):
            return True
    
    return False

async def get_browser():
    """Get or create a browser instance."""
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

async def close_browsers():
    """Close all browser instances."""
    if 'browser' in browser_instances:
        logger.info("Closing Playwright browser instance")
        await browser_instances['browser'].close()
        await browser_instances['playwright'].stop()
        browser_instances.clear()

async def crawl_with_browser(url, timeout=30):
    """
    Crawl a URL using a headless browser for JavaScript-rendered content.
    
    Args:
        url (str): URL to crawl
        timeout (int): Timeout in seconds
        
    Returns:
        dict: Extracted content and metadata or None if failed
    """
    browser = await get_browser()
    
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
        await page.goto(url, timeout=timeout * 1000, wait_until='networkidle')
        
        # Wait for content to load
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
        
        # Scroll down to load lazy content
        await page.evaluate("""
            window.scrollTo(0, document.body.scrollHeight / 2);
        """)
        await asyncio.sleep(1)
        await page.evaluate("""
            window.scrollTo(0, document.body.scrollHeight);
        """)
        await asyncio.sleep(1)
        
        # Get the page content
        html_content = await page.content()
        
        # Get page title
        title = await page.title()
        
        # Parse and extract content
        soup = BeautifulSoup(html_content, 'lxml')
        from modules.async_crawler import extract_main_content, extract_domain
        
        content = extract_main_content(soup)
        domain = extract_domain(url)
        
        # Close context to free resources
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
            'rendered_with_browser': True
        }
        
    except Exception as e:
        logger.error(f"Error during browser crawling of {url}: {str(e)}")
        return None