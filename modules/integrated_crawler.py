import asyncio
import logging
from quart import current_app
from modules.async_crawler import crawl_urls as async_crawl, get_session, close_session
from modules.browser_crawler import crawl_with_browser, should_use_browser, close_browsers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def integrated_crawl(urls, target_count=1):
    """
    Integrated crawling function that combines regular and browser-based crawling.
    
    This function first attempts regular HTTP crawling, and if that fails or
    detects a JavaScript-heavy site, it falls back to browser-based crawling.
    
    Args:
        urls (list): List of URLs to crawl
        target_count (int): Number of sources to collect
        
    Returns:
        list: List of dictionaries containing extracted content and metadata
    """
    sources = []
    attempted_urls = set()
    browser_crawled = set()
    
    # Queue to track URLs that need browser crawling
    browser_queue = asyncio.Queue()
    
    # First attempt: Try regular crawling for all URLs
    logger.info(f"Starting regular crawling for {len(urls)} URLs")
    
    # Request more URLs than needed to account for failures
    buffer_factor = 2
    initial_urls = urls[:target_count * buffer_factor]
    
    # Perform regular crawling
    regular_sources = await async_crawl(initial_urls, target_count)
    attempted_urls.update([source['url'] for source in regular_sources])
    
    # Add successful sources to our list
    sources.extend(regular_sources)
    
    # Check if we need more sources
    remaining = target_count - len(sources)
    
    if remaining > 0:
        logger.info(f"Regular crawling got {len(sources)}/{target_count} sources. Need {remaining} more.")
        
        # Get URLs we haven't tried yet
        remaining_urls = [url for url in urls if url not in attempted_urls]
        
        # If we have remaining URLs, try browser crawling
        if remaining_urls:
            # Prioritize browser crawling for remaining sources
            crawl_tasks = []
            for url in remaining_urls[:remaining]:
                task = asyncio.create_task(browser_crawl_task(url, browser_crawled))
                crawl_tasks.append(task)
                
            # Wait for browser crawling tasks to complete
            browser_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)
            
            # Process successful results
            for result in browser_results:
                if result and not isinstance(result, Exception):
                    sources.append(result)
                    if len(sources) >= target_count:
                        break
    
    # Sort sources by the order they appear in the original URL list
    # This maintains the search relevance order
    url_order = {url: i for i, url in enumerate(urls)}
    sources.sort(key=lambda s: url_order.get(s['url'], float('inf')))
    
    # Return only up to the target count
    return sources[:target_count]

async def browser_crawl_task(url, browser_crawled):
    """Task for browser-based crawling of a URL."""
    try:
        # Skip if already crawled with browser
        if url in browser_crawled:
            return None
            
        logger.info(f"Attempting browser crawling for: {url}")
        browser_crawled.add(url)
        
        # Use browser crawler
        result = await crawl_with_browser(url)
        
        if result:
            logger.info(f"Successfully crawled with browser: {url}")
            return result
        else:
            logger.warning(f"Browser crawling failed for: {url}")
            return None
            
    except Exception as e:
        logger.error(f"Error in browser crawling task for {url}: {str(e)}")
        return None

async def cleanup_crawlers():
    """Clean up all crawler resources."""
    await close_session()
    await close_browsers()