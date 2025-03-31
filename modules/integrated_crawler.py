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
    """
    sources = []
    attempted_urls = set()
    browser_crawled = set()
    
    logger.info(f"Starting integrated crawling for {len(urls)} URLs with target of {target_count} sources")
    
    # Process URLs in smaller batches until we reach the target
    url_index = 0
    while len(sources) < target_count and url_index < len(urls):
        # Calculate how many more sources we need
        remaining_sources = target_count - len(sources)
        
        # Determine how many URLs to process in this batch (use a multiplier for safety)
        batch_size = min(remaining_sources * 2, len(urls) - url_index)
        batch_urls = urls[url_index:url_index + batch_size]
        url_index += batch_size
        
        logger.info(f"Processing batch of {len(batch_urls)} URLs (have {len(sources)}/{target_count} sources)")
        
        # Try regular crawling first
        regular_sources = await async_crawl(batch_urls, remaining_sources)
        attempted_urls.update([source['url'] for source in regular_sources])
        sources.extend(regular_sources)
        
        # If we still need more sources, try browser crawling for any URLs we haven't attempted
        if len(sources) < target_count:
            remaining_sources = target_count - len(sources)
            remaining_urls = [url for url in batch_urls if url not in attempted_urls]
            
            if remaining_urls:
                # Limit to just what we need
                browser_urls = remaining_urls[:remaining_sources]
                
                # Create browser crawling tasks
                crawl_tasks = []
                for url in browser_urls:
                    task = asyncio.create_task(browser_crawl_task(url, browser_crawled))
                    crawl_tasks.append(task)
                
                # Wait for all browser crawling tasks to complete
                browser_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)
                
                # Add successful results
                for result in browser_results:
                    if result and not isinstance(result, Exception):
                        sources.append(result)
                        if len(sources) >= target_count:
                            break
    
    # Log results
    logger.info(f"Integrated crawl completed: {len(sources)}/{target_count} sources collected")
    
    # Sort sources by the order they appear in the original URL list
    url_order = {url: i for i, url in enumerate(urls)}
    sources.sort(key=lambda s: url_order.get(s['url'], float('inf')))
    
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