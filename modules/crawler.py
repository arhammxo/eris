import requests
from bs4 import BeautifulSoup
import logging
import time
from flask import current_app
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crawl_urls(urls):
    """
    Crawl a list of URLs and extract relevant content.
    
    Args:
        urls (list): List of URLs to crawl
        
    Returns:
        list: List of dictionaries containing extracted content and metadata
    """
    sources = []
    timeout = current_app.config.get('CRAWL_TIMEOUT', 10)
    user_agent = current_app.config.get('USER_AGENT', 'WebSummarizerBot/1.0')
    
    headers = {
        'User-Agent': user_agent
    }
    
    for url in urls:
        try:
            logger.info(f"Crawling URL: {url}")
            
            # Skip if URL is invalid
            if not url or not url.startswith(('http://', 'https://')):
                continue
                
            # Make the request with timeout
            response = requests.get(url, headers=headers, timeout=timeout)
            
            # Skip if response is not successful
            if response.status_code != 200:
                logger.warning(f"Received status code {response.status_code} for URL: {url}")
                continue
                
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract the main content
            content = extract_main_content(soup)
            
            if not content:
                logger.warning(f"No content extracted from URL: {url}")
                continue
                
            # Get metadata
            title = extract_title(soup)
            domain = extract_domain(url)
            
            # Add to sources
            sources.append({
                'url': url,
                'title': title,
                'domain': domain,
                'content': content,
                'crawled_at': time.time()
            })
            
            # Sleep briefly to avoid hammering servers
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error crawling URL {url}: {str(e)}")
            continue
            
    return sources

def extract_main_content(soup):
    """
    Extract the main textual content from a webpage.
    
    This is a simplified implementation. A production version would use
    more sophisticated content extraction algorithms.
    """
    # Try to find main content containers
    main_tags = soup.find_all(['main', 'article', 'div', 'section'])
    
    # Find the tag with most paragraphs
    max_paragraphs = 0
    main_content = ""
    
    for tag in main_tags:
        paragraphs = tag.find_all('p')
        if len(paragraphs) > max_paragraphs:
            max_paragraphs = len(paragraphs)
            main_content = " ".join([p.get_text() for p in paragraphs])
    
    # If we didn't find any paragraphs in common containers, just get all paragraphs
    if not main_content:
        paragraphs = soup.find_all('p')
        main_content = " ".join([p.get_text() for p in paragraphs])
    
    # Clean the content
    main_content = clean_content(main_content)
    
    return main_content

def clean_content(text):
    """Clean extracted text by removing extra whitespace and common noise."""
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove common scripts and ads text (simplified)
    noise_phrases = [
        "accept cookies",
        "cookie policy",
        "privacy policy",
        "all rights reserved",
        "terms of service",
        "copyright",
        "subscribe to our newsletter"
    ]
    
    for phrase in noise_phrases:
        text = text.replace(phrase, "")
    
    return text

def extract_title(soup):
    """Extract the page title."""
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    
    # Try h1 if no title tag
    h1_tag = soup.find('h1')
    if h1_tag:
        return h1_tag.get_text().strip()
    
    return "Untitled Page"

def extract_domain(url):
    """Extract the domain name from a URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
        
    return domain