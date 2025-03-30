import requests
from bs4 import BeautifulSoup
import logging
import time
import random
from flask import current_app
from urllib.parse import urlparse, urljoin
import re
import urllib.robotparser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crawl_urls(urls, target_count=1):
    """
    Crawl a list of URLs and extract relevant content.
    
    Args:
        urls (list): List of URLs to crawl
        target_count (int): Number of sources to collect (default: 1)
        
    Returns:
        list: List of dictionaries containing extracted content and metadata
    """
    sources = []
    timeout = current_app.config.get('CRAWL_TIMEOUT', 10)
    
    # List of common browsers user agents
    browser_user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
    ]
    
    # Use a random user agent from the list or fallback to config
    user_agent = random.choice(browser_user_agents)
    
    # Define headers that mimic a real browser
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',  # Do Not Track
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    
    # Create a session to maintain cookies across requests
    session = requests.Session()
    session.headers.update(headers)
    
    # Create a dictionary to store robots.txt parsers
    robots_parsers = {}
    
    # Keep track of crawled URLs
    attempted_urls = set()
    
    # Continue crawling until we have enough sources or run out of URLs
    for url in urls:
        # Check if we've reached the target count
        if len(sources) >= target_count:
            logger.info(f"Reached target of {target_count} sources, stopping crawl")
            break
            
        # Skip if we've already attempted this URL
        if url in attempted_urls:
            continue
            
        attempted_urls.add(url)
        
        try:
            logger.info(f"Crawling URL: {url} ({len(sources)}/{target_count} sources collected)")
            
            # Skip if URL is invalid
            if not url or not url.startswith(('http://', 'https://')):
                logger.warning(f"Invalid URL format: {url}")
                continue
            
            # Parse the URL to get the base domain
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Check robots.txt to see if we're allowed to crawl
            if base_url not in robots_parsers:
                robots_url = urljoin(base_url, "/robots.txt")
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    robots_parsers[base_url] = rp
                except Exception as e:
                    logger.warning(f"Could not read robots.txt for {base_url}: {e}")
                    # If we can't read robots.txt, assume we're allowed
                    robots_parsers[base_url] = None
            
            # Check if we're allowed to crawl
            rp = robots_parsers[base_url]
            if rp and not rp.can_fetch(user_agent, url):
                logger.warning(f"Robots.txt disallows crawling of URL: {url}")
                continue
            
            # Add a delay to respect rate limits
            delay = random.uniform(1.0, 3.0)  # Random delay between 1-3 seconds
            time.sleep(delay)
            
            # Make the request with timeout and retries
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Request attempt {attempt+1} for URL: {url}")
                    
                    # Make the request with the session to maintain cookies
                    response = session.get(url, timeout=timeout, allow_redirects=True)
                    
                    # Check if we got a successful response
                    if response.status_code == 200:
                        break
                    
                    # Handle specific status codes
                    if response.status_code == 403:
                        # Try a different user agent for the next attempt
                        new_user_agent = random.choice(browser_user_agents)
                        session.headers.update({'User-Agent': new_user_agent})
                        logger.info(f"Switching user agent for retry. New agent: {new_user_agent[:20]}...")
                    
                    # Exponential backoff
                    retry_delay *= 2
                    time.sleep(retry_delay)
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed (attempt {attempt+1}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
            
            # Skip if all retries failed or response is not successful
            if not response or response.status_code != 200:
                logger.warning(f"Received status code {response.status_code if response else 'unknown'} for URL: {url}")
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
            
            # We've already added delay with random timing above
            
        except Exception as e:
            logger.error(f"Error crawling URL {url}: {str(e)}")
            continue
            
    # Log crawl statistics
    logger.info(f"Crawl completed: {len(sources)}/{target_count} sources collected")
    logger.info(f"Attempted {len(attempted_urls)} URLs in total")
    
    # If we still don't have enough sources, log a warning
    if len(sources) < target_count:
        logger.warning(f"Could not collect requested {target_count} sources, only found {len(sources)}")
    
    # Return only up to the target count (in case we collected more)
    return sources[:target_count]

def extract_main_content(soup):
    """
    Extract the main textual content from a webpage.
    
    This is an improved implementation with better content identification.
    """
    # Remove unwanted elements that typically contain non-content
    for element in soup.select('nav, footer, header, aside, script, style, iframe, .sidebar, .menu, .navigation, .footer, .header, .ads, .banner'):
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
            if paragraphs and len(paragraphs) > 3:  # Ensure we have enough paragraphs
                main_content = " ".join([p.get_text() for p in paragraphs])
                if len(main_content) > 200:  # Ensure we have enough content
                    return clean_content(main_content)
    
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
        return clean_content(main_content)
    
    # Plan C: Get all paragraphs from the page
    paragraphs = soup.find_all('p')
    if paragraphs:
        main_content = " ".join([p.get_text() for p in paragraphs])
        return clean_content(main_content)
    
    # Last resort: get all text from the body
    body = soup.find('body')
    if body:
        return clean_content(body.get_text())
    
    return ""

def clean_content(text):
    """Clean extracted text by removing extra whitespace and common noise."""
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove common scripts and ads text (expanded list)
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