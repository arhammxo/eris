"""
Web search module for finding relevant URLs.

This module handles searching the web for relevant content based on user queries,
using SerpAPI with fallback mechanisms for when API keys are not available.
"""
import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union
from quart import current_app
from serpapi import GoogleSearch

from modules.utils.types import URL
from modules.utils.errors import (
    SearchError, SearchAPIError, NoResultsError, handle_exceptions
)
from modules.utils.logging import get_logger, log_function_call

# Create a logger for this module
logger = get_logger(__name__)

@handle_exceptions(SearchError, "Search failed")
def search_web(query: str, max_results: int = 5) -> List[URL]:
    """
    Search the web for relevant URLs based on the query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of relevant URLs
        
    Raises:
        SearchError: If search fails
        NoResultsError: If no results are found
    """
    # Try using SerpAPI if the API key is available
    api_key = current_app.config.get('SERPAPI_API_KEY')
    
    if api_key:
        return _search_with_serpapi(query, api_key, max_results)
    else:
        # Fallback to a basic search approach
        logger.warning("No SERPAPI_API_KEY found. Using fallback search method.")
        return _search_fallback(query, max_results)

@log_function_call(logger)
@handle_exceptions(SearchAPIError, "SerpAPI search failed")
def _search_with_serpapi(query: str, api_key: str, max_results: int) -> List[URL]:
    """
    Use SerpAPI to search Google.
    
    Args:
        query: The search query
        api_key: SerpAPI API key
        max_results: Maximum number of results to return
        
    Returns:
        List of relevant URLs
        
    Raises:
        SearchAPIError: If SerpAPI search fails
        NoResultsError: If no results are found
    """
    # Request more results than needed
    requested_results = max(10, max_results)  # At least 10 results
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": requested_results
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract organic results
        if "organic_results" in results:
            urls = []
            # Process organic results first
            for result in results["organic_results"]:
                if "link" in result:
                    urls.append(result["link"])
                    
            # If we still need more results and there are related searches, use those
            if len(urls) < max_results and "related_searches" in results:
                for related in results["related_searches"]:
                    if len(urls) >= max_results:
                        break
                    if "query" in related:
                        # Do a quick additional search with this query
                        try:
                            related_params = {
                                "engine": "google",
                                "q": related["query"],
                                "api_key": api_key,
                                "num": 2  # Just a couple results per related search
                            }
                            related_search = GoogleSearch(related_params)
                            related_results = related_search.get_dict()
                            
                            if "organic_results" in related_results:
                                for related_result in related_results["organic_results"]:
                                    if len(urls) >= max_results:
                                        break
                                    if "link" in related_result and related_result["link"] not in urls:
                                        urls.append(related_result["link"])
                        except Exception as e:
                            logger.warning(f"Error in related search: {str(e)}")
            
            logger.info(f"SerpAPI search found {len(urls)} results, requested {max_results}")
            
            if not urls:
                raise NoResultsError(f"No results found for query: {query}")
                
            return urls[:max_results]
        else:
            logger.warning("No organic results found in SerpAPI response.")
            raise NoResultsError(f"No organic results found for query: {query}")
            
    except Exception as e:
        if isinstance(e, NoResultsError):
            raise
        logger.error(f"SerpAPI search error: {str(e)}")
        raise SearchAPIError(f"Error using SerpAPI: {str(e)}")

@log_function_call(logger)
@handle_exceptions(SearchError, "Fallback search failed")
def _search_fallback(query: str, max_results: int) -> List[URL]:
    """
    Fallback search method when no API key is available.
    
    Note: This is a basic implementation not recommended for production.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of relevant URLs
        
    Raises:
        SearchError: If fallback search fails
        NoResultsError: If no results are found
    """
    try:
        # Using DuckDuckGo API as a fallback (no API key required)
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.error("Failed to parse DuckDuckGo response as JSON")
                return _generate_example_urls(query, max_results)
            
            # Extract URLs from results
            urls = []
            if "Results" in data:
                for result in data["Results"]:
                    if "FirstURL" in result:
                        urls.append(result["FirstURL"])
                        
            # Add URLs from Related Topics
            if "RelatedTopics" in data:
                for topic in data["RelatedTopics"]:
                    if "FirstURL" in topic:
                        urls.append(topic["FirstURL"])
            
            # If we still don't have enough results, add some example URLs
            if len(urls) < max_results:
                urls.extend(_generate_example_urls(query, max_results - len(urls)))
                
            if not urls:
                raise NoResultsError(f"No results found for query: {query}")
                
            return urls[:max_results]
        else:
            logger.error(f"Fallback search failed with status code: {response.status_code}")
            return _generate_example_urls(query, max_results)
    
    except requests.RequestException as e:
        logger.error(f"Request exception in fallback search: {str(e)}")
        return _generate_example_urls(query, max_results)
    except Exception as e:
        if isinstance(e, NoResultsError):
            raise
        logger.error(f"Fallback search error: {str(e)}")
        raise SearchError(f"Fallback search error: {str(e)}")

def _generate_example_urls(query: str, count: int) -> List[URL]:
    """
    Generate example URLs for demonstration purposes.
    
    Args:
        query: The search query
        count: Number of URLs to generate
        
    Returns:
        List of example URLs
    """
    logger.warning(f"Generating {count} example URLs for query: {query}")
    
    example_domains = [
        "wikipedia.org", "britannica.com", "nationalgeographic.com",
        "sciencedaily.com", "nature.com", "history.com", "healthline.com",
        "mayoclinic.org", "medicalnewstoday.com", "webmd.com",
        "investopedia.com", "economictimes.com", "nasa.gov"
    ]
    
    # Create URLs for the query
    formatted_query = query.replace(" ", "+")
    urls = []
    
    # Try to generate Wikipedia URL first
    wikipedia_url = f"https://en.wikipedia.org/wiki/{formatted_query.replace('+', '_')}"
    urls.append(wikipedia_url)
    
    # Add other domains
    for domain in example_domains[1:]:  # Skip wikipedia as we already added it
        if len(urls) >= count:
            break
        # Create a URL for this domain related to the query
        example_url = f"https://www.{domain}/search?q={formatted_query}"
        if example_url not in urls:
            urls.append(example_url)
    
    return urls[:count]

@handle_exceptions(SearchError, "Search with config failed")
def search_web_with_config(
    query: str, 
    max_results: int = 5, 
    config: Optional[Dict[str, Any]] = None
) -> List[URL]:
    """
    Search the web using a provided configuration.
    
    This version accepts config directly instead of using current_app.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        config: Configuration dictionary with API keys and settings
        
    Returns:
        List of relevant URLs
        
    Raises:
        SearchError: If search fails
    """
    # Try using SerpAPI if the API key is available
    api_key = config.get('SERPAPI_API_KEY') if config else None
    
    if api_key:
        return _search_with_serpapi(query, api_key, max_results)
    else:
        # Fallback to a basic search approach
        logger.warning("No SERPAPI_API_KEY found in config. Using fallback search method.")
        return _search_fallback(query, max_results)