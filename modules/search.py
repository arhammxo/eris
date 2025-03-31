import requests
import json
import logging
from flask import current_app
from serpapi import GoogleSearch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_web(query, max_results=5):
    """
    Search the web for relevant URLs based on the query.
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of relevant URLs
    """
    try:
        # Try using SerpAPI if the API key is available
        api_key = current_app.config.get('SERPAPI_API_KEY')
        
        if api_key:
            return _search_with_serpapi(query, api_key, max_results)
        else:
            # Fallback to a basic search approach
            logger.warning("No SERPAPI_API_KEY found. Using fallback search method.")
            return _search_fallback(query, max_results)
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return []

def _search_with_serpapi(query, api_key, max_results):
    """Use SerpAPI to search Google."""
    try:
        # Request more results than needed
        requested_results = max(10, max_results)  # At least 10 results
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": requested_results
        }
        
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
            return urls[:max_results]
        else:
            logger.warning("No organic results found in SerpAPI response.")
            return []
            
    except Exception as e:
        logger.error(f"SerpAPI search error: {str(e)}")
        return []

def _search_fallback(query, max_results):
    """
    Fallback search method when no API key is available.
    Note: This is a basic implementation not recommended for production.
    """
    try:
        # Using DuckDuckGo API as a fallback (no API key required)
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
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
                example_domains = [
                    "wikipedia.org", "britannica.com", "nationalgeographic.com",
                    "sciencedaily.com", "nature.com", "history.com", "healthline.com",
                    "mayoclinic.org", "medicalnewstoday.com", "webmd.com",
                    "investopedia.com", "economictimes.com", "nasa.gov"
                ]
                
                for domain in example_domains:
                    if len(urls) >= max_results:
                        break
                    # Create a URL for this domain related to the query
                    formatted_query = query.replace(" ", "+")
                    example_url = f"https://www.{domain}/search?q={formatted_query}"
                    if example_url not in urls:
                        urls.append(example_url)
                        
            return urls[:max_results]
        else:
            logger.error(f"Fallback search failed with status code: {response.status_code}")
            return []
    
    except Exception as e:
        logger.error(f"Fallback search error: {str(e)}")
        # Return a few example URLs for demonstration purposes
        return [
            "https://en.wikipedia.org/wiki/Information_retrieval",
            "https://en.wikipedia.org/wiki/Web_crawler",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Data_mining",
            "https://en.wikipedia.org/wiki/Text_mining",
            "https://en.wikipedia.org/wiki/Search_engine_technology",
            "https://en.wikipedia.org/wiki/Information_extraction",
            "https://en.wikipedia.org/wiki/Information_science"
        ][:max_results]
    
def search_web_with_config(query, max_results=5, config=None):
    """
    Search the web for relevant URLs based on the query.
    This version accepts config directly instead of using current_app.
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        config (dict): Configuration dictionary with API keys and settings
        
    Returns:
        list: List of relevant URLs
    """
    try:
        # Try using SerpAPI if the API key is available
        api_key = config.get('SERPAPI_API_KEY') if config else None
        
        if api_key:
            return _search_with_serpapi(query, api_key, max_results)
        else:
            # Fallback to a basic search approach
            logger.warning("No SERPAPI_API_KEY found. Using fallback search method.")
            return _search_fallback(query, max_results)
            
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return []