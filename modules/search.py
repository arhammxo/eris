import requests
import json
import logging
from flask import current_app
from serpapi.google_search import GoogleSearch

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
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max_results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract organic results
        if "organic_results" in results:
            urls = [result["link"] for result in results["organic_results"][:max_results]]
            return urls
        else:
            logger.warning("No organic results found in SerpAPI response.")
            return []
            
    except Exception as e:
        logger.error(f"SerpAPI search error: {str(e)}")
        return []

def _search_fallback(query, max_results):
    """
    Fallback search method when no API key is available.
    Note: This is a very basic implementation and not recommended for production.
    """
    # This is a demonstration fallback - in reality you'd want to implement
    # a more robust solution or use a different API
    try:
        # Using DuckDuckGo API as a fallback (no API key required)
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract URLs from results
            urls = []
            if "Results" in data:
                for result in data["Results"][:max_results]:
                    if "FirstURL" in result:
                        urls.append(result["FirstURL"])
                        
            # If we didn't get enough results, add some from Related Topics
            if len(urls) < max_results and "RelatedTopics" in data:
                for topic in data["RelatedTopics"]:
                    if len(urls) >= max_results:
                        break
                    if "FirstURL" in topic:
                        urls.append(topic["FirstURL"])
                        
            return urls
        else:
            logger.error(f"Fallback search failed with status code: {response.status_code}")
            return []
    
    except Exception as e:
        logger.error(f"Fallback search error: {str(e)}")
        # Return a few example URLs for demonstration purposes
        return [
            "https://en.wikipedia.org/wiki/Information_retrieval",
            "https://en.wikipedia.org/wiki/Web_crawler",
            "https://en.wikipedia.org/wiki/Natural_language_processing"
        ]