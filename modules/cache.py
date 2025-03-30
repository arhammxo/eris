import time
import logging
from flask import current_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultCache:
    """Simple wrapper for Flask-Caching for summary results."""
    
    @staticmethod
    def get_cache():
        """Get the Flask cache instance."""
        return current_app.extensions['cache']
    
    @staticmethod
    def set(key, data, timeout=None):
        """
        Save data to the cache.
        
        Args:
            key (str): Cache key
            data (dict): Data to cache
            timeout (int, optional): Cache timeout in seconds
        """
        try:
            if timeout is None:
                timeout = current_app.config.get('CACHE_DEFAULT_TIMEOUT', 3600)
                
            cache = ResultCache.get_cache()
            cache.set(key, data, timeout=timeout)
            logger.info(f"Saved data to cache with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
            return False
    
    @staticmethod
    def get(key):
        """
        Get data from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            dict: Cached data or None if not found
        """
        try:
            cache = ResultCache.get_cache()
            data = cache.get(key)
            if data:
                logger.info(f"Retrieved data from cache with key: {key}")
            return data
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
    
    @staticmethod
    def delete(key):
        """
        Delete data from the cache.
        
        Args:
            key (str): Cache key
        """
        try:
            cache = ResultCache.get_cache()
            cache.delete(key)
            logger.info(f"Deleted data from cache with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False
    
    @staticmethod
    def search_by_query(query):
        """
        Search the cache for entries matching a query.
        
        This is a simple implementation that requires iterating through all keys,
        which isn't efficient for large caches. A production version would use
        a more sophisticated indexing mechanism.
        
        Args:
            query (str): Search query
            
        Returns:
            list: List of matching cache keys
        """
        matching_keys = []
        cache = ResultCache.get_cache()
        
        # This only works with SimpleCache which exposes the _cache dictionary
        if hasattr(cache, '_cache'):
            for key, value in cache._cache.items():
                # Check if the value contains our query
                if isinstance(value, tuple) and len(value) > 1:
                    # The value is (data, expiry_time)
                    data = value[0]
                    if isinstance(data, dict) and data.get('query') == query:
                        matching_keys.append(key)
        
        return matching_keys