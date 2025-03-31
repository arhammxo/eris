"""
Cache module for storing and retrieving summary results.

This module provides a wrapper around Flask/Quart caching to handle
summary results and progress tracking data.
"""
import time
from typing import Any, Dict, List, Optional, Union
from quart import current_app

from modules.utils.errors import CacheError, CacheAccessError, handle_exceptions
from modules.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)

class ResultCache:
    """
    Simple wrapper for Flask/Quart caching for summary results.
    
    This class provides a consistent interface for caching operations
    and adds error handling and logging.
    """
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to get cache instance")
    def get_cache():
        """
        Get the Flask/Quart cache instance.
        
        Returns:
            Cache instance
            
        Raises:
            CacheError: If cache cannot be accessed
        """
        try:
            return current_app.extensions['cache']
        except (KeyError, AttributeError) as e:
            raise CacheAccessError("Cache extension not found in application") from e
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to set cache value")
    def set(key: str, data: Any, timeout: Optional[int] = None) -> bool:
        """
        Save data to the cache.
        
        Args:
            key: Cache key
            data: Data to cache
            timeout: Cache timeout in seconds
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            CacheError: If operation fails
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
            raise
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to get cache value")
    def get(key: str) -> Optional[Any]:
        """
        Get data from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
            
        Raises:
            CacheError: If operation fails
        """
        try:
            cache = ResultCache.get_cache()
            data = cache.get(key)
            if data:
                logger.info(f"Retrieved data from cache with key: {key}")
            return data
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            raise
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to delete cache value")
    def delete(key: str) -> bool:
        """
        Delete data from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            CacheError: If operation fails
        """
        try:
            cache = ResultCache.get_cache()
            cache.delete(key)
            logger.info(f"Deleted data from cache with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            raise
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to search cache")
    def search_by_query(query: str) -> List[str]:
        """
        Search the cache for entries matching a query.
        
        This is a simple implementation that requires iterating through all keys,
        which isn't efficient for large caches. A production version would use
        a more sophisticated indexing mechanism.
        
        Args:
            query: Search query
            
        Returns:
            List of matching cache keys
            
        Raises:
            CacheError: If operation fails
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
            
            logger.info(f"Found {len(matching_keys)} cache entries matching query: {query}")
        else:
            logger.warning("Cache implementation does not support searching")
            
        return matching_keys
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to clear cache")
    def clear() -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            CacheError: If operation fails
        """
        try:
            cache = ResultCache.get_cache()
            cache.clear()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to get progress")
    def get_progress(search_id: str) -> Optional[Dict[str, Any]]:
        """
        Get progress data for a search.
        
        Args:
            search_id: Search ID
            
        Returns:
            Progress data or None if not found
            
        Raises:
            CacheError: If operation fails
        """
        return ResultCache.get(f"progress_{search_id}")
    
    @staticmethod
    @handle_exceptions(CacheError, "Failed to update progress")
    def update_progress(
        search_id: str, 
        status: str, 
        progress: int = 0, 
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update progress data for a search.
        
        Args:
            search_id: Search ID
            status: Status string ('starting', 'searching', etc.)
            progress: Progress percentage (0-100)
            data: Additional data to include
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            CacheError: If operation fails
        """
        progress_data = {
            'status': status,
            'progress': progress,
            'updated_at': time.time(),
            'data': data or {}
        }
        
        return ResultCache.set(f"progress_{search_id}", progress_data, timeout=3600)  # 1 hour timeout