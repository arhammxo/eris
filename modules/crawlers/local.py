# modules/crawlers/file.py
"""
File-based crawler for local content extraction.

This module provides capabilities for crawling and extracting content
from local files in various formats, following the same patterns as
web crawling.
"""
import os
import time
import asyncio
import logging
import mimetypes
from typing import Dict, List, Optional, Set, Any
from pathlib import Path

from modules.utils.types import URL, Source
from modules.utils.errors import CrawlerError, FileExtractionError, handle_async_exceptions
from modules.utils.logging import get_logger
from modules.file_extractors import get_extractor_for_file

# Create a logger for this module
logger = get_logger(__name__)

class FileCrawler:
    """
    File-based crawler for local content extraction.
    
    This crawler traverses directories and extracts content from files
    using the appropriate extractor for each file type.
    """
    
    def __init__(
        self,
        base_directories: List[str] = None,
        max_concurrent_extractions: int = 5,
        max_file_size_mb: int = 20,
        supported_extensions: List[str] = None
    ):
        """
        Initialize the file crawler.
        
        Args:
            base_directories: List of base directories to search
            max_concurrent_extractions: Maximum concurrent file extractions
            max_file_size_mb: Maximum file size to process in MB
            supported_extensions: List of file extensions to process
        """
        self.base_directories = base_directories or []
        self.max_concurrent_extractions = max_concurrent_extractions
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.supported_extensions = supported_extensions or [
            '.txt', '.pdf', '.docx', '.doc', '.md', '.csv', '.json', '.xml', 
            '.html', '.htm', '.rtf'
        ]
        self.extraction_semaphore = asyncio.Semaphore(max_concurrent_extractions)
        
    async def crawl_files(self, query: str, file_paths: Optional[List[str]] = None, 
                         target_count: int = 5) -> List[Source]:
        """
        Crawl files matching the query and extract content.
        
        Args:
            query: Search query to match files against
            file_paths: Specific file paths to process, or None to search base directories
            target_count: Number of matching files to collect
            
        Returns:
            List of sources with extracted content
        """
        sources: List[Source] = []
        files_to_process: List[str] = []
        
        # If specific files are provided, use those
        if file_paths:
            files_to_process = file_paths
        else:
            # Otherwise search the base directories
            files_to_process = await self._find_matching_files(query)
            
        # Limit the number of files to process
        files_to_process = files_to_process[:min(target_count * 2, len(files_to_process))]
        
        # Process files concurrently
        tasks = [self.extract_file_content(file_path) for file_path in files_to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        for result in results:
            if result is not None and not isinstance(result, Exception):
                sources.append(result)
                if len(sources) >= target_count:
                    break
                    
        return sources[:target_count]
    
    async def _find_matching_files(self, query: str) -> List[str]:
        """
        Find files in base directories that might match the query.
        
        Performs a simple string matching search in filenames and directories.
        
        Args:
            query: Search query
            
        Returns:
            List of matching file paths
        """
        matching_files: List[str] = []
        query_terms = query.lower().split()
        
        # Search each base directory
        for base_dir in self.base_directories:
            if not os.path.exists(base_dir):
                logger.warning(f"Base directory does not exist: {base_dir}")
                continue
                
            for root, _, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip files that are too large
                    if os.path.getsize(file_path) > self.max_file_size_bytes:
                        continue
                        
                    # Skip unsupported extensions
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext not in self.supported_extensions:
                        continue
                        
                    # Simple matching based on filename and directory
                    file_lower = file.lower()
                    path_lower = os.path.relpath(root, base_dir).lower()
                    
                    # Consider a match if any term appears in filename or path
                    if any(term in file_lower or term in path_lower for term in query_terms):
                        matching_files.append(file_path)
        
        return matching_files
    
    @handle_async_exceptions(FileExtractionError, "Failed to extract file content")
    async def extract_file_content(self, file_path: str) -> Optional[Source]:
        """
        Extract content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Source with extracted content or None if extraction failed
        """
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return None
            
        # Get file info
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size_bytes:
            logger.warning(f"File exceeds size limit: {file_path}")
            return None
            
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_extensions:
            logger.warning(f"Unsupported file extension: {file_ext}")
            return None
            
        async with self.extraction_semaphore:
            try:
                # Convert file path to URL-like format for consistency with web sources
                file_url = f"file://{os.path.abspath(file_path)}"
                
                # Get appropriate extractor
                extractor = get_extractor_for_file(file_path)
                if not extractor:
                    logger.warning(f"No extractor available for: {file_path}")
                    return None
                
                # Extract content
                content = await extractor.extract(file_path)
                if not content:
                    logger.warning(f"No content extracted from: {file_path}")
                    return None
                
                # Create source
                return {
                    'url': file_url,
                    'title': os.path.basename(file_path),
                    'domain': 'local',
                    'content': content,
                    'crawled_at': time.time(),
                    'original_length': len(content),
                    'file_path': file_path,
                    'file_type': file_ext[1:].upper()  # Remove dot and uppercase
                }
                
            except Exception as e:
                logger.error(f"Error extracting content from {file_path}: {str(e)}")
                raise FileExtractionError(f"Error extracting content: {str(e)}")
