# modules/file_extractors/text.py
"""
Text file content extractor.
"""
import os
import asyncio
from typing import Optional

from modules.file_extractors.base import BaseFileExtractor
from modules.utils.logging import get_logger

logger = get_logger(__name__)

class TextFileExtractor(BaseFileExtractor):
    """
    Extractor for plain text files.
    
    Handles basic text files like .txt, .md, .csv, etc.
    """
    
    async def extract(self, file_path: str) -> Optional[str]:
        """
        Extract content from a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string or None if extraction failed
        """
        try:
            # Use asyncio to read the file without blocking
            content = await asyncio.to_thread(self._read_file, file_path)
            return content
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return None
    
    def _read_file(self, file_path: str) -> str:
        """
        Read a text file with encoding detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        # Try UTF-8 first, then fall back to other encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            
        # If all encodings fail, try binary mode and decode as best as possible
        with open(file_path, 'rb') as f:
            binary_data = f.read()
            return binary_data.decode('utf-8', errors='replace')