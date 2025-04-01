# modules/file_extractors/base.py
"""
Base class for file content extractors.
"""
import os
from abc import ABC, abstractmethod
from typing import Optional

class BaseFileExtractor(ABC):
    """Base class for file content extractors."""
    
    @abstractmethod
    async def extract(self, file_path: str) -> Optional[str]:
        """
        Extract content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted content or None if extraction failed
        """
        pass