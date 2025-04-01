# modules/file_extractors/__init__.py
"""
File content extraction modules.

This package provides extractors for different file types,
each implementing a common interface.
"""
import os
import importlib
from typing import Dict, Optional, Type

from modules.file_extractors.base import BaseFileExtractor
from modules.file_extractors.text import TextFileExtractor
from modules.utils.logging import get_logger

logger = get_logger(__name__)

# Registry of extractors
_EXTRACTORS: Dict[str, Type[BaseFileExtractor]] = {}

def register_extractor(extensions: list, extractor_class: Type[BaseFileExtractor]):
    """
    Register a file extractor for specific extensions.
    
    Args:
        extensions: List of file extensions (with dot)
        extractor_class: Extractor class
    """
    for ext in extensions:
        _EXTRACTORS[ext.lower()] = extractor_class

# Register built-in extractors
register_extractor(['.txt', '.md', '.csv', '.json', '.xml', '.log'], TextFileExtractor)

def get_extractor_for_file(file_path: str) -> Optional[BaseFileExtractor]:
    """
    Get the appropriate extractor for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extractor instance or None if no suitable extractor found
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    extractor_class = _EXTRACTORS.get(ext)
    if extractor_class:
        return extractor_class()
    return None

# Try to load optional extractors if dependencies are available
try:
    from modules.file_extractors.pdf import PDFFileExtractor
    register_extractor(['.pdf'], PDFFileExtractor)
except ImportError:
    logger.info("PDF extractor not available. Install PyPDF2 for PDF support.")

try:
    from modules.file_extractors.docx import DocxFileExtractor
    register_extractor(['.docx'], DocxFileExtractor)
except ImportError:
    logger.info("DOCX extractor not available. Install python-docx for DOCX support.")