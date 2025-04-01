# modules/file_extractors/pdf.py
"""
PDF file content extractor.
"""
import os
import asyncio
from typing import Optional

from modules.file_extractors.base import BaseFileExtractor
from modules.utils.logging import get_logger

logger = get_logger(__name__)

class PDFFileExtractor(BaseFileExtractor):
    """
    Extractor for PDF files.
    """
    
    async def extract(self, file_path: str) -> Optional[str]:
        """
        Extract content from a PDF file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string or None if extraction failed
        """
        try:
            import PyPDF2
            
            # Use asyncio to read the file without blocking
            content = await asyncio.to_thread(self._extract_pdf, file_path)
            return content
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return None
    
    def _extract_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
            return text