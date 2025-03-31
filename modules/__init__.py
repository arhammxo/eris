"""
Web Summarizer application modules.

This package provides the core functionality for the Web Summarizer application:

- Web search (search.py)
- Content extraction (crawlers/)
- Text processing (processor.py)
- Content quality scoring (content_quality.py)
- AI-powered summarization (summarizer.py)
- Caching (cache.py)
- Utilities (utils/)
"""

# Import key components for easy access
from modules.search import search_web
from modules.cache import ResultCache
from modules.content_quality import ContentQualityScorer
from modules.processor import process_text
from modules.summarizer import generate_summary
from modules.crawlers import crawl_urls, cleanup_crawlers

# Version information
__version__ = '0.3.0'