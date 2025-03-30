# Package initialization file
from flask import Flask

# Import module functionality
from .search import search_web
from .crawler import crawl_urls
from .processor import process_text
from .summarizer import generate_summary
from .cache import ResultCache

# Version information
__version__ = '0.1.0'