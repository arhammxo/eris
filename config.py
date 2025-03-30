import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_for_development_only')
    
    # API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY')
    
    # Cache settings
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 3600  # 1 hour
    
    # Crawler settings
    MAX_URLS_TO_CRAWL = 5
    CRAWL_TIMEOUT = 10  # seconds
    
    # Summarization settings
    MAX_TOKENS = 1000
    SUMMARY_MODEL = "gpt-4o"
    
    # Text processing
    CHUNK_SIZE = 4000  # characters
    
    # User-Agent for crawler
    USER_AGENT = "Mozilla/5.0 WebSummarizerBot/1.0"