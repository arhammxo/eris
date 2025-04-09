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
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 3600))  # 1 hour
    
    # Asynchronous settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get('MAX_CONCURRENT_REQUESTS', 10))
    MAX_CONCURRENT_PER_DOMAIN = int(os.environ.get('MAX_CONCURRENT_PER_DOMAIN', 3))
    
    # Crawler settings
    MAX_URLS_TO_CRAWL = int(os.environ.get('MAX_URLS_TO_CRAWL', 5))
    CRAWL_TIMEOUT = int(os.environ.get('CRAWL_TIMEOUT', 15))  # seconds
    RESPECT_ROBOTS_TXT = os.environ.get('RESPECT_ROBOTS_TXT', 'True').lower() == 'true'
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
    
    # Browser crawler settings
    USE_BROWSER_CRAWLER = os.environ.get('USE_BROWSER_CRAWLER', 'True').lower() == 'true'
    BROWSER_INSTANCES = int(os.environ.get('BROWSER_INSTANCES', 2))
    BROWSER_TIMEOUT = int(os.environ.get('BROWSER_TIMEOUT', 30))  # seconds
    
    # Content quality settings
    QUALITY_THRESHOLD = float(os.environ.get('QUALITY_THRESHOLD', 0.4))  # Minimum quality score
    DOMAIN_QUALITY_DB = os.environ.get('DOMAIN_QUALITY_DB', 'data/domain_quality.db')
    
    # Summarization settings
    MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 1000))
    SUMMARY_MODEL = os.environ.get('SUMMARY_MODEL', "gpt-3.5-turbo")
    
    # Text processing
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 4000))  # characters

    # Meta-Chunking settings
    USE_META_CHUNKING = os.environ.get('USE_META_CHUNKING', 'True').lower() == 'true'
    META_CHUNKING_THRESHOLD = float(os.environ.get('META_CHUNKING_THRESHOLD', 0.5))
    USE_DYNAMIC_COMBINATION = os.environ.get('USE_DYNAMIC_COMBINATION', 'True').lower() == 'true'
    USE_OPENAI_FOR_PPL = os.environ.get('USE_OPENAI_FOR_PPL', 'False').lower() == 'true'
    
    # User-Agent for crawler
    USER_AGENT = os.environ.get('USER_AGENT', "Mozilla/5.0 WebSummarizerBot/1.0")
    
    # Progress tracking
    ENABLE_PROGRESS_TRACKING = os.environ.get('ENABLE_PROGRESS_TRACKING', 'True').lower() == 'true'

    # File search settings
    FILE_SEARCH_ENABLED = os.environ.get('FILE_SEARCH_ENABLED', 'True').lower() == 'true'
    BASE_DIRECTORIES = os.environ.get('BASE_DIRECTORIES', '')
    if not BASE_DIRECTORIES:
        # Add default directories if none specified
        if os.name == 'nt':  # Windows
            BASE_DIRECTORIES = os.path.join(os.path.expanduser('~'), 'Documents')
        else:  # Unix-like
            BASE_DIRECTORIES = os.path.join(os.path.expanduser('~'), 'Documents')
            
    BASE_DIRECTORIES = BASE_DIRECTORIES.split(',')
    MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', 20))
    SUPPORTED_FILE_EXTENSIONS = os.environ.get('SUPPORTED_FILE_EXTENSIONS', 
                                             '.txt,.pdf,.docx,.doc,.md,.csv,.json,.xml,.html,.htm,.rtf').split(',')
    MAX_CONCURRENT_EXTRACTIONS = int(os.environ.get('MAX_CONCURRENT_EXTRACTIONS', 5))