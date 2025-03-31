# Getting Started with Enhanced Web Summarizer

This guide will help you implement and deploy the Phase 1 improvements to the Web Summarizer application. These improvements include asynchronous processing, content quality scoring, headless browser support, and prompt optimization.

## Implementation Guide

### Step 1: Environment Setup

1. **Update Your Python Environment**
   Make sure you have Python 3.8+ and set up a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright for Browser Support**
   ```bash
   playwright install chromium
   ```

4. **Prepare Data Directory**
   ```bash
   mkdir -p data
   ```

### Step 2: Configuration

1. **Update Environment Variables**
   Create or update your `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPAPI_API_KEY=your_serpapi_api_key
   SECRET_KEY=your_secure_random_key
   
   # Asynchronous settings
   MAX_CONCURRENT_REQUESTS=10
   MAX_CONCURRENT_PER_DOMAIN=3
   
   # Browser crawler settings
   USE_BROWSER_CRAWLER=True
   BROWSER_INSTANCES=2
   BROWSER_TIMEOUT=30
   
   # Content quality settings
   QUALITY_THRESHOLD=0.4
   
   # Cache settings
   CACHE_TYPE=SimpleCache
   CACHE_DEFAULT_TIMEOUT=3600
   
   # Progress tracking
   ENABLE_PROGRESS_TRACKING=True
   ```

2. **Adjust Configuration Values for Your Environment**
   - For lower-resource environments, reduce `MAX_CONCURRENT_REQUESTS` to 5
   - For faster performance, increase `MAX_CONCURRENT_REQUESTS` if your system has more resources
   - Set `USE_BROWSER_CRAWLER=False` if you're experiencing memory issues

### Step 3: Code Integration

1. **Replace Original Files with Enhanced Versions**
   - Replace `app.py` with `enhanced_app.py` (or rename)
   - Update `config.py` with the enhanced version
   - Add all new modules to the `modules/` directory

2. **Add the New Template**
   - Add `progress.html` to your templates directory

3. **Create Required SQLite Database**
   The application will create the domain quality database automatically on first run.

### Step 4: Testing

1. **Run the Application**
   ```bash
   python enhanced_app.py
   ```

2. **Test Basic Functionality**
   - Verify search form works
   - Test progress tracking
   - Ensure results display correctly

3. **Test Advanced Features**
   - Try a search query for a JavaScript-heavy site to test browser integration
   - Check content quality scores in the search results metadata
   - Test different query types to verify prompt optimization

## Feature-Specific Implementation Details

### Asynchronous Processing

The application has been converted from Flask to Quart to enable true asynchronous processing. Key changes include:

- Routes use `async/await` pattern
- Background tasks run concurrently
- WebSockets provide real-time updates
- Progress tracking for improved UX

If you prefer using Flask, you can use the `flask-async-views` extension as an alternative:
```bash
pip install flask-async-views
```

### Content Quality Scoring

The quality scoring system evaluates sources based on multiple factors:

- Content length (with penalties for both too short and too long)
- Readability (using Flesch reading ease and grade level)
- Citations and references (detecting academic citation patterns)
- Text coherence and structure
- Domain reputation from seed database

You can add custom domain quality scores to the SQLite database:
```sql
INSERT INTO domain_quality (domain, quality_score, last_updated)
VALUES ('example.com', 0.85, CURRENT_TIMESTAMP);
```

### Headless Browser Integration

Browser-based crawling is enabled for JavaScript-heavy sites:

- Automatically detects JS-heavy sites
- Uses Playwright to render pages with JavaScript
- Manages browser resources efficiently
- Falls back to regular HTTP requests when appropriate

The system includes a list of known JS-required domains that you can extend in `browser_crawler.py`.

### Prompt Optimization

The summarizer now includes enhanced prompt engineering:

- Detects query type (factual, comparison, instructional, opinion)
- Customizes prompts based on query type
- Adjusts system messages for different tasks
- Prioritizes higher quality sources in summaries
- Improves source attribution

You can add additional query types and patterns in `QUERY_TYPES` in `async_summarizer.py`.

## Deployment Considerations

For production deployment, consider the following:

1. **Use a Production Cache Backend**
   ```python
   # In config.py
   CACHE_TYPE = 'RedisCache'
   CACHE_REDIS_URL = 'redis://localhost:6379/0'
   ```
   You'll need to install the redis package:
   ```bash
   pip install redis
   ```

2. **Use a Production ASGI Server**
   ```bash
   pip install hypercorn
   hypercorn enhanced_app:app --bind 0.0.0.0:8000
   ```

3. **Implement Rate Limiting**
   Add rate limiting middleware to prevent abuse.

4. **Set up Monitoring**
   Configure logging and monitoring for performance tracking.

## Troubleshooting

### Browser Crawling Issues
- If you see "No usable sandbox" errors, add `--no-sandbox` to the browser launch options
- Memory issues: Reduce `BROWSER_INSTANCES` to 1
- Timeout errors: Increase `BROWSER_TIMEOUT` value

### Asynchronous Processing Issues
- Event loop errors: Ensure you're not mixing sync and async code
- Concurrency issues: Reduce `MAX_CONCURRENT_REQUESTS`

### Content Quality Scoring
- If quality scores seem incorrect, check `content_quality.py` scoring functions
- Domain reputation issues: Inspect the domain_quality.db file

### Prompt Optimization
- If summaries are poor for specific query types, adjust the system prompts
- Token limit issues: Adjust `max_tokens` values

## Next Steps

After successfully implementing Phase 1, consider exploring:
- Adding user authentication
- Implementing Redis-based caching
- Setting up a task queue with Celery
- Creating containerized deployment with Docker