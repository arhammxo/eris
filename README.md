# Enhanced Web Summarizer App

A Flask/Quart-based application that demonstrates web crawling, information retrieval, and AI-powered summarization with advanced features including asynchronous processing, content quality scoring, headless browser support, and optimized prompt engineering.

## New Features in Phase 1

### 1. Asynchronous Processing
- Converted to Quart framework for full async/await support
- Implemented concurrent web crawling with per-domain rate limiting
- Added background task processing with progress tracking
- Added WebSocket support for real-time progress updates

### 2. Content Quality Scoring
- Implemented multi-factor quality assessment system
- Content scoring based on length, readability, citations, and coherence
- Domain credibility database with seed data for reliable sources
- Quality-weighted summarization prioritizing higher quality sources

### 3. Headless Browser Integration
- Added Playwright support for JavaScript-rendered websites
- Smart detection of JavaScript-heavy sites
- Browser instance pooling and resource management
- Integration with regular crawler for fallback capabilities

### 4. Prompt Optimization
- Added query type detection for context-specific prompt engineering
- Enhanced system prompts with task-specific guidelines
- Improved source attribution and citation formatting
- Quality-weighted source prioritization in combined summaries

## Prerequisites

- Python 3.8+
- API keys:
  - OpenAI API key
  - SerpAPI key (optional, fallback search is available)
- Playwright (for headless browser support)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/web-summarizer.git
   cd web-summarizer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Playwright browsers:
   ```bash
   playwright install chromium
   ```

5. Download NLTK data (first time only):
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

6. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPAPI_API_KEY=your_serpapi_api_key
   SECRET_KEY=your_flask_secret_key
   USE_BROWSER_CRAWLER=True
   ```

## Running the Enhanced Application

1. Start the Quart application:
   ```bash
   python enhanced_app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Project Structure

- `enhanced_app.py` - Main Quart application with async routes
- `config.py` - Enhanced configuration settings
- `modules/` - Application functionality modules
  - `search.py` - Web search functionality
  - `async_crawler.py` - Asynchronous content extraction
  - `browser_crawler.py` - Headless browser integration
  - `integrated_crawler.py` - Combined crawler orchestration
  - `async_processor.py` - Asynchronous text processing
  - `content_quality.py` - Quality scoring system
  - `async_summarizer.py` - Enhanced LLM summarization
  - `cache.py` - Result caching
- `templates/` - HTML templates
  - `progress.html` - New real-time progress tracking UI
- `static/` - CSS, JavaScript, and other static assets
- `data/` - Application data storage
  - `domain_quality.db` - SQLite database for domain quality scores

## Configuration Options

The application supports numerous configuration options available in `config.py` or via environment variables:

- **Async Settings:**
  - `MAX_CONCURRENT_REQUESTS`: Maximum concurrent requests overall
  - `MAX_CONCURRENT_PER_DOMAIN`: Maximum concurrent requests per domain

- **Browser Crawler:**
  - `USE_BROWSER_CRAWLER`: Enable/disable headless browser (True/False)
  - `BROWSER_INSTANCES`: Maximum number of concurrent browser instances
  - `BROWSER_TIMEOUT`: Timeout for browser-based crawling in seconds

- **Content Quality:**
  - `QUALITY_THRESHOLD`: Minimum quality score for sources (0.0-1.0)
  - `DOMAIN_QUALITY_DB`: Path to domain quality database

## API Usage

The enhanced application provides a more robust API with asynchronous processing:

```
POST /api/search
Content-Type: application/json

{
  "query": "artificial intelligence",
  "depth": 3
}
```

Response:
```json
{
  "search_id": "unique-search-id",
  "status": "processing",
  "message": "Search is being processed",
  "poll_url": "/api/status/unique-search-id"
}
```

Check status:
```
GET /api/status/unique-search-id
```

## Performance Considerations

- The application performs best on systems with at least 4GB of RAM
- Browser-based crawling requires additional memory and CPU resources
- Consider increasing `MAX_CONCURRENT_REQUESTS` on more powerful systems
- For production deployment, use a real cache backend (Redis recommended)

## License

This project is licensed under the MIT License - see the LICENSE file for details