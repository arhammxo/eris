# Web Summarizer

A modern asynchronous web application for searching, crawling, and summarizing web content using AI.

## Features

- **Asynchronous Processing**: Built with Quart for full async/await support
- **Intelligent Web Crawling**: 
  - HTTP-based crawling for standard websites
  - Browser-based crawling for JavaScript-heavy sites
  - Smart content extraction
- **Content Quality Scoring**:
  - Multi-factor analysis (readability, structure, citations)
  - Domain reputation database
  - Quality-weighted summaries
- **AI-Powered Summarization**:
  - Context-aware prompt engineering
  - Query type detection
  - Source synthesis and attribution
- **Robust Error Handling**:
  - Structured exception hierarchy
  - Graceful fallbacks
  - Detailed logging
- **Real-time Progress Tracking**:
  - WebSocket updates
  - Step-by-step progress visualization

## Project Structure

```
web-summarizer/
├── app.py                          # Main Quart application
├── config.py                       # Application configuration
├── modules/                        # Application modules
│   ├── crawlers/                   # Web crawling modules
│   │   ├── http.py                 # HTTP-based crawler
│   │   ├── browser.py              # Browser-based crawler
│   │   └── integrated.py           # Combined crawler orchestration
│   ├── utils/                      # Utility modules
│   │   ├── errors.py               # Error handling
│   │   ├── logging.py              # Logging configuration
│   │   └── types.py                # Type definitions
│   ├── cache.py                    # Caching functionality
│   ├── content_quality.py          # Content quality scoring
│   ├── processor.py                # Text processing
│   ├── search.py                   # Web search
│   └── summarizer.py               # AI summarization
├── static/                         # Static assets
│   ├── css/
│   └── js/
├── templates/                      # HTML templates
└── tests/                          # Test suite
```

## Prerequisites

- Python 3.8+
- API keys:
  - OpenAI API key (for summarization)
  - SerpAPI key (optional, for better search results)
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

5. Download NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

6. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPAPI_API_KEY=your_serpapi_api_key
   SECRET_KEY=your_secure_random_key
   ```

## Running the Application

```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000`

## API Usage

```bash
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "depth": 3}'
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
```bash
curl http://127.0.0.1:5000/api/status/unique-search-id
```

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=modules
```

## Configuration Options

Edit `config.py` or set environment variables to configure:

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.