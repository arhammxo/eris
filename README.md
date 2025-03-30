# Web Summarizer Demo App

A Flask-based application that demonstrates web crawling, information retrieval, and AI-powered summarization.

## Features

- Search web for relevant information
- Extract and process content from multiple sources
- Generate comprehensive summaries using OpenAI's API
- Present results with source attribution
- Cache search results for performance

## Prerequisites

- Python 3.8+
- API keys:
  - OpenAI API key
  - SerpAPI key (optional, fallback search is available)

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

4. Download NLTK data (first time only):
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPAPI_API_KEY=your_serpapi_api_key
   SECRET_KEY=your_flask_secret_key
   ```

## Running the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Project Structure

- `app.py` - Main Flask application
- `config.py` - Configuration settings
- `modules/` - Application functionality modules
  - `search.py` - Web search functionality
  - `crawler.py` - URL content extraction
  - `processor.py` - Text processing
  - `summarizer.py` - LLM summarization
  - `cache.py` - Result caching
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and other static assets

## Extending the Application

This demo app provides a foundation that can be extended in various ways:

- Add user authentication and saved searches
- Further enhance crawler capabilities:
  - Implement proxy support for IP rotation
  - Add browser automation for JavaScript-heavy sites
  - Develop site-specific extraction rules for popular domains
- Add different summarization models
- Create a more advanced caching system
- Add visualization of source relationships
- Implement a feedback mechanism for summary quality
- Expand the API for integration with other applications

## API Usage

The application provides a simple API endpoint that can be used to perform searches programmatically:

```
POST /api/search
Content-Type: application/json

{
  "query": "artificial intelligence",
  "depth": 3
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.