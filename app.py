from flask import Flask, render_template, request, jsonify, session
from flask_caching import Cache
import time
import uuid
from datetime import datetime

# Import application modules
from modules.search import search_web
from modules.crawler import crawl_urls
from modules.processor import process_text
from modules.summarizer import generate_summary
from config import Config

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)

# Initialize cache
cache = Cache(app)

# Register Jinja2 filters
@app.template_filter('timestamp_to_datetime')
def timestamp_to_datetime(timestamp):
    """Convert a Unix timestamp to a formatted datetime string."""
    if not timestamp:
        return ''
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/', methods=['GET'])
def index():
    """Render the search form."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Process the search query and return results."""
    # Get query from form
    query = request.form.get('query', '')
    depth = int(request.form.get('depth', 3))
    summary_length = request.form.get('summary_length', 'medium')
    
    if not query:
        return render_template('index.html', error="Please enter a search query.")
    
    # Generate unique ID for this search
    search_id = str(uuid.uuid4())
    
    # Start the search process
    # Step 1: Search the web for relevant URLs
    urls = search_web(query, max_results=depth)
    
    if not urls:
        return render_template('index.html', 
                               error="No relevant results found. Please try a different query.")
    
    # Step 2: Crawl the URLs to extract content
    sources = crawl_urls(urls)
    
    if not sources:
        return render_template('index.html', 
                               error="Could not extract content from search results.")
    
    # Step 3: Process the extracted text
    processed_content = process_text(sources)
    
    # Step 4: Generate summary using LLM
    summary, metadata = generate_summary(
        query, 
        processed_content,
        length=summary_length
    )
    
    # Step 5: Cache results
    result = {
        'query': query,
        'timestamp': time.time(),
        'sources': sources,
        'summary': summary,
        'metadata': metadata
    }
    cache.set(search_id, result)
    
    return render_template('results.html', 
                          query=query,
                          summary=summary,
                          sources=sources,
                          metadata=metadata)

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for search queries."""
    data = request.json
    query = data.get('query', '')
    depth = int(data.get('depth', 3))
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Same processing logic as the web route
    urls = search_web(query, max_results=depth)
    if not urls:
        return jsonify({'error': 'No results found'}), 404
    
    sources = crawl_urls(urls)
    processed_content = process_text(sources)
    summary, metadata = generate_summary(query, processed_content)
    
    return jsonify({
        'query': query,
        'summary': summary,
        'sources': [s['url'] for s in sources],
        'metadata': metadata
    })

if __name__ == '__main__':
    app.run(debug=True)