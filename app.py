from quart import Quart, render_template, request, jsonify, websocket
from quart_cache import Cache
import time
import uuid
import asyncio
from datetime import datetime

# Import application modules
from modules.search import search_web
from modules.async_crawler import crawl_urls, close_session
from modules.processor import process_text
from modules.summarizer import generate_summary
from config import Config

# Initialize Quart application
app = Quart(__name__)
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
async def index():
    """Render the search form."""
    return await render_template('index.html')

@app.route('/search', methods=['POST'])
async def search():
    """Process the search query and return results."""
    # Get form data (in Quart, this is handled differently)
    form = await request.form
    query = form.get('query', '')
    depth = int(form.get('depth', 3))
    summary_length = form.get('summary_length', 'medium')
    
    if not query:
        return await render_template('index.html', error="Please enter a search query.")
    
    # Generate unique ID for this search
    search_id = str(uuid.uuid4())
    
    # Start the search process
    # Step a: Search the web for relevant URLs (still synchronous for now)
    # Request more URLs than needed to account for crawling failures
    buffer_factor = 2  # Request 2x the URLs to account for failures
    urls = await asyncio.to_thread(search_web, query, max_results=depth * buffer_factor)
    
    if not urls:
        return await render_template('index.html', 
                               error="No relevant results found. Please try a different query.")
    
    # Step b: Asynchronously crawl the URLs to extract content
    sources = await crawl_urls(urls, target_count=depth)
    
    if not sources:
        return await render_template('index.html', 
                               error="Could not extract content from search results.")
    
    # Step c: Process the extracted text (currently synchronous)
    processed_content = await asyncio.to_thread(process_text, sources)
    
    # Step d: Generate summary using LLM (currently synchronous)
    summary, metadata = await asyncio.to_thread(
        generate_summary,
        query, 
        processed_content,
        length=summary_length
    )
    
    # Step e: Cache results
    result = {
        'query': query,
        'timestamp': time.time(),
        'sources': sources,
        'summary': summary,
        'metadata': metadata
    }
    await cache.set(search_id, result)
    
    return await render_template('results.html', 
                          query=query,
                          summary=summary,
                          sources=sources,
                          metadata=metadata)

@app.route('/api/search', methods=['POST'])
async def api_search():
    """API endpoint for search queries."""
    data = await request.json
    query = data.get('query', '')
    depth = int(data.get('depth', 3))
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Same processing logic as the web route, but asynchronous
    urls = await asyncio.to_thread(search_web, query, max_results=depth * 2)
    if not urls:
        return jsonify({'error': 'No results found'}), 404
    
    sources = await crawl_urls(urls, target_count=depth)
    if not sources:
        return jsonify({'error': 'Could not extract content'}), 500
    
    processed_content = await asyncio.to_thread(process_text, sources)
    summary, metadata = await asyncio.to_thread(generate_summary, query, processed_content)
    
    return jsonify({
        'query': query,
        'summary': summary,
        'sources': [s['url'] for s in sources],
        'metadata': metadata
    })

# WebSocket endpoint for providing progress updates
@app.websocket('/ws/progress')
async def ws_progress():
    while True:
        data = await websocket.receive_json()
        search_id = data.get('search_id')
        if search_id:
            # Retrieve progress for the given search_id
            progress_data = await cache.get(f"progress_{search_id}") or {"status": "unknown"}
            await websocket.send_json(progress_data)
        else:
            await websocket.send_json({"error": "No search_id provided"})

# Background task tracker
background_tasks = {}

# Helper function to update progress
def update_progress(search_id, status, progress=0, data=None):
    """
    Update the progress of a search task.
    
    Args:
        search_id (str): The search ID
        status (str): Current status (starting, searching, crawling, etc.)
        progress (int): Percentage complete (0-100)
        data (dict): Additional data to store
    """
    if not app.config.get('ENABLE_PROGRESS_TRACKING'):
        return
        
    progress_data = {
        'status': status,
        'progress': progress,
        'updated_at': time.time(),
        'data': data or {}
    }
    
    # Synchronous cache set
    cache.set(f"progress_{search_id}", progress_data, timeout=3600)  # 1 hour timeout
    return True

# Cleanup on shutdown
@app.before_serving
async def startup():
    # Any startup tasks
    pass

@app.after_serving
async def shutdown():
    # Close the aiohttp session
    await close_session()

if __name__ == '__main__':
    app.run(debug=True)