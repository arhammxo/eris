from quart import Quart, render_template, request, jsonify, websocket
from flask_caching import Cache
import time
import uuid
import asyncio
import logging
from datetime import datetime

# Import application modules
from modules.search import search_web, search_web_with_config
from modules.integrated_crawler import integrated_crawl, cleanup_crawlers
from modules.async_processor import process_text
from modules.async_summarizer import generate_summary
from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Quart application
app = Quart(__name__)
app.config.from_object(Config)

# Initialize cache
cache = Cache(app)

# Background tasks store
background_tasks = {}

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

async def process_search(search_id, query, depth, summary_length):
    """
    Asynchronous background task to process a search query.
    
    Updates progress in the cache as each step completes.
    """
    try:
        # Update progress - Starting search
        update_progress(search_id, 'searching', 10, {
            'message': 'Searching for relevant sources...'
        })
        
        # Step 1: Search the web for relevant URLs
        # Use app context for any function that might access current_app
        buffer_factor = 2  # Request 2x the URLs to account for failures
        
        # Create a copy of the app's config values needed for search
        # so we don't need current_app inside the thread
        search_config = {}
        async with app.app_context():
            search_config['SERPAPI_API_KEY'] = app.config.get('SERPAPI_API_KEY')
            search_config['MAX_URLS_TO_CRAWL'] = app.config.get('MAX_URLS_TO_CRAWL')
        
        # Pass the config directly to avoid needing current_app
        urls = await asyncio.to_thread(
            search_web_with_config, 
            query, 
            max_results=depth * buffer_factor,
            config=search_config
        )
        
        if not urls:
            update_progress(search_id, 'error', 100, {
                'message': 'No relevant results found. Please try a different query.'
            })
            return None
        
        # Update progress - Starting crawling
        update_progress(search_id, 'crawling', 30, {
            'message': 'Extracting content from sources...',
            'urls_found': len(urls)
        })
        
        # Step 2: Crawl the URLs to extract content
        # Create app context for the async crawling
        async with app.app_context():
            sources = await integrated_crawl(urls, target_count=depth)
        
        if not sources:
            update_progress(search_id, 'error', 100, {
                'message': 'Could not extract content from search results.'
            })
            return None
        
        # Update progress - Processing content
        update_progress(search_id, 'processing', 50, {
            'message': 'Processing and analyzing content...',
            'sources_found': len(sources)
        })
        
        # Step 3: Process the extracted text
        async with app.app_context():
            processed_content = await process_text(sources)
        
        # Update progress - Generating summary
        update_progress(search_id, 'summarizing', 70, {
            'message': 'Generating comprehensive summary...',
            'sources_processed': len(processed_content)
        })
        
        # Step 4: Generate summary using LLM
        async with app.app_context():
            summary, metadata = await generate_summary(
                query, 
                processed_content,
                length=summary_length
            )
        
        # Update progress - Complete
        update_progress(search_id, 'complete', 100, {
            'message': 'Search complete!',
            'summary_length': len(summary)
        })
        
        # Step 5: Cache results
        result = {
            'query': query,
            'timestamp': time.time(),
            'search_id': search_id,
            'sources': sources,
            'summary': summary,
            'metadata': metadata
        }
        cache.set(search_id, result, timeout=86400)  # 24 hour cache
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing search {search_id}: {str(e)}")
        update_progress(search_id, 'error', 100, {
            'message': f"An error occurred: {str(e)}"
        })
        return None
    finally:
        # Remove from active background tasks
        if search_id in background_tasks:
            del background_tasks[search_id]

@app.route('/search', methods=['POST'])
async def search():
    """Process the search query and return initial response."""
    # Get form data
    form = await request.form
    query = form.get('query', '')
    depth = int(form.get('depth', 3))
    summary_length = form.get('summary_length', 'medium')
    
    if not query:
        return await render_template('index.html', error="Please enter a search query.")
    
    # Generate unique ID for this search
    search_id = str(uuid.uuid4())
    
    # Store initial progress
    update_progress(search_id, 'starting', 0, {
        'query': query,
        'depth': depth,
        'summary_length': summary_length
    })
    
    # Start background task for processing
    task = asyncio.create_task(process_search(search_id, query, depth, summary_length))
    background_tasks[search_id] = task
    
    # Return template with search_id for progress tracking
    return await render_template('progress.html', 
                              search_id=search_id,
                              query=query)

@app.route('/results/<search_id>', methods=['GET'])
async def results(search_id):
    """Show results for a completed search."""
    # Get results from cache
    result = cache.get(search_id)
    
    if not result:
        # Check if search is still in progress
        progress = cache.get(f"progress_{search_id}")
        if progress and progress['status'] not in ['complete', 'error']:
            return await render_template('progress.html', 
                                      search_id=search_id,
                                      query=progress.get('data', {}).get('query', 'your search'))
        else:
            return await render_template('index.html', 
                                error="Search results not found or expired. Please try a new search.")
    
    # Return results template with data
    return await render_template('results.html', 
                              search_id=search_id,
                              query=result['query'],
                              summary=result['summary'],
                              sources=result['sources'],
                              metadata=result['metadata'])

@app.route('/api/search', methods=['POST'])
async def api_search():
    """API endpoint for search queries."""
    data = await request.json
    query = data.get('query', '')
    depth = int(data.get('depth', 3))
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Generate unique ID for this search
    search_id = str(uuid.uuid4())
    
    # Start processing in background
    task = asyncio.create_task(process_search(search_id, query, depth, 'medium'))
    background_tasks[search_id] = task
    
    # Return initial response with search_id
    return jsonify({
        'search_id': search_id,
        'status': 'processing',
        'message': 'Search is being processed',
        'poll_url': f'/api/status/{search_id}'
    })

@app.route('/api/status/<search_id>', methods=['GET'])
async def api_status(search_id):
    """API endpoint to check search status."""
    # Check progress
    progress = cache.get(f"progress_{search_id}")
    
    if not progress:
        return jsonify({'error': 'Search not found'}), 404
        
    # If complete, include results
    if progress['status'] == 'complete':
        result = cache.get(search_id)
        if result:
            return jsonify({
                'status': 'complete',
                'search_id': search_id,
                'query': result['query'],
                'summary': result['summary'],
                'sources': [s['url'] for s in result['sources']],
                'metadata': result['metadata']
            })
    
    # Return progress information
    return jsonify({
        'status': progress['status'],
        'progress': progress['progress'],
        'message': progress.get('data', {}).get('message', ''),
        'updated_at': progress['updated_at']
    })

@app.websocket('/ws/progress/<search_id>')
async def ws_progress(search_id):
    """WebSocket endpoint for real-time progress updates."""
    while True:
        try:
            # Get current progress
            progress = cache.get(f"progress_{search_id}")
            
            if progress:
                # Send progress update
                await websocket.send_json(progress)
                
                # If processing is complete or errored, close connection
                if progress['status'] in ['complete', 'error']:
                    await websocket.send_json({
                        'status': progress['status'],
                        'message': 'Processing finished',
                        'redirect': f'/results/{search_id}' if progress['status'] == 'complete' else None
                    })
                    break
            else:
                # Search not found
                await websocket.send_json({
                    'status': 'error',
                    'message': 'Search not found'
                })
                break
                
            # Wait before next update
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            break

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
    
    # Use synchronous cache.set (no await)
    cache.set(f"progress_{search_id}", progress_data, timeout=3600)  # 1 hour timeout
    return True

# Cleanup on shutdown
@app.before_serving
async def startup():
    # Any startup tasks
    pass

@app.after_serving
async def shutdown():
    # Clean up resources
    await cleanup_crawlers()
    
    # Cancel any running background tasks
    for task in background_tasks.values():
        if not task.done():
            task.cancel()

if __name__ == '__main__':
    app.run(debug=True)