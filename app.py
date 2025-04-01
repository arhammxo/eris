"""
Web Summarizer: Main application entry point.

This Quart-based application provides a web interface for searching,
crawling, and summarizing web content with asynchronous processing.
"""
from quart import Quart, render_template, request, jsonify, websocket
from flask_caching import Cache
import time
import uuid
import asyncio
from datetime import datetime
import logging
import os

# Import application modules
from modules.search import search_web, search_web_with_config, search_combined
from modules.crawlers import crawl_urls, cleanup_crawlers, IntegratedCrawler
from modules.crawlers.file import FileCrawler
from modules.processor import process_text
from modules.summarizer import generate_summary
from modules.utils.errors import WebSummarizerError, create_error_response
from modules.utils.logging import setup_logging, get_logger
from config import Config

# Initialize logging
setup_logging(
    level=logging.INFO, 
    structured=False,
    log_file=os.path.join('logs', 'web_summarizer.log') if os.path.exists('logs') else None
)

# Get logger for this module
logger = get_logger(__name__)

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

async def process_search(search_id, query, depth, summary_length, search_scope='web'):
    """
    Asynchronous background task to process a search query.
    
    Updates progress in the cache as each step completes.
    
    Args:
        search_id: Unique identifier for this search
        query: Search query
        depth: Number of sources to analyze
        summary_length: Desired summary length ('short', 'medium', 'long')
        
    Returns:
        Result dictionary or None if processing failed
    """
    try:

        # Log search scope
        logger.info(f"Processing search with scope: {search_scope}")

        # Update progress - Starting search
        update_progress(search_id, 'searching', 10, {
            'message': 'Searching for relevant sources...'
        })

        # Configure search based on scope
        include_web = search_scope in ['web', 'both']
        include_files = search_scope in ['files', 'both']

        # Log base directories when file search is enabled
        if include_files:
            async with app.app_context():
                base_dirs = app.config.get('BASE_DIRECTORIES', [])
                logger.info(f"File search enabled with base directories: {base_dirs}")
                
                # Check if directories exist
                valid_dirs = [d for d in base_dirs if d and os.path.exists(d)]
                logger.info(f"Valid base directories: {valid_dirs}")
                
                if not valid_dirs:
                    logger.warning("No valid base directories found for file search")
                    update_progress(search_id, 'warning', 10, {
                        'message': 'No valid directories configured for file search.'
                    })
        
        urls = []
        has_file_results = False
        
        # Step 1: Search based on specified scope
        if include_web:
            # Web search
            buffer_factor = 2
            search_config = {}
            async with app.app_context():
                search_config['SERPAPI_API_KEY'] = app.config.get('SERPAPI_API_KEY')
                search_config['MAX_URLS_TO_CRAWL'] = app.config.get('MAX_URLS_TO_CRAWL')
            
            urls = await asyncio.to_thread(
                search_web_with_config, 
                query, 
                max_results=depth * buffer_factor,
                config=search_config
            )
        
        if include_files:
            has_file_results = True
            logger.info(f"File search enabled for query: {query}")
        
        # If no web results and not including files, show error
        if not urls and not has_file_results:
            update_progress(search_id, 'error', 100, {
                'message': 'No relevant results found. Please try a different query.'
            })
            return None
        
        # Update progress - Starting crawling
        update_progress(search_id, 'crawling', 30, {
            'message': 'Extracting content from sources...',
            'urls_found': len(urls),
            'includes_files': has_file_results
        })
        
        # Step 2: Crawl URLs and/or local files to extract content
        sources = []
        async with app.app_context():
            # Create base directories list from config
            base_dirs = app.config.get('BASE_DIRECTORIES', [])

            # Local-only search path
            if search_scope == 'files' and not urls:
                logger.info("Performing local-only file search")
                file_crawler = FileCrawler(
                    base_directories=base_dirs,
                    max_concurrent_extractions=app.config.get('MAX_CONCURRENT_EXTRACTIONS', 5)
                )
                
                # Directly search for files
                sources = await file_crawler.crawl_files(
                    query=query,
                    target_count=depth
                )
                
                logger.info(f"Local file search complete, found {len(sources)} sources")
            else:
                # Regular integrated crawling
                crawler = IntegratedCrawler(
                    use_browser=app.config.get('USE_BROWSER_CRAWLER', True) and include_web,
                    use_file_search=include_files,
                    max_concurrent_requests=app.config.get('MAX_CONCURRENT_REQUESTS', 10),
                    max_concurrent_per_domain=app.config.get('MAX_CONCURRENT_PER_DOMAIN', 3),
                    base_directories=base_dirs
                )
                
                # Crawl sources
                sources = await crawler.crawl_urls(
                    urls, 
                    query=query, 
                    target_count=depth,
                    include_files=has_file_results
                )

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
        logger.error(f"Error processing search {search_id}: {str(e)}", exc_info=True)
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
    search_scope = form.get('search_scope', 'web')  # Get the search scope
    
    if not query:
        return await render_template('index.html', error="Please enter a search query.")
    
    # Generate unique ID for this search
    search_id = str(uuid.uuid4())
    
    # Store initial progress
    update_progress(search_id, 'starting', 0, {
        'query': query,
        'depth': depth,
        'summary_length': summary_length,
        'search_scope': search_scope  # Add search scope to progress data
    })
    
    # Start background task for processing
    task = asyncio.create_task(process_search(search_id, query, depth, summary_length, search_scope))
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
    try:
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
    except Exception as e:
        logger.error(f"API search error: {str(e)}", exc_info=True)
        error_response = create_error_response(e)
        return jsonify(error_response), 500

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
            logger.error(f"WebSocket error: {str(e)}", exc_info=True)
            break

def update_progress(search_id, status, progress=0, data=None):
    """
    Update the progress of a search task.
    
    Args:
        search_id: The search ID
        status: Current status (starting, searching, crawling, etc.)
        progress: Percentage complete (0-100)
        data: Additional data to store
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

# Lifecycle hooks
@app.before_serving
async def startup():
    """Execute before the server starts accepting connections."""
    logger.info("Starting Web Summarizer server")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

@app.after_serving
async def shutdown():
    """Execute after the server stops accepting connections."""
    logger.info("Shutting down Web Summarizer server")
    
    # Clean up resources
    await cleanup_crawlers()
    
    # Cancel any running background tasks
    for task_id, task in list(background_tasks.items()):
        if not task.done():
            logger.info(f"Canceling background task {task_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    background_tasks.clear()

if __name__ == '__main__':
    app.run(debug=True)