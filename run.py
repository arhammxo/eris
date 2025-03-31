#!/usr/bin/env python
"""
Runner script for Web Summarizer application.

This script provides a convenient way to run the application with proper
error handling, environment setup, and command-line options.
"""
import os
import sys
import argparse
import logging
import asyncio
from dotenv import load_dotenv

# Make sure modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environment variables from .env file
load_dotenv()

# Import application
from app import app
from modules.utils.logging import setup_logging

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Web Summarizer application")
parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
parser.add_argument("--debug", action="store_true", help="Run in debug mode")
parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                    help="Logging level")
parser.add_argument("--log-file", help="Log to file instead of console")
parser.add_argument("--structured-logs", action="store_true", help="Use structured JSON logging")

args = parser.parse_args()

# Configure logging
log_level = getattr(logging, args.log_level)
setup_logging(
    level=log_level,
    structured=args.structured_logs,
    log_file=args.log_file,
    log_to_console=not args.log_file
)

logger = logging.getLogger(__name__)

# Check for required environment variables
required_env_vars = ["SECRET_KEY"]
optional_env_vars = ["OPENAI_API_KEY", "SERPAPI_API_KEY"]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please check your .env file or environment settings")
    sys.exit(1)

missing_optional = [var for var in optional_env_vars if not os.getenv(var)]
if missing_optional:
    logger.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    logger.warning("Some features may be limited or unavailable")

# Configure application
if args.debug:
    app.debug = True

# Run application
if __name__ == "__main__":
    logger.info(f"Starting Web Summarizer on {args.host}:{args.port} (debug={app.debug})")
    
    try:
        import hypercorn.asyncio
        from hypercorn.config import Config
        
        # Use Hypercorn for production if available
        logger.info("Using Hypercorn ASGI server")
        config = Config()
        config.bind = [f"{args.host}:{args.port}"]
        config.workers = args.workers
        config.accesslog = "-"  # Log to stdout
        
        asyncio.run(hypercorn.asyncio.serve(app, config))
    except ImportError:
        # Fall back to built-in Quart development server
        logger.info("Using Quart development server (not recommended for production)")
        app.run(host=args.host, port=args.port)