#!/usr/bin/env python
"""
Migration script for Web Summarizer refactoring.

This script helps with the transition from the old project structure
to the new refactored structure, including:
- Creating new directories
- Moving files to their new locations
- Backing up old files
- Setting up initial database structure
"""
import os
import shutil
import argparse
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define project structure
NEW_DIRECTORIES = [
    "modules/crawlers",
    "modules/utils",
    "tests",
    "logs",
    "data",
    "scripts"
]

# Files that need to be moved or renamed
FILE_MOVES = {
    "enhanced_app.py": "app.py",
    "modules/async_crawler.py": "modules/crawlers/http.py",
    "modules/browser_crawler.py": "modules/crawlers/browser.py",
    "modules/integrated_crawler.py": "modules/crawlers/integrated.py",
    "modules/async_processor.py": "modules/processor.py",
    "modules/async_summarizer.py": "modules/summarizer.py"
}

# Files that should be backed up before overwriting
BACKUP_FILES = [
    "app.py",
    "modules/crawler.py",
    "modules/processor.py",
    "modules/summarizer.py",
    "modules/__init__.py",
    "README.md",
    "requirements.txt"
]

def create_directories(base_path):
    """Create the new directory structure."""
    logger.info("Creating new directory structure...")
    
    for directory in NEW_DIRECTORIES:
        full_path = os.path.join(base_path, directory)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            logger.info(f"Created directory: {full_path}")
        else:
            logger.info(f"Directory already exists: {full_path}")
    
    return True

def backup_files(base_path):
    """Backup files that will be overwritten."""
    logger.info("Backing up existing files...")
    
    backup_dir = os.path.join(base_path, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(backup_dir)
    logger.info(f"Created backup directory: {backup_dir}")
    
    for file_path in BACKUP_FILES:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            backup_path = os.path.join(backup_dir, file_path)
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy file to backup
            shutil.copy2(full_path, backup_path)
            logger.info(f"Backed up: {file_path}")
    
    return backup_dir

def move_files(base_path):
    """Move files to their new locations."""
    logger.info("Moving files to new locations...")
    
    for old_path, new_path in FILE_MOVES.items():
        full_old_path = os.path.join(base_path, old_path)
        full_new_path = os.path.join(base_path, new_path)
        
        if os.path.exists(full_old_path):
            # Create parent directories if needed
            os.makedirs(os.path.dirname(full_new_path), exist_ok=True)
            
            # Copy file to new location
            shutil.copy2(full_old_path, full_new_path)
            logger.info(f"Moved: {old_path} → {new_path}")
        else:
            logger.warning(f"Source file not found: {old_path}")
    
    return True

def setup_database(base_path):
    """Set up the initial database structure."""
    logger.info("Setting up initial database...")
    
    db_path = os.path.join(base_path, "data/domain_quality.db")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create database and tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create domain quality table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS domain_quality (
        domain TEXT PRIMARY KEY,
        quality_score REAL,
        last_updated INTEGER
    )
    ''')
    
    # Seed with some known reliable domains if table is empty
    cursor.execute("SELECT COUNT(*) FROM domain_quality")
    if cursor.fetchone()[0] == 0:
        now = int(datetime.now().timestamp())
        seed_domains = [
            ('wikipedia.org', 0.95, now),
            ('britannica.com', 0.92, now),
            ('smithsonianmag.com', 0.9, now),
            ('nature.com', 0.95, now),
            ('science.org', 0.93, now),
            ('mayoclinic.org', 0.92, now),
            ('nih.gov', 0.95, now),
            ('cdc.gov', 0.93, now),
            ('who.int', 0.93, now),
            ('edu', 0.85, now),  # Generic .edu domains
            ('gov', 0.85, now),  # Generic .gov domains
        ]
        
        cursor.executemany(
            "INSERT INTO domain_quality (domain, quality_score, last_updated) VALUES (?, ?, ?)",
            seed_domains
        )
        logger.info(f"Seeded database with {len(seed_domains)} domains")
    
    conn.commit()
    conn.close()
    
    return True

def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate Web Summarizer to new structure")
    parser.add_argument("--path", default=".", help="Base path of the project")
    args = parser.parse_args()
    
    base_path = os.path.abspath(args.path)
    logger.info(f"Starting migration in: {base_path}")
    
    try:
        # Backup existing files
        backup_dir = backup_files(base_path)
        
        # Create new directory structure
        create_directories(base_path)
        
        # Move files to new locations
        move_files(base_path)
        
        # Set up database
        setup_database(base_path)
        
        logger.info("Migration completed successfully!")
        logger.info(f"Backup of original files saved to: {backup_dir}")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()