"""
Text processor for web content analysis.

This module handles the processing of extracted web content, including
cleaning, sentence extraction, chunking, and quality scoring.
"""
import asyncio
import re
from typing import Dict, List, Optional, Any

# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt', quiet=True)
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords', quiet=True)

import nltk
from nltk.corpus import stopwords

from quart import current_app

from modules.utils.types import Source, ProcessedSource
from modules.utils.errors import ProcessorError, TextAnalysisError, handle_async_exceptions
from modules.utils.logging import get_logger
from modules.content_quality import ContentQualityScorer
# Import the new meta_chunking module
from modules.meta_chunking import meta_chunk_text

# Create a logger for this module
logger = get_logger(__name__)

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords")
    nltk.download('stopwords', quiet=True)

# Initialize content quality scorer
quality_scorer = ContentQualityScorer()

async def process_text(sources: List[Source]) -> List[ProcessedSource]:
    """
    Process the extracted text from multiple sources asynchronously.
    
    Args:
        sources: List of dictionaries containing source content and metadata
        
    Returns:
        Processed content organized by source
    """
    if not sources:
        logger.warning("No sources provided for processing")
        return []
        
    processed_content: List[ProcessedSource] = []
    
    # Create tasks for processing each source
    tasks = [process_source(source) for source in sources]
    
    # Process all sources concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and collect successful results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error during processing: {str(result)}")
        elif result:  # Skip None results
            processed_content.append(result)
    
    # Sort by quality score in descending order
    processed_content.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    return processed_content

@handle_async_exceptions(ProcessorError, "Failed to process source")
async def process_source(source: Source) -> Optional[ProcessedSource]:
    """
    Process a single source asynchronously.
    
    Args:
        source: Source data with content and metadata
        
    Returns:
        Processed source data
    """
    
    # Get the content
    content = source.get('content', '')
    
    # Skip if no content
    if not content:
        logger.warning(f"No content to process for {source.get('url', 'unknown')}")
        return None
    
    # Process the content (these can be CPU-bound)
    cleaned_text = await asyncio.to_thread(clean_text, content)
    important_sentences = await asyncio.to_thread(extract_important_sentences, cleaned_text)
    
    # Use app context if available to get chunking parameters
    use_meta_chunking = True
    chunk_size = 4000
    
    try:
        use_meta_chunking = current_app.config.get('USE_META_CHUNKING', True)
        chunk_size = current_app.config.get('CHUNK_SIZE', 4000)
    except RuntimeError:
        # No app context
        pass
    
    # Choose chunking method based on configuration
    if use_meta_chunking:
        # Use the new Meta-Chunking approach
        chunks = await meta_chunk_text(
            cleaned_text,
            target_chunk_size=chunk_size
        )
        logger.info(f"Meta-chunked content into {len(chunks)} chunks")
    else:
        # Use traditional chunking
        chunks = await asyncio.to_thread(create_chunks, cleaned_text, chunk_size)
        logger.info(f"Traditional chunking created {len(chunks)} chunks")
    
    # Logging block moved here, after chunks are created
    if use_meta_chunking:
        logger.info(f"META-CHUNKING: Created {len(chunks)} chunks using perplexity-based segmentation for {source.get('url', 'unknown')}")
        logger.info(f"META-CHUNKING: Average chunk size: {sum(len(c) for c in chunks)/len(chunks) if chunks else 0:.1f} characters")
    else:
        logger.info(f"TRADITIONAL: Created {len(chunks)} chunks using sentence-based segmentation for {source.get('url', 'unknown')}")
        logger.info(f"TRADITIONAL: Average chunk size: {sum(len(c) for c in chunks)/len(chunks) if chunks else 0:.1f} characters")
    
    # Score the content quality
    quality_score = await asyncio.to_thread(quality_scorer.score_content, source)
    
    # Create enhanced chunks with metadata
    enhanced_chunks = []
    current_pos = 0
    for idx, chunk_content in enumerate(chunks):
        # Create a unique chunk ID
        # Note: hash() is not stable across Python processes/versions.
        # Consider a more robust hashing like hashlib.sha1 if needed.
        chunk_id = f"{source.get('domain', 'no_domain')}_{hash(source.get('url', 'no_url'))}_{idx}"

        # Find start position more robustly
        start_pos = cleaned_text.find(chunk_content, current_pos)
        if start_pos == -1:
            # Fallback if the exact chunk isn't found (e.g., due to minor cleaning differences)
            # Use the start of the first sentence if possible
            first_sentence = chunk_content.split('.')[0]
            start_pos = cleaned_text.find(first_sentence, current_pos)
            if start_pos == -1:
                 # If still not found, log a warning and use approximate position
                 logger.warning(f"Could not accurately find start position for chunk {idx} in {source.get('url', 'unknown')}. Using previous end position.")
                 start_pos = current_pos # Approximate start

        end_pos = start_pos + len(chunk_content)
        current_pos = end_pos # Update current position for next search

        # Create enhanced chunk dictionary
        enhanced_chunk = {
            "chunk_id": chunk_id,
            "source_url": source.get('url'),
            "source_title": source.get('title'),
            "source_domain": source.get('domain'),
            "content": chunk_content,
            "position": idx,
            "start_char": start_pos if start_pos != -1 else None, # Use None if not found
            "end_char": end_pos if start_pos != -1 else None, # Use None if not found
            "quality_score": quality_score  # Inherit from source for now
        }
        enhanced_chunks.append(enhanced_chunk)

    # Create processed source, now using enhanced_chunks
    return {
        'url': source.get('url'),
        'title': source.get('title'),
        'domain': source.get('domain'),
        'cleaned_text': cleaned_text, # Keep the full cleaned text if needed elsewhere
        'important_sentences': important_sentences,
        'chunks': enhanced_chunks,  # Use the list of chunk dictionaries
        'original_length': len(content),
        'processed_length': len(cleaned_text),
        'quality_score': quality_score
    }

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags if any remains
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s.]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_important_sentences(text: str, num_sentences: int = 5) -> List[str]:
    """
    Extract the most important sentences from the text.
    
    Uses frequency-based ranking to identify key sentences.
    
    Args:
        text: Cleaned text content
        num_sentences: Number of sentences to extract
        
    Returns:
        List of important sentences
    """
    try:
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        
        # If we have very few sentences, return all of them
        if len(sentences) <= num_sentences:
            return sentences
            
        # Tokenize into words
        words = nltk.word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
        
        # Calculate word frequencies
        word_frequencies: Dict[str, int] = {}
        for word in words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
        # Normalize frequencies
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
            
        # Score sentences
        sentence_scores: Dict[str, float] = {}
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
                        
        # Get top sentences
        import heapq
        important_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        
        return important_sentences
        
    except Exception as e:
        logger.warning(f"Error in NLTK extraction: {str(e)}")
        # Simple fallback - just split by period and take first few sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[:num_sentences]

def create_chunks(text: str, chunk_size: int = 4000) -> List[str]:
    """
    Break text into chunks of specified size.
    
    This is the traditional chunking method, kept for compatibility.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of text chunks
    """
    # Tokenize into sentences to avoid breaking in the middle of a sentence
    sentences = nltk.sent_tokenize(text)
    
    chunks: List[str] = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the chunk size, start a new chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks