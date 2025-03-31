import nltk
import re
import logging
import asyncio
from quart import current_app
from modules.content_quality import ContentQualityScorer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize content quality scorer
quality_scorer = ContentQualityScorer()

async def process_text(sources):
    """
    Process the extracted text from multiple sources asynchronously.
    
    Args:
        sources (list): List of dictionaries containing source content and metadata
        
    Returns:
        dict: Processed content organized by source
    """
    processed_content = []
    
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

async def process_source(source):
    """Process a single source asynchronously."""
    try:
        # Get the content
        content = source['content']
        
        # Skip if no content
        if not content:
            return None
        
        # Process the content (these can be CPU-bound)
        cleaned_text = await asyncio.to_thread(clean_text, content)
        important_sentences = await asyncio.to_thread(extract_important_sentences, cleaned_text)
        chunks = await asyncio.to_thread(create_chunks, cleaned_text)
        
        # Score the content quality
        quality_score = await asyncio.to_thread(quality_scorer.score_content, source)
        
        # Add to processed content
        return {
            'url': source['url'],
            'title': source['title'],
            'domain': source['domain'],
            'cleaned_text': cleaned_text,
            'important_sentences': important_sentences,
            'chunks': chunks,
            'original_length': len(content),
            'processed_length': len(cleaned_text),
            'quality_score': quality_score
        }
            
    except Exception as e:
        logger.error(f"Error processing content from {source.get('url', 'unknown')}: {str(e)}")
        return None

def clean_text(text):
    """Clean and normalize text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags if any remains
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s.]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_important_sentences(text, num_sentences=5):
    """
    Extract the most important sentences from the text.
    
    This is a simplified implementation using frequency-based ranking.
    A production version would use more sophisticated algorithms like TextRank.
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
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words = [word for word in words if word.lower() not in stopwords]
        
        # Calculate word frequencies
        word_frequencies = {}
        for word in words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
        # Normalize frequencies
        maximum_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / maximum_frequency
            
        # Score sentences
        sentence_scores = {}
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
        logger.error(f"Error extracting important sentences: {str(e)}")
        # If anything fails, return the first few sentences
        sentences = nltk.sent_tokenize(text)
        return sentences[:num_sentences]

def create_chunks(text, chunk_size=None):
    """
    Break text into chunks of specified size.
    
    This is important for working with LLMs that have context length limitations.
    """
    if chunk_size is None:
        chunk_size = current_app.config.get('CHUNK_SIZE', 4000)
        
    # Tokenize into sentences to avoid breaking in the middle of a sentence
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
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