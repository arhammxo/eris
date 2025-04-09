# test_meta_chunking_demo.py
import sys
import os
import nltk
from pprint import pprint

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.meta_chunking import MetaChunker
from modules.processor import create_chunks

def highlight_chunk_boundaries(text, chunks):
    """Highlight chunk boundaries in the original text for visualization."""
    sentences = nltk.sent_tokenize(text)
    marked_text = []
    
    chunk_start_indices = []
    current_pos = 0
    
    # Find start positions of chunks in the original text
    for chunk in chunks:
        first_sentence = nltk.sent_tokenize(chunk)[0]
        found_pos = text.find(first_sentence, current_pos)
        if found_pos != -1:
            chunk_start_indices.append(found_pos)
            current_pos = found_pos + 1
    
    # Add markers to the text
    last_pos = 0
    for i, pos in enumerate(chunk_start_indices):
        # Add text up to this chunk
        marked_text.append(text[last_pos:pos])
        # Add chunk marker
        marked_text.append(f"\n{'='*80}\n[CHUNK {i+1} STARTS HERE]\n{'='*80}\n")
        last_pos = pos
    
    # Add remaining text
    marked_text.append(text[last_pos:])
    
    return "".join(marked_text)

def main():
    # Sample text for testing
    sample_text = """
    The Web Summarizer application processes content from multiple sources. It searches the web for relevant documents.
    These documents are then extracted and cleaned. The cleaning process removes HTML tags and other noise.
    
    Next, the application analyzes the content for quality. High-quality sources are given more weight in the final summary.
    This analysis considers factors like readability, citations, and domain reputation.
    
    After processing, the system generates a comprehensive summary. The summary synthesizes information from all sources.
    It highlights key points and maintains logical connections between ideas. This approach ensures that users receive
    accurate and concise information about their query.
    
    The Web Summarizer is built using a modular architecture. Each component handles a specific task in the pipeline.
    This design makes the system extensible and maintainable. New features can be added without disrupting existing functionality.
    
    The search module finds relevant sources across the web. It uses search APIs and follows links to discover content.
    The crawler module extracts content from these sources. It can handle both regular web pages and JavaScript-heavy sites.
    
    The processor module cleans and analyzes the extracted content. It removes irrelevant elements and structures the text.
    The quality scoring module evaluates each source based on multiple factors. It assigns a score that reflects the reliability and relevance of the source.
    
    Finally, the summarizer module generates the comprehensive summary. It uses advanced NLP techniques to synthesize information.
    The final output provides users with a concise overview of the topic based on multiple high-quality sources.
    """
    
    print("Testing Meta-Chunking with different thresholds:")
    
    # Test with different thresholds
    for threshold in [0.3, 0.5, 1.0]:
        print(f"\n\nThreshold: {threshold}")
        
        # Use direct MetaChunker
        chunker = MetaChunker(threshold=threshold)
        chunks = chunker.chunk_text(sample_text)
        
        print(f"Generated {len(chunks)} chunks")
        print("Chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk)} chars):")
            print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)
        
        # Show the original text with chunk boundaries
        print("\nOriginal text with chunk boundaries:")
        marked_text = highlight_chunk_boundaries(sample_text, chunks)
        print(marked_text)
    
    # Compare with original chunking
    print("\n\nComparing with original chunking:")
    
    # Mock current_app for testing
    os.environ['META_CHUNKING_ENABLED'] = 'False'
    
    # Original chunking
    original_chunks = create_chunks(sample_text, chunk_size=500)
    print(f"\nOriginal chunking generated {len(original_chunks)} chunks:")
    for i, chunk in enumerate(original_chunks):
        print(f"\nOriginal Chunk {i+1} ({len(chunk)} chars):")
        print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)
    
    print("\nOriginal chunking with boundaries:")
    marked_original = highlight_chunk_boundaries(sample_text, original_chunks)
    print(marked_original)

if __name__ == "__main__":
    main()