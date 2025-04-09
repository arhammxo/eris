# performance_test.py
import time
import os
import sys
import psutil
import nltk
from tqdm import tqdm
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.meta_chunking import MetaChunker
from modules.processor import create_chunks

def generate_sample_text(paragraphs=20, sentences_per_paragraph=5, words_per_sentence=15):
    """Generate random text for testing."""
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
             "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
             "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation",
             "ullamco", "laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo", "consequat"]
    
    connectives = ["However, ", "Therefore, ", "Furthermore, ", "Nevertheless, ", "In addition, ", 
                  "Consequently, ", "Thus, ", "Indeed, ", "Moreover, ", "As a result, "]
    
    text = []
    
    for p in range(paragraphs):
        paragraph = []
        for s in range(sentences_per_paragraph):
            # Occasionally add a connective at the beginning
            sentence = random.choice(connectives) if random.random() < 0.3 and s > 0 else ""
            
            # Add random words
            sentence += " ".join(random.choices(words, k=random.randint(words_per_sentence-5, words_per_sentence+5)))
            sentence += "."
            paragraph.append(sentence)
        
        text.append(" ".join(paragraph))
    
    return "\n\n".join(text)

def measure_performance(func, *args, **kwargs):
    """Measure execution time and memory usage of a function."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    execution_time = end_time - start_time
    memory_used = mem_after - mem_before
    
    return result, execution_time, memory_used

def main():
    print("Running performance tests for chunking methods...")
    
    # Test with different text sizes
    sizes = [1000, 5000, 10000, 50000]
    
    results = {
        "Original": {"time": [], "memory": [], "chunks": []},
        "MetaChunking (0.3)": {"time": [], "memory": [], "chunks": []},
        "MetaChunking (0.5)": {"time": [], "memory": [], "chunks": []},
        "MetaChunking (1.0)": {"time": [], "memory": [], "chunks": []}
    }
    
    for size in sizes:
        print(f"\nTesting with {size} words...")
        
        # Generate sample text
        paragraphs = size // 50  # Approx. 50 words per paragraph
        text = generate_sample_text(paragraphs)
        
        # Set chunk size proportional to text size
        chunk_size = size // 10
        
        # Test original chunking
        print("Testing original chunking...")
        original_chunks, original_time, original_memory = measure_performance(
            create_chunks, text, chunk_size
        )
        results["Original"]["time"].append(original_time)
        results["Original"]["memory"].append(original_memory)
        results["Original"]["chunks"].append(len(original_chunks))
        
        # Test Meta-Chunking with different thresholds
        for threshold in [0.3, 0.5, 1.0]:
            print(f"Testing Meta-Chunking (threshold={threshold})...")
            chunker = MetaChunker(threshold=threshold, max_tokens_per_chunk=chunk_size)
            meta_chunks, meta_time, meta_memory = measure_performance(
                chunker.chunk_text, text
            )
            
            results[f"MetaChunking ({threshold})"]["time"].append(meta_time)
            results[f"MetaChunking ({threshold})"]["memory"].append(meta_memory)
            results[f"MetaChunking ({threshold})"]["chunks"].append(len(meta_chunks))
    
    # Print results
    print("\n\nPerformance Results:")
    print("--------------------")
    
    print("\nExecution Time (seconds):")
    print(f"{'Text Size':<10}", end="")
    for method in results:
        print(f"{method:<20}", end="")
    print()
    
    for i, size in enumerate(sizes):
        print(f"{size:<10}", end="")
        for method in results:
            print(f"{results[method]['time'][i]:.4f}{' '*16}", end="")
        print()
    
    print("\nMemory Usage (MB):")
    print(f"{'Text Size':<10}", end="")
    for method in results:
        print(f"{method:<20}", end="")
    print()
    
    for i, size in enumerate(sizes):
        print(f"{size:<10}", end="")
        for method in results:
            print(f"{results[method]['memory'][i]:.4f}{' '*16}", end="")
        print()
    
    print("\nNumber of Chunks:")
    print(f"{'Text Size':<10}", end="")
    for method in results:
        print(f"{method:<20}", end="")
    print()
    
    for i, size in enumerate(sizes):
        print(f"{size:<10}", end="")
        for method in results:
            print(f"{results[method]['chunks'][i]}{' '*19}", end="")
        print()

if __name__ == "__main__":
    main()