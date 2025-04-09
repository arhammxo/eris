"""
Benchmark script for comparing different chunking methods.

This script compares traditional chunking with Meta-Chunking,
measuring processing time and analyzing chunk quality.
"""
import sys
import os
import time
import asyncio
import argparse
from typing import List, Dict, Any
import nltk
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.processor import create_chunks
from modules.meta_chunking import MetaChunker

class ChunkingBenchmark:
    """Benchmark different chunking methods."""
    
    def __init__(self, sample_texts: List[str]):
        """
        Initialize with sample texts.
        
        Args:
            sample_texts: List of texts to benchmark
        """
        self.sample_texts = sample_texts
        
    async def run_benchmark(self):
        """Run the benchmarking tests."""
        results = []
        
        for i, text in enumerate(self.sample_texts):
            print(f"Testing sample {i+1}/{len(self.sample_texts)}...")
            
            # Measure traditional chunking
            start_time = time.time()
            trad_chunks = create_chunks(text)
            trad_time = time.time() - start_time
            
            # Measure Meta-Chunking
            meta_chunker = MetaChunker(threshold=0.5, target_chunk_size=4000)
            start_time = time.time()
            meta_chunks = await meta_chunker.chunk_text(text)
            meta_time = time.time() - start_time
            
            # Calculate metrics
            result = {
                "sample_id": i+1,
                "text_length": len(text),
                "sentence_count": len(nltk.sent_tokenize(text)),
                "traditional": {
                    "time": trad_time,
                    "chunk_count": len(trad_chunks),
                    "avg_chunk_size": np.mean([len(c) for c in trad_chunks]),
                    "std_chunk_size": np.std([len(c) for c in trad_chunks]),
                },
                "meta_chunking": {
                    "time": meta_time,
                    "chunk_count": len(meta_chunks),
                    "avg_chunk_size": np.mean([len(c) for c in meta_chunks]),
                    "std_chunk_size": np.std([len(c) for c in meta_chunks]),
                }
            }
            
            results.append(result)
            print(f"Sample {i+1} results:")
            print(f"  Traditional: {result['traditional']['time']:.4f}s, {result['traditional']['chunk_count']} chunks")
            print(f"  Meta-Chunking: {result['meta_chunking']['time']:.4f}s, {result['meta_chunking']['chunk_count']} chunks")
            
        return results
    
    def visualize_results(self, results: List[Dict[str, Any]]):
        """
        Generate visualizations of the benchmark results.
        
        Args:
            results: Benchmark results
        """
        # Extract data
        sample_ids = [r["sample_id"] for r in results]
        trad_times = [r["traditional"]["time"] for r in results]
        meta_times = [r["meta_chunking"]["time"] for r in results]
        
        trad_counts = [r["traditional"]["chunk_count"] for r in results]
        meta_counts = [r["meta_chunking"]["chunk_count"] for r in results]
        
        trad_avg_sizes = [r["traditional"]["avg_chunk_size"] for r in results]
        meta_avg_sizes = [r["meta_chunking"]["avg_chunk_size"] for r in results]
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot processing times
        axs[0].bar(
            np.array(sample_ids) - 0.2, 
            trad_times, 
            width=0.4, 
            label='Traditional',
            color='blue',
            alpha=0.7
        )
        axs[0].bar(
            np.array(sample_ids) + 0.2, 
            meta_times, 
            width=0.4, 
            label='Meta-Chunking',
            color='green',
            alpha=0.7
        )
        axs[0].set_xlabel('Sample ID')
        axs[0].set_ylabel('Processing Time (s)')
        axs[0].set_title('Chunking Processing Time Comparison')
        axs[0].legend()
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot chunk counts
        axs[1].bar(
            np.array(sample_ids) - 0.2, 
            trad_counts, 
            width=0.4, 
            label='Traditional',
            color='blue',
            alpha=0.7
        )
        axs[1].bar(
            np.array(sample_ids) + 0.2, 
            meta_counts, 
            width=0.4, 
            label='Meta-Chunking',
            color='green',
            alpha=0.7
        )
        axs[1].set_xlabel('Sample ID')
        axs[1].set_ylabel('Number of Chunks')
        axs[1].set_title('Chunk Count Comparison')
        axs[1].legend()
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot average chunk sizes
        axs[2].bar(
            np.array(sample_ids) - 0.2, 
            trad_avg_sizes, 
            width=0.4, 
            label='Traditional',
            color='blue',
            alpha=0.7
        )
        axs[2].bar(
            np.array(sample_ids) + 0.2, 
            meta_avg_sizes, 
            width=0.4, 
            label='Meta-Chunking',
            color='green',
            alpha=0.7
        )
        axs[2].set_xlabel('Sample ID')
        axs[2].set_ylabel('Average Chunk Size (chars)')
        axs[2].set_title('Average Chunk Size Comparison')
        axs[2].legend()
        axs[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('chunking_benchmark_results.png')
        plt.close()
        
        print(f"Benchmark visualization saved to 'chunking_benchmark_results.png'")


async def load_sample_texts(file_paths: List[str]) -> List[str]:
    """
    Load sample texts from files.
    
    Args:
        file_paths: Paths to files containing sample texts
        
    Returns:
        List of text content
    """
    texts = []
    
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)
        except Exception as e:
            print(f"Error loading file {path}: {str(e)}")
    
    return texts


async def main():
    parser = argparse.ArgumentParser(description='Benchmark chunking methods')
    parser.add_argument('--files', nargs='+', help='Paths to text files for benchmarking')
    parser.add_argument('--output', default='benchmark_results.txt', help='Output file for results')
    
    args = parser.parse_args()
    
    if not args.files:
        print("No input files specified. Please use --files option.")
        return
    
    # Load sample texts
    sample_texts = await load_sample_texts(args.files)
    
    if not sample_texts:
        print("No sample texts loaded. Please check the input files.")
        return
    
    # Run benchmark
    benchmark = ChunkingBenchmark(sample_texts)
    results = await benchmark.run_benchmark()
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results saved to {args.output}")
    
    # Create visualizations
    benchmark.visualize_results(results)


if __name__ == "__main__":
    asyncio.run(main())