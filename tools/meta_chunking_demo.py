#!/usr/bin/env python
"""
Standalone demo for Meta-Chunking functionality.

This script provides a complete demonstration of the Meta-Chunking implementation,
comparing it with traditional chunking and visualizing the results.
"""
import sys
import os
import time
import asyncio
import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from colorama import Fore, Style, init

# Initialize colorama
init()

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create a minimal mock for current_app to avoid dependency issues
class MockApp:
    def __init__(self):
        self.config = {
            'USE_META_CHUNKING': True,
            'META_CHUNKING_THRESHOLD': 0.5,
            'USE_DYNAMIC_COMBINATION': True,
            'USE_OPENAI_FOR_PPL': False,
            'CHUNK_SIZE': 4000
        }

# Mock the current_app for standalone execution
sys.modules['quart'] = type('MockQuart', (), {'current_app': MockApp()})

# Now import our modules after setting up the mock
from modules.meta_chunking import MetaChunker
import nltk

# Sample text for demonstration - multiple paragraphs with clear logical sections
SAMPLE_TEXT = """
Artificial Intelligence (AI) has evolved dramatically over the past decade, transforming from a niche research field into a pervasive technology affecting nearly every industry. The development of machine learning algorithms, particularly deep learning approaches, has been central to this revolution. Neural networks with multiple layers can now recognize patterns in vast amounts of data with unprecedented accuracy, enabling applications from image recognition to natural language processing.

One of the most significant breakthroughs has been in natural language processing (NLP). Modern language models can understand and generate human language with remarkable fluency. This has enabled applications such as automated customer service, content generation, and sophisticated translation services. The introduction of transformer-based architectures, beginning with models like BERT and GPT, has been particularly transformative, allowing models to capture long-range dependencies in text and generate coherent paragraphs.

Computer vision has seen similarly impressive advances. Convolutional neural networks have revolutionized image recognition, enabling systems that can identify objects, faces, and activities in images and videos with human-level accuracy. These technologies power everything from self-driving cars to medical imaging diagnostics, fundamentally changing how machines perceive and interact with the visual world.

Reinforcement learning represents another frontier in AI development. By learning through trial and error with a reward system, AI agents can master complex tasks without explicit programming. This approach has led to dramatic achievements, from defeating world champions at Go to optimizing energy consumption in data centers. Reinforcement learning's ability to find novel solutions to complex problems makes it particularly valuable for scenarios where the optimal approach isn't obvious to human experts.

The ethical implications of advanced AI systems have become increasingly important as these technologies impact society more broadly. Issues of bias, privacy, transparency, and autonomy require careful consideration as AI systems make decisions affecting human lives. Researchers and policymakers are working to develop frameworks that ensure AI technologies are developed and deployed responsibly, with appropriate safeguards and oversight.

Looking to the future, several research directions hold particular promise. Multimodal models that can process and integrate different types of data—text, images, audio—are advancing rapidly. Unsupervised learning techniques continue to improve, reducing the need for labeled training data. And quantum computing may eventually provide computational resources that enable entirely new AI approaches. As these technologies mature, the boundary between what is uniquely human and what machines can accomplish continues to shift.
"""

# Additional sample with more technical content and different structure
TECHNICAL_SAMPLE = """
Quantum computing leverages the principles of quantum mechanics to process information in fundamentally different ways than classical computers. Unlike classical bits, which exist in a state of either 0 or 1, quantum bits or "qubits" can exist in a superposition of both states simultaneously. This property, along with quantum entanglement and quantum interference, allows quantum computers to perform certain calculations exponentially faster than classical computers.

The quantum computing stack consists of several layers. At the hardware level, there are various approaches to implementing qubits, including superconducting circuits, trapped ions, photonic systems, and topological qubits. Each has distinct advantages and challenges related to coherence time, error rates, and scalability. The control layer includes the systems necessary to manipulate qubits through precisely calibrated microwave or laser pulses. Error correction represents another critical layer, as quantum systems are highly susceptible to environmental noise and decoherence.

Quantum algorithms exploit the unique properties of quantum systems to solve specific problems efficiently. Shor's algorithm for integer factorization and Grover's search algorithm are canonical examples, with the former threatening to undermine current cryptographic systems. Other algorithms show promise for simulation of quantum systems, optimization problems, and machine learning applications. The development of these algorithms has proceeded somewhat independently from hardware, creating a gap between theoretical capabilities and practical implementations.

Current quantum computers exist in the NISQ (Noisy Intermediate-Scale Quantum) era, characterized by limited qubit counts and high error rates. Despite these limitations, researchers are exploring potential quantum advantage in specific applications, especially in materials science and chemistry where quantum effects are intrinsic to the problems being solved. Variational algorithms, which combine classical and quantum computing in hybrid approaches, have emerged as promising methods for extracting useful work from imperfect quantum hardware.

The quantum computing field faces significant engineering and theoretical challenges. Scaling beyond a few hundred noisy qubits to the millions of error-corrected qubits needed for transformative applications requires breakthroughs in materials science, cryogenics, control systems, and error correction protocols. The development of a comprehensive theory of quantum computational advantage—clearly delineating which problems quantum computers can solve more efficiently and by how much—remains an active area of research with many open questions.

Commercial interest in quantum computing has accelerated in recent years, with major technology companies, specialized startups, and government initiatives investing billions in development. This has raised concerns about hype and unrealistic timelines, as practical applications with clear advantages over classical approaches remain limited. Nevertheless, the field continues to advance rapidly, with regular announcements of new qubit counts, fidelity improvements, and algorithmic innovations suggesting that quantum computing's long-promised potential may eventually be realized.
"""

class ChunkVisualizer:
    """Visualize and compare chunking methods."""
    
    @staticmethod
    def display_chunks(text: str, chunk_boundaries: list, title: str):
        """
        Display text with chunk boundaries highlighted.
        
        Args:
            text: The text being chunked
            chunk_boundaries: List of indices where chunks end
            title: Title for the visualization
        """
        sentences = nltk.sent_tokenize(text)
        
        print(f"\n{Fore.CYAN}{'='*20} {title} {'='*20}{Style.RESET_ALL}")
        
        current_chunk = 1
        boundary_indices = set(chunk_boundaries)
        
        for i, sentence in enumerate(sentences):
            # Determine if this is the start of a new chunk
            if i > 0 and (i-1) in boundary_indices:
                print(f"\n{Fore.GREEN}--- Chunk {current_chunk} End ---{Style.RESET_ALL}\n")
                current_chunk += 1
                print(f"{Fore.GREEN}--- Chunk {current_chunk} Start ---{Style.RESET_ALL}\n")
            
            # Print the sentence
            print(sentence)
            
        if sentences and len(sentences) - 1 in boundary_indices:
            print(f"\n{Fore.GREEN}--- Chunk {current_chunk} End ---{Style.RESET_ALL}\n")
    
    @staticmethod
    def calculate_ppl_scores(sentences: list) -> list:
        """
        Calculate local approximation of perplexity scores.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of perplexity scores
        """
        # Simple implementation of perplexity estimation for visualization
        ppl_scores = []
        
        # Create a simple word frequency model from all text
        full_text = " ".join(sentences)
        words = re.findall(r'\w+', full_text.lower())
        word_counts = Counter(words)
        total_words = len(words)
        word_probs = {word: count/total_words for word, count in word_counts.items()}
        
        # For each sentence, calculate approximate perplexity
        for i, sentence in enumerate(sentences):
            # Skip first sentence as it has no context
            if i == 0:
                ppl_scores.append(1.0)
                continue
                
            # Get words in this sentence
            sent_words = re.findall(r'\w+', sentence.lower())
            
            if not sent_words:
                ppl_scores.append(1.0)
                continue
                
            # Calculate how unexpected this sentence is given previous context
            context_until_now = " ".join(sentences[:i])
            context_words = re.findall(r'\w+', context_until_now.lower())
            context_counts = Counter(context_words)
            
            # Check how many words in this sentence are in the context
            context_word_count = sum(1 for word in sent_words if word in context_counts)
            context_ratio = context_word_count / len(sent_words) if sent_words else 0
            
            # Sentences with less contextual words have higher perplexity
            perplexity = 1.0 / (context_ratio + 0.1)  # Add small constant to avoid division by zero
            ppl_scores.append(perplexity)
        
        # Normalize scores for visualization
        if ppl_scores:
            min_score = min(ppl_scores)
            max_score = max(ppl_scores)
            if min_score != max_score:
                ppl_scores = [(score - min_score) / (max_score - min_score) * 5 + 1 for score in ppl_scores]
        
        return ppl_scores
    
    @staticmethod
    def plot_perplexity_and_chunks(text: str, meta_chunker: MetaChunker, output_file: str = None):
        """
        Plot perplexity scores and chunk boundaries.
        
        Args:
            text: Text to analyze
            meta_chunker: MetaChunker instance
            output_file: Optional file to save the plot
        """
        sentences = nltk.sent_tokenize(text)
        ppl_scores = ChunkVisualizer.calculate_ppl_scores(sentences)
        
        # Find chunk boundaries based on perplexity
        chunk_boundaries = []
        threshold = meta_chunker.threshold
        
        for i in range(1, len(ppl_scores) - 1):
            # Check if this point is a local minimum in perplexity
            is_minimum = ppl_scores[i] < ppl_scores[i-1] and ppl_scores[i] < ppl_scores[i+1]
            
            # Check if the difference exceeds threshold
            left_diff = ppl_scores[i-1] - ppl_scores[i]
            right_diff = ppl_scores[i+1] - ppl_scores[i]
            
            if is_minimum and (left_diff > threshold or right_diff > threshold):
                chunk_boundaries.append(i)
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot perplexity scores
        plt.plot(ppl_scores, marker='o', label='Perplexity Score')
        
        # Mark chunk boundaries
        for boundary in chunk_boundaries:
            plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.5)
        
        # Add threshold reference line
        plt.axhline(y=threshold, color='g', linestyle='-.', label=f'Threshold: {threshold}')
        
        # Add vertical lines at paragraph breaks (approximated by double newlines in the original text)
        paragraphs = text.split('\n\n')
        current_sent_idx = 0
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            para_sentences = nltk.sent_tokenize(paragraph)
            current_sent_idx += len(para_sentences)
            if current_sent_idx < len(sentences):
                plt.axvline(x=current_sent_idx, color='blue', linestyle=':', alpha=0.3)
        
        plt.title('Sentence Perplexity and Chunk Boundaries')
        plt.xlabel('Sentence Index')
        plt.ylabel('Perplexity Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file)
            print(f"Perplexity plot saved to {output_file}")
        else:
            plt.show()

def traditional_chunk_indices(text: str, chunk_size: int = 4000) -> list:
    """
    Find chunk boundary indices using traditional chunking.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size
        
    Returns:
        List of sentence indices where chunks end
    """
    sentences = nltk.sent_tokenize(text)
    boundaries = []
    
    current_chunk_size = 0
    for i, sentence in enumerate(sentences):
        sentence_size = len(sentence)
        
        if current_chunk_size + sentence_size > chunk_size and current_chunk_size > 0:
            boundaries.append(i - 1)  # End chunk at previous sentence
            current_chunk_size = sentence_size
        else:
            current_chunk_size += sentence_size
    
    # Add the last boundary if there's content after the last boundary
    if boundaries and boundaries[-1] < len(sentences) - 1:
        boundaries.append(len(sentences) - 1)
        
    return boundaries

async def find_meta_chunk_boundaries(text: str, meta_chunker: MetaChunker) -> list:
    """
    Find chunk boundary indices using Meta-Chunking.
    
    Args:
        text: Text to chunk
        meta_chunker: MetaChunker instance
        
    Returns:
        List of sentence indices where chunks end
    """
    sentences = nltk.sent_tokenize(text)
    
    # Calculate perplexity scores
    ppl_scores = ChunkVisualizer.calculate_ppl_scores(sentences)
    
    # Find chunk boundaries based on perplexity
    boundaries = []
    for i in range(1, len(ppl_scores) - 1):
        # Check if this point is a local minimum in perplexity
        is_minimum = ppl_scores[i] < ppl_scores[i-1] and ppl_scores[i] < ppl_scores[i+1]
        
        # Check if the difference exceeds threshold
        left_diff = ppl_scores[i-1] - ppl_scores[i]
        right_diff = ppl_scores[i+1] - ppl_scores[i]
        
        if is_minimum and (left_diff > meta_chunker.threshold or right_diff > meta_chunker.threshold):
            boundaries.append(i)
    
    # Add the last boundary if not already present
    if boundaries and boundaries[-1] < len(sentences) - 1:
        boundaries.append(len(sentences) - 1)
        
    return boundaries

async def run_demo():
    """Run the Meta-Chunking demonstration."""
    print(f"{Fore.YELLOW}Meta-Chunking Demonstration{Style.RESET_ALL}")
    print("This demo will compare traditional chunking with Meta-Chunking")
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    print("\nInitializing MetaChunker...")
    meta_chunker = MetaChunker(threshold=0.5, target_chunk_size=4000)
    
    # Process the first sample
    print(f"\n{Fore.YELLOW}Processing sample 1 (General AI text){Style.RESET_ALL}")
    
    # Find chunk boundaries using both methods
    traditional_boundaries = traditional_chunk_indices(SAMPLE_TEXT)
    meta_boundaries = await find_meta_chunk_boundaries(SAMPLE_TEXT, meta_chunker)
    
    # Display chunk results
    ChunkVisualizer.display_chunks(SAMPLE_TEXT, traditional_boundaries, "Traditional Chunking")
    ChunkVisualizer.display_chunks(SAMPLE_TEXT, meta_boundaries, "Meta-Chunking")
    
    # Generate perplexity visualization
    print(f"\n{Fore.YELLOW}Generating perplexity visualization for sample 1...{Style.RESET_ALL}")
    ChunkVisualizer.plot_perplexity_and_chunks(SAMPLE_TEXT, meta_chunker, "sample1_perplexity.png")
    
    # Process the second sample
    print(f"\n{Fore.YELLOW}Processing sample 2 (Technical text on quantum computing){Style.RESET_ALL}")
    
    # Find chunk boundaries using both methods for second sample
    traditional_boundaries2 = traditional_chunk_indices(TECHNICAL_SAMPLE)
    meta_boundaries2 = await find_meta_chunk_boundaries(TECHNICAL_SAMPLE, meta_chunker)
    
    # Display chunk results for second sample
    ChunkVisualizer.display_chunks(TECHNICAL_SAMPLE, traditional_boundaries2, "Traditional Chunking")
    ChunkVisualizer.display_chunks(TECHNICAL_SAMPLE, meta_boundaries2, "Meta-Chunking")
    
    # Generate perplexity visualization for second sample
    print(f"\n{Fore.YELLOW}Generating perplexity visualization for sample 2...{Style.RESET_ALL}")
    ChunkVisualizer.plot_perplexity_and_chunks(TECHNICAL_SAMPLE, meta_chunker, "sample2_perplexity.png")
    
    # Compare statistics
    print(f"\n{Fore.YELLOW}Chunking Statistics:{Style.RESET_ALL}")
    print(f"Sample 1 (General AI):")
    print(f"  Traditional Chunking: {len(traditional_boundaries)} chunks")
    print(f"  Meta-Chunking: {len(meta_boundaries)} chunks")
    print(f"Sample 2 (Quantum Computing):")
    print(f"  Traditional Chunking: {len(traditional_boundaries2)} chunks")
    print(f"  Meta-Chunking: {len(meta_boundaries2)} chunks")
    
    print(f"\n{Fore.GREEN}Demo completed successfully!{Style.RESET_ALL}")
    print("The perplexity plots have been saved as 'sample1_perplexity.png' and 'sample2_perplexity.png'")

if __name__ == "__main__":
    asyncio.run(run_demo())