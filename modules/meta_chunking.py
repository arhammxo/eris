"""
Meta-Chunking module for advanced text segmentation.

This module implements the Meta-Chunking strategy based on the paper
"META-CHUNKING: LEARNING EFFICIENT TEXT SEGMENTATION VIA LOGICAL PERCEPTION",
using perplexity-inspired heuristics to identify logical boundaries between chunks of text.
"""
import logging
import nltk
from typing import List, Optional
import re

from modules.utils.errors import ProcessorError, handle_exceptions
from modules.utils.logging import get_logger

# Create a logger for this module
logger = get_logger(__name__)

class MetaChunker:
    """
    MetaChunker implements perplexity-based text chunking that identifies
    logical boundaries in text by analyzing linguistic features.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        max_tokens_per_chunk: int = 4000,
        use_dynamic_combination: bool = True
    ):
        """
        Initialize the MetaChunker.
        
        Args:
            threshold: Minimum difference in perplexity to identify a chunk boundary
            max_tokens_per_chunk: Maximum size of a chunk in tokens
            use_dynamic_combination: Whether to use dynamic combination of meta-chunks
        """
        self.threshold = threshold
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.use_dynamic_combination = use_dynamic_combination
    
    @handle_exceptions(ProcessorError, "Meta-Chunking failed")
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using perplexity-inspired Meta-Chunking.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        
        if not sentences:
            return [text] if text else []
        
        # Calculate pseudo-perplexity for each sentence
        perplexities = self._calculate_pseudo_perplexities(sentences)
        
        # Identify chunk boundaries based on perplexity
        chunk_boundaries = self._find_chunk_boundaries(perplexities)
        
        # Create meta-chunks based on boundaries
        meta_chunks = self._create_meta_chunks(sentences, chunk_boundaries)
        
        # If using dynamic combination, combine meta-chunks to respect max size
        if self.use_dynamic_combination:
            return self._dynamic_combine_chunks(meta_chunks)
        else:
            return meta_chunks
    
    def _calculate_pseudo_perplexities(self, sentences: List[str]) -> List[float]:
        """
        Calculate pseudo-perplexity for each sentence based on heuristic features.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of pseudo-perplexity scores for each sentence
        """
        perplexities = []
        
        for i, sentence in enumerate(sentences):
            if i == 0:
                # First sentence has no context, assign a default perplexity
                perplexities.append(5.0)
                continue
                
            # Features that might indicate higher perplexity
            # 1. Sentence length (shorter sentences often have higher perplexity)
            length_score = 1.0 + 5.0 / max(len(sentence.split()), 1)
            
            # 2. Starting with connectives might indicate lower perplexity (more expected)
            connectives = ["and", "but", "or", "so", "because", "therefore", "thus", "however",
                          "although", "moreover", "furthermore", "additionally"]
            first_word = sentence.lower().split()[0] if sentence.split() else ""
            connective_factor = 0.8 if first_word in connectives else 1.0
            
            # 3. Topical shift can be detected by lack of common words with previous sentence
            prev_words = set(sentences[i-1].lower().split())
            current_words = set(sentence.lower().split())
            common_ratio = len(prev_words.intersection(current_words)) / max(len(prev_words.union(current_words)), 1)
            topic_shift_factor = 2.0 - common_ratio  # Higher score for less overlap
            
            # 4. Presence of entities, numbers, or special terms can increase perplexity
            has_numbers = bool(re.search(r'\d', sentence))
            has_special = bool(re.search(r'[A-Z][a-z]+', sentence))  # Potential proper nouns
            entity_factor = 1.2 if (has_numbers or has_special) else 1.0
            
            # Combine factors to create pseudo-perplexity
            pseudo_perplexity = length_score * connective_factor * topic_shift_factor * entity_factor
            
            perplexities.append(pseudo_perplexity)
        
        # Normalize the perplexities
        if perplexities:
            mean_ppl = sum(perplexities) / len(perplexities)
            perplexities = [p / mean_ppl * 5.0 for p in perplexities]  # Scale around 5.0
            
        return perplexities
    
    def _find_chunk_boundaries(self, perplexities: List[float]) -> List[int]:
        """
        Find chunk boundaries based on perplexity values.
        
        Args:
            perplexities: List of perplexity values for each sentence
            
        Returns:
            List of indices where chunks should be split
        """
        boundaries = []
        
        for i in range(1, len(perplexities)-1):
            # A boundary is a local minimum in perplexity beyond the threshold
            is_local_min = min(perplexities[i-1], perplexities[i+1]) - perplexities[i] > self.threshold
            is_descending_plateau = perplexities[i-1] - perplexities[i] > self.threshold and perplexities[i+1] == perplexities[i]
            
            if is_local_min or is_descending_plateau:
                boundaries.append(i)
        
        return boundaries
    
    def _create_meta_chunks(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """
        Create meta-chunks based on identified boundaries.
        
        Args:
            sentences: List of sentences
            boundaries: List of boundary indices
            
        Returns:
            List of meta-chunks
        """
        meta_chunks = []
        start_idx = 0
        
        # Add 1 to boundaries to get the actual split points
        split_points = [b + 1 for b in boundaries]
        
        # Add the end index
        split_points.append(len(sentences))
        
        for end_idx in split_points:
            chunk = " ".join(sentences[start_idx:end_idx])
            meta_chunks.append(chunk)
            start_idx = end_idx
        
        return meta_chunks
    
    def _dynamic_combine_chunks(self, meta_chunks: List[str]) -> List[str]:
        """
        Dynamically combine meta-chunks to respect maximum chunk size.
        
        Args:
            meta_chunks: List of meta-chunks
            
        Returns:
            List of combined chunks
        """
        combined_chunks = []
        current_chunk = ""
        
        for chunk in meta_chunks:
            # If adding the chunk would exceed the limit, start a new combined chunk
            if len(current_chunk) + len(chunk) > self.max_tokens_per_chunk and current_chunk:
                combined_chunks.append(current_chunk)
                current_chunk = chunk
            else:
                current_chunk += " " + chunk if current_chunk else chunk
        
        # Add the last chunk
        if current_chunk:
            combined_chunks.append(current_chunk)
        
        return combined_chunks