"""
Meta-Chunking module for efficient text segmentation via logical perception.

This module implements Meta-Chunking, a method of text segmentation that identifies
logical boundaries within text using perplexity-based analysis to preserve
semantic coherence between sentences.
"""
import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
import nltk
from quart import current_app

from modules.utils.errors import ProcessorError, handle_async_exceptions
from modules.utils.logging import get_logger, log_async_function_call

# Create a logger for this module
logger = get_logger(__name__)

class MetaChunker:
    """
    Meta-Chunking implementation using perplexity-based analysis.
    
    This class provides methods to segment text based on its logical structure,
    maintaining the coherence of related sentences.
    """
    
    def __init__(
        self,
        threshold: float = 0.0,
        target_chunk_size: int = 4000,
        use_dynamic_combination: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize the MetaChunker.
        
        Args:
            threshold: Perplexity threshold for chunk boundaries
            target_chunk_size: Target size in characters for final chunks
            use_dynamic_combination: Whether to use dynamic combination for chunk size control
            cache_size: Maximum size of the KV cache for perplexity calculation
        """
        self.threshold = threshold
        self.target_chunk_size = target_chunk_size
        self.use_dynamic_combination = use_dynamic_combination
        self.cache_size = cache_size
        self.ppl_cache = {}
        
    async def chunk_text(self, text: str) -> List[str]:
        """
        Segment text into logical chunks using Meta-Chunking.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        # Split into sentences first
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
        
        # Calculate perplexity scores for sentences
        ppl_scores = await self._calculate_ppl_scores(sentences)
        
        # Find chunk boundaries based on perplexity distribution
        chunk_boundaries = self._find_chunk_boundaries(ppl_scores)
        
        # Create initial meta-chunks
        meta_chunks = self._create_meta_chunks(sentences, chunk_boundaries)
        
        # If dynamic combination is enabled, merge chunks to target size
        if self.use_dynamic_combination:
            final_chunks = self._combine_chunks_to_target_size(meta_chunks)
            return final_chunks
        
        return meta_chunks
    
    @log_async_function_call(logger)
    async def _calculate_ppl_scores(self, sentences: List[str]) -> List[float]:
        """
        Calculate perplexity scores for each sentence based on its context.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of perplexity scores
        """
        # Method 1: Using OpenAI API for perplexity (higher accuracy, higher cost)
        if current_app.config.get('USE_OPENAI_FOR_PPL', False):
            return await self._calculate_ppl_with_openai(sentences)
        
        # Method 2: Using local approximation (faster, lower cost)
        return await self._calculate_ppl_local(sentences)
    
    async def _calculate_ppl_with_openai(self, sentences: List[str]) -> List[float]:
        """
        Calculate perplexity using OpenAI API.
        """
        import openai
        from modules.summarizer import get_openai_client
        
        # Get OpenAI client
        client = get_openai_client()
        
        # Initialize scores list
        ppl_scores = []
        
        # Calculate perplexity for each sentence based on preceding context
        for i in range(len(sentences)):
            # Create context from previous sentences
            context = " ".join(sentences[:i]) if i > 0 else ""
            
            try:
                # Use OpenAI's logprobs to calculate perplexity
                response = await asyncio.to_thread(
                    client.completions.create,
                    model="gpt-3.5-turbo-instruct",  # Use instruct model for logprobs
                    prompt=context,
                    max_tokens=1,
                    temperature=0,
                    logprobs=1,
                    echo=False
                )
                
                # Extract log probability
                logprob = response.logprobs.token_logprobs[0] if response.logprobs else 0
                
                # Convert to perplexity: PPL = 2^(-logprob)
                perplexity = 2 ** (-logprob) if logprob else float('inf')
                ppl_scores.append(perplexity)
                
            except Exception as e:
                logger.warning(f"Error calculating perplexity with OpenAI: {str(e)}")
                # Fallback: assign average or previous score
                if ppl_scores:
                    ppl_scores.append(np.mean(ppl_scores))
                else:
                    ppl_scores.append(1.0)  # Default score
        
        return ppl_scores
    
    async def _calculate_ppl_local(self, sentences: List[str]) -> List[float]:
        """
        Calculate a local approximation of perplexity without external API calls.
        
        This method uses statistical properties of the text to approximate
        perplexity without requiring an LLM API.
        """
        import re
        from collections import Counter
        
        # Initialize scores
        ppl_scores = []
        
        # Create a simple N-gram model from the full text
        full_text = " ".join(sentences)
        words = re.findall(r'\w+', full_text.lower())
        
        # Count word frequencies
        word_counts = Counter(words)
        total_words = len(words)
        
        # Calculate a basic probability distribution
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
            context_ratio = context_word_count / len(sent_words)
            
            # Sentences with less contextual words have higher perplexity
            perplexity = 1.0 / (context_ratio + 0.1)  # Add small constant to avoid division by zero
            ppl_scores.append(perplexity)
        
        # Normalize scores
        if ppl_scores:
            min_score = min(ppl_scores)
            max_score = max(ppl_scores)
            if min_score != max_score:
                ppl_scores = [(score - min_score) / (max_score - min_score) * 5 + 1 for score in ppl_scores]
        
        return ppl_scores
    
    def _find_chunk_boundaries(self, ppl_scores: List[float]) -> List[int]:
        """
        Find chunk boundaries based on perplexity distribution.
        
        Args:
            ppl_scores: List of perplexity scores
            
        Returns:
            List of indices representing chunk boundaries
        """
        boundaries = []
        
        for i in range(1, len(ppl_scores) - 1):
            # Check if this point is a local minimum in perplexity
            is_minimum = ppl_scores[i] < ppl_scores[i-1] and ppl_scores[i] < ppl_scores[i+1]
            
            # Check if the difference exceeds threshold
            left_diff = ppl_scores[i-1] - ppl_scores[i]
            right_diff = ppl_scores[i+1] - ppl_scores[i]
            
            if is_minimum and (left_diff > self.threshold or right_diff > self.threshold):
                boundaries.append(i)
                
        # Always add the last sentence as a boundary
        if ppl_scores and len(ppl_scores) > 1:
            boundaries.append(len(ppl_scores) - 1)
            
        return boundaries
    
    def _create_meta_chunks(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """
        Create meta-chunks based on identified boundaries.
        
        Args:
            sentences: List of sentences
            boundaries: List of indices representing chunk boundaries
            
        Returns:
            List of text chunks
        """
        chunks = []
        start_idx = 0
        
        for boundary in boundaries:
            chunk_sentences = sentences[start_idx:boundary+1]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)
            start_idx = boundary + 1
            
        # Add remaining sentences if any
        if start_idx < len(sentences):
            chunk_text = " ".join(sentences[start_idx:])
            chunks.append(chunk_text)
            
        return chunks
    
    def _combine_chunks_to_target_size(self, chunks: List[str]) -> List[str]:
        """
        Combine meta-chunks to reach target chunk size.
        
        Args:
            chunks: List of meta-chunks
            
        Returns:
            List of combined chunks meeting target size
        """
        if not chunks:
            return []
            
        combined_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            # If adding this chunk would exceed target size, start a new combined chunk
            if len(current_chunk) + len(chunk) > self.target_chunk_size and current_chunk:
                combined_chunks.append(current_chunk)
                current_chunk = chunk
            else:
                # Add space separator if current_chunk is not empty
                separator = " " if current_chunk else ""
                current_chunk += separator + chunk
                
        # Add the last chunk if not empty
        if current_chunk:
            combined_chunks.append(current_chunk)
            
        return combined_chunks


# Convenience function for integration with the rest of the application
@handle_async_exceptions(ProcessorError, "Meta-chunking failed")
async def meta_chunk_text(
    text: str,
    threshold: float = None,
    target_chunk_size: int = None,
    use_dynamic_combination: bool = None
) -> List[str]:
    """
    Segment text into logical chunks using Meta-Chunking.
    
    Args:
        text: Input text to be chunked
        threshold: Perplexity threshold for chunk boundaries
        target_chunk_size: Target size in characters for final chunks
        use_dynamic_combination: Whether to use dynamic combination
        
    Returns:
        List of text chunks
    """
    # Get configuration from current_app if parameters not provided
    if threshold is None:
        threshold = current_app.config.get('META_CHUNKING_THRESHOLD', 0.0)
        
    if target_chunk_size is None:
        target_chunk_size = current_app.config.get('CHUNK_SIZE', 4000)
        
    if use_dynamic_combination is None:
        use_dynamic_combination = current_app.config.get('USE_DYNAMIC_COMBINATION', True)
    
    # Create chunker and process text
    chunker = MetaChunker(
        threshold=threshold,
        target_chunk_size=target_chunk_size,
        use_dynamic_combination=use_dynamic_combination
    )
    
    return await chunker.chunk_text(text)