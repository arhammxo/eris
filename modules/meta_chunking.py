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
from collections import Counter, defaultdict
import re
import math

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
        cache_size: int = 1000,
        ngram_order: int = 2
    ):
        """
        Initialize the MetaChunker.
        
        Args:
            threshold: Perplexity threshold for chunk boundaries
            target_chunk_size: Target size in characters for final chunks
            use_dynamic_combination: Whether to use dynamic combination for chunk size control
            cache_size: Maximum size of the KV cache for perplexity calculation
            ngram_order: Maximum n-gram order to use (2 for bigrams, 3 for trigrams)
        """
        self.threshold = threshold
        self.target_chunk_size = target_chunk_size
        self.use_dynamic_combination = use_dynamic_combination
        self.cache_size = cache_size
        self.ppl_cache = {}
        self.ngram_order = min(3, max(1, ngram_order))  # Limit between 1-3 for performance
        
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
        
        # Method 2: Using enhanced local approximation (faster, lower cost)
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
        Calculate an enhanced local approximation of perplexity without external API calls.
        
        This method uses n-gram models and statistical properties of the text to
        approximate perplexity without requiring an LLM API.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of perplexity scores for each sentence
        """
        # Ensure we have sentences
        if not sentences:
            return []
        if len(sentences) == 1:
            return [1.0]  # Single sentence gets neutral score
        
        # Preprocess all sentences for better tokenization
        processed_sentences = []
        for sentence in sentences:
            # Clean and normalize
            clean_text = sentence.lower()
            # Remove special characters but keep dots and commas which can be relevant
            clean_text = re.sub(r'[^\w\s.,?!]', '', clean_text)
            # Normalize whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            processed_sentences.append(clean_text)
        
        # Generate n-grams for all sentences
        ngrams_by_sentence = []
        for sentence in processed_sentences:
            words = self._tokenize_sentence(sentence)
            sentence_ngrams = []
            
            # Generate n-grams of different orders up to self.ngram_order
            for n in range(1, self.ngram_order + 1):
                if len(words) < n:
                    continue
                
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i:i+n])
                    sentence_ngrams.append(ngram)
            
            ngrams_by_sentence.append(sentence_ngrams)
        
        # Build n-gram model from the full text
        all_ngrams = [ngram for sentence_ngrams in ngrams_by_sentence for ngram in sentence_ngrams]
        ngram_counts = Counter(all_ngrams)
        total_ngrams = len(all_ngrams)
        
        # Calculate n-gram probabilities with Laplace smoothing
        vocabulary_size = len(set(ngram for ngram in all_ngrams if len(ngram) == 1))
        ngram_probs = {}
        
        for ngram, count in ngram_counts.items():
            if len(ngram) == 1:  # Unigram
                ngram_probs[ngram] = (count + 1) / (total_ngrams + vocabulary_size)
            else:  # Higher-order n-gram
                prefix = ngram[:-1]
                prefix_count = sum(1 for n in all_ngrams if n[:len(prefix)] == prefix)
                # Apply Laplace smoothing
                ngram_probs[ngram] = (count + 1) / (prefix_count + vocabulary_size)
        
        # Calculate perplexity scores using a sliding context window
        ppl_scores = []
        context_ngrams = Counter()  # Start with empty context
        
        for i, sentence_ngrams in enumerate(ngrams_by_sentence):
            if i == 0:
                # First sentence has no previous context
                ppl_scores.append(1.0)
                # Add it to context for next sentence
                context_ngrams.update(sentence_ngrams)
                continue
            
            # Calculate perplexity based on previous context
            if not sentence_ngrams:
                ppl_scores.append(1.0)  # Default score for empty sentence
                continue
            
            # Calculate sentence probability using n-gram model
            log_probability = 0.0
            sentence_length = 0
            
            for ngram in sentence_ngrams:
                sentence_length += 1
                
                # Calculate probability of this n-gram given context
                if ngram in context_ngrams:
                    # This n-gram exists in context, use context probability
                    probability = (context_ngrams[ngram] + 1) / (sum(context_ngrams.values()) + vocabulary_size)
                else:
                    # N-gram doesn't exist in context, use global probability with higher smoothing
                    probability = 1 / (sum(context_ngrams.values()) + vocabulary_size * 2)
                
                # Add log probability (avoid log(0))
                log_probability += math.log(max(probability, 1e-10))
            
            # Calculate perplexity: 2^(-log_prob/length)
            if sentence_length > 0:
                # Normalize by sentence length
                norm_log_prob = log_probability / sentence_length
                perplexity = 2 ** (-norm_log_prob)
            else:
                perplexity = 1.0  # Default for empty sentence
            
            ppl_scores.append(perplexity)
            
            # Update context with current sentence n-grams
            context_ngrams.update(sentence_ngrams)
            
            # Limit context size for efficiency with long texts
            if len(context_ngrams) > self.cache_size * 10:
                # Keep most frequent n-grams
                context_ngrams = Counter(dict(context_ngrams.most_common(self.cache_size)))
        
        # Normalize scores for better comparison and visualization
        if len(ppl_scores) > 1:
            min_score = min(ppl_scores)
            max_score = max(ppl_scores)
            if min_score != max_score:
                ppl_scores = [(score - min_score) / (max_score - min_score) * 5 + 1 for score in ppl_scores]
        
        # Enhance with additional cues from the text
        ppl_scores = self._enhance_with_discourse_cues(sentences, ppl_scores)
        
        return ppl_scores
    
    def _tokenize_sentence(self, sentence: str) -> List[str]:
        """
        Tokenize a sentence into words.
        
        Args:
            sentence: The sentence to tokenize
            
        Returns:
            List of words
        """
        # Simple tokenization by splitting on whitespace
        words = sentence.split()
        
        # Filter out very short tokens and convert to lowercase
        words = [word.lower() for word in words if len(word) > 1]
        
        return words
    
    def _enhance_with_discourse_cues(self, sentences: List[str], scores: List[float]) -> List[float]:
        """
        Enhance perplexity scores with discourse cues.
        
        This looks for discourse markers, topic shifts, and other linguistic features
        that can indicate logical boundaries.
        
        Args:
            sentences: List of original sentences
            scores: Initial perplexity scores
            
        Returns:
            Enhanced perplexity scores
        """
        # If we have less than 3 sentences, no need to enhance
        if len(sentences) < 3:
            return scores
        
        enhanced_scores = scores.copy()
        
        # Discourse markers that often indicate topic shifts
        topic_shift_markers = [
            r'\b(however|nevertheless|conversely|in contrast|on the other hand)\b',
            r'\b(furthermore|moreover|in addition|additionally)\b',
            r'\b(first|firstly|second|secondly|third|finally|lastly)\b',
            r'\b(for example|for instance|specifically|in particular)\b',
            r'\b(in conclusion|to summarize|in summary|to conclude)\b'
        ]
        
        # Check each sentence for discourse markers
        for i in range(1, len(sentences)):
            sentence = sentences[i].lower()
            
            # Check for discourse markers that suggest boundaries
            for marker_pattern in topic_shift_markers:
                if re.search(marker_pattern, sentence, re.IGNORECASE):
                    # Increase the score to make this more likely to be a boundary
                    enhanced_scores[i] *= 1.2
                    break
            
            # Check for quotes or reported speech which often indicate boundaries
            if (sentence.startswith('"') or sentence.startswith("'") or 
                sentence.startswith('"') or re.search(r'\bsaid\b|\bstated\b', sentence)):
                enhanced_scores[i] *= 1.15
            
            # Check for sentence starters that suggest a new thought
            if re.match(r'^(the|a|an|this|these|those|one|it)\b', sentence):
                enhanced_scores[i] *= 0.9  # Less likely to be a boundary
                
            # Check for connective words suggesting continuation
            if re.match(r'^(and|but|or|so|because|since)\b', sentence):
                enhanced_scores[i] *= 0.8  # Less likely to be a boundary
        
        # Look for sentence length shifts which can indicate structure changes
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        for i in range(1, len(sentences) - 1):
            prev_length = sentence_lengths[i-1]
            curr_length = sentence_lengths[i]
            next_length = sentence_lengths[i+1]
            
            # If this sentence is significantly different in length from neighbors
            if (abs(curr_length - prev_length) > 10 and abs(curr_length - next_length) > 10):
                enhanced_scores[i] *= 1.1  # Slightly increase boundary probability
        
        return enhanced_scores
    
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