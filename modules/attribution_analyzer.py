"""
Attribution Analyzer for Web Summarizer.

This module provides functionality for analyzing the attribution of content
in summaries to their source chunks, improving the transparency of the summarization process.
"""
import re
import nltk
from typing import Dict, List, Optional, Set, Any, Tuple
import logging

from modules.utils.logging import get_logger, log_function_call
from modules.utils.errors import handle_exceptions

# Create a logger for this module
logger = get_logger(__name__)

class AttributionAnalyzer:
    """
    Analyzes attribution between summary content and source chunks.
    
    This class provides methods to:
    1. Identify which chunks contributed to which sentences
    2. Calculate confidence scores for attributions
    3. Detect potential attributions even when not explicitly cited
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize the AttributionAnalyzer.
        
        Args:
            similarity_threshold: Threshold for semantic similarity matching
        """
        self.similarity_threshold = similarity_threshold
        
        # Ensure required NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt', quiet=True)
    
    @log_function_call(logger)
    @handle_exceptions(ValueError, "Attribution analysis failed")
    def analyze_summary_attribution(
        self, 
        summary: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the attribution between a summary and its source chunks.
        
        Args:
            summary: The generated summary text
            chunks: List of chunk dictionaries with content and metadata
            
        Returns:
            Dictionary with attribution mapping and confidence scores
        """
        # Extract explicit citations from the summary
        citation_map = self._extract_citations(summary)
        
        # Split the summary into sentences
        sentences = nltk.sent_tokenize(summary)
        
        # Map each sentence to its explicit citations
        sentence_attribution = self._map_sentences_to_citations(sentences, citation_map)
        
        # Find additional attributions based on content similarity
        enhanced_attribution = self._analyze_semantic_attribution(sentences, chunks, sentence_attribution)
        
        # Calculate confidence scores for each attribution
        attribution_confidence = self._calculate_attribution_confidence(enhanced_attribution, chunks)
        
        return {
            'attribution_mapping': enhanced_attribution,
            'attribution_confidence': attribution_confidence
        }
    
    def _extract_citations(self, summary: str) -> Dict[int, List[str]]:
        """
        Extract explicit citations from the summary text.
        
        Args:
            summary: The summary text with citations
            
        Returns:
            Dictionary mapping character positions to chunk IDs
        """
        citation_map = {}
        
        # Find all citations with their positions
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        for match in re.finditer(citation_pattern, summary):
            start_pos = match.start()
            citation_text = match.group(1)
            
            # Handle multiple chunk IDs in a single citation
            chunk_ids = [chunk_id.strip() for chunk_id in citation_text.split(',')]
            citation_map[start_pos] = chunk_ids
        
        return citation_map
    
    def _map_sentences_to_citations(
        self, 
        sentences: List[str], 
        citation_map: Dict[int, List[str]]
    ) -> Dict[int, List[str]]:
        """
        Map each sentence to its explicit citations.
        
        Args:
            sentences: List of sentences from the summary
            citation_map: Dictionary mapping character positions to chunk IDs
            
        Returns:
            Dictionary mapping sentence indices to chunk IDs
        """
        sentence_attribution = {}
        
        # Track the character position
        current_pos = 0
        
        # For each sentence, find citations that fall within its range
        for i, sentence in enumerate(sentences):
            sentence_start = current_pos
            sentence_end = current_pos + len(sentence)
            
            # Find citations within this sentence's range
            sentence_chunks = set()
            for citation_pos, chunk_ids in citation_map.items():
                if sentence_start <= citation_pos < sentence_end:
                    sentence_chunks.update(chunk_ids)
            
            # Save attribution if any citations found
            if sentence_chunks:
                sentence_attribution[i] = list(sentence_chunks)
            
            # Update position for next sentence (add space for separator)
            current_pos = sentence_end + 1
        
        return sentence_attribution
    
    def _analyze_semantic_attribution(
        self, 
        sentences: List[str], 
        chunks: List[Dict[str, Any]], 
        explicit_attribution: Dict[int, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Find additional attributions based on content similarity.
        
        This method looks for semantic connections between sentences and chunks,
        even when not explicitly cited.
        
        Args:
            sentences: List of sentences from the summary
            chunks: List of chunk dictionaries
            explicit_attribution: Existing attributions from explicit citations
            
        Returns:
            Enhanced attribution mapping with additional connections
        """
        enhanced_attribution = {}
        
        # Create a simple term frequency model for each chunk
        chunk_term_maps = {}
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            content = chunk['content'].lower()
            terms = content.split()
            term_freq = {}
            for term in terms:
                if len(term) > 3:  # Only consider meaningful terms
                    term_freq[term] = term_freq.get(term, 0) + 1
            chunk_term_maps[chunk_id] = term_freq
        
        # For each sentence, look for semantic connections to chunks
        for i, sentence in enumerate(sentences):
            # Skip if already has attribution
            if i in explicit_attribution:
                enhanced_attribution[str(i)] = explicit_attribution[i]
                continue
            
            # Calculate similarity with each chunk
            sentence_chunks = []
            sentence_terms = set(term for term in sentence.lower().split() if len(term) > 3)
            
            for chunk_id, term_freq in chunk_term_maps.items():
                # Calculate term overlap
                overlap_terms = set(term_freq.keys()) & sentence_terms
                if len(overlap_terms) >= 2:  # At least 2 significant terms overlap
                    sentence_chunks.append(chunk_id)
            
            # Add this implicit attribution if found
            if sentence_chunks:
                enhanced_attribution[str(i)] = sentence_chunks
        
        return enhanced_attribution
    
    def _calculate_attribution_confidence(
        self, 
        attribution_mapping: Dict[str, List[str]], 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence scores for each attribution.
        
        Args:
            attribution_mapping: Dictionary mapping sentence indices to chunk IDs
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with confidence scores for each attribution
        """
        confidence_scores = {}
        
        # Create a lookup for chunks by ID
        chunk_map = {chunk['chunk_id']: chunk for chunk in chunks}
        
        for sentence_idx, chunk_ids in attribution_mapping.items():
            sentence_scores = {}
            
            for chunk_id in chunk_ids:
                # Base confidence on chunk quality score if available
                if chunk_id in chunk_map:
                    chunk = chunk_map[chunk_id]
                    base_score = chunk.get('quality_score', 0.5)
                else:
                    base_score = 0.3  # Lower confidence for unrecognized chunk IDs
                
                # Higher confidence for explicit citations (assumed for now)
                # In a more sophisticated version, we would check if this was explicit
                is_explicit = True  # Simplified assumption
                confidence = base_score * (1.2 if is_explicit else 0.8)
                
                # Cap at 1.0
                sentence_scores[chunk_id] = min(1.0, confidence)
            
            confidence_scores[sentence_idx] = sentence_scores
        
        return confidence_scores

def analyze_attribution(summary: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to analyze attribution for a summary.
    
    Args:
        summary: The summary text
        chunks: List of chunk dictionaries
        
    Returns:
        Attribution analysis results
    """
    analyzer = AttributionAnalyzer()
    return analyzer.analyze_summary_attribution(summary, chunks)