# tests/test_meta_chunking.py
import pytest
from modules.meta_chunking import MetaChunker
import nltk

class TestMetaChunking:
    """Tests for the Meta-Chunking module."""
    
    @pytest.fixture
    def sample_text(self):
        return """
        The Web Summarizer application processes content from multiple sources. It searches the web for relevant documents.
        These documents are then extracted and cleaned. The cleaning process removes HTML tags and other noise.
        
        Next, the application analyzes the content for quality. High-quality sources are given more weight in the final summary.
        This analysis considers factors like readability, citations, and domain reputation.
        
        After processing, the system generates a comprehensive summary. The summary synthesizes information from all sources.
        It highlights key points and maintains logical connections between ideas. This approach ensures that users receive
        accurate and concise information about their query.
        """
    
    def test_meta_chunker_initialization(self):
        """Test the initialization of MetaChunker with various parameters."""
        chunker = MetaChunker()
        assert chunker.threshold == 0.5
        assert chunker.max_tokens_per_chunk == 4000
        assert chunker.use_dynamic_combination == True
        
        custom_chunker = MetaChunker(threshold=0.8, max_tokens_per_chunk=2000, use_dynamic_combination=False)
        assert custom_chunker.threshold == 0.8
        assert custom_chunker.max_tokens_per_chunk == 2000
        assert custom_chunker.use_dynamic_combination == False
    
    def test_pseudo_perplexity_calculation(self, sample_text):
        """Test the calculation of pseudo-perplexity scores."""
        chunker = MetaChunker()
        sentences = nltk.sent_tokenize(sample_text)
        
        perplexities = chunker._calculate_pseudo_perplexities(sentences)
        
        # Basic validation
        assert len(perplexities) == len(sentences)
        assert all(p > 0 for p in perplexities)
        
        # Check that sentences with connectives tend to have lower perplexity
        sentences_with_connectives = [i for i, s in enumerate(sentences) if any(c in s.lower().split()[:3] for c in ["and", "but", "however", "therefore"])]
        other_sentences = [i for i in range(len(sentences)) if i not in sentences_with_connectives and i > 0]
        
        if sentences_with_connectives and other_sentences:
            avg_connective_ppl = sum(perplexities[i] for i in sentences_with_connectives) / len(sentences_with_connectives)
            avg_other_ppl = sum(perplexities[i] for i in other_sentences) / len(other_sentences)
            
            # Sentences with connectives should generally have lower perplexity
            assert avg_connective_ppl <= avg_other_ppl * 1.2
    
    def test_chunk_boundary_detection(self):
        """Test the detection of chunk boundaries based on perplexity values."""
        chunker = MetaChunker(threshold=0.5)
        
        # Mock perplexity values with clear local minima
        perplexities = [5.0, 6.0, 3.0, 7.0, 8.0, 4.0, 9.0]
        
        boundaries = chunker._find_chunk_boundaries(perplexities)
        
        # The local minima are at indices 2 and 5
        assert 2 in boundaries
        assert 5 in boundaries
    
    def test_meta_chunking_output(self, sample_text):
        """Test the complete chunking process and validate the output."""
        chunker = MetaChunker(threshold=1.0)  # Higher threshold for clearer breaks
        
        chunks = chunker.chunk_text(sample_text)
        
        # Check basic properties
        assert len(chunks) > 0
        assert all(chunk.strip() for chunk in chunks)
        
        # Each chunk should be a complete sentence or set of sentences
        for chunk in chunks:
            sentences = nltk.sent_tokenize(chunk)
            assert len(sentences) > 0
            
        # Combine all chunks and compare to original (should maintain all content)
        combined = " ".join(chunks)
        original_sentences = set([s.strip() for s in nltk.sent_tokenize(sample_text) if s.strip()])
        chunked_sentences = set([s.strip() for s in nltk.sent_tokenize(combined) if s.strip()])
        assert original_sentences == chunked_sentences