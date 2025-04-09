"""
Tests for the processor module.
"""
import pytest
from unittest.mock import patch, MagicMock

from modules.processor import (
    process_text,
    process_source,
    clean_text,
    extract_important_sentences,
    create_chunks
)
from modules.utils.errors import ProcessorError, TextAnalysisError

def test_clean_text():
    """Test text cleaning functionality."""
    raw_text = """
    This is Some Text with UPPERCASE and <html> tags.
    It has multiple   spaces and special characters like $%^&.
    Numbers like 123 should be removed.
    """
    
    cleaned = clean_text(raw_text)
    
    # Check lowercase conversion
    assert "UPPERCASE" not in cleaned
    assert "uppercase" in cleaned
    
    # Check HTML tag removal
    assert "<html>" not in cleaned
    
    # Check special character removal
    assert "$%^&" not in cleaned
    
    # Check whitespace normalization
    assert "  " not in cleaned
    
    # Check digit removal
    assert "123" not in cleaned

def test_extract_important_sentences():
    """Test extraction of important sentences."""
    text = """
    This is the first sentence with important keywords.
    Second sentence is just filler.
    Third sentence contains crucial information and keywords.
    Fourth sentence is more filler text.
    Fifth sentence has significant details and crucial information.
    """
    
    # Extract top 3 sentences
    important = extract_important_sentences(text, num_sentences=3)
    
    # Check that we got 3 sentences
    assert len(important) == 3
    
    # Check that sentences with important keywords are included
    assert any("important keywords" in s for s in important)
    assert any("crucial information" in s for s in important)
    assert any("significant details" in s for s in important)

def test_create_chunks():
    """Test text chunking functionality."""
    # Create a long text with multiple sentences
    sentences = ["This is sentence " + str(i) + "." for i in range(20)]
    text = " ".join(sentences)
    
    # Create chunks with a small chunk size to force multiple chunks
    chunks = create_chunks(text, chunk_size=50)
    
    # Check that we have multiple chunks
    assert len(chunks) > 1
    
    # Check that the total content is preserved
    combined = " ".join(chunks)
    assert len(combined) >= len(text)
    
    # Check that sentences aren't broken mid-sentence
    for chunk in chunks:
        assert chunk.startswith("This is sentence ")
        assert chunk.endswith(".")

@pytest.mark.asyncio
@patch('modules.processor.quality_scorer.score_content')
async def test_process_source(mock_score_content, mock_source):
    """Test processing of a single source."""
    # Setup mock
    mock_score_content.return_value = 0.75
    
    # Process the source
    result = await process_source(mock_source)
    
    # Check the result
    assert result is not None
    assert result['url'] == mock_source['url']
    assert result['title'] == mock_source['title']
    assert result['domain'] == mock_source['domain']
    assert 'cleaned_text' in result
    assert 'important_sentences' in result
    assert 'chunks' in result
    assert result['original_length'] == len(mock_source['content'])
    assert 'processed_length' in result
    assert result['quality_score'] == 0.75
    
    # Check that the content was cleaned
    assert result['cleaned_text'].islower()
    
    # Quality scorer should have been called
    mock_score_content.assert_called_once_with(mock_source)

@pytest.mark.asyncio
@patch('modules.processor.process_source')
async def test_process_text(mock_process_source, mock_source):
    """Test processing of multiple sources."""
    # Setup mock
    mock_process_result = {
        'url': mock_source['url'],
        'title': mock_source['title'],
        'domain': mock_source['domain'],
        'cleaned_text': 'cleaned content',
        'important_sentences': ['sentence 1', 'sentence 2'],
        'chunks': ['chunk 1', 'chunk 2'],
        'original_length': len(mock_source['content']),
        'processed_length': 14,
        'quality_score': 0.75
    }
    mock_process_source.return_value = mock_process_result
    
    # Process multiple sources
    sources = [mock_source, mock_source]
    results = await process_text(sources)
    
    # Check the results
    assert len(results) == 2
    assert results[0] == mock_process_result
    assert results[1] == mock_process_result
    
    # process_source should be called for each source
    assert mock_process_source.call_count == 2

@pytest.mark.asyncio
@patch('modules.processor.process_source')
async def test_process_text_with_errors(mock_process_source, mock_source):
    """Test processing with some failing sources."""
    # Setup mock to alternate between success and failure
    mock_process_result = {
        'url': mock_source['url'],
        'title': mock_source['title'],
        'domain': mock_source['domain'],
        'cleaned_text': 'cleaned content',
        'important_sentences': ['sentence 1', 'sentence 2'],
        'chunks': ['chunk 1', 'chunk 2'],
        'original_length': len(mock_source['content']),
        'processed_length': 14,
        'quality_score': 0.75
    }
    
    # Make the mock alternate between success and error
    mock_process_source.side_effect = [
        mock_process_result,
        ProcessorError("Test error"),
        mock_process_result
    ]
    
    # Process multiple sources
    sources = [mock_source, mock_source, mock_source]
    results = await process_text(sources)
    
    # Check the results - should only have successful ones
    assert len(results) == 2
    assert results[0] == mock_process_result
    assert results[1] == mock_process_result
    
    # process_source should be called for each source
    assert mock_process_source.call_count == 3

@pytest.mark.asyncio
@patch('modules.processor.extract_important_sentences')
async def test_process_source_with_errors(mock_extract_sentences, mock_source):
    """Test process_source handling of errors."""
    # Setup mock to raise an error
    mock_extract_sentences.side_effect = TextAnalysisError("Test error")
    
    # Process the source - should raise ProcessorError
    with pytest.raises(ProcessorError):
        await process_source(mock_source)
    
    # extract_important_sentences should have been called
    mock_extract_sentences.assert_called_once()

def test_create_chunks_with_meta_chunking():
    """Test the create_chunks function with Meta-Chunking enabled."""
    # Mock the app configuration
    with patch('quart.current_app') as mock_app:
        mock_app.config = {
            'META_CHUNKING_ENABLED': True,
            'META_CHUNKING_THRESHOLD': 0.5,
            'META_CHUNKING_DYNAMIC_COMBINATION': True
        }
        
        # Create a test text with distinct logical sections
        text = """
        Section 1: This is the first logical section. It contains related sentences that should stay together.
        These sentences all talk about the same topic and have strong connections.
        
        Section 2: This begins a new logical section. The topic has shifted significantly from the previous content.
        It introduces new concepts unrelated to what was discussed before.
        
        Section 3: Another distinct shift in topic occurs here. This content is separate from both previous sections.
        The logical flow of the document has clearly moved to a new area of discussion.
        """
        
        # Run chunking
        chunks = create_chunks(text, chunk_size=200)
        
        # Verify chunks maintain logical sections
        assert len(chunks) >= 3  # Should have at least 3 chunks for 3 logical sections
        
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.endswith('.') or chunk.endswith('?') or chunk.endswith('!')
        
        # Verify logical coherence - each chunk should primarily contain one section
        section_markers = ["Section 1:", "Section 2:", "Section 3:"]
        for chunk in chunks:
            # Count section markers in each chunk - should typically only have one
            markers_in_chunk = sum(1 for marker in section_markers if marker in chunk)
            assert markers_in_chunk <= 1