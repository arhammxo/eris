"""
Tests for the summarizer module.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import openai

from modules.summarizer import (
    generate_summary,
    detect_query_type,
    summarize_source,
    generate_combined_summary
)
from modules.utils.errors import SummarizerError, ModelAPIError

def test_detect_query_type():
    """Test query type detection."""
    # Test factual queries
    assert detect_query_type("What is artificial intelligence?") == "factual"
    assert detect_query_type("Who is Marie Curie?") == "factual"
    assert detect_query_type("Explain quantum physics") == "factual"
    
    # Test comparison queries
    assert detect_query_type("Compare React vs Angular") == "comparison"
    assert detect_query_type("What's the difference between prokaryotes and eukaryotes?") == "comparison"
    assert detect_query_type("Pros and cons of electric vehicles") == "comparison"
    
    # Test instructional queries
    assert detect_query_type("How to bake a cake") == "instructional"
    assert detect_query_type("Steps to apply for a passport") == "instructional"
    assert detect_query_type("Guide to basic programming") == "instructional"
    
    # Test opinion queries
    assert detect_query_type("Review of iPhone 14") == "opinion"
    assert detect_query_type("Best restaurants in New York") == "opinion"
    assert detect_query_type("Is Python worth learning?") == "opinion"
    
    # Test default behavior for ambiguous queries
    assert detect_query_type("Solar system") == "factual"
    assert detect_query_type("Climate change") == "factual"

@pytest.mark.asyncio
@patch('openai.OpenAI')
@patch('modules.summarizer.generate_combined_summary')
@patch('modules.summarizer.summarize_source')
async def test_generate_summary(mock_summarize_source, mock_combined_summary, mock_openai, 
                               mock_processed_source, test_config):
    """Test the main summary generation flow."""
    # Setup mocks
    source_summary = {
        'url': mock_processed_source['url'],
        'title': mock_processed_source['title'],
        'domain': mock_processed_source['domain'],
        'summary': 'This is a source summary.',
        'quality_score': 0.75
    }
    mock_summarize_source.return_value = source_summary
    
    combined_text = "This is the combined summary of all sources."
    mock_combined_summary.return_value = combined_text
    
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Create Quart app context with config
    with patch('quart.current_app') as mock_app:
        mock_app.config = test_config
        
        # Generate summary
        processed_sources = [mock_processed_source]
        summary, metadata = await generate_summary("What is artificial intelligence?", processed_sources)
        
        # Check the result
        assert summary == combined_text
        assert metadata['sources_count'] == 1
        assert metadata['query_type'] == 'factual'
        assert 'total_content_length' in metadata
        assert 'generated_at' in metadata
        assert 'model_used' in metadata
        assert 'sources' in metadata
        
        # Check that the summarize_source was called
        mock_summarize_source.assert_called_once()
        
        # Check that generate_combined_summary was called
        mock_combined_summary.assert_called_once()

@pytest.mark.asyncio
@patch('openai.OpenAI')
async def test_summarize_source(mock_openai, mock_processed_source):
    """Test summarization of a single source."""
    # Setup mock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "This is a summary of the source."
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Summarize source
    summary = await summarize_source(
        "What is artificial intelligence?",
        mock_processed_source,
        "factual",
        client=mock_client,
        model="gpt-3.5-turbo",
        max_tokens=100
    )
    
    # Check the result
    assert summary['url'] == mock_processed_source['url']
    assert summary['title'] == mock_processed_source['title']
    assert summary['domain'] == mock_processed_source['domain']
    assert summary['summary'] == "This is a summary of the source."
    assert summary['quality_score'] == mock_processed_source['quality_score']
    
    # Check that the API was called with the right parameters
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args['model'] == "gpt-3.5-turbo"
    assert call_args['max_tokens'] == 100
    assert call_args['temperature'] == 0.5
    assert len(call_args['messages']) == 2
    assert call_args['messages'][0]['role'] == "system"
    assert call_args['messages'][1]['role'] == "user"
    assert "The user searched for: \"What is artificial intelligence?\"" in call_args['messages'][1]['content']

@pytest.mark.asyncio
@patch('openai.OpenAI')
async def test_generate_combined_summary(mock_openai):
    """Test generation of combined summary from multiple sources."""
    # Setup mock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "This is a combined summary of all sources."
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Create source summaries
    source_summaries = [
        {
            'url': 'https://example.com/1',
            'title': 'Page 1',
            'domain': 'example.com',
            'summary': 'Summary of source 1.',
            'quality_score': 0.8
        },
        {
            'url': 'https://example.com/2',
            'title': 'Page 2',
            'domain': 'example.com',
            'summary': 'Summary of source 2.',
            'quality_score': 0.6
        }
    ]
    
    # Generate combined summary
    summary = await generate_combined_summary(
        "What is artificial intelligence?",
        source_summaries,
        "factual",
        client=mock_client,
        model="gpt-3.5-turbo",
        max_tokens=200
    )
    
    # Check the result
    assert summary == "This is a combined summary of all sources."
    
    # Check that the API was called with the right parameters
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args['model'] == "gpt-3.5-turbo"
    assert call_args['max_tokens'] == 200
    assert call_args['temperature'] == 0.7
    assert len(call_args['messages']) == 2
    assert call_args['messages'][0]['role'] == "system"
    assert call_args['messages'][1]['role'] == "user"
    assert "The user searched for: \"What is artificial intelligence?\"" in call_args['messages'][1]['content']
    assert "Source 1 (example.com, high quality)" in call_args['messages'][1]['content']
    assert "Source 2 (example.com, good quality)" in call_args['messages'][1]['content']

@pytest.mark.asyncio
@patch('openai.OpenAI')
async def test_summarize_source_api_error(mock_openai, mock_processed_source):
    """Test handling of API errors in source summarization."""
    # Setup mock to raise an error
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=openai.BadRequestError(message="API error", response=None))
    
    # Try to summarize source - should raise ModelAPIError
    with pytest.raises(ModelAPIError):
        await summarize_source(
            "What is artificial intelligence?",
            mock_processed_source,
            "factual",
            client=mock_client,
            model="gpt-3.5-turbo",
            max_tokens=100
        )

@pytest.mark.asyncio
@patch('openai.OpenAI')
@patch('modules.summarizer.generate_combined_summary')
@patch('modules.summarizer.summarize_source')
async def test_generate_summary_with_source_errors(mock_summarize_source, mock_combined_summary, 
                                                 mock_openai, mock_processed_source, test_config):
    """Test summary generation with some source summarization failures."""
    # Setup mocks to alternate between success and failure
    source_summary = {
        'url': mock_processed_source['url'],
        'title': mock_processed_source['title'],
        'domain': mock_processed_source['domain'],
        'summary': 'This is a source summary.',
        'quality_score': 0.75
    }
    
    mock_summarize_source.side_effect = [
        source_summary,
        ModelAPIError("API error"),
        source_summary
    ]
    
    combined_text = "This is the combined summary of all sources."
    mock_combined_summary.return_value = combined_text
    
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Create Quart app context with config
    with patch('quart.current_app') as mock_app:
        mock_app.config = test_config
        
        # Generate summary
        processed_sources = [mock_processed_source, mock_processed_source, mock_processed_source]
        summary, metadata = await generate_summary("What is artificial intelligence?", processed_sources)
        
        # Check the result - should still get a summary with placeholder for failed source
        assert summary == combined_text
        assert metadata['sources_count'] == 3
        
        # Check that summarize_source was called for each source
        assert mock_summarize_source.call_count == 3
        
        # Check that generate_combined_summary was called with correct summaries
        call_args = mock_combined_summary.call_args[0]
        assert len(call_args[1]) == 3  # All sources included
        assert "Failed to summarize this source" in call_args[1][1]['summary']  # Placeholder for failed source