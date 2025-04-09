"""
AI-powered text summarization module with enhanced attribution.

This module provides functions for summarizing web content using
OpenAI's API with query-specific prompt optimization and improved source attribution.
"""
import re
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union

import openai
from quart import current_app

from modules.utils.types import ProcessedSource, SourceSummary, SummaryMetadata
from modules.utils.errors import (
    SummarizerError, ModelAPIError, PromptGenerationError, handle_async_exceptions
)
from modules.utils.logging import get_logger, log_async_function_call

# Create a logger for this module
logger = get_logger(__name__)

# Query type patterns for prompt customization
QUERY_TYPES = {
    'factual': [
        r'what is', r'who is', r'where is', r'when was', r'why does',
        r'explain', r'describe', r'definition of', r'facts about',
        r'how does', r'history of'
    ],
    'comparison': [
        r'compare', r'difference between', r'similarities between',
        r'versus', r'vs', r'pros and cons', r'advantages and disadvantages'
    ],
    'instructional': [
        r'how to', r'steps to', r'guide to', r'tutorial', r'instructions for',
        r'ways to', r'methods to', r'process of', r'procedure for'
    ],
    'opinion': [
        r'review of', r'opinion on', r'thoughts on', r'analysis of',
        r'evaluation of', r'assessment of', r'critique of', r'perspective on',
        r'best', r'worst', r'should', r'worth it'
    ]
}

def get_openai_client():
    """Get an initialized OpenAI client using the API key from config."""
    api_key = current_app.config.get('OPENAI_API_KEY')
    if not api_key:
        raise ModelAPIError("OpenAI API key not configured")
    return openai.OpenAI(api_key=api_key)

@handle_async_exceptions(SummarizerError, "Failed to generate summary")
async def generate_summary(
    query: str, 
    processed_content: List[ProcessedSource], 
    length: str = 'medium'
) -> Tuple[str, SummaryMetadata]:
    """
    Generate a summary of processed content using OpenAI's API, tracking chunk usage.
    
    Args:
        query: The original search query
        processed_content: Processed content organized by source, including chunks
        length: Desired summary length ('short', 'medium', 'long')
        
    Returns:
        Tuple of (summary text, metadata including used chunks)
        
    Raises:
        SummarizerError: If summary generation fails
    """
    if not processed_content:
        raise SummarizerError("No content provided for summarization")
        
    try:
        # Get OpenAI client
        client = get_openai_client()
        
        # Determine the max tokens based on length
        max_tokens = {
            'short': 150,
            'medium': 300,
            'long': 500
        }.get(length, 300)
        
        # Determine the model
        model = current_app.config.get('SUMMARY_MODEL', 'gpt-3.5-turbo')
        
        # Detect query type for prompt customization
        query_type = detect_query_type(query)
        logger.info(f"Detected query type: {query_type}")
        
        # Flatten all chunks from all sources for easier tracking
        all_chunks = []
        total_content_length = 0
        source_metadata = []

        # Check if processed_content actually contains chunks
        if not processed_content or not isinstance(processed_content[0].get('chunks'), list):
             logger.error("Processed content does not contain 'chunks'. Cannot proceed with chunk-based summarization.")
             raise SummarizerError("Invalid content format: Missing chunks.")

        for source in processed_content:
            # Ensure chunks have necessary keys
            if not all(isinstance(chunk, dict) and 'chunk_id' in chunk and 'content' in chunk and 'source_url' in chunk and 'source_title' in chunk for chunk in source.get('chunks', [])):
                 logger.error(f"Invalid chunk format found in source: {source.get('url')}")
                 raise SummarizerError("Invalid chunk format.")
            
            all_chunks.extend(source['chunks'])
            total_content_length += source.get('original_length', 0)
            source_metadata.append({
                 'url': source['url'], 
                 'title': source['title'], 
                 'quality_score': source.get('quality_score', 0)
            })

        if not all_chunks:
            raise SummarizerError("No chunks found in processed content.")

        # Enhanced system prompts with stronger attribution instructions
        system_prompts = {
             'factual': """
             You are a comprehensive research assistant that synthesizes factual information from provided text chunks.
             
             IMPORTANT ATTRIBUTION INSTRUCTIONS:
             1. For EVERY statement or fact in your summary, cite the specific chunk ID using [Source: chunk_id].
             2. NEVER make a statement without citing its source chunk.
             3. Citations MUST be placed immediately after the specific information they support.
             4. If multiple chunks support a statement, cite all relevant chunks, e.g., [Source: chunk_id1, chunk_id2].
             5. Do not combine information from different chunks without clearly attributing each piece.
             6. If information appears in multiple chunks, cite the chunk with the most complete information.
             
             Additional Guidelines:
             - Synthesize information accurately without distorting the original meaning.
             - Focus on clarity, accuracy, and conciseness.
             - Maintain a neutral, informative tone.
             - Output ONLY the summary with citations.
             - Format the summary using markdown.
             """,
             'comparison': """
             You are a balanced analysis assistant that synthesizes comparison information from provided text chunks.
             
             IMPORTANT ATTRIBUTION INSTRUCTIONS:
             1. For EVERY comparison point in your summary, cite the specific chunk ID using [Source: chunk_id].
             2. NEVER make a statement without citing its source chunk.
             3. Citations MUST be placed immediately after the specific information they support.
             4. If multiple chunks support a statement, cite all relevant chunks, e.g., [Source: chunk_id1, chunk_id2].
             5. For contrasting viewpoints, clearly attribute each perspective to its source.
             6. When information appears in multiple chunks, cite the most authoritative or detailed source.

             Additional Guidelines:
             - Organize the summary around key points of comparison relevant to the user query.
             - Present similarities and differences clearly.
             - Maintain neutrality when presenting different perspectives.
             - Output ONLY the summary with citations.
             - Format the summary using markdown.
             """,
             'instructional': """
             You are a clear instructional assistant that synthesizes how-to information from provided text chunks.
             
             IMPORTANT ATTRIBUTION INSTRUCTIONS:
             1. For EVERY step or instruction in your summary, cite the specific chunk ID using [Source: chunk_id].
             2. NEVER include a step or instruction without citing its source chunk.
             3. Citations MUST be placed immediately after the specific information they support.
             4. If multiple chunks support a step, cite all relevant chunks, e.g., [Source: chunk_id1, chunk_id2].
             5. For alternative approaches to the same task, clearly attribute each method to its source.
             6. When steps appear in multiple chunks, cite the source that explains it most clearly.

             Additional Guidelines:
             - Present information as a cohesive guide addressing the user query.
             - Organize steps logically in a sequence.
             - Focus on practical, actionable information.
             - Output ONLY the guide with citations.
             - Format the instructions using markdown with numbered steps.
             """,
             'opinion': """
             You are a balanced review assistant that synthesizes opinions and evaluations from provided text chunks.
             
             IMPORTANT ATTRIBUTION INSTRUCTIONS:
             1. For EVERY opinion or evaluation in your summary, cite the specific chunk ID using [Source: chunk_id].
             2. NEVER state an opinion without citing its source chunk.
             3. Citations MUST be placed immediately after the specific opinion they support.
             4. If multiple chunks express the same opinion, cite all relevant chunks, e.g., [Source: chunk_id1, chunk_id2].
             5. For contrasting opinions, clearly attribute each perspective to its source.
             6. When presenting consensus views, cite all chunks that express that view.

             Additional Guidelines:
             - Present the range of opinions relevant to the user query.
             - Note consensus or disagreement where applicable.
             - Maintain neutrality while presenting different viewpoints.
             - Output ONLY the summary with citations.
             - Format the summary using markdown with clear sections.
             """
         }
        system_prompt = system_prompts.get(query_type, system_prompts['factual'])

        # Enhanced user prompt with explicit citation formatting instructions
        user_prompt = f"""
        The user searched for: "{query}"

        Synthesize the following text chunks to create a summary answering the query.
        
        CITATION FORMATTING RULES:
        1. EVERY fact, statement, or opinion MUST be followed by a citation in this format: [Source: chunk_id]
        2. For information from multiple chunks, use: [Source: chunk_id1, chunk_id2]
        3. Place citations immediately after the specific statement they support
        4. Do not place citations at the end of paragraphs - each claim needs its own citation
        5. Citations must be inline (not as footnotes or endnotes)
        
        Here are the chunks with their IDs:
        """

        # Add chunks with IDs to the prompt
        chunk_texts = []
        for chunk in all_chunks:
             if len(chunk.get('content', '')) > 10:
                 chunk_texts.append(f"CHUNK {chunk['chunk_id']}: {chunk['content']}")
        
        if not chunk_texts:
            raise SummarizerError("No suitable chunks found after filtering.")

        user_prompt += "\n\n" + "\n\n".join(chunk_texts)
        
        user_prompt += f"""

        FINAL REMINDERS:
        - Create a coherent summary answering "{query}".
        - Cite EVERY piece of information with its source chunk ID.
        - Use the exact format [Source: chunk_id] for citations.
        - Only include information that appears in the provided chunks.
        - If the chunks contain conflicting information, acknowledge the different perspectives and cite the sources.
        - Output ONLY the summary with citations.
        """

        # Call OpenAI API - FIX: Use the correct method for API call
        logger.info(f"Generating summary using {model} with {len(all_chunks)} total chunks.")
        
        # FIX: Instead of using await with .create(), use a synchronous call
        response = client.chat.completions.create(
             model=model,
             messages=[
                 {"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}
             ],
             max_tokens=max_tokens,
             temperature=0.5,  # Lower temperature for more consistent citations
         )
        
        combined_summary = response.choices[0].message.content.strip()
        logger.info(f"Raw summary generated. Length: {len(combined_summary)}")

        # Extract used chunk IDs from the summary with improved regex
        # This regex handles both single citations [Source: id] and multiple citations [Source: id1, id2]
        chunk_references = re.findall(r'\[Source:\s*(.*?)(?=\])', combined_summary)
        
        # Process to handle multiple chunk IDs in a single citation
        referenced_chunk_ids = set()
        for ref in chunk_references:
            # Split by comma and clean up each ID
            ids = [chunk_id.strip() for chunk_id in ref.split(',')]
            for chunk_id in ids:
                if chunk_id:  # Ensure we're not adding empty strings
                    referenced_chunk_ids.add(chunk_id)
        
        logger.info(f"Found {len(referenced_chunk_ids)} unique chunk references: {referenced_chunk_ids}")

        # Create list of used chunks with metadata
        used_chunks_map = {chunk['chunk_id']: chunk for chunk in all_chunks}
        used_chunks_details = []
        found_ids = set()

        for chunk_id in referenced_chunk_ids:
            if chunk_id in used_chunks_map:
                chunk = used_chunks_map[chunk_id]
                used_chunks_details.append({
                    "chunk_id": chunk_id,
                    "source_url": chunk['source_url'],
                    "source_title": chunk['source_title'],
                    "content": chunk['content'],
                    "relevance_score": calculate_relevance_score(chunk, combined_summary)
                })
                found_ids.add(chunk_id)
            else:
                 logger.warning(f"Referenced chunk_id '{chunk_id}' not found in the original chunks list.")
        
        if len(referenced_chunk_ids) > 0 and not used_chunks_details:
             logger.warning("Chunk references were found in the summary, but none matched the provided chunk IDs.")

        # Create sentence-level attribution mapping
        attribution_mapping = analyze_sentence_attribution(combined_summary, all_chunks)

        # Prepare metadata
        metadata: SummaryMetadata = {
            'sources_count': len(processed_content),
            'total_content_length': total_content_length,
            'generated_at': time.time(),
            'model_used': model,
            'query_type': query_type,
            'sources': source_metadata,
            'used_chunks': used_chunks_details,
            'referenced_chunk_ids': list(referenced_chunk_ids),
            'attribution_mapping': attribution_mapping
        }
        
        return combined_summary, metadata
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise ModelAPIError(f"OpenAI API error: {str(e)}")
    except SummarizerError as e:
        logger.error(f"Summarization error: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error generating summary: {str(e)}")
        raise SummarizerError(f"An unexpected error occurred: {str(e)}")

def detect_query_type(query: str) -> str:
    """
    Detect the type of query to customize the prompt.
    
    Args:
        query: User search query
        
    Returns:
        Query type ('factual', 'comparison', 'instructional', or 'opinion')
    """
    query = query.lower()
    
    for query_type, patterns in QUERY_TYPES.items():
        for pattern in patterns:
            if re.search(rf'\b{pattern}\b', query):
                return query_type
    
    # Default to factual if no pattern matches
    return 'factual'

def calculate_relevance_score(chunk: Dict[str, Any], summary: str) -> float:
    """
    Calculate a relevance score for a chunk based on its usage in the summary.
    
    Args:
        chunk: The chunk data dictionary
        summary: The generated summary text
        
    Returns:
        A relevance score between 0 and 1
    """
    # Count occurrences of this chunk's ID in the summary
    chunk_id = chunk['chunk_id']
    citation_count = summary.count(f"[Source: {chunk_id}]")
    
    # Also count when it appears in multi-source citations
    pattern = r'\[Source:.*?' + re.escape(chunk_id) + r'.*?\]'
    multi_citations = re.findall(pattern, summary)
    citation_count += len(multi_citations)
    
    # Calculate base score from citation frequency
    base_score = min(1.0, citation_count / 5)  # Cap at 1.0, with 5 citations being "maximum relevance"
    
    # Adjust score based on chunk quality if available
    quality_factor = chunk.get('quality_score', 0.5)
    
    # Calculate weighted score
    return 0.7 * base_score + 0.3 * quality_factor

def analyze_sentence_attribution(summary: str, chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Analyze which sentences in the summary are attributed to which chunks.
    
    Args:
        summary: The generated summary text
        chunks: List of all available chunks
        
    Returns:
        Dictionary mapping sentence indices to lists of chunk IDs
    """
    # Create a mapping of chunk IDs to their content for quick lookup
    chunk_content_map = {chunk['chunk_id']: chunk['content'] for chunk in chunks}
    
    # Split summary into sentences
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    sentences = nltk.sent_tokenize(summary)
    
    # Analyze each sentence for citations
    attribution_map = {}
    for i, sentence in enumerate(sentences):
        # Find all citations in this sentence
        citations = re.findall(r'\[Source:\s*(.*?)(?=\])', sentence)
        
        # Process multiple chunk IDs in citations
        chunk_ids = set()
        for citation in citations:
            ids = [chunk_id.strip() for chunk_id in citation.split(',')]
            chunk_ids.update(id for id in ids if id)
        
        if chunk_ids:
            attribution_map[str(i)] = list(chunk_ids)
    
    return attribution_map