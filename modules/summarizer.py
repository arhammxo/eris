"""
AI-powered text summarization module.

This module provides functions for summarizing web content using
OpenAI's API with query-specific prompt optimization.
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
        # Set OpenAI API key
        api_key = current_app.config.get('OPENAI_API_KEY')
        if not api_key:
            logger.error("OpenAI API key not found")
            raise ModelAPIError("API key not configured")
            
        client = openai.OpenAI(api_key=api_key)
        
        # Determine the max tokens based on length
        max_tokens = {
            'short': 150,
            'medium': 300,
            'long': 500
        }.get(length, 300) # Note: This might need adjustment based on the number of chunks
        
        # Determine the model
        model = current_app.config.get('SUMMARY_MODEL', 'gpt-3.5-turbo')
        
        # Detect query type for prompt customization (can still be useful for system prompt)
        query_type = detect_query_type(query)
        logger.info(f"Detected query type: {query_type}")
        
        # --- Start Edit 1: Flatten chunks and build new prompt ---
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
            total_content_length += source.get('original_length', 0) # Keep track of original total length
            source_metadata.append({ # Store source info for final metadata
                 'url': source['url'], 
                 'title': source['title'], 
                 'quality_score': source.get('quality_score', 0)
            })

        if not all_chunks:
            raise SummarizerError("No chunks found in processed content.")

        # Customize system prompt based on query type (similar to combined summary prompt)
        # This helps guide the overall synthesis style
        system_prompts = {
             'factual': """
             You are a comprehensive research assistant that synthesizes factual information from provided text chunks.
             Guidelines:
             1. Synthesize information from the chunks to answer the user query.
             2. For each significant point in your summary, cite the source chunk ID using [Source: chunk_id].
             3. Focus on accuracy, clarity, and conciseness.
             4. Maintain a neutral, informative tone.
             5. Output ONLY the summary with citations. Do not add introductory or concluding remarks.
             6. Output ALWAYS in a formated markdown view.
             """,
             'comparison': """
             You are a balanced analysis assistant that synthesizes comparison information from provided text chunks.
             Guidelines:
             1. Organize the summary around key points of comparison relevant to the user query.
             2. Present similarities and differences clearly.
             3. For each comparison point, cite the supporting chunk IDs using [Source: chunk_id].
             4. Maintain neutrality.
             5. Output ONLY the summary with citations.
             6. Output ALWAYS in a formated markdown view.
             """,
             'instructional': """
             You are a clear instructional assistant that synthesizes how-to information from provided text chunks.
             Guidelines:
             1. Present information as a cohesive guide addressing the user query.
             2. Organize steps logically.
             3. Cite the chunk ID for each step or key piece of information using [Source: chunk_id].
             4. Focus on practical, actionable information.
             5. Output ONLY the guide with citations.
             6. Output ALWAYS in a formated markdown view.
             """,
             'opinion': """
             You are a balanced review assistant that synthesizes opinions and evaluations from provided text chunks.
             Guidelines:
             1. Present the range of opinions relevant to the user query.
             2. Note consensus or disagreement where applicable.
             3. For each key opinion or evaluation, cite the source chunk ID using [Source: chunk_id].
             4. Maintain neutrality.
             5. Output ONLY the summary with citations.
             6. Output ALWAYS in a formated markdown view.
             """
         }
        system_prompt = system_prompts.get(query_type, system_prompts['factual'])

        # Construct the user prompt with chunks and instructions
        user_prompt = f"""
        The user searched for: "{query}"

        Synthesize the following text chunks to create a summary answering the query.
        For each significant point in your summary, include a reference to the chunk ID 
        that provided that information using the format [Source: chunk_id].

        Here are the chunks with their IDs:
        """

        # Add chunks with IDs to the prompt
        # Consider token limits here - this could become very large
        # A potential improvement is to select relevant chunks first
        chunk_texts = []
        for chunk in all_chunks:
             # Basic check for chunk content length
             if len(chunk.get('content', '')) > 10: # Ignore very short/empty chunks
                 chunk_texts.append(f"CHUNK {chunk['chunk_id']}: {chunk['content']}")
        
        # Check if adding chunks exceeds a reasonable prompt size (estimate tokens)
        # This is a rough estimate; precise token counting is better
        estimated_prompt_tokens = len(system_prompt.split()) + len(user_prompt.split()) + sum(len(c.split()) for c in chunk_texts)
        # Example: Leave ~1000 tokens headroom below model limit (e.g., 4096 for gpt-3.5-turbo)
        # MAX_PROMPT_TOKENS = 3000 
        # if estimated_prompt_tokens > MAX_PROMPT_TOKENS:
        #    logger.warning(f"Estimated prompt tokens ({estimated_prompt_tokens}) exceed limit. Truncating chunks.")
        #    # Implement truncation or selection logic here if needed
        #    pass 

        if not chunk_texts:
            raise SummarizerError("No suitable chunks found after filtering.")

        user_prompt += "\n\n" + "\n\n".join(chunk_texts)
        
        user_prompt += f"""

        Reminder: Create a coherent summary answering "{query}". Cite every piece of information 
        using the specific CHUNK ID in the format [Source: chunk_id].
        Output ONLY the summary with citations.
        """

        # Call OpenAI API (single call with all chunks)
        logger.info(f"Generating summary using {model} with {len(all_chunks)} total chunks.")
        response = await asyncio.to_thread(
             client.chat.completions.create,
             model=model,
             messages=[
                 {"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}
             ],
             max_tokens=max_tokens,
             temperature=0.6, # Slightly lower temp might help with citation accuracy
         )
        
        combined_summary = response.choices[0].message.content.strip()
        logger.info(f"Raw summary generated. Length: {len(combined_summary)}")

        # --- End Edit 1 ---

        # --- Start Edit 2: Extract chunk references and update metadata ---
        # Extract used chunk IDs from the summary
        # This regex finds patterns like [Source: anything_not_a_closing_bracket]
        chunk_references = re.findall(r'\[Source:\s*([^\]]+)\]', combined_summary)
        # Normalize IDs (e.g., remove leading/trailing spaces)
        referenced_chunk_ids = {ref.strip() for ref in chunk_references}
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
                    # Add relevance score later if needed
                })
                found_ids.add(chunk_id)
            else:
                 logger.warning(f"Referenced chunk_id '{chunk_id}' not found in the original chunks list.")
        
        if len(referenced_chunk_ids) > 0 and not used_chunks_details:
             logger.warning("Chunk references were found in the summary, but none matched the provided chunk IDs.")

        # Prepare metadata
        metadata: SummaryMetadata = {
            'sources_count': len(processed_content),
            'total_content_length': total_content_length, # Use calculated total length
            'generated_at': time.time(),
            'model_used': model,
            'query_type': query_type,
            'sources': source_metadata, # Use the stored source metadata
            'used_chunks': used_chunks_details, # List of dicts with chunk details
            'referenced_chunk_ids': list(referenced_chunk_ids) # List of unique string IDs found
        }
        # --- End Edit 2 ---
        
        return combined_summary, metadata
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise ModelAPIError(f"OpenAI API error: {str(e)}")
    except SummarizerError as e: # Catch specific internal errors
        logger.error(f"Summarization error: {str(e)}")
        raise # Re-raise specific error
    except Exception as e:
        logger.exception(f"Unexpected error generating summary: {str(e)}") # Log full traceback for unexpected errors
        raise SummarizerError(f"An unexpected error occurred: {str(e)}") # Wrap in SummarizerError

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