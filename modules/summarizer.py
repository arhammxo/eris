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
    Generate a summary of processed content using OpenAI's API.
    
    Args:
        query: The original search query
        processed_content: Processed content organized by source
        length: Desired summary length ('short', 'medium', 'long')
        
    Returns:
        Tuple of (summary text, metadata)
        
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
        }.get(length, 300)
        
        # Determine the model
        model = current_app.config.get('SUMMARY_MODEL', 'gpt-3.5-turbo')
        
        # Detect query type for prompt customization
        query_type = detect_query_type(query)
        logger.info(f"Detected query type: {query_type}")
        
        # Generate summaries for each source
        summarization_tasks = []
        for source in processed_content:
            task = asyncio.create_task(
                summarize_source(
                    query, 
                    source, 
                    query_type,
                    client=client,
                    model=model, 
                    max_tokens=max_tokens // len(processed_content)
                )
            )
            summarization_tasks.append(task)
        
        # Wait for all summaries to complete
        source_summaries_results = await asyncio.gather(*summarization_tasks, return_exceptions=True)
        
        # Process results and filter out exceptions
        source_summaries: List[SourceSummary] = []
        for i, result in enumerate(source_summaries_results):
            if isinstance(result, Exception):
                logger.error(f"Error summarizing source: {str(result)}")
                # Add a placeholder for failed summaries
                source_summaries.append({
                    'url': processed_content[i]['url'],
                    'title': processed_content[i]['title'],
                    'domain': processed_content[i]['domain'],
                    'summary': "Failed to summarize this source.",
                    'quality_score': processed_content[i].get('quality_score', 0)
                })
            else:
                source_summaries.append(result)
        
        # Generate a final combined summary
        combined_summary = await generate_combined_summary(
            query,
            source_summaries,
            query_type,
            client=client,
            model=model,
            max_tokens=max_tokens
        )
        
        # Prepare metadata
        metadata: SummaryMetadata = {
            'sources_count': len(processed_content),
            'total_content_length': sum(s.get('original_length', 0) for s in processed_content),
            'generated_at': time.time(),
            'model_used': model,
            'query_type': query_type,
            'sources': [{
                'url': s['url'], 
                'title': s['title'], 
                'quality_score': s.get('quality_score', 0)
            } for s in processed_content]
        }
        
        return combined_summary, metadata
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise ModelAPIError(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise

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

@log_async_function_call(logger)
@handle_async_exceptions(ModelAPIError, "Failed to summarize source")
async def summarize_source(
    query: str,
    source: ProcessedSource,
    query_type: str,
    client: openai.OpenAI,
    model: str,
    max_tokens: int
) -> SourceSummary:
    """
    Summarize a single source with improved prompting.
    
    Args:
        query: Original search query
        source: Processed source content
        query_type: Type of query for prompt customization
        client: OpenAI client instance
        model: Model to use
        max_tokens: Maximum tokens for the response
        
    Returns:
        Source summary information
        
    Raises:
        ModelAPIError: If API call fails
    """
    try:
        # Get quality score to emphasize for higher quality sources
        quality_score = source.get('quality_score', 0.5)
        
        # Determine source credibility level
        credibility_level = "highly credible" if quality_score > 0.8 else \
                            "credible" if quality_score > 0.6 else \
                            "moderate credibility" if quality_score > 0.4 else \
                            "limited credibility"
        
        # Customize system prompt based on query type
        system_prompts = {
            'factual': f"""
            You are a precise, knowledgeable assistant that summarizes factual information.
            Focus on accuracy, clarity, and conciseness. Include specific data points, dates, 
            and definitions where relevant. This source has {credibility_level}.
            """,
            
            'comparison': f"""
            You are a detailed, analytical assistant that summarizes comparison information.
            Focus on identifying key similarities and differences, pros and cons, 
            and presenting balanced viewpoints. Structure your summary to highlight contrasting elements.
            This source has {credibility_level}.
            """,
            
            'instructional': f"""
            You are a clear, step-by-step assistant that summarizes instructional content.
            Focus on distilling processes into clear steps, highlighting key techniques,
            tools needed, and common challenges. This source has {credibility_level}.
            """,
            
            'opinion': f"""
            You are a balanced, thoughtful assistant that summarizes opinions and reviews.
            Focus on capturing key perspectives, noting if they represent consensus or divergent views,
            and maintaining neutrality while conveying the essence of evaluations.
            This source has {credibility_level}.
            """
        }
        
        system_prompt = system_prompts.get(query_type, system_prompts['factual'])
        
        # Construct prompt for the source
        prompt = f"""
        The user searched for: "{query}"
        
        Please summarize the following content from {source['title']} ({source['domain']}):
        
        {source['cleaned_text'][:3000]}
        
        Focus on information directly relevant to the user's query.
        If the source has high quality, emphasize its findings.
        Always maintain factual accuracy and include specific details when available.
        
        Summary:
        """
        
        # Call OpenAI API
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        
        return {
            'url': source['url'],
            'title': source['title'],
            'domain': source['domain'],
            'summary': response.choices[0].message.content.strip(),
            'quality_score': quality_score
        }
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error summarizing source {source['url']}: {str(e)}")
        raise ModelAPIError(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error summarizing source {source['url']}: {str(e)}")
        raise

@log_async_function_call(logger)
@handle_async_exceptions(ModelAPIError, "Failed to generate combined summary")
async def generate_combined_summary(
    query: str,
    source_summaries: List[SourceSummary],
    query_type: str,
    client: openai.OpenAI,
    model: str,
    max_tokens: int
) -> str:
    """
    Generate a combined summary from individual source summaries.
    
    Args:
        query: Original search query
        source_summaries: List of individual source summaries
        query_type: Type of query for prompt customization
        client: OpenAI client instance
        model: Model to use
        max_tokens: Maximum tokens for the response
        
    Returns:
        Combined summary text
        
    Raises:
        ModelAPIError: If API call fails
    """
    try:
        # Sort sources by quality score
        sorted_summaries = sorted(source_summaries, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Customize system prompt based on query type
        system_prompts = {
            'factual': """
            You are a comprehensive research assistant that synthesizes factual information from multiple sources.
            
            Guidelines:
            1. Prioritize information from higher quality sources.
            2. Highlight points of consensus across sources.
            3. Note any significant factual discrepancies between sources.
            4. Include relevant dates, statistics, and specific details.
            5. Maintain a neutral, informative tone.
            6. Cite sources when stating specific facts using (Source: domain).
            """,
            
            'comparison': """
            You are a balanced analysis assistant that synthesizes comparison information from multiple sources.
            
            Guidelines:
            1. Organize the summary around key points of comparison.
            2. Present both similarities and differences clearly.
            3. Include pros and cons from multiple perspectives.
            4. Highlight trade-offs where relevant.
            5. Avoid taking a side unless there's clear consensus across sources.
            6. Cite sources when stating specific comparisons using (Source: domain).
            """,
            
            'instructional': """
            You are a clear instructional assistant that synthesizes how-to information from multiple sources.
            
            Guidelines:
            1. Present information as a cohesive guide rather than separate summaries.
            2. Organize steps in a logical sequence.
            3. Include alternative approaches when mentioned by different sources.
            4. Highlight important cautions or tips.
            5. Focus on practical, actionable information.
            6. Cite sources for specific techniques using (Source: domain).
            """,
            
            'opinion': """
            You are a balanced review assistant that synthesizes opinions and evaluations from multiple sources.
            
            Guidelines:
            1. Present the range of opinions across sources.
            2. Note where there's consensus or disagreement.
            3. Include both positive and negative perspectives.
            4. Avoid amplifying extreme opinions unless they represent a significant viewpoint.
            5. Maintain neutrality while accurately conveying evaluations.
            6. Cite sources for specific opinions using (Source: domain).
            """
        }
        
        system_prompt = system_prompts.get(query_type, system_prompts['factual'])
        
        # Construct prompt for combined summary
        prompt = f"""
        The user searched for: "{query}"
        
        Here are summaries from multiple sources, ordered by credibility (highest first):
        
        """
        
        # Add each source summary
        for i, summary in enumerate(sorted_summaries, 1):
            quality_label = "high quality" if summary.get('quality_score', 0) > 0.8 else \
                           "good quality" if summary.get('quality_score', 0) > 0.6 else \
                           "medium quality" if summary.get('quality_score', 0) > 0.4 else \
                           "limited quality"
                           
            prompt += f"""
            Source {i} ({summary['domain']}, {quality_label}): 
            {summary['summary']}
            
            """
            
        prompt += f"""
        Please provide a comprehensive and accurate summary synthesizing all these sources,
        addressing the user's original query: "{query}"
        
        Remember to:
        1. Emphasize information from higher quality sources
        2. Note consensus and disagreements between sources
        3. Cite sources for specific facts using (Source: domain) format
        4. Organize the information logically 
        5. Maintain a balanced, helpful tone
        
        Combined Summary:
        """
        
        # Call OpenAI API
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error generating combined summary: {str(e)}")
        raise ModelAPIError(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating combined summary: {str(e)}")
        raise