import openai
import logging
import time
from flask import current_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_summary(query, processed_content, length='medium'):
    """
    Generate a summary of the processed content using OpenAI's API.
    
    Args:
        query (str): The original search query
        processed_content (list): Processed content organized by source
        length (str): Desired summary length ('short', 'medium', 'long')
        
    Returns:
        tuple: (summary, metadata)
    """
    try:
        # Set OpenAI API key
        api_key = current_app.config.get('OPENAI_API_KEY')
        if not api_key:
            logger.error("OpenAI API key not found")
            return "Unable to generate summary: API key not configured.", {}
            
        client = openai.OpenAI(api_key=api_key)
        
        # Determine the max tokens based on length
        max_tokens = {
            'short': 150,
            'medium': 300,
            'long': 500
        }.get(length, 300)
        
        # Determine the model
        model = current_app.config.get('SUMMARY_MODEL', 'gpt-3.5-turbo')
        
        # Generate summaries for each source
        source_summaries = []
        for source in processed_content:
            source_summary = summarize_source(
                query, 
                source, 
                client=client,
                model=model, 
                max_tokens=max_tokens // len(processed_content)
            )
            source_summaries.append({
                'url': source['url'],
                'title': source['title'],
                'domain': source['domain'],
                'summary': source_summary
            })
        
        # Generate a final combined summary
        combined_summary = generate_combined_summary(
            query,
            source_summaries,
            client=client,
            model=model,
            max_tokens=max_tokens
        )
        
        # Prepare metadata
        metadata = {
            'sources_count': len(processed_content),
            'total_content_length': sum(s['original_length'] for s in processed_content),
            'generated_at': time.time(),
            'model_used': model,
            'sources': [{'url': s['url'], 'title': s['title']} for s in processed_content]
        }
        
        return combined_summary, metadata
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"An error occurred while generating the summary: {str(e)}", {}

def summarize_source(query, source, client, model, max_tokens):
    """Summarize a single source."""
    try:
        # Construct prompt for the source
        prompt = f"""
        The user searched for: "{query}"
        
        Please summarize the following content from {source['title']} ({source['domain']}):
        
        {source['cleaned_text'][:3000]}
        
        Summary:
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes web content accurately."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error summarizing source {source['url']}: {str(e)}")
        return f"Failed to summarize this source."

def generate_combined_summary(query, source_summaries, client, model, max_tokens):
    """Generate a combined summary from individual source summaries."""
    try:
        # Construct prompt for combined summary
        prompt = f"""
        The user searched for: "{query}"
        
        Here are summaries from multiple sources:
        
        """
        
        # Add each source summary
        for i, summary in enumerate(source_summaries, 1):
            prompt += f"""
            Source {i} ({summary['domain']}): 
            {summary['summary']}
            
            """
            
        prompt += f"""
        Please provide a comprehensive and accurate summary of all these sources,
        addressing the user's original query: "{query}"
        
        Combined Summary:
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates comprehensive summaries from multiple sources. Always cite the sources when stating facts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating combined summary: {str(e)}")
        return f"Failed to generate combined summary due to an error: {str(e)}"