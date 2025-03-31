import logging
import re
import textstat
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sqlite3
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentQualityScorer:
    """
    Evaluates the quality of web content based on multiple metrics.
    """
    
    def __init__(self, db_path=None):
        """Initialize the quality scorer with optional domain reputation database."""
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), '../data/domain_quality.db')
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create the domain quality database if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS domain_quality (
            domain TEXT PRIMARY KEY,
            quality_score REAL,
            last_updated INTEGER
        )
        ''')
        
        # Seed with some known reliable domains if table is empty
        cursor.execute("SELECT COUNT(*) FROM domain_quality")
        if cursor.fetchone()[0] == 0:
            seed_domains = [
                ('wikipedia.org', 0.95),
                ('britannica.com', 0.92),
                ('smithsonianmag.com', 0.9),
                ('nature.com', 0.95),
                ('science.org', 0.93),
                ('mayoclinic.org', 0.92),
                ('nih.gov', 0.95),
                ('cdc.gov', 0.93),
                ('who.int', 0.93),
                ('edu', 0.85),  # Generic .edu domains
                ('gov', 0.85),  # Generic .gov domains
            ]
            
            now = int(time.time())
            cursor.executemany(
                "INSERT INTO domain_quality (domain, quality_score, last_updated) VALUES (?, ?, ?)",
                [(domain, score, now) for domain, score in seed_domains]
            )
        
        conn.commit()
        conn.close()
    
    def get_domain_quality(self, domain):
        """Get the quality score for a domain from the database."""
        base_domain = self._extract_base_domain(domain)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Try exact match first
        cursor.execute("SELECT quality_score FROM domain_quality WHERE domain = ?", (domain,))
        result = cursor.fetchone()
        
        # If no exact match, try base domain or TLD
        if not result:
            cursor.execute("SELECT quality_score FROM domain_quality WHERE domain = ?", (base_domain,))
            result = cursor.fetchone()
            
            # If still no match, try the TLD
            if not result:
                tld = domain.split('.')[-1] if '.' in domain else None
                if tld:
                    cursor.execute("SELECT quality_score FROM domain_quality WHERE domain = ?", (tld,))
                    result = cursor.fetchone()
        
        conn.close()
        
        # Return the score or a neutral default
        return result[0] if result else 0.5
    
    def _extract_base_domain(self, domain):
        """Extract the base domain (e.g., example.com from sub.example.com)."""
        parts = domain.split('.')
        if len(parts) > 2:
            return '.'.join(parts[-2:])
        return domain
    
    def score_content(self, source):
        """
        Calculate a quality score for content based on multiple factors.
        
        Args:
            source (dict): Source dictionary with content and metadata
            
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            content = source.get('content', '')
            url = source.get('url', '')
            domain = source.get('domain', '')
            
            # Skip if no content
            if not content:
                return 0.0
                
            # Calculate individual metrics
            length_score = self._score_content_length(content)
            readability_score = self._score_readability(content)
            citation_score = self._score_citations(content)
            domain_score = self.get_domain_quality(domain)
            coherence_score = self._score_coherence(content)
            
            # Combine scores with weights
            weights = {
                'length': 0.15,
                'readability': 0.25,
                'citations': 0.2,
                'domain': 0.25,
                'coherence': 0.15
            }
            
            combined_score = (
                weights['length'] * length_score +
                weights['readability'] * readability_score +
                weights['citations'] * citation_score +
                weights['domain'] * domain_score +
                weights['coherence'] * coherence_score
            )
            
            # Log individual scores for debugging
            logger.info(f"Content quality scores for {domain}: " + 
                       f"length={length_score:.2f}, readability={readability_score:.2f}, " +
                       f"citations={citation_score:.2f}, domain={domain_score:.2f}, " +
                       f"coherence={coherence_score:.2f}, combined={combined_score:.2f}")
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error calculating content quality score: {str(e)}")
            return 0.5  # Neutral score on error
    
    def _score_content_length(self, content):
        """
        Score based on content length - penalize both too short and too long.
        
        Ideal length is around 1500-3000 characters.
        """
        length = len(content)
        
        # Too short (< 500 chars)
        if length < 500:
            return max(0.2, length / 500)
        
        # Ideal range (500-5000 chars)
        if 500 <= length <= 5000:
            # Scale to 0.7-1.0 within this range
            normalized = (length - 500) / (5000 - 500)
            # Bell curve with peak around 2500 chars
            return 0.7 + 0.3 * (1 - abs(normalized - 0.5) * 2)
        
        # Too long but still valuable (5000-10000 chars)
        if 5000 < length <= 10000:
            return 0.8 - (length - 5000) / (10000 - 5000) * 0.2
        
        # Extremely long (> 10000 chars)
        return 0.6
    
    def _score_readability(self, content):
        """
        Score based on readability metrics.
        
        Uses textstat library to calculate readability scores.
        """
        try:
            # Get multiple readability metrics
            flesch_score = textstat.flesch_reading_ease(content)
            grade_level = textstat.text_standard(content, float_output=True)
            
            # Normalize Flesch score (0-100 scale where higher is more readable)
            # Convert to 0-1 scale with a sweet spot between 50-70
            if flesch_score < 30:
                flesch_normalized = max(0.3, flesch_score / 30)
            elif 30 <= flesch_score <= 70:
                flesch_normalized = 0.7 + (flesch_score - 30) / (70 - 30) * 0.3
            else:  # > 70
                flesch_normalized = 1.0 - (flesch_score - 70) / (100 - 70) * 0.2
                
            # Normalize grade level (typically 0-16 scale)
            # Convert to 0-1 scale with sweet spot between grades 8-12
            if grade_level < 6:
                grade_normalized = 0.5  # Too simple
            elif 6 <= grade_level <= 12:
                grade_normalized = 0.7 + (grade_level - 6) / (12 - 6) * 0.3
            else:  # > 12
                grade_normalized = max(0.5, 1.0 - (grade_level - 12) / 8 * 0.5)
            
            # Combine the scores
            return (flesch_normalized * 0.6) + (grade_normalized * 0.4)
            
        except Exception as e:
            logger.warning(f"Error calculating readability score: {str(e)}")
            return 0.5
    
    def _score_citations(self, content):
        """
        Score based on presence of citations, references, and links.
        """
        # Look for indicators of citations
        citation_indicators = [
            r'\[\d+\]',                # [1], [2], etc.
            r'\(\d{4}\)',              # (2020), (2021), etc.
            r'\b(et al\.|cited in)\b', # Academic citation patterns
            r'according to',
            r'cited by',
            r'reference',
            r'bibliography',
            r'source',
            r'study (by|from)',
            r'research (by|from)',
            r'published in'
        ]
        
        # Count matches
        citation_count = 0
        for pattern in citation_indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citation_count += len(matches)
        
        # Count links
        link_count = len(re.findall(r'https?://\S+', content))
        
        # Calculate the score based on citation density
        content_length = len(content)
        expected_citations = max(1, content_length / 1000)  # Expect more citations in longer content
        
        total_references = citation_count + link_count
        citation_ratio = min(1.0, total_references / expected_citations)
        
        return citation_ratio
    
    def _score_coherence(self, content):
        """
        Score based on text coherence and structure.
        
        Looks for indicators of well-structured content.
        """
        # Split into sentences and paragraphs
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        paragraphs = re.split(r'\n+', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not sentences or not paragraphs:
            return 0.3
        
        # Calculate average sentence length (penalize very short or long sentences)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if avg_sentence_length < 5:
            sentence_score = 0.3  # Too short
        elif 5 <= avg_sentence_length <= 25:
            sentence_score = 0.5 + (avg_sentence_length - 5) / (25 - 5) * 0.5
            # Bell curve with peak around 15 words
            normalized = (avg_sentence_length - 5) / (25 - 5)
            sentence_score = 0.5 + 0.5 * (1 - abs(normalized - 0.5) * 2)
        else:  # > 25
            sentence_score = max(0.3, 1.0 - (avg_sentence_length - 25) / 15 * 0.7)
        
        # Check paragraph structure (ideal is 2-5 sentences per paragraph)
        if len(paragraphs) < 2:
            structure_score = 0.4  # Poor structure, just one paragraph
        else:
            avg_sentences_per_para = len(sentences) / len(paragraphs)
            if avg_sentences_per_para < 1.5:
                structure_score = 0.4  # Too fragmented
            elif 1.5 <= avg_sentences_per_para <= 5:
                structure_score = 0.7 + (avg_sentences_per_para - 1.5) / (5 - 1.5) * 0.3
            else:  # > 5
                structure_score = max(0.4, 1.0 - (avg_sentences_per_para - 5) / 5 * 0.6)
        
        # Look for section headers
        header_indicators = [
            r'\n[A-Z][^.!?]+\n',  # Capitalized text on its own line
            r'\*\*[^*]+\*\*',      # **Bold text**
            r'#{1,3} ',            # Markdown headers
            r'^[A-Z][^a-z]+:',     # SECTION: format
        ]
        
        has_headers = any(re.search(pattern, content) for pattern in header_indicators)
        header_score = 0.7 if has_headers else 0.4
        
        # Combine the scores
        return (sentence_score * 0.4) + (structure_score * 0.4) + (header_score * 0.2)