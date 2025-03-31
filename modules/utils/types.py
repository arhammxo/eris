"""
Type definitions for Web Summarizer.

This module provides common type aliases and protocols used throughout the application
to improve type checking and code readability.
"""
from typing import Any, Dict, List, Optional, TypedDict, Union
from typing_extensions import NotRequired, Protocol, TypeAlias

# Basic type aliases
URL: TypeAlias = str
DomainName: TypeAlias = str
JSON: TypeAlias = Dict[str, Any]
HTMLContent: TypeAlias = str

# Source-related types
class SourceMetadata(TypedDict):
    """Metadata for a web source."""
    url: URL
    title: str
    domain: DomainName
    crawled_at: float  # Unix timestamp
    rendered_with_browser: NotRequired[bool]

class SourceContent(TypedDict):
    """Content extracted from a web source."""
    content: str
    original_length: int

class Source(SourceMetadata, SourceContent):
    """Complete source with metadata and content."""
    pass

# Processed content types
class ProcessedSource(TypedDict):
    """Source after text processing."""
    url: URL
    title: str
    domain: DomainName
    cleaned_text: str
    important_sentences: List[str]
    chunks: List[str]
    original_length: int
    processed_length: int
    quality_score: float

# Summary types
class SourceSummary(TypedDict):
    """Summary of a single source."""
    url: URL
    title: str
    domain: DomainName
    summary: str
    quality_score: NotRequired[float]

class SummaryMetadata(TypedDict):
    """Metadata for a generated summary."""
    sources_count: int
    total_content_length: int
    generated_at: float  # Unix timestamp
    model_used: str
    query_type: NotRequired[str]
    sources: List[Dict[str, Union[str, float]]]

# Progress tracking types
class ProgressData(TypedDict):
    """Data for progress tracking."""
    query: NotRequired[str]
    depth: NotRequired[int]
    summary_length: NotRequired[str]
    message: NotRequired[str]
    urls_found: NotRequired[int]
    sources_found: NotRequired[int]
    sources_processed: NotRequired[int]
    summary_length: NotRequired[int]

class ProgressInfo(TypedDict):
    """Information about the progress of a task."""
    status: str  # 'starting', 'searching', 'crawling', 'processing', 'summarizing', 'complete', 'error'
    progress: int  # 0-100 percentage
    updated_at: float  # Unix timestamp
    data: ProgressData

# Interface protocols
class SourceExtractor(Protocol):
    """Protocol for source content extractors."""
    
    async def extract(self, url: URL) -> Optional[Source]:
        """Extract content from a URL."""
        ...

class ContentProcessor(Protocol):
    """Protocol for content processors."""
    
    async def process(self, source: Source) -> Optional[ProcessedSource]:
        """Process a source's content."""
        ...

class Summarizer(Protocol):
    """Protocol for summarizers."""
    
    async def summarize(self, 
                        query: str, 
                        processed_sources: List[ProcessedSource], 
                        length: str = 'medium') -> tuple[str, SummaryMetadata]:
        """Generate a summary from processed sources."""
        ...