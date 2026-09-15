# Meta-Chunking in Eris

This document outlines the implementation of Meta-Chunking in Eris, including usage instructions, benefits, and testing procedures.

## What is Meta-Chunking?

Meta-Chunking is an advanced text segmentation approach that divides content based on semantic meaning and logical structure rather than arbitrary character counts. It operates at a granularity between sentences and paragraphs, preserving deep linguistic logical connections between sentences.

Unlike traditional chunking methods, Meta-Chunking:

- Preserves semantic relationships between sentences
- Maintains logical coherence within chunks
- Adapts to the natural structure of the text
- Improves downstream tasks like summarization and retrieval

## Implementation Details

Our implementation uses Perplexity (PPL) Chunking, which:

1. Calculates the perplexity of each sentence based on its context
2. Identifies text chunk boundaries by analyzing the characteristics of perplexity distribution
3. Optionally combines smaller chunks to meet target size requirements

### Key Components

- `modules/meta_chunking.py`: Core implementation of the Meta-Chunking algorithm
- Updated `modules/processor.py`: Integration with the existing processing pipeline
- Configuration options in `config.py`: Controls Meta-Chunking behavior

### Performance Optimizations

Our implementation includes several performance-focused features:

- **Local Perplexity Calculation**: Uses statistical approximation instead of API calls
- **Bounded Context Window**: The running n-gram context is capped (`cache_size`) and pruned to
  its most frequent entries, so memory stays flat on long documents
- **Dynamic Combination**: Balances chunk size with logical coherence
- **Configurable Thresholds**: Allows fine-tuning for different content types

## Configuration Options

In `config.py`, you can adjust the following settings:

```python
# Meta-Chunking settings (each reads the same-named environment variable)
USE_META_CHUNKING = True                  # Enable/disable Meta-Chunking
META_CHUNKING_THRESHOLD = 0.5             # Perplexity threshold for chunk boundaries
USE_DYNAMIC_COMBINATION = True            # Enable/disable dynamic chunk size management
USE_OPENAI_FOR_PPL = False                # Use OpenAI API for perplexity calculation
```

Set them in `.env` rather than editing `config.py`; see `.env.example`.

## Running the Demo

The standalone demo script demonstrates the difference between traditional chunking and Meta-Chunking:

```bash
# Install required packages
pip install colorama matplotlib

# Run the demo
python tools/meta_chunking_demo.py
```

The demo will:
1. Process two sample texts with different characteristics
2. Display the chunks identified by each method
3. Generate visualizations of perplexity scores and chunk boundaries
4. Provide statistics comparing the methods

## Running Tests

We provide two types of tests for the Meta-Chunking implementation:

### 1. Integration Tests

```bash
# Install pytest and required dependencies
pip install pytest pytest-asyncio

# Run the integration tests
pytest tests/test_meta_chunking_integration.py -v
```

These tests verify that Meta-Chunking works correctly when integrated with the rest of the application.

### 2. Benchmarking

```bash
# Run the benchmarking script with sample files
python tools/benchmark_chunking.py --files your_text_a.txt your_text_b.txt
```

The benchmarking tool compares traditional chunking with Meta-Chunking across multiple dimensions:
- Processing time
- Number of chunks generated
- Average chunk size
- Chunk size distribution

## Benefits for Eris

The Meta-Chunking approach provides several key benefits for our application:

1. **Improved Summary Quality**: By preserving semantic relationships between sentences, summaries become more coherent and accurate.

2. **Enhanced Source Synthesis**: When combining information from multiple websites, Meta-Chunking helps maintain the integrity of each source's perspective.

3. **Better Handling of Complex Queries**: For multi-faceted topics, Meta-Chunking provides more meaningful organization of information.

4. **Optimized LLM Token Usage**: More intelligently grouped content makes better use of the context window, potentially reducing API costs.

5. **Improved Retrieval in RAG**: Chunks with intact logical structure yield better search results.

## Future Improvements

Potential future enhancements to our Meta-Chunking implementation:

1. **Adaptive Thresholds**: Dynamically adjust perplexity thresholds based on text characteristics
2. **Document Structure Awareness**: Integrate awareness of headings, sections, and other HTML elements
3. **Multilingual Support**: Extend perplexity analysis to handle multiple languages
4. **Real-time Parameter Tuning**: Optimize chunking parameters based on feedback from summarization results