"""
Universal Document Ingestion Pipeline
======================================
A modular, generalizable document chunking system for:
- Documentation, manuals, guides
- Rules, regulations, policies  
- Research papers, academic texts
- Technical specifications

Architecture:
- Extractors: Pull content from documents (text, OCR, visual - pluggable)
- Chunkers: Split content into semantic units
- Filters: Optional quality/signal filtering
- Formatters: Structure output for downstream use
"""

import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Protocol, Any
from abc import ABC, abstractmethod

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunking behavior."""
    min_chars: int = 300
    max_chars: int = 4000
    overlap_chars: int = 200
    
    # Sentence splitting patterns (supports multiple scripts)
    sentence_pattern: str = r'(?<=[.!?。！？।])\s+'
    
    # Header detection patterns (regex list - matched in order)
    header_patterns: list[str] = field(default_factory=lambda: [
        r'^\d+(\.\d+)*\s+',           # Numbered: 1.2.3 Header
        r'^[A-Z][A-Z\s]{10,}$',       # ALL CAPS headers
        r'^(Chapter|Section|Part)\s+\d+',  # Chapter 1, Section 2
        r'^#{1,6}\s+',                # Markdown headers
    ])
    
    # Scope detection patterns (persistent context) - Universally Robust
    scope_patterns: list[str] = field(default_factory=lambda: [
        r'^[A-Za-z]+\s+(?:\d+|[IVX]+)$',        # Label + Number (e.g., "Chapter 1", "Semester IV", "Unit 5")
        r'^(?:PART|SECTION|CHAPTER|VOLUME)\s+', # Explicit Standard Divisions
        r'^[A-Z][A-Z\s0-9]{3,50}$',             # Short ALL CAPS headers (e.g., "SYLLABUS", "APPENDIX A")
    ])
    
    # Categories to treat as section breaks
    section_categories: set[str] = field(default_factory=lambda: {
        "Title", "SectionHeader", "Header"
    })
    
    # Categories to filter out entirely
    exclude_categories: set[str] = field(default_factory=lambda: {
        "Header", "Footer", "PageNumber"
    })


@dataclass  
class IngestConfig:
    """Configuration for the full ingestion pipeline."""
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)
    
    # Filtering options
    filter_short_chunks: bool = True
    filter_low_signal: bool = False  # Disabled by default - too aggressive
    min_word_count: int = 20
    
    # Metadata
    language: str = "en"
    
    # Future extensibility
    enable_ocr: bool = False
    enable_visual_embedding: bool = False


# =============================================================================
# EXTRACTOR PROTOCOL (Pluggable)
# =============================================================================

class ContentExtractor(Protocol):
    """Protocol for content extractors. Implement this for OCR, visual, etc."""
    
    def extract(self, file_path: str) -> list[dict]:
        """
        Extract content elements from a file.
        
        Returns list of dicts with at minimum:
        - text: str
        - category: str (e.g., "Title", "NarrativeText", "Table")
        - metadata: dict (optional extra info)
        """
        ...


class TextExtractor:
    """Default text extractor using unstructured library."""
    
    def __init__(self, exclude_categories: set[str] | None = None):
        self.exclude_categories = exclude_categories or {"Header", "Footer"}
    
    def extract(self, file_path: str) -> list[dict]:
        from unstructured.partition.auto import partition
        
        elements = partition(file_path)
        
        result = []
        for el in elements:
            if el.category in self.exclude_categories:
                continue
            
            result.append({
                "text": str(el),
                "category": el.category,
                "metadata": getattr(el, "metadata", {}).__dict__ if hasattr(el, "metadata") else {}
            })
        
        return result


# Placeholder for future extractors
class OCRExtractor:
    """Placeholder for OCR-based extraction."""
    
    def extract(self, file_path: str) -> list[dict]:
        raise NotImplementedError("OCR extraction not yet implemented")


class VisualExtractor:
    """Placeholder for visual/image embedding preparation."""
    
    def extract(self, file_path: str) -> list[dict]:
        raise NotImplementedError("Visual extraction not yet implemented")


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    from unstructured.cleaners.core import clean_extra_whitespace
    
    text = clean_extra_whitespace(text)
    
    # Remove excessive dots (table of contents artifacts)
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'(\.\s){3,}', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def is_section_header(element: dict, config: ChunkConfig) -> bool:
    """Determine if an element represents a section header."""
    category = element.get("category", "")
    text = element.get("text", "").strip()
    
    # Check by category
    if category in config.section_categories:
        # Validate it's substantial enough to be a real header
        if len(text) >= 5:
            return True
    
    # Check by pattern matching
    for pattern in config.header_patterns:
        if re.match(pattern, text, re.MULTILINE):
            return True
    
    return False


def get_element_scope(text: str, config: ChunkConfig) -> str | None:
    """Check if text defines a new scope (e.g., 'Semester IV')."""
    # Simple check: must be short enough to be a header
    if len(text) > 100:
        return None
        
    for pattern in config.scope_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Clean up the match (remove extra whitespace)
            return re.sub(r'\s+', ' ', match.group(0).upper().strip())
    return None


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

@dataclass
class Chunk:
    """Represents a document chunk."""
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


def chunk_by_sections(elements: list[dict], config: ChunkConfig) -> list[Chunk]:
    """
    Create chunks based on section headers with hierarchical scope tracking.
    """
    chunks = []
    current_texts = []
    current_title = "Introduction"
    current_scope = ""
    
    for el in elements:
        text = el.get("text", "").strip()
        if not text:
            continue
            
        # Update hierarchy scope if detected
        new_scope = get_element_scope(text, config)
        if new_scope:
            current_scope = new_scope
        
        if is_section_header(el, config):
            # Save current chunk before starting new section
            if current_texts:
                combined = clean_text(" ".join(current_texts))
                if combined:
                    # Inject scope context if not already present
                    final_text = combined
                    if current_scope and current_scope not in combined.upper():
                        final_text = f"[{current_scope}] {combined}"
                    
                    chunks.append(Chunk(
                        text=final_text, 
                        title=current_title,
                        metadata={"scope": current_scope}
                    ))
            
            # Start new section
            current_title = text
            current_texts = [text]
        else:
            current_texts.append(text)
    
    # Don't forget the last chunk
    if current_texts:
        combined = clean_text(" ".join(current_texts))
        if combined:
            final_text = combined
            if current_scope and current_scope not in combined.upper():
                final_text = f"[{current_scope}] {combined}"
            
            chunks.append(Chunk(
                text=final_text, 
                title=current_title,
                metadata={"scope": current_scope}
            ))
    
    return chunks


def split_by_sentences(chunk: Chunk, config: ChunkConfig) -> list[Chunk]:
    """Split a chunk at sentence boundaries if it exceeds max_chars."""
    text = chunk.text
    
    if len(text) <= config.max_chars:
        return [chunk]
    
    # Split into sentences
    sentences = re.split(config.sentence_pattern, text)
    
    parts = []
    current = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Would adding this sentence exceed the limit?
        test_len = len(current) + len(sentence) + 1
        
        if current and test_len > config.max_chars:
            # Save current part
            parts.append(current.strip())
            
            # Start new part with overlap
            if config.overlap_chars > 0:
                overlap = current[-config.overlap_chars:]
                current = overlap + " " + sentence
            else:
                current = sentence
        else:
            current = (current + " " + sentence).strip()
    
    if current:
        parts.append(current.strip())
    
    # Handle edge case: no sentence boundaries found, do word-boundary split
    final_parts = []
    for part in parts:
        if len(part) > config.max_chars:
            final_parts.extend(_hard_split_at_words(part, config.max_chars))
        else:
            final_parts.append(part)
    
    return [Chunk(text=p, title=chunk.title, metadata=chunk.metadata) for p in final_parts]


def _hard_split_at_words(text: str, max_chars: int) -> list[str]:
    """Fallback: split at word boundaries when no sentences found."""
    words = text.split()
    parts = []
    current = ""
    
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            if current:
                parts.append(current.strip())
            current = word
        else:
            current = (current + " " + word).strip()
    
    if current:
        parts.append(current.strip())
    
    return parts


def merge_small_chunks(chunks: list[Chunk], config: ChunkConfig) -> list[Chunk]:
    """Merge chunks smaller than min_chars with adjacent chunks."""
    if not chunks:
        return []
    
    merged = []
    buffer: Chunk | None = None
    
    for chunk in chunks:
        if len(chunk.text) < config.min_chars:
            # Accumulate small chunk
            if buffer is None:
                buffer = Chunk(text=chunk.text, title=chunk.title)
            else:
                buffer.text = (buffer.text + " " + chunk.text).strip()
        else:
            # Process buffer first
            if buffer is not None:
                combined_len = len(buffer.text) + len(chunk.text) + 1
                if combined_len <= config.max_chars:
                    # Merge buffer into current chunk
                    chunk = Chunk(
                        text=(buffer.text + " " + chunk.text).strip(),
                        title=buffer.title or chunk.title
                    )
                else:
                    # Buffer is big enough on its own, or append to previous
                    if merged and len(merged[-1].text) + len(buffer.text) + 1 <= config.max_chars:
                        merged[-1].text += " " + buffer.text
                    else:
                        merged.append(buffer)
                buffer = None
            
            merged.append(chunk)
    
    # Handle remaining buffer
    if buffer is not None:
        if merged and len(merged[-1].text) + len(buffer.text) + 1 <= config.max_chars:
            merged[-1].text += " " + buffer.text
        else:
            merged.append(buffer)
    
    return merged


# =============================================================================
# FILTERS (Optional)
# =============================================================================

def filter_short_chunks(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Remove chunks below minimum character threshold."""
    return [c for c in chunks if len(c.text) >= min_chars]


def filter_low_signal(chunks: list[Chunk], min_words: int = 20) -> list[Chunk]:
    """
    Remove chunks with low informational content.
    Conservative filter - only removes obvious noise.
    """
    filtered = []
    
    for chunk in chunks:
        text = chunk.text
        words = text.split()
        
        # Too few words
        if len(words) < min_words:
            continue
        
        # Mostly numbers or single characters (likely table of contents)
        alpha_ratio = sum(1 for w in words if w.isalpha()) / max(len(words), 1)
        if alpha_ratio < 0.3:
            continue
        
        filtered.append(chunk)
    
    return filtered


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

@dataclass
class IngestedChunk:
    """Final output format for ingested chunks."""
    id: str
    text: str
    metadata: dict


def format_output(
    chunks: list[Chunk],
    source_id: str,
    source_type: str,
    config: IngestConfig
) -> list[IngestedChunk]:
    """Format chunks into final output structure."""
    output = []
    
    for i, chunk in enumerate(chunks):
        output.append(IngestedChunk(
            id=str(uuid.uuid4()),
            text=chunk.text,
            metadata={
                "source_id": source_id,
                "source_type": source_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk.text),
                "word_count": len(chunk.text.split()),
                "title": chunk.title,
                "language": config.language,
                "extraction_method": "text",  # Will be "ocr" or "visual" when extended
            }
        ))
    
    return output


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def ingest_document(
    file_path: str,
    config: IngestConfig | None = None,
    extractor: ContentExtractor | None = None,
    source_id: str | None = None,
) -> list[IngestedChunk]:
    """
    Main ingestion pipeline.
    
    Args:
        file_path: Path to the document
        config: Ingestion configuration (uses defaults if None)
        extractor: Content extractor to use (uses TextExtractor if None)
        source_id: Identifier for the source document
    
    Returns:
        List of IngestedChunk objects ready for embedding/storage
    """
    config = config or IngestConfig()
    chunk_config = config.chunk_config
    
    # Determine source info
    path = Path(file_path)
    source_id = source_id or str(path)
    source_type = path.suffix.lower().lstrip(".")
    
    # Step 1: Extract content
    if extractor is None:
        extractor = TextExtractor(exclude_categories=chunk_config.exclude_categories)
    
    elements = extractor.extract(file_path)
    
    if not elements:
        return []
    
    # Step 2: Chunk by sections
    chunks = chunk_by_sections(elements, chunk_config)
    
    # Step 3: Merge small chunks (first pass)
    chunks = merge_small_chunks(chunks, chunk_config)
    
    # Step 4: Split large chunks at sentence boundaries
    split_chunks = []
    for chunk in chunks:
        split_chunks.extend(split_by_sentences(chunk, chunk_config))
    chunks = split_chunks
    
    # Step 5: Merge again (cleanup after splitting)
    chunks = merge_small_chunks(chunks, chunk_config)
    
    # Step 6: Apply filters
    if config.filter_short_chunks:
        chunks = filter_short_chunks(chunks, chunk_config.min_chars)
    
    if config.filter_low_signal:
        chunks = filter_low_signal(chunks, config.min_word_count)
    
    # Step 7: Format output
    return format_output(chunks, source_id, source_type, config)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def ingest_with_defaults(file_path: str) -> list[IngestedChunk]:
    """Quick ingestion with sensible defaults."""
    return ingest_document(file_path)


def ingest_research_paper(file_path: str) -> list[IngestedChunk]:
    """Preset for research papers - smaller chunks, more overlap."""
    config = IngestConfig(
        chunk_config=ChunkConfig(
            min_chars=200,
            max_chars=2000,
            overlap_chars=150,
        ),
        filter_low_signal=True,
    )
    return ingest_document(file_path, config=config)


def ingest_documentation(file_path: str) -> list[IngestedChunk]:
    """Preset for technical documentation - larger chunks, less overlap."""
    config = IngestConfig(
        chunk_config=ChunkConfig(
            min_chars=400,
            max_chars=5000,
            overlap_chars=200,
            header_patterns=[
                r'^#{1,6}\s+',                # Markdown
                r'^\d+(\.\d+)*\s+',            # Numbered
                r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*$',  # Title Case headers
            ],
        ),
    )
    return ingest_document(file_path, config=config)


def ingest_regulations(file_path: str) -> list[IngestedChunk]:
    """Preset for rules, regulations, policies."""
    config = IngestConfig(
        chunk_config=ChunkConfig(
            min_chars=300,
            max_chars=4000,
            overlap_chars=200,
            header_patterns=[
                r'^\d+(\.\d+)*\s+',
                r'^(Article|Section|Rule|Clause)\s+\d+',
                r'^[A-Z][A-Z\s]{5,}$',
            ],
        ),
    )
    return ingest_document(file_path, config=config)


# =============================================================================
# CLI / DEBUG
# =============================================================================

if __name__ == "__main__":
    import sys
    
    file_path = sys.argv[1] if len(sys.argv) > 1 else "./data/Academic-Regulations-2018.pdf"
    
    print(f"Ingesting: {file_path}")
    print("-" * 50)
    
    chunks = ingest_document(file_path)
    
    print(f"Created {len(chunks)} chunks")
    print()
    
    # Show size distribution
    sizes = [c.metadata["char_count"] for c in chunks]
    print(f"Size range: {min(sizes)} - {max(sizes)} chars")
    print(f"Average: {sum(sizes) // len(sizes)} chars")
    print()
    
    # Show first few chunks
    for c in chunks[70:73]:
        print(f"--- Chunk {c.metadata['chunk_index'] + 1} ({c.metadata['char_count']} chars) ---")
        print(f"Title: {c.metadata['title']}")
        print(c.text[:200], "...")
        print()
