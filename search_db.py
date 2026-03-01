"""
Search Database Module
======================
Semantic search over the Qdrant vector database with metadata filtering.
Optimized for local execution - uses the same embedding model as ingestion.

Usage:
    python search_db.py "your search query" [--collection name] [--top-k 5]
    python search_db.py "your query" --filter source_type=pdf
    python search_db.py "your query" --filter source_id=path/to/doc.pdf
"""

import sys
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    Range,
    ScoredPoint,
)

from embed import EmbedConfig, embed_texts, get_model


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for search operations."""
    
    # Qdrant connection
    qdrant_path: str = "./qdrant_data"
    
    # Collection to search
    collection_name: str = "documents"
    
    # Number of results to return
    top_k: int = 5
    
    # Minimum similarity score (0-1 for cosine)
    score_threshold: float | None = None
    
    # Embedding config (should match what was used for ingestion)
    embed_config: EmbedConfig = field(default_factory=EmbedConfig)


# =============================================================================
# SEARCH RESULT
# =============================================================================

@dataclass
class SearchResult:
    """A single search result with metadata."""
    id: str
    text: str
    score: float
    metadata: dict
    
    def __str__(self) -> str:
        return (
            f"[Score: {self.score:.4f}] "
            f"[{self.metadata.get('title', 'Untitled')[:50]}]\n"
            f"{self.text[:200]}..."
        )
    
    @property
    def source_id(self) -> str:
        return self.metadata.get("source_id", "")
    
    @property
    def title(self) -> str:
        return self.metadata.get("title", "")
    
    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", -1)


# =============================================================================
# CLIENT MANAGEMENT
# =============================================================================

_client_cache: dict[str, QdrantClient] = {}


def get_client(qdrant_path: str) -> QdrantClient:
    """Get or create Qdrant client (cached)."""
    if qdrant_path not in _client_cache:
        path = Path(qdrant_path)
        if not path.exists():
            raise FileNotFoundError(f"Qdrant data not found at: {qdrant_path}")
        _client_cache[qdrant_path] = QdrantClient(path=str(path))
    return _client_cache[qdrant_path]


def cleanup_clients():
    """Explicitly close all cached clients to avoid shutdown errors."""
    for client in _client_cache.values():
        try:
            client.close()
        except Exception:
            pass
    _client_cache.clear()


# Register cleanup on exit
import atexit
atexit.register(cleanup_clients)


# =============================================================================
# FILTER BUILDERS
# =============================================================================

def build_filter(
    source_id: str | None = None,
    source_type: str | None = None,
    title_contains: str | None = None,
    language: str | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    custom_filters: list[FieldCondition] | None = None,
) -> Filter | None:
    """
    Build a Qdrant filter from common filter parameters.
    
    Args:
        source_id: Filter by exact source document path
        source_type: Filter by file type (pdf, docx, txt, etc.)
        title_contains: Filter by title containing text
        language: Filter by language code
        min_chars: Minimum character count
        max_chars: Maximum character count
        custom_filters: Additional custom FieldConditions
    
    Returns:
        Filter object or None if no filters specified
    """
    conditions = []
    
    if source_id:
        conditions.append(
            FieldCondition(key="source_id", match=MatchValue(value=source_id))
        )
    
    if source_type:
        conditions.append(
            FieldCondition(key="source_type", match=MatchValue(value=source_type))
        )
    
    if title_contains:
        conditions.append(
            FieldCondition(key="title", match=MatchText(text=title_contains))
        )
    
    if language:
        conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language))
        )
    
    if min_chars is not None or max_chars is not None:
        conditions.append(
            FieldCondition(
                key="char_count",
                range=Range(
                    gte=min_chars,
                    lte=max_chars,
                )
            )
        )
    
    if custom_filters:
        conditions.extend(custom_filters)
    
    if not conditions:
        return None
    
    return Filter(must=conditions)


def parse_filter_string(filter_str: str) -> tuple[str, str]:
    """Parse a filter string like 'key=value' into (key, value)."""
    if "=" not in filter_str:
        raise ValueError(f"Invalid filter format: {filter_str}. Expected 'key=value'")
    key, value = filter_str.split("=", 1)
    return key.strip(), value.strip()


def build_filter_from_strings(filter_strings: list[str]) -> Filter | None:
    """Build a filter from CLI-style 'key=value' strings."""
    if not filter_strings:
        return None
    
    conditions = []
    for fs in filter_strings:
        key, value = parse_filter_string(fs)
        conditions.append(
            FieldCondition(key=key, match=MatchValue(value=value))
        )
    
    return Filter(must=conditions)


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def search(
    query: str,
    config: SearchConfig | None = None,
    query_filter: Filter | None = None,
) -> list[SearchResult]:
    """
    Perform semantic search over the vector database.
    
    Args:
        query: The search query text
        config: Search configuration
        query_filter: Optional Qdrant filter
    
    Returns:
        List of SearchResult objects ranked by similarity
    """
    config = config or SearchConfig()
    
    # Get client
    client = get_client(config.qdrant_path)
    
    # Generate query embedding
    query_embedding = embed_texts([query], config.embed_config)[0]
    
    # Perform search using query_points (newer API)
    results = client.query_points(
        collection_name=config.collection_name,
        query=query_embedding,
        query_filter=query_filter,
        limit=config.top_k,
        score_threshold=config.score_threshold,
        with_payload=True,
    )
    
    # Convert to SearchResult objects
    return [
        SearchResult(
            id=str(point.id),
            text=point.payload.get("text", ""),
            score=point.score,
            metadata={k: v for k, v in point.payload.items() if k != "text"},
        )
        for point in results.points
    ]


def search_with_filters(
    query: str,
    config: SearchConfig | None = None,
    source_id: str | None = None,
    source_type: str | None = None,
    title_contains: str | None = None,
    language: str | None = None,
) -> list[SearchResult]:
    """
    Convenience function: search with common filter parameters.
    
    Args:
        query: The search query text
        config: Search configuration
        source_id: Filter by exact source document
        source_type: Filter by file type
        title_contains: Filter by title text
        language: Filter by language
    
    Returns:
        List of SearchResult objects
    """
    query_filter = build_filter(
        source_id=source_id,
        source_type=source_type,
        title_contains=title_contains,
        language=language,
    )
    
    return search(query, config=config, query_filter=query_filter)


def search_in_document(
    query: str,
    document_path: str,
    config: SearchConfig | None = None,
) -> list[SearchResult]:
    """
    Search within a specific document only.
    
    Args:
        query: The search query text
        document_path: Path to the document to search within
        config: Search configuration
    
    Returns:
        List of SearchResult objects from that document only
    """
    return search_with_filters(query, config=config, source_id=document_path)


# =============================================================================
# RETRIEVAL FOR RAG
# =============================================================================

def retrieve_context(
    query: str,
    config: SearchConfig | None = None,
    query_filter: Filter | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Retrieve context for RAG (Retrieval-Augmented Generation).
    Returns concatenated text from top results.
    
    Args:
        query: The search query
        config: Search configuration
        query_filter: Optional filter
        max_tokens: Approximate max tokens (chars / 4)
    
    Returns:
        Concatenated context string
    """
    results = search(query, config=config, query_filter=query_filter)
    
    if not results:
        return ""
    
    context_parts = []
    total_chars = 0
    max_chars = (max_tokens * 4) if max_tokens else None
    
    for result in results:
        source = Path(result.source_id).name if result.source_id else "Unknown"
        chunk_text = f"[Source: {source} | Section: {result.title}]\n{result.text}\n"
        
        if max_chars and total_chars + len(chunk_text) > max_chars:
            # Truncate to fit
            remaining = max_chars - total_chars
            if remaining > 100:
                context_parts.append(chunk_text[:remaining] + "...")
            break
        
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)
    
    return "\n---\n".join(context_parts)


def retrieve_for_rag(
    query: str,
    config: SearchConfig | None = None,
    query_filter: Filter | None = None,
) -> dict:
    """
    Retrieve context and metadata for RAG pipeline.
    
    Returns:
        Dict with 'context', 'sources', and 'results'
    """
    results = search(query, config=config, query_filter=query_filter)
    
    context = retrieve_context(query, config=config, query_filter=query_filter)
    
    sources = list(set(r.source_id for r in results if r.source_id))
    
    return {
        "query": query,
        "context": context,
        "sources": sources,
        "num_results": len(results),
        "results": results,
    }


# =============================================================================
# COLLECTION INFO
# =============================================================================

def list_collections(qdrant_path: str = "./qdrant_data") -> list[str]:
    """List all collections in the database."""
    client = get_client(qdrant_path)
    collections = client.get_collections().collections
    return [c.name for c in collections]


def get_collection_stats(
    collection_name: str,
    qdrant_path: str = "./qdrant_data"
) -> dict:
    """Get statistics about a collection."""
    client = get_client(qdrant_path)
    info = client.get_collection(collection_name)
    
    # Handle different qdrant-client versions
    vector_size = None
    if info.config and info.config.params:
        vectors = info.config.params.vectors
        if hasattr(vectors, 'size'):
            vector_size = vectors.size
        elif isinstance(vectors, dict) and '' in vectors:
            vector_size = vectors[''].size
    
    return {
        "name": collection_name,
        "points_count": info.points_count,
        "status": info.status.name if hasattr(info.status, 'name') else str(info.status),
        "vector_size": vector_size,
    }


def list_sources(
    collection_name: str,
    qdrant_path: str = "./qdrant_data"
) -> list[str]:
    """List all unique source documents in a collection."""
    client = get_client(qdrant_path)
    
    # Scroll through all points to collect unique sources
    sources = set()
    offset = None
    
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=["source_id"],
            with_vectors=False,
        )
        
        for point in points:
            if "source_id" in point.payload:
                sources.add(point.payload["source_id"])
        
        if offset is None:
            break
    
    return sorted(list(sources))


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Search the Qdrant vector database"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query text",
    )
    parser.add_argument(
        "--collection", "-c",
        default="documents",
        help="Collection name (default: documents)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    parser.add_argument(
        "--filter", "-f",
        action="append",
        help="Filter in 'key=value' format (can be used multiple times)",
    )
    parser.add_argument(
        "--source",
        help="Filter by source document path",
    )
    parser.add_argument(
        "--type",
        help="Filter by source type (pdf, docx, etc.)",
    )
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_data",
        help="Path to Qdrant data (default: ./qdrant_data)",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all collections",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List all source documents in collection",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Output as RAG context (concatenated text)",
    )
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list_collections:
        collections = list_collections(args.qdrant_path)
        print("Collections:")
        for c in collections:
            print(f"  - {c}")
        return
    
    if args.stats:
        stats = get_collection_stats(args.collection, args.qdrant_path)
        print(f"Collection: {stats['name']}")
        print(f"  Points: {stats['points_count']}")
        print(f"  Vector size: {stats['vector_size']}")
        print(f"  Status: {stats['status']}")
        return
    
    if args.list_sources:
        sources = list_sources(args.collection, args.qdrant_path)
        print(f"Sources in '{args.collection}':")
        for s in sources:
            print(f"  - {s}")
        return
    
    # Search mode requires a query
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    # Build config
    config = SearchConfig(
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        top_k=args.top_k,
    )
    
    # Build filter
    query_filter = None
    filter_conditions = []
    
    if args.source:
        filter_conditions.append(
            FieldCondition(key="source_id", match=MatchValue(value=args.source))
        )
    
    if args.type:
        filter_conditions.append(
            FieldCondition(key="source_type", match=MatchValue(value=args.type))
        )
    
    if args.filter:
        for f in args.filter:
            key, value = parse_filter_string(f)
            filter_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
    
    if filter_conditions:
        query_filter = Filter(must=filter_conditions)
    
    # Perform search
    print(f"Searching: '{args.query}'")
    if query_filter:
        print(f"Filters: {len(filter_conditions)} active")
    print("-" * 50)
    
    if args.context:
        # RAG context mode
        context = retrieve_context(args.query, config=config, query_filter=query_filter)
        print(context)
    else:
        # Normal search mode
        results = search(args.query, config=config, query_filter=query_filter)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result.score:.4f}")
            print(f"    Source: {Path(result.source_id).name if result.source_id else 'Unknown'}")
            print(f"    Title: {result.title[:60]}...")
            print(f"    Text: {result.text[:200]}...")


if __name__ == "__main__":
    main()
