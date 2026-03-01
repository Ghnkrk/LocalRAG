"""
Embedding Module
================
Generates embeddings for document chunks using sentence-transformers.
Optimized for local execution on modest hardware (i5, 20GB RAM, iGPU).

Uses all-MiniLM-L6-v2:
- 384 dimensions
- Fast inference on CPU
- Good quality for semantic search
"""

import os
from dataclasses import dataclass, field
from typing import Iterator
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ingest import ingest_document, IngestConfig, IngestedChunk


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmbedConfig:
    """Configuration for embedding generation."""
    
    # Local model path - prioritize project root, then home directory
    model_name: str = "all-MiniLM-L6-v2"  # HuggingFace name
    
    # Path resolution logic (expanded in get_model)
    local_model_path: str = "./embed_models/all-MiniLM-L6-v2"
    home_model_path: str = os.path.expanduser("~/ai/embed_models/all-MiniLM-L6-v2")
    
    # Batch size for embedding - lower for memory constrained systems
    batch_size: int = 32
    
    # Device: 'cpu' for local, or 'cuda' if GPU available
    device: str = "cpu"
    
    # Normalize embeddings for cosine similarity
    normalize: bool = True
    
    # Show progress bar
    show_progress: bool = True
    
    # Cache directory for model
    cache_dir: str | None = None


# =============================================================================
# EMBEDDING MODEL (Singleton for efficiency)
# =============================================================================

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(config: EmbedConfig) -> SentenceTransformer:
    """
    Get or create embedding model (cached singleton).
    Avoids reloading model on every call.
    """
    # Resolve model path:
    # 1. Try local project path
    # 2. Try home directory path
    # 3. Fallback to model name (may trigger download)
    
    model_path = config.model_name
    
    # Prioritize local models for Git portability
    local_path = os.path.expanduser(config.local_model_path)
    home_path = os.path.expanduser(config.home_model_path)
    
    if os.path.isdir(local_path):
        model_path = local_path
        print(f"Using local project embedding model: {model_path}")
    elif os.path.isdir(home_path):
        model_path = home_path
        print(f"Using local home embedding model: {model_path}")
    else:
        print(f"Local model not found. Using HuggingFace: {config.model_name}")
        
    cache_key = f"{model_path}_{config.device}"
    
    if cache_key not in _model_cache:
        _model_cache[cache_key] = SentenceTransformer(
            model_path,
            device=config.device,
            cache_folder=config.cache_dir,
        )
        dim = _model_cache[cache_key].get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {dim}")
    
    return _model_cache[cache_key]


# =============================================================================
# EMBEDDED CHUNK DATA STRUCTURE
# =============================================================================

@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""
    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    
    @property
    def dimension(self) -> int:
        return len(self.embedding)


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def embed_texts(
    texts: list[str],
    config: EmbedConfig | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        config: Embedding configuration
    
    Returns:
        List of embedding vectors (as Python lists)
    """
    config = config or EmbedConfig()
    model = get_model(config)
    
    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=config.show_progress,
        normalize_embeddings=config.normalize,
        convert_to_numpy=True,
    )
    
    # Convert to Python lists for JSON serialization
    return [emb.tolist() for emb in embeddings]


def embed_chunks(
    chunks: list[IngestedChunk],
    config: EmbedConfig | None = None,
) -> list[EmbeddedChunk]:
    """
    Generate embeddings for a list of ingested chunks.
    
    Args:
        chunks: List of IngestedChunk from ingest module
        config: Embedding configuration
    
    Returns:
        List of EmbeddedChunk with embeddings attached
    """
    config = config or EmbedConfig()
    
    # Extract texts
    texts = [chunk.text for chunk in chunks]
    
    # Generate embeddings
    embeddings = embed_texts(texts, config)
    
    # Create embedded chunks
    return [
        EmbeddedChunk(
            id=chunk.id,
            text=chunk.text,
            embedding=emb,
            metadata=chunk.metadata,
        )
        for chunk, emb in zip(chunks, embeddings)
    ]


def embed_chunks_streaming(
    chunks: list[IngestedChunk],
    config: EmbedConfig | None = None,
) -> Iterator[EmbeddedChunk]:
    """
    Generate embeddings in a streaming fashion (memory efficient).
    Yields chunks one batch at a time.
    
    Args:
        chunks: List of IngestedChunk from ingest module
        config: Embedding configuration
    
    Yields:
        EmbeddedChunk objects one at a time
    """
    config = config or EmbedConfig()
    model = get_model(config)
    
    batch_size = config.batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=config.normalize,
            convert_to_numpy=True,
        )
        
        for chunk, emb in zip(batch, embeddings):
            yield EmbeddedChunk(
                id=chunk.id,
                text=chunk.text,
                embedding=emb.tolist(),
                metadata=chunk.metadata,
            )


# =============================================================================
# MAIN PIPELINE: FILE -> EMBEDDED CHUNKS
# =============================================================================

def embed_document(
    file_path: str,
    embed_config: EmbedConfig | None = None,
    ingest_config: IngestConfig | None = None,
) -> list[EmbeddedChunk]:
    """
    Full pipeline: Ingest document -> Generate embeddings.
    
    Args:
        file_path: Path to the document to process
        embed_config: Embedding configuration
        ingest_config: Ingestion configuration
    
    Returns:
        List of EmbeddedChunk ready for vector database
    """
    embed_config = embed_config or EmbedConfig()
    
    # Step 1: Ingest document into chunks
    print(f"Ingesting: {file_path}")
    chunks = ingest_document(file_path, config=ingest_config)
    print(f"Created {len(chunks)} chunks")
    
    if not chunks:
        print("Warning: No chunks extracted from document")
        return []
    
    # Step 2: Generate embeddings
    print(f"Generating embeddings with {embed_config.model_name}...")
    embedded = embed_chunks(chunks, embed_config)
    print(f"Generated {len(embedded)} embeddings (dim={embedded[0].dimension})")
    
    return embedded


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_embedding_dimension(config: EmbedConfig | None = None) -> int:
    """Get the embedding dimension for the configured model."""
    config = config or EmbedConfig()
    model = get_model(config)
    return model.get_sentence_embedding_dimension()


def warmup_model(config: EmbedConfig | None = None) -> None:
    """Pre-load the model to avoid first-call latency."""
    config = config or EmbedConfig()
    get_model(config)
    print("Model warmed up and ready")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embed.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Run embedding pipeline
    embedded_chunks = embed_document(file_path)
    
    # Show summary
    print("\n" + "=" * 50)
    print(f"Processed: {file_path}")
    print(f"Total chunks: {len(embedded_chunks)}")
    if embedded_chunks:
        print(f"Embedding dimension: {embedded_chunks[0].dimension}")
        
        # Sample
        print("\nSample chunk:")
        sample = embedded_chunks[0]
        print(f"  ID: {sample.id}")
        print(f"  Text: {sample.text[:100]}...")
        print(f"  Embedding (first 5): {sample.embedding[:5]}")
