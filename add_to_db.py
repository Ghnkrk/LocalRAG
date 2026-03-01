"""
Add to Database Module
======================
Main entry point for adding documents to the Qdrant vector database.
Handles the full pipeline: ingest -> embed -> store.

Supports:
- Creating new collections
- Upserting to existing collections
- Batch processing for efficiency
- Local Qdrant (file-based or server mode)
- Optional ADE (Agentic Document Extraction) for structured facts

Usage:
    python add_to_db.py <file_path> [--collection <name>] [--recreate]
    python add_to_db.py <file_path> --ade  # Enable structured extraction
"""

import sys
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    CollectionInfo,
)

from ingest import IngestConfig
from embed import (
    embed_document,
    EmbedConfig,
    EmbeddedChunk,
    EmbeddedChunk,
    get_embedding_dimension,
)

# Structure / Extraction Pipeline Integration
try:
    from extraction.pipeline import ExtractionPipeline, PipelineConfig
    from structure.fact_store import EntityStore
    EXTRACTION_AVAILABLE = True
except ImportError:
    EXTRACTION_AVAILABLE = False

# Legacy: ADE imports (deprecated - will be removed)
ADE_AVAILABLE = False  # Disabled in favor of new extraction pipeline

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DBConfig:
    """Configuration for vector database operations."""
    
    # Qdrant connection
    # For local file-based storage (no server needed):
    qdrant_path: str = "./qdrant_data"
    
    # Or for Qdrant server (uncomment and set):
    # qdrant_url: str = "http://localhost:6333"
    
    # Collection settings
    collection_name: str = "documents"
    
    # Distance metric for similarity search
    distance: Distance = Distance.COSINE
    
    # Batch size for upserting points
    upsert_batch_size: int = 100
    
    # Whether to recreate collection if exists
    recreate_collection: bool = False
    
    # Entity Extraction (replaces old ADE and table extraction)
    enable_extraction: bool = True  # Run structured entity extraction
    entity_store_path: str = "./fact_store/entities.db"  # Entity store location
    
    # Legacy settings (deprecated)
    enable_ade: bool = False  # Disabled
    ade_store_path: str = "./ade_store"


# =============================================================================
# DATABASE CLIENT
# =============================================================================

_client_cache: dict[str, QdrantClient] = {}


def get_client(config: DBConfig) -> QdrantClient:
    """Get or create Qdrant client (cached)."""
    cache_key = config.qdrant_path
    
    if cache_key not in _client_cache:
        # Create path if using file-based storage
        path = Path(config.qdrant_path)
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"Connecting to Qdrant at: {config.qdrant_path}")
        _client_cache[cache_key] = QdrantClient(path=str(path))
    
    return _client_cache[cache_key]


# =============================================================================
# COLLECTION MANAGEMENT
# =============================================================================

def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists."""
    try:
        collections = client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    except Exception:
        return False


def create_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int,
    distance: Distance = Distance.COSINE,
) -> None:
    """Create a new collection."""
    print(f"Creating collection: {collection_name} (dim={dimension})")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dimension,
            distance=distance,
        ),
    )


def ensure_collection(
    client: QdrantClient,
    config: DBConfig,
    embed_config: EmbedConfig,
) -> None:
    """Ensure collection exists, create if needed."""
    exists = collection_exists(client, config.collection_name)
    
    if exists and config.recreate_collection:
        print(f"Recreating collection: {config.collection_name}")
        client.delete_collection(config.collection_name)
        exists = False
    
    if not exists:
        dimension = get_embedding_dimension(embed_config)
        create_collection(
            client,
            config.collection_name,
            dimension,
            config.distance,
        )
    else:
        print(f"Using existing collection: {config.collection_name}")


def get_collection_info(client: QdrantClient, collection_name: str) -> CollectionInfo:
    """Get information about a collection."""
    return client.get_collection(collection_name)


# =============================================================================
# DOCUMENT HASHING (for deduplication)
# =============================================================================

def compute_document_hash(file_path: str) -> str:
    """Compute a hash of the document for tracking."""
    path = Path(file_path)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def compute_chunk_hash(text: str, source_id: str) -> str:
    """Compute a hash for a chunk (for deduplication)."""
    content = f"{source_id}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# UPSERT OPERATIONS
# =============================================================================

def chunks_to_points(
    chunks: list[EmbeddedChunk],
    source_hash: str,
) -> list[PointStruct]:
    """Convert embedded chunks to Qdrant points."""
    points = []
    
    for chunk in chunks:
        # Add source hash to metadata for tracking
        metadata = {
            **chunk.metadata,
            "source_hash": source_hash,
        }
        
        points.append(PointStruct(
            id=chunk.id,  # UUID as string
            vector=chunk.embedding,
            payload={
                "text": chunk.text,
                **metadata,
            },
        ))
    
    return points


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[EmbeddedChunk],
    source_hash: str,
    batch_size: int = 100,
) -> int:
    """
    Upsert embedded chunks to the collection.
    
    Returns:
        Number of points upserted
    """
    points = chunks_to_points(chunks, source_hash)
    
    total_upserted = 0
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )
        
        total_upserted += len(batch)
        print(f"  Upserted {total_upserted}/{len(points)} points")
    
    return total_upserted


def delete_by_source(
    client: QdrantClient,
    collection_name: str,
    source_id: str,
) -> None:
    """Delete all chunks from a specific source."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_id),
                )
            ]
        ),
    )
    print(f"Deleted points from source: {source_id}")


# =============================================================================
# ADE INTEGRATION
# =============================================================================

def run_ade_on_chunks(
    chunks: list[EmbeddedChunk],
    source_id: str,
    store_path: str = "./ade_store",
) -> dict:
    """
    Run ADE extraction on embedded chunks.
    Only processes chunks that pass the router heuristics.
    
    Args:
        chunks: List of embedded chunks
        source_id: Source document identifier
        store_path: Path for ADE store
        
    Returns:
        Stats dict with extraction results
    """
    if not ADE_AVAILABLE:
        logger.warning("ADE not available - skipping extraction")
        return {"error": "ADE not available"}
    
    from ade.ade_store import StoreConfig
    
    print("\nRunning ADE extraction...")
    
    stats = {
        "total_chunks": len(chunks),
        "routed": 0,
        "extracted": 0,
        "entities": 0,
        "rules": 0,
        "lists": 0,
    }
    
    # Initialize store
    store_config = StoreConfig(storage_path=store_path)
    store = ADEStore(store_config)
    
    # Initialize ADE config (low temperature, small output)
    ade_config = ADEConfig(
        temperature=0.0,
        max_tokens=400,
    )
    
    for chunk in chunks:
        text = chunk.text
        chunk_id = chunk.id
        
        # Check if chunk should be processed
        if not should_run_ade(text):
            continue
        
        stats["routed"] += 1
        
        # Run extraction
        result = run_ade(
            text,
            config=ade_config,
            source_id=source_id,
            chunk_id=chunk_id,
        )
        
        if result is None or result.is_empty():
            continue
        
        # Store result
        store.store(result)
        
        stats["extracted"] += 1
        stats["entities"] += len(result.entities)
        stats["rules"] += len(result.rules)
        stats["lists"] += len(result.lists)
        
        logger.info(f"ADE chunk {chunk_id}: {result.summary()}")
    
    print(f"  Routed to ADE: {stats['routed']}/{stats['total_chunks']} chunks")
    print(f"  Extracted: {stats['extracted']} chunks")
    print(f"  Facts: {stats['entities']} entities, {stats['rules']} rules, {stats['lists']} lists")
    
    return stats


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def add_document_to_db(
    file_path: str,
    db_config: DBConfig | None = None,
    embed_config: EmbedConfig | None = None,
    ingest_config: IngestConfig | None = None,
    replace_existing: bool = True,
) -> dict:
    """
    Full pipeline: Ingest -> Embed -> Store in Qdrant -> (optional) ADE.
    
    Args:
        file_path: Path to the document
        db_config: Database configuration
        embed_config: Embedding configuration
        ingest_config: Ingestion configuration
        replace_existing: If True, delete existing chunks from same source before adding
    
    Returns:
        Summary dict with operation results
    """
    db_config = db_config or DBConfig()
    embed_config = embed_config or EmbedConfig()
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    source_id = str(path.absolute())
    source_hash = compute_document_hash(file_path)
    
    print("=" * 60)
    print(f"Adding document to database")
    print(f"  File: {file_path}")
    print(f"  Collection: {db_config.collection_name}")
    print(f"  Source hash: {source_hash}")
    if db_config.enable_ade:
        print(f"  ADE: Disabled (use enable_extraction instead)")
    if db_config.enable_extraction:
        print(f"  Entity Extraction: Enabled")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # STEP 0: Entity Extraction Pipeline (Tables, NER, Relations)
    # -------------------------------------------------------------------------
    ingest_config = ingest_config or IngestConfig()
    extraction_stats = None
    
    if db_config.enable_extraction and EXTRACTION_AVAILABLE:
        print(f"Running extraction pipeline on: {file_path}")
        try:
            pipeline_config = PipelineConfig(store_path=db_config.entity_store_path)
            pipeline = ExtractionPipeline(pipeline_config)
            extraction_stats = pipeline.process_document(file_path, clear_existing=True)
            
            print(f"  ✓ Entities: {extraction_stats['entities_extracted']} extracted, {extraction_stats['entities_stored']} stored")
            print(f"  ✓ Relations: {extraction_stats['relations_extracted']} extracted")
            
            # Exclude tables from vector embedding to avoid duplication
            ingest_config.chunk_config.exclude_categories.add("Table")
        except Exception as e:
            import traceback
            print(f"  ! Extraction error: {e}")
            traceback.print_exc()

    # Step 1: Embed document (includes ingestion)
    embedded_chunks = embed_document(
        file_path,
        embed_config=embed_config,
        ingest_config=ingest_config,
    )
    
    if not embedded_chunks:
        return {
            "status": "error",
            "message": "No chunks extracted from document",
            "file_path": file_path,
        }
    
    # Step 2: Connect to Qdrant
    client = get_client(db_config)
    
    # Step 3: Ensure collection exists
    ensure_collection(client, db_config, embed_config)
    
    # Step 4: Delete existing chunks from same source (if replace mode)
    if replace_existing:
        try:
            delete_by_source(client, db_config.collection_name, source_id)
        except Exception as e:
            # Collection might be empty or source doesn't exist - that's fine
            pass
    
    # Step 5: Upsert new chunks
    print("\nUpserting to database...")
    upserted = upsert_chunks(
        client,
        db_config.collection_name,
        embedded_chunks,
        source_hash,
        db_config.upsert_batch_size,
    )
    
    # Step 6: Run ADE extraction (optional)
    ade_stats = None
    if db_config.enable_ade:
        ade_stats = run_ade_on_chunks(
            embedded_chunks,
            source_id=source_id,
            store_path=db_config.ade_store_path,
        )
    
    # Step 7: Get final stats
    info = get_collection_info(client, db_config.collection_name)
    
    result = {
        "status": "success",
        "file_path": file_path,
        "source_id": source_id,
        "source_hash": source_hash,
        "chunks_added": upserted,
        "collection_name": db_config.collection_name,
        "total_points_in_collection": info.points_count,
        "embedding_dimension": embedded_chunks[0].dimension,
    }
    
    if ade_stats:
        result["ade"] = ade_stats
    
    if extraction_stats:
        result["extraction"] = extraction_stats
    
    print("\n" + "=" * 60)
    print("✓ Document added successfully")
    print(f"  Chunks added: {upserted}")
    print(f"  Total in collection: {info.points_count}")
    if extraction_stats:
        print(f"  Entities: {extraction_stats['entities_stored']} stored")
        print(f"  Relations: {extraction_stats['relations_stored']} stored")
    print("=" * 60)
    
    return result


def add_directory_to_db(
    directory_path: str,
    extensions: list[str] | None = None,
    db_config: DBConfig | None = None,
    embed_config: EmbedConfig | None = None,
    ingest_config: IngestConfig | None = None,
) -> list[dict]:
    """
    Add all documents from a directory to the database.
    
    Args:
        directory_path: Path to directory
        extensions: File extensions to include (e.g., ['.pdf', '.docx'])
        db_config: Database configuration
        embed_config: Embedding configuration
        ingest_config: Ingestion configuration
    
    Returns:
        List of result dicts for each file
    """
    extensions = extensions or [".pdf", ".docx", ".txt", ".md", ".html"]
    
    path = Path(directory_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")
    
    results = []
    
    for ext in extensions:
        for file_path in path.glob(f"**/*{ext}"):
            try:
                result = add_document_to_db(
                    str(file_path),
                    db_config=db_config,
                    embed_config=embed_config,
                    ingest_config=ingest_config,
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "file_path": str(file_path),
                    "message": str(e),
                })
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add documents to Qdrant vector database"
    )
    parser.add_argument(
        "file_path",
        help="Path to document or directory",
    )
    parser.add_argument(
        "--collection", "-c",
        default="documents",
        help="Collection name (default: documents)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if exists",
    )
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_data",
        help="Path for Qdrant data (default: ./qdrant_data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--ade",
        action="store_true",
        help="Enable ADE (Agentic Document Extraction) for structured facts",
    )
    parser.add_argument(
        "--ade-store-path",
        default="./ade_store",
        help="Path for ADE storage (default: ./ade_store)",
    )
    
    args = parser.parse_args()
    
    # Check ADE availability
    if args.ade and not ADE_AVAILABLE:
        print("Error: ADE module not available. Please check ade/ directory.")
        sys.exit(1)
    
    db_config = DBConfig(
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        recreate_collection=args.recreate,
        enable_ade=args.ade,
        ade_store_path=args.ade_store_path,
    )
    
    embed_config = EmbedConfig(
        batch_size=args.batch_size,
    )
    
    path = Path(args.file_path)
    
    if path.is_dir():
        results = add_directory_to_db(
            args.file_path,
            db_config=db_config,
            embed_config=embed_config,
        )
        
        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success
        print(f"\nProcessed {len(results)} files: {success} success, {failed} failed")
    else:
        result = add_document_to_db(
            args.file_path,
            db_config=db_config,
            embed_config=embed_config,
        )
        
        if result["status"] != "success":
            print(f"Error: {result.get('message', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
