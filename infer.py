"""
RAG Inference Module
====================
Local RAG (Retrieval-Augmented Generation) pipeline using:
- Qdrant for vector search
- sentence-transformers for embeddings  
- llama.cpp for local LLM inference

Usage:
    python infer.py "What is the attendance policy?"
    python infer.py "grading system" --collection test_docs
    python infer.py --interactive  # Chat mode
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from llama_cpp import Llama

from search_db import (
    search,
    retrieve_context,
    SearchConfig,
    build_filter,
    Filter,
)
from embed import EmbedConfig

# Structure / Entity Store Integration
try:
    from structure.fact_store import EntityStore
    STRUCTURE_AVAILABLE = True
except ImportError:
    STRUCTURE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for local LLM."""
    
    # Model path - automatically searched in ./models or ~/ai/models
    model_path: str = None
    
    # Context window size
    n_ctx: int = 16384
    
    # GPU layers (for Intel iGPU, keep low or 0)
    n_gpu_layers: int = 0
    
    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Verbose output from llama.cpp
    verbose: bool = False


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    embed_config: EmbedConfig = field(default_factory=EmbedConfig)
    
    # Number of chunks to retrieve (reduced to 3 to make room for ADE context)
    top_k: int = 3
    
    # Max context tokens (approximate - chars / 4)
    max_context_tokens: int = 2000
    
    # Entity Store Integration (replaces old ADE and Fact Store)
    use_entities: bool = True
    entity_store_path: str = "./fact_store/entities.db"
    max_entities: int = 20  # Max entities to include in context
    
    # Legacy settings (deprecated)
    use_ade: bool = False
    ade_store_path: str = "./ade_store"
    use_tables: bool = False  # Replaced by use_entities
    fact_store_path: str = "./fact_store/facts.db"  # Deprecated


# =============================================================================
# RAG PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the context provided below
2. If the context doesn't contain enough information, say "I don't have enough information to answer that"
3. Be concise and direct
4. Cite the source/section when relevant
5. Do NOT make up information not in the context"""

RAG_USER_TEMPLATE = """Context from documents:
---
{context}
---

Question: {question}

Answer based on the context above:"""

NO_CONTEXT_TEMPLATE = """Question: {question}

Note: No relevant context was found in the knowledge base. Answer based on your general knowledge, or indicate if you cannot answer."""


# =============================================================================
# LLM SINGLETON
# =============================================================================

def find_llm_model(provided_path: str = None) -> str:
    """Resolve LLM model path by checking relative and home directories."""
    default_name = "qwen2.5-3b-instruct-q6_k.gguf"
    
    # Use provided path if exists
    if provided_path and os.path.exists(provided_path):
        return provided_path
    
    # Check relative path
    rel_path = f"./models/{default_name}"
    if os.path.exists(rel_path):
        return rel_path
    
    # Check home directory path
    home_path = os.path.expanduser(f"~/ai/models/{default_name}")
    if os.path.exists(home_path):
        return home_path
        
    return provided_path or home_path


_llm_cache: dict[str, Llama] = {}


def get_llm(config: LLMConfig) -> Llama:
    """Initialize or retrieve local LLM."""
    model_path = find_llm_model(config.model_path)
    
    if model_path not in _llm_cache:
        print(f"Loading LLM from: {model_path}")
        try:
            _llm_cache[model_path] = Llama(
                model_path=model_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                verbose=config.verbose,
            )
            print("LLM loaded successfully")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            raise
            
    return _llm_cache[model_path]


# =============================================================================
# ADE RETRIEVAL
# =============================================================================

# Try to import ADE store
try:
    from ade import ADEStore
    from ade.ade_store import StoreConfig
    ADE_AVAILABLE = True
except ImportError:
    ADE_AVAILABLE = False


def retrieve_ade_facts(
    query: str,
    config: RAGConfig,
) -> tuple[str, dict]:
    """
    Retrieve relevant structured facts from ADE store.
    
    Uses keyword matching to find relevant entities and rules.
    
    Args:
        query: User query
        config: RAG configuration
        
    Returns:
        Tuple of (formatted_facts_string, stats_dict)
    """
    if not ADE_AVAILABLE or not config.use_ade:
        return "", {}
    
    from pathlib import Path
    
    # Check if ADE store exists
    store_path = Path(config.ade_store_path)
    if not store_path.exists():
        return "", {}
    
    try:
        store_config = StoreConfig(storage_path=config.ade_store_path)
        store = ADEStore(store_config)
    except Exception:
        return "", {}
    
    # Extract keywords from query for entity search
    # Simple approach: use words > 3 chars as keywords
    words = query.lower().split()
    keywords = [w.strip("?.,!") for w in words if len(w) > 3]
    
    found_entities = []
    found_rules = []
    
    # Search for entities by each keyword
    for keyword in keywords[:5]:  # Limit keywords to check
        try:
            results = store.search_entities(keyword)
            for entity in results:
                if entity not in found_entities:
                    found_entities.append(entity)
        except Exception:
            continue
    
    # Get rules from the store
    try:
        sources = store.get_all_sources()
        for source_id in sources[:1]:  # Just first source for now
            data = store.get(source_id)
            if data and 'extractions' in data:
                for chunk_id, extraction in data['extractions'].items():
                    rules = extraction.get('data', {}).get('rules', [])
                    for rule in rules:
                        if rule not in found_rules:
                            # Check if rule is relevant to query
                            if any(kw in rule.lower() for kw in keywords):
                                found_rules.append(rule)
    except Exception:
        pass
    
    # Limit results
    found_entities = found_entities[:config.ade_max_entities]
    found_rules = found_rules[:config.ade_max_rules]
    
    if not found_entities and not found_rules:
        return "", {}
    
    # Format facts for context
    facts_parts = []
    
    if found_entities:
        facts_parts.append("=== Extracted Structured Facts ===")
        for entity in found_entities:
            entity_type = entity.get('entity_type', 'item')
            name = entity.get('name', 'Unknown')
            attrs = entity.get('attributes', {})
            
            entity_str = f"• [{entity_type}] {name}"
            if attrs:
                attr_strs = [f"  - {k}: {v}" for k, v in list(attrs.items())[:3]]
                entity_str += "\n" + "\n".join(attr_strs)
            facts_parts.append(entity_str)
    
    if found_rules:
        facts_parts.append("\n=== Relevant Rules/Policies ===")
        for rule in found_rules:
            # Truncate long rules
            if len(rule) > 150:
                rule = rule[:150] + "..."
            facts_parts.append(f"• {rule}")
    
    facts_context = "\n".join(facts_parts)
    
    stats = {
        "entities_found": len(found_entities),
        "rules_found": len(found_rules),
    }
    
    return facts_context, stats


def retrieve_entities(
    query: str,
    config: RAGConfig,
) -> tuple[str, dict]:
    """
    Retrieve entities from the Entity Store.
    
    Uses smart query detection to determine entity type and scope filters.
    """
    if not STRUCTURE_AVAILABLE or not config.use_entities:
        return "", {}
        
    from pathlib import Path
    if not Path(config.entity_store_path).exists():
        return "", {}
        
    try:
        store = EntityStore(config.entity_store_path)
    except Exception:
        return "", {}
    
    query_lower = query.lower()
    found_entities = []
    
    # Detect query type for smart filtering
    listing_keywords = ['semester', 'course', 'papers', 'subjects', 'list', 'all', 'what are']
    is_listing_query = any(kw in query_lower for kw in listing_keywords)
    
    # Detect scope from query (e.g., "semester 1", "semester IV")
    import re
    scope_match = re.search(r'semester\s*([IVXLCDM]+|\d+)', query, re.IGNORECASE)
    scope_filter = None
    if scope_match:
        scope_filter = f"Semester {scope_match.group(1).upper()}"
    
    if is_listing_query:
        # For listing queries, return course entities with scope filtering
        found_entities = store.search_entities(
            entity_type='course',
            scope=scope_filter,
            limit=config.max_entities,
        )
    else:
        # General search - search by query text
        search_terms = [w for w in query.split() if len(w) > 3 and w.lower() not in ['what', 'which', 'where', 'when', 'that', 'this', 'from']]
        
        for term in search_terms[:3]:
            results = store.search_entities(
                query=term,
                limit=config.max_entities // 3,
            )
            for e in results:
                if not any(existing.id == e.id for existing in found_entities):
                    found_entities.append(e)
        
        found_entities = found_entities[:config.max_entities]
    
    if not found_entities:
        return "", {}
    
    # Format entities for context
    lines = ["=== Extracted Entities ==="]
    
    for entity in found_entities:
        e_str = f"• [{entity.type.upper()}] {entity.canonical_name}"
        
        # Add key attributes
        attrs = []
        for key, value in entity.attributes.items():
            if key not in ['pattern', 'original_text'] and value:
                attrs.append(f"{key}: {value}")
        
        if attrs:
            e_str += f" ({', '.join(attrs[:3])})"  # Limit to 3 attrs
        
        if entity.scope:
            e_str += f" [Scope: {entity.scope}]"
        
        lines.append(e_str)
    
    stats = {"entities_found": len(found_entities)}
    return "\n".join(lines), stats


# Legacy alias for backwards compatibility
def retrieve_table_facts(query: str, config: RAGConfig) -> tuple[str, dict]:
    """Legacy function - redirects to retrieve_entities."""
    return retrieve_entities(query, config)


# =============================================================================
# RAG PIPELINE
# =============================================================================

def retrieve(
    query: str,
    config: RAGConfig,
    query_filter: Filter | None = None,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant context for a query.
    
    Combines:
    1. Entities from Entity Store (structured extraction)
    2. Vector-retrieved chunks from Qdrant
    
    Returns:
        Tuple of (context_string, list_of_sources)
    """
    context_parts = []
    sources = []
    total_chars = 0
    max_chars = config.max_context_tokens * 4
    
    # Step 1: Retrieve Entities (from structured extraction)
    entity_context, entity_stats = retrieve_entities(query, config)
    if entity_context:
        context_parts.append(entity_context)
        total_chars += len(entity_context)
        if entity_stats.get('entities_found'):
            sources.append({
                "source": "Entity Store",
                "title": f"{entity_stats.get('entities_found')} entities",
                "score": 1.0
            })
    
    # Step 2: Vector search for relevant chunks
    search_config = SearchConfig(
        qdrant_path=config.search_config.qdrant_path,
        collection_name=config.search_config.collection_name,
        top_k=config.top_k,
        embed_config=config.embed_config,
    )
    
    results = search(query, config=search_config, query_filter=query_filter)
    
    if results:
        context_parts.append("\n=== Retrieved Document Sections ===")
        
        for result in results:
            source_name = Path(result.source_id).name if result.source_id else "Unknown"
            title = result.title[:50] if result.title else "Untitled"
            
            chunk_text = f"[Source: {source_name} | Section: {title}]\n{result.text}\n"
            
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            sources.append({
                "source": source_name,
                "title": result.title,
                "score": result.score,
            })
            total_chars += len(chunk_text)
    
    if not context_parts:
        return "", []
    
    context = "\n---\n".join(context_parts)
    return context, sources


def generate(
    query: str,
    context: str,
    config: RAGConfig,
    stream: bool = False,
) -> str:
    """
    Generate a response using the LLM with retrieved context.
    """
    llm = get_llm(config.llm_config)
    
    # Build prompt
    if context:
        user_content = RAG_USER_TEMPLATE.format(
            context=context,
            question=query,
        )
    else:
        user_content = NO_CONTEXT_TEMPLATE.format(question=query)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    if stream:
        return generate_stream(llm, messages, config.llm_config)
    
    # Generate response (non-streaming)
    response = llm.create_chat_completion(
        messages,
        max_tokens=config.llm_config.max_tokens,
        temperature=config.llm_config.temperature,
        top_p=config.llm_config.top_p,
    )
    
    return response["choices"][0]["message"]["content"]


def generate_stream(llm: Llama, messages: list, llm_config: LLMConfig) -> str:
    """
    Generate streaming response - prints tokens as they arrive.
    Returns the complete response.
    """
    response_text = ""
    
    for chunk in llm.create_chat_completion(
        messages,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            token = delta["content"]
            print(token, end="", flush=True)
            response_text += token
    
    print()  # Newline after streaming completes
    return response_text


def stream_response(llm: Llama, query: str, llm_config: LLMConfig) -> str:
    """
    Stream a direct LLM response (no RAG).
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    return generate_stream(llm, messages, llm_config)


def rag_query(
    query: str,
    config: RAGConfig | None = None,
    query_filter: Filter | None = None,
    verbose: bool = False,
) -> dict:
    """
    Full RAG pipeline: Retrieve -> Generate.
    
    Args:
        query: User question
        config: RAG configuration
        query_filter: Optional metadata filter
        verbose: Print intermediate steps
    
    Returns:
        Dict with 'answer', 'sources', 'context'
    """
    config = config or RAGConfig()
    
    # Step 1: Retrieve context
    if verbose:
        print(f"Retrieving context for: {query}")
    
    context, sources = retrieve(query, config, query_filter)
    
    if verbose:
        print(f"Retrieved {len(sources)} sources")
        for s in sources:
            print(f"  - {s['source']}: {s['title'][:40]}... (score: {s['score']:.3f})")
    
    # Step 2: Generate answer
    if verbose:
        print("Generating response...")
    
    answer = generate(query, context, config)
    
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "context": context,
        "has_context": bool(context),
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def ask(
    question: str,
    collection: str = "documents",
    top_k: int = 5,
    verbose: bool = False,
) -> str:
    """
    Simple interface for asking questions.
    
    Args:
        question: Your question
        collection: Qdrant collection name
        top_k: Number of chunks to retrieve
        verbose: Show retrieval details
    
    Returns:
        Answer string
    """
    config = RAGConfig(
        search_config=SearchConfig(collection_name=collection),
        top_k=top_k,
    )
    
    result = rag_query(question, config=config, verbose=verbose)
    return result["answer"]


def ask_document(
    question: str,
    document_path: str,
    collection: str = "documents",
    verbose: bool = False,
) -> str:
    """
    Ask a question about a specific document only.
    """
    config = RAGConfig(
        search_config=SearchConfig(collection_name=collection),
    )
    
    query_filter = build_filter(source_id=document_path)
    result = rag_query(question, config=config, query_filter=query_filter, verbose=verbose)
    return result["answer"]


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(config: RAGConfig, use_rag: bool = False):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    if use_rag:
        print("Interactive Mode (RAG Enabled)")
        print(f"Collection: {config.search_config.collection_name}")
    else:
        print("Interactive Mode (Direct LLM)")
    print(f"Model: {Path(config.llm_config.model_path).name}")
    print("Type 'quit' or 'exit' to end")
    if use_rag:
        print("Type 'sources' to show last sources")
    print("=" * 60 + "\n")
    
    # Pre-load LLM
    llm = get_llm(config.llm_config)
    
    last_result = None
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        if query.lower() == "sources" and last_result and use_rag:
            print("\nSources from last answer:")
            for s in last_result["sources"]:
                print(f"  - {s['source']}: {s['title'][:50]}... (score: {s['score']:.3f})")
            continue
        
        if use_rag:
            # Run RAG query with streaming
            context, sources = retrieve(query, config)
            last_result = {"sources": sources}
            
            print("\nAssistant: ", end="", flush=True)
            generate(query, context, config, stream=True)
            
            if sources:
                print(f"\n[{len(sources)} sources used]")
        else:
            # Direct LLM query with streaming
            print("\nAssistant: ", end="", flush=True)
            stream_response(llm, query, config.llm_config)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG inference with local LLM"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask",
    )
    parser.add_argument(
        "--collection", "-c",
        default="documents",
        help="Qdrant collection name (default: documents)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive chat mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieval details",
    )
    parser.add_argument(
        "--rag", "-r",
        action="store_true",
        help="Enable RAG mode (retrieve context from knowledge base)",
    )
    parser.add_argument(
        "--no-ade",
        action="store_true",
        help="Disable ADE structured facts retrieval",
    )
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_data",
        help="Path to Qdrant data (default: ./qdrant_data)",
    )
    
    # LLM generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=f"Max tokens to generate (default: {LLMConfig.max_tokens})",
    )
    parser.add_argument(
        "--temperature", "--temp",
        type=float,
        help=f"Sampling temperature (default: {LLMConfig.temperature})",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help=f"Top-p (nucleus) sampling (default: {LLMConfig.top_p})",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        help=f"Context window size (default: {LLMConfig.n_ctx})",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        help=f"Number of GPU layers (default: {LLMConfig.n_gpu_layers})",
    )
    
    args = parser.parse_args()
    
    # Build config
    llm_config = LLMConfig()
    if args.model:
        llm_config.model_path = args.model
    if args.max_tokens:
        llm_config.max_tokens = args.max_tokens
    if args.temperature:
        llm_config.temperature = args.temperature
    if args.top_p:
        llm_config.top_p = args.top_p
    if args.n_ctx:
        llm_config.n_ctx = args.n_ctx
    if args.n_gpu_layers is not None:
        llm_config.n_gpu_layers = args.n_gpu_layers
    
    search_config = SearchConfig(
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
    )
    
    config = RAGConfig(
        llm_config=llm_config,
        search_config=search_config,
        top_k=args.top_k,
        use_ade=not args.no_ade,
    )
    
    # Interactive mode
    if args.interactive:
        interactive_mode(config, use_rag=args.rag)
        return
    
    # Single query mode
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    if args.rag:
        # Full RAG query with streaming
        if args.verbose:
            print(f"Retrieving context for: {args.query}")
        
        context, sources = retrieve(args.query, config)
        
        if args.verbose:
            print(f"Retrieved {len(sources)} sources")
            for s in sources:
                print(f"  - {s['source']}: {s['title'][:40]}... (score: {s['score']:.3f})")
            print("Generating response...")
        
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        generate(args.query, context, config, stream=True)
        
        if sources:
            print("\n" + "-" * 60)
            print("Sources:")
            for s in sources:
                print(f"  - {s['source']}: {s['title'][:50]}...")
    else:
        # Direct LLM query with streaming (default)
        llm = get_llm(config.llm_config)
        stream_response(llm, args.query, config.llm_config)


if __name__ == "__main__":
    main()
