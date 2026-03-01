"""
Inference Latency Benchmark
===========================
Measures detailed latency of the inference pipeline:
- Query embedding generation
- Vector search (retrieval)
- LLM generation (time to first token, tokens/sec, total)
- Total pipeline time

Usage:
    python benchmark.py "your query" --collection test_docs
    python benchmark.py --runs 5  # Multiple runs for averaging
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llama_cpp import Llama

from embed import EmbedConfig, embed_texts, get_model as get_embed_model
from search_db import SearchConfig, get_client, Filter
from qdrant_client.models import FieldCondition, MatchValue
from infer import (
    LLMConfig,
    RAGConfig,
    get_llm,
    SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    retrieve_ade_facts,
    retrieve_table_facts,
    ADE_AVAILABLE,
    STRUCTURE_AVAILABLE,
)


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    duration_ms: float
    details: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.name}: {self.duration_ms:.2f}ms"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def result(self, **details) -> TimingResult:
        return TimingResult(self.name, self.duration_ms, details)


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_embedding(query: str, config: EmbedConfig) -> TimingResult:
    """Benchmark query embedding generation."""
    with Timer("Query Embedding") as t:
        embedding = embed_texts([query], config)[0]
    
    return t.result(
        embedding_dim=len(embedding),
        query_length=len(query),
    )


def benchmark_search(
    query_embedding: list[float],
    collection_name: str,
    qdrant_path: str,
    top_k: int = 5,
) -> tuple[TimingResult, list]:
    """Benchmark vector search."""
    client = get_client(qdrant_path)
    
    with Timer("Vector Search") as t:
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
    
    return t.result(
        results_count=len(results.points),
        top_k=top_k,
    ), results.points


def benchmark_context_building(results: list) -> tuple[TimingResult, str]:
    """Benchmark context string building."""
    with Timer("Context Building") as t:
        context_parts = []
        for point in results:
            source = Path(point.payload.get("source_id", "")).name
            title = point.payload.get("title", "Untitled")[:50]
            text = point.payload.get("text", "")
            context_parts.append(f"[Source: {source} | Section: {title}]\n{text}\n")
        context = "\n---\n".join(context_parts)
    
    return t.result(
        context_length=len(context),
        chunks_used=len(results),
    ), context


def benchmark_ade_retrieval(
    query: str,
    config: RAGConfig,
) -> tuple[TimingResult, str, dict]:
    """Benchmark ADE structured facts retrieval."""
    with Timer("ADE Retrieval") as t:
        ade_context, ade_stats = retrieve_ade_facts(query, config)
    
    return t.result(
        entities_found=ade_stats.get('entities_found', 0),
        rules_found=ade_stats.get('rules_found', 0),
        context_length=len(ade_context),
    ), ade_context, ade_stats


def benchmark_fact_store_retrieval(
    query: str,
    config: RAGConfig,
) -> tuple[TimingResult, str, dict]:
    """Benchmark Fact Store (table facts) retrieval."""
    with Timer("Fact Store Retrieval") as t:
        fact_context, fact_stats = retrieve_table_facts(query, config)
    
    return t.result(
        facts_found=fact_stats.get('table_facts_found', 0),
        context_length=len(fact_context),
    ), fact_context, fact_stats


def benchmark_llm_generation(
    query: str,
    context: str,
    llm: Llama,
    llm_config: LLMConfig,
) -> TimingResult:
    """Benchmark LLM generation with detailed token metrics."""
    
    # Build prompt
    user_content = RAG_USER_TEMPLATE.format(
        context=context,
        question=query,
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    # Measure generation
    tokens_generated = 0
    first_token_time = None
    response_text = ""
    
    start_time = time.perf_counter()
    
    for chunk in llm.create_chat_completion(
        messages,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        stream=True,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            tokens_generated += 1
            response_text += delta["content"]
    
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    generation_time_ms = (end_time - first_token_time) * 1000 if first_token_time else total_time_ms
    tokens_per_sec = tokens_generated / (generation_time_ms / 1000) if generation_time_ms > 0 else 0
    
    return TimingResult(
        name="LLM Generation",
        duration_ms=total_time_ms,
        details={
            "time_to_first_token_ms": ttft_ms,
            "generation_time_ms": generation_time_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_sec,
            "response_length": len(response_text),
            "prompt_length": len(user_content),
        }
    )


def benchmark_direct_llm(
    query: str,
    llm: Llama,
    llm_config: LLMConfig,
) -> TimingResult:
    """Benchmark direct LLM generation (no RAG)."""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    
    tokens_generated = 0
    first_token_time = None
    response_text = ""
    
    start_time = time.perf_counter()
    
    for chunk in llm.create_chat_completion(
        messages,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        stream=True,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            tokens_generated += 1
            response_text += delta["content"]
    
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    generation_time_ms = (end_time - first_token_time) * 1000 if first_token_time else total_time_ms
    tokens_per_sec = tokens_generated / (generation_time_ms / 1000) if generation_time_ms > 0 else 0
    
    return TimingResult(
        name="LLM Generation (Direct)",
        duration_ms=total_time_ms,
        details={
            "time_to_first_token_ms": ttft_ms,
            "generation_time_ms": generation_time_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_sec,
            "response_length": len(response_text),
        }
    )


# =============================================================================
# FULL PIPELINE BENCHMARK
# =============================================================================

def run_rag_benchmark(
    query: str,
    collection: str = "documents",
    qdrant_path: str = "./qdrant_data",
    top_k: int = 3,
    llm_config: LLMConfig | None = None,
    embed_config: EmbedConfig | None = None,
    use_ade: bool = True,
    ade_store_path: str = "./ade_store",
    verbose: bool = True,
) -> dict:
    """
    Run full RAG pipeline benchmark.
    
    Returns dict with all timing results and totals.
    """
    llm_config = llm_config or LLMConfig()
    embed_config = embed_config or EmbedConfig()
    
    results = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RAG Pipeline Benchmark")
        print(f"{'='*60}")
        print(f"Query: {query[:50]}...")
        print(f"Collection: {collection}")
        print(f"Top-K: {top_k}")
        print(f"ADE: {'Enabled' if use_ade and ADE_AVAILABLE else 'Disabled'}")
        print(f"{'='*60}\n")
    
    # Create RAG config for ADE and Fact Store
    rag_config = RAGConfig(
        use_ade=use_ade and ADE_AVAILABLE,
        ade_store_path=ade_store_path,
        use_tables=STRUCTURE_AVAILABLE,
    )
    
    # 1. Query Embedding
    if verbose:
        print("1. Generating query embedding...")
    
    with Timer("Query Embedding") as t:
        query_embedding = embed_texts([query], embed_config)[0]
    results["embedding"] = t.result(dim=len(query_embedding))
    
    if verbose:
        print(f"   ✓ {results['embedding']}")
    
    # 2. Vector Search
    if verbose:
        print("2. Searching vector database...")
    
    search_result, points = benchmark_search(
        query_embedding, collection, qdrant_path, top_k
    )
    results["search"] = search_result
    
    if verbose:
        print(f"   ✓ {results['search']} ({search_result.details['results_count']} results)")
    
    # 3. ADE Retrieval (optional)
    ade_context = ""
    if use_ade and ADE_AVAILABLE:
        if verbose:
            print("3. Retrieving ADE structured facts...")
        
        ade_result, ade_context, ade_stats = benchmark_ade_retrieval(query, rag_config)
        results["ade"] = ade_result
        
        if verbose:
            entities = ade_result.details.get('entities_found', 0)
            rules = ade_result.details.get('rules_found', 0)
            print(f"   ✓ {results['ade']} ({entities} entities, {rules} rules)")
    
    # 3b. Fact Store Retrieval
    fact_context = ""
    if STRUCTURE_AVAILABLE:
        step_label = "3b" if use_ade and ADE_AVAILABLE else "3"
        if verbose:
            print(f"{step_label}. Retrieving Fact Store facts...")
        
        fact_result, fact_context, fact_stats = benchmark_fact_store_retrieval(query, rag_config)
        results["fact_store"] = fact_result
        
        if verbose:
            facts_found = fact_result.details.get('facts_found', 0)
            print(f"   ✓ {results['fact_store']} ({facts_found} facts)")
    
    # 4. Context Building
    step_num = 4 if use_ade and ADE_AVAILABLE else 3
    if verbose:
        print(f"{step_num}. Building context...")
    
    context_result, vector_context = benchmark_context_building(points)
    results["context"] = context_result
    
    # Combine Fact Store + ADE + vector context
    if fact_context:
        context = fact_context + "\n\n" + vector_context
    else:
        context = vector_context
        
    if ade_context:
        context = ade_context + "\n\n" + context
    
    if verbose:
        print(f"   ✓ {results['context']} ({len(context)} chars total)")
    
    # 5. LLM Generation
    step_num = 5 if use_ade and ADE_AVAILABLE else 4
    if verbose:
        print(f"{step_num}. Generating LLM response...")
    
    llm = get_llm(llm_config)
    llm_result = benchmark_llm_generation(query, context, llm, llm_config)
    results["llm"] = llm_result
    
    if verbose:
        print(f"   ✓ {results['llm']}")
        print(f"      - Time to first token: {llm_result.details['time_to_first_token_ms']:.2f}ms")
        print(f"      - Tokens generated: {llm_result.details['tokens_generated']}")
        print(f"      - Tokens/sec: {llm_result.details['tokens_per_second']:.2f}")
    
    # Calculate totals
    total_ms = sum(r.duration_ms for r in results.values() if isinstance(r, TimingResult))
    
    retrieval_ms = results["embedding"].duration_ms + results["search"].duration_ms + results["context"].duration_ms
    ade_ms = results.get("ade", TimingResult("", 0)).duration_ms
    fact_store_ms = results.get("fact_store", TimingResult("", 0)).duration_ms
    
    results["summary"] = {
        "total_ms": total_ms,
        "retrieval_ms": retrieval_ms,
        "ade_ms": ade_ms,
        "fact_store_ms": fact_store_ms,
        "generation_ms": results["llm"].duration_ms,
        "retrieval_pct": (retrieval_ms / total_ms) * 100 if total_ms > 0 else 0,
        "ade_pct": (ade_ms / total_ms) * 100 if total_ms > 0 else 0,
        "fact_store_pct": (fact_store_ms / total_ms) * 100 if total_ms > 0 else 0,
        "generation_pct": (results["llm"].duration_ms / total_ms) * 100 if total_ms > 0 else 0,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total Pipeline Time: {total_ms:.2f}ms ({total_ms/1000:.2f}s)")
        print(f"  - Embedding: {results['embedding'].duration_ms:.2f}ms")
        print(f"  - Vector Search: {results['search'].duration_ms:.2f}ms")
        if ade_ms > 0:
            print(f"  - ADE Retrieval: {ade_ms:.2f}ms ({results['summary']['ade_pct']:.1f}%)")
        if fact_store_ms > 0:
            print(f"  - Fact Store: {fact_store_ms:.2f}ms ({results['summary']['fact_store_pct']:.1f}%)")
        print(f"  - Context Build: {results['context'].duration_ms:.2f}ms")
        print(f"  - LLM Generation: {results['llm'].duration_ms:.2f}ms ({results['summary']['generation_pct']:.1f}%)")
        print(f"{'='*60}")
    
    return results


def run_direct_benchmark(
    query: str,
    llm_config: LLMConfig | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run direct LLM benchmark (no RAG).
    """
    llm_config = llm_config or LLMConfig()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Direct LLM Benchmark (No RAG)")
        print(f"{'='*60}")
        print(f"Query: {query[:50]}...")
        print(f"{'='*60}\n")
    
    print("Loading LLM...")
    llm = get_llm(llm_config)
    
    print("Generating response...")
    result = benchmark_direct_llm(query, llm, llm_config)
    
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Total Time: {result.duration_ms:.2f}ms ({result.duration_ms/1000:.2f}s)")
        print(f"Time to First Token: {result.details['time_to_first_token_ms']:.2f}ms")
        print(f"Tokens Generated: {result.details['tokens_generated']}")
        print(f"Tokens/Second: {result.details['tokens_per_second']:.2f}")
        print(f"{'='*60}")
    
    return {"llm": result, "summary": {"total_ms": result.duration_ms}}


def run_multiple_benchmarks(
    query: str,
    runs: int = 3,
    rag: bool = True,
    collection: str = "documents",
    **kwargs,
) -> dict:
    """
    Run multiple benchmarks and compute averages.
    """
    print(f"\nRunning {runs} benchmark iterations...")
    
    all_results = []
    
    for i in range(runs):
        print(f"\n--- Run {i+1}/{runs} ---")
        
        if rag:
            result = run_rag_benchmark(query, collection=collection, verbose=False, **kwargs)
        else:
            result = run_direct_benchmark(query, verbose=False, **kwargs)
        
        all_results.append(result)
        print(f"Total: {result['summary']['total_ms']:.2f}ms")
    
    # Compute averages
    avg_total = sum(r["summary"]["total_ms"] for r in all_results) / runs
    
    print(f"\n{'='*60}")
    print(f"AVERAGE OVER {runs} RUNS")
    print(f"{'='*60}")
    print(f"Average Total Time: {avg_total:.2f}ms ({avg_total/1000:.2f}s)")
    
    if rag:
        avg_retrieval = sum(r["summary"]["retrieval_ms"] for r in all_results) / runs
        avg_generation = sum(r["summary"]["generation_ms"] for r in all_results) / runs
        avg_tps = sum(r["llm"].details["tokens_per_second"] for r in all_results) / runs
        
        print(f"Average Retrieval: {avg_retrieval:.2f}ms")
        print(f"Average Generation: {avg_generation:.2f}ms")
        print(f"Average Tokens/sec: {avg_tps:.2f}")
    else:
        avg_tps = sum(r["llm"].details["tokens_per_second"] for r in all_results) / runs
        print(f"Average Tokens/sec: {avg_tps:.2f}")
    
    print(f"{'='*60}")
    
    return {
        "runs": runs,
        "results": all_results,
        "averages": {
            "total_ms": avg_total,
            "tokens_per_second": avg_tps,
        }
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark inference pipeline latency"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the attendance policy for students?",
        help="Query to benchmark",
    )
    parser.add_argument(
        "--collection", "-c",
        default="documents",
        help="Qdrant collection (default: documents)",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=1,
        help="Number of runs for averaging (default: 1)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Benchmark direct LLM only (no retrieval)",
    )
    parser.add_argument(
        "--no-ade",
        action="store_true",
        help="Disable ADE structured facts retrieval",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_data",
        help="Path to Qdrant data",
    )
    parser.add_argument(
        "--ade-store-path",
        default="./ade_store",
        help="Path to ADE store (default: ./ade_store)",
    )
    
    args = parser.parse_args()
    
    if args.runs > 1:
        run_multiple_benchmarks(
            args.query,
            runs=args.runs,
            rag=not args.no_rag,
            collection=args.collection,
            qdrant_path=args.qdrant_path,
            top_k=args.top_k,
            use_ade=not args.no_ade,
            ade_store_path=args.ade_store_path,
        )
    elif args.no_rag:
        run_direct_benchmark(args.query)
    else:
        run_rag_benchmark(
            args.query,
            collection=args.collection,
            qdrant_path=args.qdrant_path,
            top_k=args.top_k,
            use_ade=not args.no_ade,
            ade_store_path=args.ade_store_path,
        )


if __name__ == "__main__":
    main()
