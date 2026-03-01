"""
Extraction Pipeline
===================
Orchestrates document parsing and entity extraction.

Features:
- Combines structure parsing, table extraction, and NER
- Scope propagation from sections to entities
- Deduplication and merging of entities
- Relation inference from structure
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from structure.document_tree import DocumentTree, NodeType
from structure.entity_schema import Entity, Relation, RelationTypes
from structure.parser import parse_document, ParserConfig
from structure.table_parser import extract_tables, tables_to_entities, TableConfig
from structure.fact_store import EntityStore

from .ner import extract_entities, NERConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    
    # Parser settings
    parser_config: ParserConfig = field(default_factory=ParserConfig)
    
    # Table extraction settings
    table_config: TableConfig = field(default_factory=TableConfig)
    
    # NER settings
    ner_config: NERConfig = field(default_factory=NERConfig)
    
    # Entity store path
    store_path: str = "./fact_store/entities.db"
    
    # Processing options
    extract_from_tables: bool = True
    extract_from_text: bool = True
    infer_relations: bool = True
    deduplicate: bool = True
    
    # Minimum confidence to keep
    min_confidence: float = 0.5


# =============================================================================
# EXTRACTION PIPELINE
# =============================================================================

class ExtractionPipeline:
    """
    Full document extraction pipeline.
    
    Steps:
    1. Parse document structure (sections, paragraphs)
    2. Extract tables and convert to entities
    3. Extract entities from text using NER
    4. Infer relations from structure (scope, proximity)
    5. Deduplicate and merge entities
    6. Store in EntityStore
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.store = EntityStore(self.config.store_path)
    
    def process_document(
        self, 
        file_path: str,
        clear_existing: bool = False,
    ) -> dict:
        """
        Process a document and extract all entities.
        
        Args:
            file_path: Path to document
            clear_existing: Clear existing entities from this source
            
        Returns:
            Dict with extraction statistics
        """
        source_id = str(Path(file_path).absolute())
        
        # Clear existing if requested
        if clear_existing:
            self.store.delete_by_source(source_id)
        
        all_entities: List[Entity] = []
        all_relations: List[Relation] = []
        
        # Step 1: Parse document structure
        tree = parse_document(file_path)
        
        # Track current scope for propagation
        current_scope = ""
        scope_stack: List[str] = []
        
        # Step 2: Extract from tables
        if self.config.extract_from_tables:
            tables = extract_tables(file_path, self.config.table_config)
            
            # Assign scope to tables based on preceding sections
            for section in tree.get_sections():
                if section.scope:
                    current_scope = section.scope
                    
                # Find tables on same page after this section
                for table in tables:
                    if table.page == section.page and not table.scope:
                        table.scope = current_scope
            
            # Convert tables to entities
            table_entities = tables_to_entities(tables, source_id, self.config.table_config)
            all_entities.extend(table_entities)
        
        # Step 3: Extract from text using NER
        if self.config.extract_from_text:
            for node in tree.root.walk():
                if node.type in (NodeType.PARAGRAPH, NodeType.SECTION):
                    text_entities = extract_entities(
                        node.text,
                        self.config.ner_config,
                        source_id=source_id,
                        page=node.page,
                        scope=node.scope,
                    )
                    all_entities.extend(text_entities)
        
        # Step 4: Infer relations
        if self.config.infer_relations:
            relations = self._infer_relations(all_entities, tree)
            all_relations.extend(relations)
        
        # Step 5: Deduplicate
        if self.config.deduplicate:
            all_entities = self._deduplicate_entities(all_entities)
        
        # Filter by confidence
        all_entities = [
            e for e in all_entities 
            if e.confidence >= self.config.min_confidence
        ]
        
        # Step 6: Store
        entity_count = self.store.add_entities(all_entities)
        relation_count = self.store.add_relations(all_relations)
        
        return {
            "source_id": source_id,
            "entities_extracted": len(all_entities),
            "entities_stored": entity_count,
            "relations_extracted": len(all_relations),
            "relations_stored": relation_count,
            "tree_stats": tree.stats(),
        }
    
    def _infer_relations(
        self, 
        entities: List[Entity], 
        tree: DocumentTree,
    ) -> List[Relation]:
        """Infer relations from document structure."""
        relations = []
        
        # Build entity lookup by scope
        by_scope: dict[str, List[Entity]] = {}
        for entity in entities:
            scope = entity.scope or "global"
            if scope not in by_scope:
                by_scope[scope] = []
            by_scope[scope].append(entity)
        
        # Create BELONGS_TO relations for scoped entities
        for scope, scope_entities in by_scope.items():
            if scope == "global":
                continue
            
            # Find or create scope entity
            scope_entity = next(
                (e for e in entities if e.canonical_name == scope),
                None
            )
            
            if not scope_entity:
                # Create a scope entity
                scope_entity = Entity(
                    type="scope",
                    canonical_name=scope,
                    source_id=entities[0].source_id if entities else "",
                )
                entities.append(scope_entity)
            
            # Create relations
            for entity in scope_entities:
                if entity.id != scope_entity.id:
                    relations.append(Relation(
                        type=RelationTypes.BELONGS_TO,
                        source_entity_id=entity.id,
                        target_entity_id=scope_entity.id,
                        evidence=f"{entity.canonical_name} in {scope}",
                        source_id=entity.source_id,
                        page=entity.page,
                        confidence=0.8,
                    ))
        
        return relations
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities by merging duplicates."""
        seen: dict[tuple, Entity] = {}
        
        for entity in entities:
            key = (entity.type, entity.canonical_name.lower())
            
            if key in seen:
                # Merge with existing
                seen[key] = seen[key].merge(entity)
            else:
                seen[key] = entity
        
        return list(seen.values())


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def extract_from_document(
    file_path: str,
    config: PipelineConfig = None,
    clear_existing: bool = True,
) -> dict:
    """
    Convenience function to extract entities from a document.
    
    Args:
        file_path: Path to document
        config: Pipeline configuration
        clear_existing: Clear existing entities from this source
        
    Returns:
        Extraction statistics
    """
    pipeline = ExtractionPipeline(config)
    return pipeline.process_document(file_path, clear_existing)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <document_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"Processing: {file_path}")
    print("=" * 60)
    
    result = extract_from_document(file_path)
    
    print(f"\nResults:")
    print(f"  Entities: {result['entities_extracted']} extracted, {result['entities_stored']} stored")
    print(f"  Relations: {result['relations_extracted']} extracted, {result['relations_stored']} stored")
    print(f"  Document: {result['tree_stats']}")
    
    # Show sample entities
    store = EntityStore()
    entities = store.search_entities(limit=10)
    
    print(f"\nSample entities:")
    for entity in entities[:5]:
        print(f"  [{entity.type}] {entity.canonical_name} (scope={entity.scope or 'global'})")
