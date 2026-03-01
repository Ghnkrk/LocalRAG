"""
Structure Module
================
Document structure parsing and entity extraction.

Components:
- document_tree: Hierarchical document representation
- parser: PDF to DocumentTree parser
- table_parser: Table extraction with column inference
- entity_schema: Entity/Relation data model
- fact_store: SQLite storage for entities

Usage:
    from structure import parse_document, extract_tables, Entity
"""

from .document_tree import (
    DocumentTree,
    DocumentNode,
    NodeType,
    BoundingBox,
    TableData,
)

from .entity_schema import (
    Entity,
    Relation,
    EntityTypes,
    RelationTypes,
    Fact,  # Legacy compatibility
)

from .parser import (
    parse_pdf,
    parse_document,
    parse_docx,
    parse_with_unstructured,
    ParserConfig,
)

from .table_parser import (
    extract_tables,
    tables_to_entities,
    TableConfig,
)

from .fact_store import FactStore, EntityStore

__all__ = [
    # Document Tree
    "DocumentTree",
    "DocumentNode", 
    "NodeType",
    "BoundingBox",
    "TableData",
    
    # Entity Schema
    "Entity",
    "Relation",
    "EntityTypes",
    "RelationTypes",
    "Fact",
    
    # Parser
    "parse_pdf",
    "parse_document",
    "ParserConfig",
    
    # Table Parser
    "extract_tables",
    "tables_to_entities",
    "TableConfig",
    
    # Storage
    "FactStore",
    "EntityStore",
]
