"""
Extraction Package
==================
Entity and relation extraction from documents.

Components:
- ner: Named Entity Recognition (regex-based, local)
- relations: Relation extraction patterns
- pipeline: Orchestration of extraction

All extraction is LOCAL - no cloud APIs.
"""

from .ner import (
    extract_entities,
    EntityPattern,
    NERConfig,
)

from .pipeline import (
    ExtractionPipeline,
    extract_from_document,
)

__all__ = [
    "extract_entities",
    "EntityPattern",
    "NERConfig",
    "ExtractionPipeline",
    "extract_from_document",
]
