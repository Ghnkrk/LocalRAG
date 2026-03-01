"""
Entity and Relation Schema
==========================
Unified schema for structured data extraction.

Replaces the old Fact schema with a proper Entity-Relation model
supporting typed entities, relationships, and provenance tracking.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class Entity:
    """
    A named entity extracted from a document.
    
    Examples:
        - Course: "Digital Signal Processing" with code "U18ECT3101"
        - Person: "Dr. John Smith"
        - Date: "January 2024"
        - Organization: "IEEE"
    
    Attributes:
        id: Unique identifier (UUID)
        type: Entity type (course, person, date, org, amount, etc.)
        canonical_name: Primary/normalized name
        aliases: Alternative names or mentions
        attributes: Type-specific attributes as key-value pairs
        source_id: Document source path
        page: Page number where found
        scope: Inherited context (e.g., "Semester IV")
        confidence: Extraction confidence 0-1
        created_at: Timestamp
    """
    type: str
    canonical_name: str
    attributes: dict = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    source_id: str = ""
    page: int = 0
    scope: str = ""
    confidence: float = 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        # Hash by type + canonical_name for deduplication
        return hash((self.type, self.canonical_name.lower()))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.type == other.type and 
                self.canonical_name.lower() == other.canonical_name.lower())
    
    def merge(self, other: "Entity") -> "Entity":
        """Merge another entity into this one (for deduplication)."""
        # Combine aliases
        new_aliases = list(set(self.aliases + other.aliases + [other.canonical_name]))
        new_aliases = [a for a in new_aliases if a.lower() != self.canonical_name.lower()]
        
        # Merge attributes (other takes precedence for conflicts)
        merged_attrs = {**self.attributes, **other.attributes}
        
        return Entity(
            id=self.id,
            type=self.type,
            canonical_name=self.canonical_name,
            aliases=new_aliases,
            attributes=merged_attrs,
            source_id=self.source_id or other.source_id,
            page=self.page or other.page,
            scope=self.scope or other.scope,
            confidence=max(self.confidence, other.confidence),
            created_at=self.created_at,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "source_id": self.source_id,
            "page": self.page,
            "scope": self.scope,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
            
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            type=data["type"],
            canonical_name=data["canonical_name"],
            aliases=data.get("aliases", []),
            attributes=data.get("attributes", {}),
            source_id=data.get("source_id", ""),
            page=data.get("page", 0),
            scope=data.get("scope", ""),
            confidence=data.get("confidence", 1.0),
            created_at=created_at,
        )
    
    def __repr__(self) -> str:
        attrs_str = ", ".join(f"{k}={v}" for k, v in list(self.attributes.items())[:3])
        return f"Entity({self.type}: {self.canonical_name}, {attrs_str})"


@dataclass
class Relation:
    """
    A relationship between two entities.
    
    Examples:
        - PREREQUISITE: "DSP" requires "Signals and Systems"
        - TEACHES: "Dr. Smith" teaches "DSP"
        - BELONGS_TO: "DSP" belongs_to "Semester IV"
        - HAS_CREDITS: "DSP" has_credits "4"
    
    Attributes:
        id: Unique identifier
        type: Relation type (prerequisite, teaches, belongs_to, etc.)
        source_entity_id: ID of source entity
        target_entity_id: ID of target entity
        evidence: Text snippet supporting this relation
        source_id: Document source path
        page: Page number where found
        confidence: Extraction confidence 0-1
    """
    type: str
    source_entity_id: str
    target_entity_id: str
    evidence: str = ""
    source_id: str = ""
    page: int = 0
    confidence: float = 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "evidence": self.evidence,
            "source_id": self.source_id,
            "page": self.page,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
            
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            type=data["type"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            evidence=data.get("evidence", ""),
            source_id=data.get("source_id", ""),
            page=data.get("page", 0),
            confidence=data.get("confidence", 1.0),
            created_at=created_at,
        )
    
    def __repr__(self) -> str:
        return f"Relation({self.source_entity_id} --{self.type}--> {self.target_entity_id})"


# =============================================================================
# ENTITY TYPE DEFINITIONS
# =============================================================================

class EntityTypes:
    """Standard entity types with their patterns."""
    
    COURSE = "course"
    PERSON = "person"
    ORGANIZATION = "org"
    DATE = "date"
    AMOUNT = "amount"
    POLICY = "policy"
    REQUIREMENT = "requirement"
    TOPIC = "topic"
    
    # Course-specific attributes
    COURSE_ATTRS = ["code", "credits", "hours", "category", "prerequisites"]
    
    # Person-specific attributes  
    PERSON_ATTRS = ["title", "department", "email", "role"]


class RelationTypes:
    """Standard relation types."""
    
    PREREQUISITE = "prerequisite"      # A requires B
    TEACHES = "teaches"                # Person teaches Course
    BELONGS_TO = "belongs_to"          # Entity belongs to Scope
    HAS_CREDITS = "has_credits"        # Course has N credits
    FOLLOWS = "follows"                # A comes after B
    PART_OF = "part_of"                # A is part of B
    REFERENCES = "references"          # A references B


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

@dataclass
class Fact:
    """
    Legacy Fact schema for backwards compatibility.
    
    Deprecated: Use Entity instead.
    """
    entity_type: str
    name: str
    attributes: dict = field(default_factory=dict)
    relationships: list = field(default_factory=list)
    source_id: str = ""
    chunk_id: str = ""
    
    def to_entity(self) -> Entity:
        """Convert legacy Fact to new Entity."""
        return Entity(
            type=self.entity_type,
            canonical_name=self.name,
            attributes=self.attributes,
            source_id=self.source_id,
        )
    
    @classmethod
    def from_entity(cls, entity: Entity) -> "Fact":
        """Create legacy Fact from Entity."""
        return cls(
            entity_type=entity.type,
            name=entity.canonical_name,
            attributes=entity.attributes,
            source_id=entity.source_id,
        )
