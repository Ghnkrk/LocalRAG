"""
Named Entity Recognition (NER)
==============================
Local, regex-based entity extraction.

Features:
- Configurable entity patterns
- Domain-agnostic (works with any document type)
- No external dependencies (pure Python regex)
- Optional spaCy integration for better accuracy

Entity Types:
- CODE: Alphanumeric codes (product codes, course codes, IDs)
- PERSON: Names with titles (Dr., Prof., Mr., Ms.)
- ORG: Organizations (suffixes like Inc., Ltd., University)
- DATE: Dates in various formats
- AMOUNT: Numbers with units (credits, hours, dollars)
- EMAIL: Email addresses
- PHONE: Phone numbers
- URL: Web URLs
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from collections import defaultdict

import sys
sys.path.insert(0, '..')
from structure.entity_schema import Entity, EntityTypes


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EntityPattern:
    """
    A pattern for extracting entities.
    
    Attributes:
        name: Pattern identifier
        entity_type: Type of entity this pattern extracts
        pattern: Regex pattern (compiled or string)
        normalizer: Optional function to normalize extracted text
        confidence: Base confidence for matches
        priority: Higher priority patterns take precedence
    """
    name: str
    entity_type: str
    pattern: str
    normalizer: Optional[Callable[[str], str]] = None
    confidence: float = 0.8
    priority: int = 0
    
    def __post_init__(self):
        # Compile pattern
        self._compiled = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
    
    def find_all(self, text: str) -> List[tuple[str, int, int]]:
        """Find all matches. Returns list of (match, start, end)."""
        matches = []
        for m in self._compiled.finditer(text):
            matched_text = m.group(0)
            if self.normalizer:
                matched_text = self.normalizer(matched_text)
            matches.append((matched_text, m.start(), m.end()))
        return matches


@dataclass
class NERConfig:
    """Configuration for NER extraction."""
    
    # Custom patterns (extend or override defaults)
    custom_patterns: List[EntityPattern] = field(default_factory=list)
    
    # Enable/disable pattern categories
    enable_codes: bool = True
    enable_persons: bool = True
    enable_orgs: bool = True
    enable_dates: bool = True
    enable_amounts: bool = True
    enable_contacts: bool = True
    
    # Deduplication
    deduplicate: bool = True
    
    # Minimum match length
    min_match_length: int = 2


# =============================================================================
# DEFAULT PATTERNS
# =============================================================================

def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_name(text: str) -> str:
    """Normalize person/org names."""
    text = _normalize_whitespace(text)
    # Title case
    return text.title()


# Code patterns (product codes, course codes, IDs)
CODE_PATTERNS = [
    EntityPattern(
        name="alphanumeric_code",
        entity_type="code",
        # Matches: U18ECT3101, ABC-123, PRD_456
        pattern=r'\b[A-Z][A-Z0-9]{2,4}[A-Z0-9\-_]{2,10}\b',
        confidence=0.85,
        priority=10,
    ),
    EntityPattern(
        name="section_code",
        entity_type="code",
        # Matches: Section 4.2.1, Clause 12.3
        pattern=r'(?:Section|Clause|Article|Rule)\s+(\d+(?:\.\d+)*)',
        confidence=0.9,
        priority=5,
    ),
]

# Person patterns
PERSON_PATTERNS = [
    EntityPattern(
        name="titled_person",
        entity_type="person",
        # Dr. John Smith, Prof. A. Kumar
        pattern=r'(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?|Shri|Smt\.?)\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)?',
        normalizer=_normalize_name,
        confidence=0.9,
        priority=10,
    ),
    EntityPattern(
        name="name_with_suffix",
        entity_type="person",
        # John Smith Jr., Jane Doe PhD
        pattern=r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Jr\.?|Sr\.?|PhD|MD|MBA)',
        normalizer=_normalize_name,
        confidence=0.85,
        priority=5,
    ),
]

# Organization patterns
ORG_PATTERNS = [
    EntityPattern(
        name="org_with_suffix",
        entity_type="org",
        # Acme Inc., Google LLC, MIT University
        pattern=r'[A-Z][A-Za-z\s]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|University|Institute|College|Hospital|Foundation)',
        normalizer=_normalize_whitespace,
        confidence=0.85,
        priority=10,
    ),
    EntityPattern(
        name="acronym_org",
        entity_type="org",
        # IEEE, ISO, WHO, NASA
        pattern=r'\b[A-Z]{2,6}\b(?=\s+(?:standard|committee|organization|association))',
        confidence=0.7,
        priority=5,
    ),
]

# Date patterns
DATE_PATTERNS = [
    EntityPattern(
        name="date_mdy",
        entity_type="date",
        # January 15, 2024 or Jan 15, 2024
        pattern=r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}',
        normalizer=_normalize_whitespace,
        confidence=0.95,
        priority=10,
    ),
    EntityPattern(
        name="date_dmy",
        entity_type="date",
        # 15 January 2024, 15-01-2024
        pattern=r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        confidence=0.9,
        priority=5,
    ),
    EntityPattern(
        name="semester_date",
        entity_type="date",
        # Semester IV, Year 2, Fall 2024
        pattern=r'(?:Semester|Sem|Year|Quarter|Fall|Spring|Summer|Winter)\s+(?:[IVXLCDM]+|\d{1,2}|\d{4})',
        normalizer=_normalize_whitespace,
        confidence=0.9,
        priority=8,
    ),
]

# Amount patterns
AMOUNT_PATTERNS = [
    EntityPattern(
        name="currency",
        entity_type="amount",
        # $100, Rs. 5000, €50
        pattern=r'(?:\$|Rs\.?|₹|€|£)\s*[\d,]+(?:\.\d{2})?',
        confidence=0.95,
        priority=10,
    ),
    EntityPattern(
        name="credit_hours",
        entity_type="amount",
        # 3 credits, 4 hours, 2.5 units
        pattern=r'\d+(?:\.\d+)?\s*(?:credits?|hours?|units?|points?)',
        confidence=0.9,
        priority=5,
    ),
    EntityPattern(
        name="percentage",
        entity_type="amount",
        # 85%, 99.5%
        pattern=r'\d+(?:\.\d+)?%',
        confidence=0.95,
        priority=5,
    ),
]

# Contact patterns
CONTACT_PATTERNS = [
    EntityPattern(
        name="email",
        entity_type="email",
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        confidence=0.98,
        priority=10,
    ),
    EntityPattern(
        name="phone",
        entity_type="phone",
        # +1-555-123-4567, (555) 123-4567
        pattern=r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        confidence=0.9,
        priority=10,
    ),
    EntityPattern(
        name="url",
        entity_type="url",
        pattern=r'https?://[^\s<>"{}|\\^`\[\]]+',
        confidence=0.98,
        priority=10,
    ),
]


# =============================================================================
# EXTRACTION ENGINE
# =============================================================================

def get_default_patterns(config: NERConfig) -> List[EntityPattern]:
    """Get default patterns based on config."""
    patterns = []
    
    if config.enable_codes:
        patterns.extend(CODE_PATTERNS)
    if config.enable_persons:
        patterns.extend(PERSON_PATTERNS)
    if config.enable_orgs:
        patterns.extend(ORG_PATTERNS)
    if config.enable_dates:
        patterns.extend(DATE_PATTERNS)
    if config.enable_amounts:
        patterns.extend(AMOUNT_PATTERNS)
    if config.enable_contacts:
        patterns.extend(CONTACT_PATTERNS)
    
    # Add custom patterns
    patterns.extend(config.custom_patterns)
    
    # Sort by priority (highest first)
    patterns.sort(key=lambda p: -p.priority)
    
    return patterns


@dataclass
class ExtractedEntity:
    """An entity extracted from text with position info."""
    text: str
    entity_type: str
    start: int
    end: int
    pattern_name: str
    confidence: float


def extract_entities(
    text: str,
    config: NERConfig = None,
    source_id: str = "",
    page: int = 0,
    scope: str = "",
) -> List[Entity]:
    """
    Extract entities from text using pattern matching.
    
    Args:
        text: Input text to extract from
        config: NER configuration
        source_id: Source document identifier
        page: Page number
        scope: Inherited scope context
        
    Returns:
        List of Entity objects
    """
    config = config or NERConfig()
    patterns = get_default_patterns(config)
    
    # Track matched spans to avoid overlaps
    matched_spans = set()
    extracted = []
    
    for pattern in patterns:
        matches = pattern.find_all(text)
        
        for matched_text, start, end in matches:
            # Skip if too short
            if len(matched_text) < config.min_match_length:
                continue
            
            # Skip if overlaps with higher-priority match
            span = (start, end)
            if any(
                s <= start < e or s < end <= e
                for s, e in matched_spans
            ):
                continue
            
            matched_spans.add(span)
            extracted.append(ExtractedEntity(
                text=matched_text,
                entity_type=pattern.entity_type,
                start=start,
                end=end,
                pattern_name=pattern.name,
                confidence=pattern.confidence,
            ))
    
    # Convert to Entity objects
    entities = []
    seen = set()  # For deduplication
    
    for ext in extracted:
        # Deduplicate by (type, text)
        key = (ext.entity_type, ext.text.lower())
        if config.deduplicate and key in seen:
            continue
        seen.add(key)
        
        entities.append(Entity(
            type=ext.entity_type,
            canonical_name=ext.text,
            attributes={
                "pattern": ext.pattern_name,
                "original_text": ext.text,
            },
            source_id=source_id,
            page=page,
            scope=scope,
            confidence=ext.confidence,
        ))
    
    return entities


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Course: U18ECT3101 - Digital Signal Processing
    Instructor: Dr. John Smith, Prof. A. Kumar
    Credits: 4 credits
    
    Prerequisites: U18ECT2101
    
    Department: ABC University, IEEE Member
    Contact: john.smith@university.edu
    
    Semester IV, 2024
    Fee: Rs. 5000
    Passing: 60%
    """
    
    print("=== NER Test ===")
    print(f"Input:\n{sample_text}")
    print("\n--- Extracted Entities ---")
    
    entities = extract_entities(sample_text)
    
    for entity in entities:
        print(f"  [{entity.type}] {entity.canonical_name} (conf={entity.confidence:.2f})")
