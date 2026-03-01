"""
Document Tree Data Structure
============================
Represents a document as a hierarchical tree preserving structure.

Each node represents a structural element (section, paragraph, table, list)
with inherited context from parent nodes.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class NodeType(Enum):
    """Types of document nodes."""
    DOCUMENT = "document"       # Root node
    SECTION = "section"         # Chapter, section, subsection
    PARAGRAPH = "paragraph"     # Regular text block
    TABLE = "table"             # Table container
    TABLE_ROW = "table_row"     # Single row in table
    LIST = "list"               # Ordered or unordered list
    LIST_ITEM = "list_item"     # Single list item
    HEADER = "header"           # Page header (usually filtered)
    FOOTER = "footer"           # Page footer (usually filtered)
    TITLE = "title"             # Document title


@dataclass
class BoundingBox:
    """Bounding box for layout analysis."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class DocumentNode:
    """
    A node in the document tree.
    
    Attributes:
        type: Type of node (section, paragraph, table, etc.)
        text: Text content of this node
        level: Hierarchy depth (0=document, 1=chapter, 2=section, 3=subsection...)
        page: Page number (1-indexed)
        bbox: Bounding box for layout analysis
        children: Child nodes
        metadata: Additional type-specific data
        
    Inherited Context:
        The 'scope' in metadata is inherited from parent sections.
        E.g., if parent is "Semester IV", all children inherit this scope.
    """
    type: NodeType
    text: str = ""
    level: int = 0
    page: int = 1
    bbox: Optional[BoundingBox] = None
    children: list["DocumentNode"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    # Inherited from parent during tree building
    _inherited_scope: str = field(default="", repr=False)
    
    @property
    def scope(self) -> str:
        """Get the inherited scope context."""
        return self._inherited_scope or self.metadata.get("scope", "")
    
    def add_child(self, child: "DocumentNode") -> "DocumentNode":
        """Add a child node, propagating scope."""
        # Inherit scope from parent if child doesn't have its own
        if not child._inherited_scope and self.scope:
            child._inherited_scope = self.scope
        self.children.append(child)
        return child
    
    def set_scope(self, scope: str):
        """Set scope for this node and propagate to children."""
        self._inherited_scope = scope
        for child in self.children:
            if not child.metadata.get("scope"):  # Don't override explicit scope
                child.set_scope(scope)
    
    def walk(self):
        """Iterate over all nodes in tree (depth-first)."""
        yield self
        for child in self.children:
            yield from child.walk()
    
    def get_text_recursive(self, separator: str = "\n") -> str:
        """Get all text from this node and descendants."""
        texts = [self.text] if self.text else []
        for child in self.children:
            child_text = child.get_text_recursive(separator)
            if child_text:
                texts.append(child_text)
        return separator.join(texts)
    
    def find_by_type(self, node_type: NodeType) -> list["DocumentNode"]:
        """Find all descendants of a given type."""
        results = []
        for node in self.walk():
            if node.type == node_type:
                results.append(node)
        return results
    
    def __len__(self) -> int:
        """Number of characters in this node's text."""
        return len(self.text)
    
    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"DocumentNode({self.type.value}, level={self.level}, page={self.page}, text='{text_preview}')"


@dataclass
class TableCell:
    """A cell in a table."""
    text: str
    row: int
    col: int
    bbox: Optional[BoundingBox] = None
    is_header: bool = False


@dataclass
class TableData:
    """
    Structured table data extracted from a document.
    
    Stores the table as a grid with inferred column headers.
    """
    page: int
    table_index: int  # Which table on the page (0-indexed)
    headers: list[str]  # Column headers (inferred from first row or heuristics)
    rows: list[dict[str, str]]  # List of {header: value} dicts
    bbox: Optional[BoundingBox] = None
    scope: str = ""  # Inherited from parent section
    raw_text: str = ""  # Original text for debugging
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def to_facts(self, source_id: str) -> list[dict]:
        """
        Convert table rows to fact dictionaries.
        
        Returns list of dicts ready for Entity creation.
        """
        facts = []
        for row_idx, row in enumerate(self.rows):
            facts.append({
                "source_id": source_id,
                "page": self.page,
                "table_index": self.table_index,
                "row_index": row_idx,
                "scope": self.scope,
                "columns": row,
            })
        return facts


@dataclass 
class DocumentTree:
    """
    Root container for a parsed document.
    
    Provides convenience methods for accessing document structure.
    """
    source_id: str  # File path or identifier
    root: DocumentNode = field(default_factory=lambda: DocumentNode(type=NodeType.DOCUMENT))
    tables: list[TableData] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Document-level metadata
    
    def add_section(self, text: str, level: int, page: int, **kwargs) -> DocumentNode:
        """Add a section node to the appropriate parent."""
        node = DocumentNode(
            type=NodeType.SECTION,
            text=text,
            level=level,
            page=page,
            metadata=kwargs,
        )
        # Find appropriate parent based on level
        parent = self._find_parent_for_level(level)
        parent.add_child(node)
        return node
    
    def _find_parent_for_level(self, level: int) -> DocumentNode:
        """Find the appropriate parent node for a given level."""
        # Walk backwards through sections to find parent at level-1
        if level <= 1:
            return self.root
        
        # Find last section at level-1
        candidates = []
        for node in self.root.walk():
            if node.type == NodeType.SECTION and node.level == level - 1:
                candidates.append(node)
        
        return candidates[-1] if candidates else self.root
    
    def get_all_text(self) -> str:
        """Get full document text."""
        return self.root.get_text_recursive()
    
    def get_sections(self) -> list[DocumentNode]:
        """Get all section nodes."""
        return self.root.find_by_type(NodeType.SECTION)
    
    def get_tables(self) -> list[TableData]:
        """Get all tables."""
        return self.tables
    
    def stats(self) -> dict:
        """Get document statistics."""
        all_nodes = list(self.root.walk())
        return {
            "total_nodes": len(all_nodes),
            "sections": len([n for n in all_nodes if n.type == NodeType.SECTION]),
            "paragraphs": len([n for n in all_nodes if n.type == NodeType.PARAGRAPH]),
            "tables": len(self.tables),
            "total_chars": len(self.get_all_text()),
        }
