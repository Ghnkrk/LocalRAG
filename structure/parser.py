"""
PDF Parser with Layout Analysis
================================
Parses PDF documents into DocumentTree structure using pdfminer.

Features:
- Layout-aware text extraction
- Section hierarchy detection
- Table region identification
- Scope inheritance (e.g., "Semester IV" applies to all following content)
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTPage,
    LTTextContainer,
    LTTextBoxHorizontal,
    LTTextLineHorizontal,
    LTChar,
    LTFigure,
    LTRect,
    LTLine,
)

from .document_tree import (
    DocumentTree,
    DocumentNode,
    NodeType,
    BoundingBox,
    TableData,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ParserConfig:
    """Configuration for PDF parsing."""
    
    # Section detection
    section_patterns: list[str] = None  # Regex patterns for section headers
    scope_patterns: list[str] = None    # Patterns that define scope context
    
    # Layout analysis
    line_margin: float = 0.5            # Vertical margin for line grouping
    word_margin: float = 0.1            # Horizontal margin for word grouping
    char_margin: float = 2.0            # Character margin
    
    # Filtering  
    min_text_length: int = 3            # Minimum text length to include
    exclude_patterns: list[str] = None  # Patterns to exclude (headers/footers)
    
    def __post_init__(self):
        if self.section_patterns is None:
            self.section_patterns = [
                r'^(?:CHAPTER|UNIT|SECTION|PART)\s+[IVXLCDM\d]+',  # CHAPTER I, UNIT 1
                r'^(?:SEMESTER|SEM)\s+[IVXLCDM\d]+',               # SEMESTER IV
                r'^\d+\.\s+[A-Z]',                                  # 1. Title
                r'^[IVXLCDM]+\.\s+[A-Z]',                          # I. Title
                r'^[A-Z][A-Z\s]{10,}$',                            # ALL CAPS HEADERS
            ]
        
        if self.scope_patterns is None:
            self.scope_patterns = [
                r'(?:SEMESTER|SEM)\s+([IVXLCDM]+|\d+)',            # Semester IV
                r'(?:YEAR|YR)\s+([IVXLCDM]+|\d+)',                 # Year 2
                r'(?:PART|SECTION)\s+([A-Z]|\d+)',                 # Part A
                r'(?:CHAPTER|UNIT)\s+([IVXLCDM]+|\d+)',            # Chapter 1
            ]
        
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                r'^Page\s+\d+',                                    # Page numbers
                r'^\d+$',                                          # Just numbers
                r'^https?://',                                     # URLs
            ]


# =============================================================================
# TEXT ELEMENT EXTRACTION
# =============================================================================

@dataclass
class TextElement:
    """A text element extracted from PDF with position info."""
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float = 12.0
    is_bold: bool = False
    
    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(self.x0, self.y0, self.x1, self.y1)
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)


def extract_text_elements(file_path: str, config: ParserConfig = None) -> list[TextElement]:
    """
    Extract text elements from PDF with layout information.
    
    Returns list of TextElement with position, font info.
    """
    config = config or ParserConfig()
    elements = []
    
    laparams = LAParams(
        line_margin=config.line_margin,
        word_margin=config.word_margin,
        char_margin=config.char_margin,
        detect_vertical=False,
    )
    
    for page_num, page_layout in enumerate(extract_pages(file_path, laparams=laparams), 1):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if len(text) < config.min_text_length:
                    continue
                
                # Get font info from first character
                font_size = 12.0
                is_bold = False
                
                for line in element:
                    if isinstance(line, LTTextLineHorizontal):
                        for char in line:
                            if isinstance(char, LTChar):
                                font_size = char.size
                                is_bold = "bold" in char.fontname.lower()
                                break
                        break
                
                elements.append(TextElement(
                    text=text,
                    page=page_num,
                    x0=element.x0,
                    y0=element.y0,
                    x1=element.x1,
                    y1=element.y1,
                    font_size=font_size,
                    is_bold=is_bold,
                ))
    
    return elements


# =============================================================================
# STRUCTURE DETECTION
# =============================================================================

def is_section_header(element: TextElement, config: ParserConfig) -> bool:
    """Check if element is a section header based on patterns and formatting."""
    text = element.text.strip()
    
    # Check against section patterns
    for pattern in config.section_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    # Heuristic: Large font + short text + starts with capital
    if (element.font_size > 14 and 
        len(text) < 100 and 
        text[0].isupper() and
        not text.endswith('.')):
        return True
    
    # Heuristic: Bold + short + all caps
    if element.is_bold and len(text) < 80 and text.isupper():
        return True
    
    return False


def extract_scope(text: str, config: ParserConfig) -> Optional[str]:
    """Extract scope identifier from text (e.g., 'Semester IV')."""
    for pattern in config.scope_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def should_exclude(text: str, config: ParserConfig) -> bool:
    """Check if text should be excluded (headers, footers, etc.)."""
    for pattern in config.exclude_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    return False


def detect_section_level(text: str, font_size: float) -> int:
    """Infer section level from text and formatting."""
    # Chapter/Unit = level 1
    if re.match(r'^(?:CHAPTER|UNIT)\s+[IVXLCDM\d]+', text, re.IGNORECASE):
        return 1
    
    # Semester/Part = level 1
    if re.match(r'^(?:SEMESTER|SEM|PART)\s+[IVXLCDM\d]+', text, re.IGNORECASE):
        return 1
    
    # Numbered sections
    if re.match(r'^\d+\.\s+[A-Z]', text):
        return 2
    
    # Sub-numbered sections
    if re.match(r'^\d+\.\d+\s+[A-Z]', text):
        return 3
    
    # Fall back to font size heuristic
    if font_size >= 16:
        return 1
    elif font_size >= 14:
        return 2
    else:
        return 3


# =============================================================================
# MAIN PARSER
# =============================================================================

def parse_pdf(file_path: str, config: ParserConfig = None) -> DocumentTree:
    """
    Parse a PDF file into a DocumentTree structure.
    
    Args:
        file_path: Path to PDF file
        config: Parser configuration
        
    Returns:
        DocumentTree with hierarchical structure preserved
    """
    config = config or ParserConfig()
    source_id = str(Path(file_path).absolute())
    
    # Extract raw text elements
    elements = extract_text_elements(file_path, config)
    
    if not elements:
        return DocumentTree(source_id=source_id)
    
    # Build document tree
    tree = DocumentTree(source_id=source_id)
    tree.metadata["total_pages"] = max(e.page for e in elements)
    tree.metadata["total_elements"] = len(elements)
    
    current_scope = ""
    current_section: Optional[DocumentNode] = None
    section_stack: list[DocumentNode] = []  # Stack for hierarchy
    
    for element in elements:
        text = element.text.strip()
        
        # Skip excluded content
        if should_exclude(text, config):
            continue
        
        # Check for scope change
        scope = extract_scope(text, config)
        if scope:
            current_scope = scope
        
        # Check if section header
        if is_section_header(element, config):
            level = detect_section_level(text, element.font_size)
            
            # Pop stack until we find parent at level-1
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()
            
            # Create section node
            section_node = DocumentNode(
                type=NodeType.SECTION,
                text=text,
                level=level,
                page=element.page,
                bbox=element.bbox,
            )
            
            # Set scope
            if current_scope:
                section_node._inherited_scope = current_scope
            
            # Add to tree
            if section_stack:
                section_stack[-1].add_child(section_node)
            else:
                tree.root.add_child(section_node)
            
            section_stack.append(section_node)
            current_section = section_node
            
        else:
            # Regular paragraph
            para_node = DocumentNode(
                type=NodeType.PARAGRAPH,
                text=text,
                level=current_section.level + 1 if current_section else 1,
                page=element.page,
                bbox=element.bbox,
            )
            
            if current_scope:
                para_node._inherited_scope = current_scope
            
            # Add to current section or root
            if current_section:
                current_section.add_child(para_node)
            else:
                tree.root.add_child(para_node)
    
    return tree


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_document(file_path: str) -> DocumentTree:
    """
    Parse any supported document into a DocumentTree.
    
    Supports: PDF, DOCX, TXT
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        return parse_pdf(file_path)
    
    elif suffix == ".txt" or suffix == ".md":
        # Simple text file parsing
        tree = DocumentTree(source_id=str(path.absolute()))
        text = path.read_text(encoding="utf-8", errors="ignore")
        
        # Split on blank lines for paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        for i, para in enumerate(paragraphs):
            tree.root.add_child(DocumentNode(
                type=NodeType.PARAGRAPH,
                text=para,
                page=1,
            ))
        return tree
    
    elif suffix == ".docx":
        return parse_docx(file_path)
    
    else:
        # Fallback: Try unstructured for other formats
        try:
            return parse_with_unstructured(file_path)
        except Exception:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt, .md")


def parse_docx(file_path: str) -> DocumentTree:
    """
    Parse a DOCX file into a DocumentTree.
    
    Uses python-docx for extraction with style-based section detection.
    """
    try:
        from docx import Document as DocxDocument
        from docx.enum.style import WD_STYLE_TYPE
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    
    path = Path(file_path)
    tree = DocumentTree(source_id=str(path.absolute()))
    
    doc = DocxDocument(file_path)
    config = ParserConfig()
    current_scope = ""
    current_section = None
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        # Detect sections from heading styles
        is_heading = para.style.name.startswith("Heading") if para.style else False
        
        # Check for scope
        scope = extract_scope(text, config)
        if scope:
            current_scope = scope
        
        if is_heading or (len(text) < 100 and text.isupper()):
            # Section header
            level = 1
            if para.style and para.style.name:
                try:
                    level = int(para.style.name[-1])
                except (ValueError, IndexError):
                    level = 1
            
            section_node = DocumentNode(
                type=NodeType.SECTION,
                text=text,
                level=level,
                page=1,
            )
            if current_scope:
                section_node._inherited_scope = current_scope
            
            tree.root.add_child(section_node)
            current_section = section_node
        else:
            # Regular paragraph
            para_node = DocumentNode(
                type=NodeType.PARAGRAPH,
                text=text,
                level=2 if current_section else 1,
                page=1,
            )
            if current_scope:
                para_node._inherited_scope = current_scope
            
            if current_section:
                current_section.add_child(para_node)
            else:
                tree.root.add_child(para_node)
    
    return tree


def parse_with_unstructured(file_path: str) -> DocumentTree:
    """
    Fallback parser using unstructured library.
    
    Works with many formats: PDF, DOCX, HTML, PPTX, XLSX, TXT, etc.
    """
    try:
        from unstructured.partition.auto import partition
    except ImportError:
        raise ImportError("unstructured not installed. Run: pip install unstructured")
    
    path = Path(file_path)
    tree = DocumentTree(source_id=str(path.absolute()))
    
    elements = partition(file_path)
    config = ParserConfig()
    current_scope = ""
    current_section = None
    
    for elem in elements:
        text = str(elem).strip()
        if not text:
            continue
        
        elem_type = type(elem).__name__
        
        # Check for scope
        scope = extract_scope(text, config)
        if scope:
            current_scope = scope
        
        # Headers become sections
        if "Title" in elem_type or "Header" in elem_type:
            section_node = DocumentNode(
                type=NodeType.SECTION,
                text=text,
                level=1,
                page=getattr(elem.metadata, 'page_number', 1) or 1,
            )
            if current_scope:
                section_node._inherited_scope = current_scope
            tree.root.add_child(section_node)
            current_section = section_node
        else:
            # Other elements become paragraphs
            para_node = DocumentNode(
                type=NodeType.PARAGRAPH,
                text=text,
                level=2 if current_section else 1,
                page=getattr(elem.metadata, 'page_number', 1) or 1,
            )
            if current_scope:
                para_node._inherited_scope = current_scope
            
            if current_section:
                current_section.add_child(para_node)
            else:
                tree.root.add_child(para_node)
    
    return tree


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parser.py <pdf_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    tree = parse_pdf(file_path)
    
    print(f"\nDocument Tree: {file_path}")
    print(f"{'='*60}")
    print(f"Stats: {tree.stats()}")
    print(f"\nSections:")
    
    for section in tree.get_sections():
        indent = "  " * section.level
        scope_str = f" [{section.scope}]" if section.scope else ""
        print(f"{indent}• {section.text[:60]}{scope_str}")
