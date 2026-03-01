"""
Table Parser with Column Inference
==================================
Extracts tables from PDFs using layout analysis.

Features:
- Column detection via x-coordinate clustering
- Header inference from first row
- Scope inheritance from parent sections
- Structured row-to-column mapping
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTTextContainer,
    LTTextBoxHorizontal,
    LTTextLineHorizontal,
    LTChar,
)

from .document_tree import TableData, BoundingBox
from .entity_schema import Entity, EntityTypes


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TableConfig:
    """Configuration for table extraction."""
    
    # Row grouping tolerance (vertical)
    row_tolerance: float = 8.0
    
    # Column detection
    min_columns: int = 2
    column_gap_threshold: float = 20.0  # Min gap between columns
    
    # Content patterns
    table_indicator_patterns: list[str] = None  # Patterns that indicate table content
    
    # Course-specific patterns (for syllabus documents)
    course_code_pattern: str = r'^[A-Z][A-Z0-9]{4,15}$'
    
    def __post_init__(self):
        if self.table_indicator_patterns is None:
            self.table_indicator_patterns = [
                r'^S\.?\s*No\.?$',           # S.No, S. No.
                r'^(?:Course|Subject)\s*(?:Code|No\.?)?',
                r'^(?:Course|Subject)\s*(?:Title|Name)',
                r'^Credits?$',
                r'^Hours?$',
                r'^L\s*T\s*P',               # L T P (lecture, tutorial, practical)
            ]


# =============================================================================
# LAYOUT ANALYSIS
# =============================================================================

@dataclass
class TextCell:
    """A text cell with position information."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0


def extract_text_cells(file_path: str) -> dict[int, list[TextCell]]:
    """
    Extract all text cells from PDF, grouped by page.
    
    Returns:
        Dict mapping page number to list of TextCells
    """
    laparams = LAParams(
        line_margin=0.3,
        word_margin=0.1,
        char_margin=2.0,
        detect_vertical=False,
    )
    
    cells_by_page = defaultdict(list)
    
    for page_num, page_layout in enumerate(extract_pages(file_path, laparams=laparams), 1):
        page_height = page_layout.height
        
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if not text:
                    continue
                
                # Normalize y-coordinates (PDF has origin at bottom)
                cells_by_page[page_num].append(TextCell(
                    text=text,
                    x0=element.x0,
                    y0=page_height - element.y1,  # Flip to top-origin
                    x1=element.x1,
                    y1=page_height - element.y0,
                    page=page_num,
                ))
    
    return dict(cells_by_page)


# =============================================================================
# ROW AND COLUMN DETECTION
# =============================================================================

def group_cells_into_rows(cells: list[TextCell], tolerance: float = 8.0) -> list[list[TextCell]]:
    """
    Group cells into rows based on y-coordinate clustering.
    
    Cells with similar y-center are grouped into the same row.
    """
    if not cells:
        return []
    
    # Sort by y-coordinate (top to bottom)
    sorted_cells = sorted(cells, key=lambda c: c.y0)
    
    rows = []
    current_row = [sorted_cells[0]]
    current_y = sorted_cells[0].center_y
    
    for cell in sorted_cells[1:]:
        if abs(cell.center_y - current_y) <= tolerance:
            current_row.append(cell)
        else:
            # Sort current row by x-coordinate (left to right)
            current_row.sort(key=lambda c: c.x0)
            rows.append(current_row)
            current_row = [cell]
            current_y = cell.center_y
    
    if current_row:
        current_row.sort(key=lambda c: c.x0)
        rows.append(current_row)
    
    return rows


def detect_column_boundaries(rows: list[list[TextCell]], gap_threshold: float = 20.0) -> list[float]:
    """
    Detect column boundaries from row data.
    
    Analyzes x-coordinates across all rows to find consistent column positions.
    
    Returns:
        List of x-coordinate boundaries (left edges of columns)
    """
    if not rows:
        return []
    
    # Collect all x0 positions
    all_x0 = []
    for row in rows:
        for cell in row:
            all_x0.append(cell.x0)
    
    if not all_x0:
        return []
    
    # Cluster x0 positions
    all_x0.sort()
    
    boundaries = [all_x0[0]]
    for x in all_x0[1:]:
        if x - boundaries[-1] > gap_threshold:
            boundaries.append(x)
    
    return boundaries


def assign_cells_to_columns(row: list[TextCell], boundaries: list[float]) -> dict[int, str]:
    """
    Assign cells to columns based on their x-position.
    
    Returns:
        Dict mapping column index to cell text
    """
    result = {}
    
    for cell in row:
        # Find which column this cell belongs to
        col_idx = 0
        for i, boundary in enumerate(boundaries):
            if cell.x0 >= boundary - 10:  # Allow some tolerance
                col_idx = i
        
        # Combine if multiple cells in same column
        if col_idx in result:
            result[col_idx] += " " + cell.text
        else:
            result[col_idx] = cell.text
    
    return result


# =============================================================================
# TABLE DETECTION
# =============================================================================

def is_table_row(row: list[TextCell], config: TableConfig) -> bool:
    """
    Determine if a row of cells is part of a table.
    
    Uses heuristics:
    - Multiple columns
    - Contains table indicator patterns
    - Contains course codes or numeric data
    """
    if len(row) < config.min_columns:
        return False
    
    row_text = " ".join(cell.text for cell in row)
    
    # Check for table indicators
    for pattern in config.table_indicator_patterns:
        if re.search(pattern, row_text, re.IGNORECASE):
            return True
    
    # Check for course code pattern
    for cell in row:
        if re.match(config.course_code_pattern, cell.text.strip()):
            return True
    
    # Check for numeric content (credits, hours, etc.)
    numeric_cells = sum(1 for cell in row if re.match(r'^\d+$', cell.text.strip()))
    if numeric_cells >= 2:
        return True
    
    return False


def find_table_regions(
    rows: list[list[TextCell]], 
    config: TableConfig
) -> list[tuple[int, int]]:
    """
    Find contiguous regions that appear to be tables.
    
    Returns:
        List of (start_row_idx, end_row_idx) tuples
    """
    if not rows:
        return []
    
    regions = []
    in_table = False
    start_idx = 0
    
    for i, row in enumerate(rows):
        is_table = is_table_row(row, config)
        
        if is_table and not in_table:
            start_idx = i
            in_table = True
        elif not is_table and in_table:
            if i - start_idx >= 2:  # Minimum 2 rows for a table
                regions.append((start_idx, i))
            in_table = False
    
    # Handle table at end of page
    if in_table and len(rows) - start_idx >= 2:
        regions.append((start_idx, len(rows)))
    
    return regions


# =============================================================================
# HEADER INFERENCE
# =============================================================================

def infer_headers(rows: list[list[TextCell]], boundaries: list[float]) -> list[str]:
    """
    Infer column headers from the first row or patterns.
    
    Returns:
        List of header names for each column
    """
    if not rows:
        return []
    
    first_row = rows[0]
    col_data = assign_cells_to_columns(first_row, boundaries)
    
    # Check if first row looks like headers
    header_patterns = [
        r'^S\.?\s*No\.?',
        r'^(?:Course|Subject)',
        r'^(?:Code|Title|Name|Category)',
        r'^Credits?',
        r'^Hours?',
        r'^L$|^T$|^P$|^J$|^C$',
        r'^Pre.?req',
    ]
    
    first_row_text = " ".join(cell.text for cell in first_row)
    is_header_row = any(
        re.search(p, first_row_text, re.IGNORECASE) 
        for p in header_patterns
    )
    
    if is_header_row:
        # Use first row as headers
        headers = []
        for i in range(len(boundaries)):
            text = col_data.get(i, f"Column_{i}")
            # Clean header text
            text = re.sub(r'\s+', ' ', text).strip()
            headers.append(text)
        return headers
    else:
        # Generate generic headers
        return [f"Column_{i}" for i in range(len(boundaries))]


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_tables(
    file_path: str, 
    config: TableConfig = None,
    scope: str = ""
) -> list[TableData]:
    """
    Extract all tables from a document file.
    
    Currently supports PDF only. Returns empty list for other formats.
    
    Args:
        file_path: Path to document file
        config: Table extraction configuration
        scope: Default scope for all tables (e.g., "Semester IV")
        
    Returns:
        List of TableData objects (empty for non-PDF files)
    """
    from pathlib import Path
    
    # Only PDFs are supported for table extraction
    if not file_path.lower().endswith('.pdf'):
        return []
    
    config = config or TableConfig()
    
    try:
        cells_by_page = extract_text_cells(file_path)
    except Exception:
        # Failed to extract cells (corrupted PDF, etc.)
        return []
    
    all_tables = []
    
    for page_num, cells in cells_by_page.items():
        # Group into rows
        rows = group_cells_into_rows(cells, config.row_tolerance)
        
        if not rows:
            continue
        
        # Find table regions
        regions = find_table_regions(rows, config)
        
        for table_idx, (start_idx, end_idx) in enumerate(regions):
            table_rows = rows[start_idx:end_idx]
            
            if not table_rows:
                continue
            
            # Detect column boundaries
            boundaries = detect_column_boundaries(table_rows, config.column_gap_threshold)
            
            if len(boundaries) < config.min_columns:
                continue
            
            # Infer headers
            headers = infer_headers(table_rows, boundaries)
            
            # Extract data rows (skip header if detected)
            data_rows = []
            start_data = 1 if headers[0] != "Column_0" else 0
            
            for row in table_rows[start_data:]:
                col_data = assign_cells_to_columns(row, boundaries)
                row_dict = {}
                for i, header in enumerate(headers):
                    row_dict[header] = col_data.get(i, "")
                data_rows.append(row_dict)
            
            # Calculate bounding box
            all_x0 = [c.x0 for row in table_rows for c in row]
            all_y0 = [c.y0 for row in table_rows for c in row]
            all_x1 = [c.x1 for row in table_rows for c in row]
            all_y1 = [c.y1 for row in table_rows for c in row]
            
            bbox = BoundingBox(
                min(all_x0), min(all_y0),
                max(all_x1), max(all_y1)
            ) if all_x0 else None
            
            # Raw text for debugging
            raw_text = "\n".join(
                " | ".join(c.text for c in row)
                for row in table_rows
            )
            
            all_tables.append(TableData(
                page=page_num,
                table_index=table_idx,
                headers=headers,
                rows=data_rows,
                bbox=bbox,
                scope=scope,
                raw_text=raw_text,
            ))
    
    return all_tables


# =============================================================================
# ENTITY EXTRACTION FROM TABLES
# =============================================================================

def tables_to_entities(
    tables: list[TableData], 
    source_id: str,
    config: TableConfig = None
) -> list[Entity]:
    """
    Convert extracted tables into Entity objects.
    
    Intelligently identifies entity types based on table content.
    """
    config = config or TableConfig()
    entities = []
    
    for table in tables:
        for row_idx, row in enumerate(table.rows):
            entity = extract_entity_from_row(
                row, 
                table.headers, 
                source_id=source_id,
                page=table.page,
                scope=table.scope,
                config=config,
            )
            if entity:
                entities.append(entity)
    
    return entities


def extract_entity_from_row(
    row: dict[str, str],
    headers: list[str],
    source_id: str,
    page: int,
    scope: str,
    config: TableConfig
) -> Optional[Entity]:
    """
    Extract an entity from a single table row.
    
    Identifies entity type and maps columns to attributes.
    """
    # Find course code
    code = None
    name = None
    
    for header, value in row.items():
        value = value.strip()
        if not value:
            continue
        
        header_lower = header.lower()
        
        # Course code detection
        if re.match(config.course_code_pattern, value):
            code = value
        elif 'code' in header_lower and not code:
            code = value
        
        # Course name detection  
        elif 'title' in header_lower or 'name' in header_lower:
            name = value
        elif 'subject' in header_lower or 'course' in header_lower:
            if not re.match(config.course_code_pattern, value):
                name = value
    
    # If we found a code, this is likely a course
    if code:
        # Clean up name
        if not name:
            # Try to find longest text that isn't the code
            candidates = [v for v in row.values() if v and v != code and len(v) > 5]
            # Filter out things that look like numeric/tabular data
            candidates = [
                c for c in candidates 
                if not re.match(r'^[\d\s\-]+$', c)  # Pure numeric
                and not re.match(r'^[A-Z]{2,4}\s+\d', c)  # "BS 2 0 2 0" pattern
                and not re.match(r'^\d+\s+\d+\s+\d+', c)  # "3 0 2 0" pattern
            ]
            if candidates:
                # Prefer candidates that look like actual titles (contain lowercase or long)
                title_candidates = [c for c in candidates if any(ch.islower() for ch in c) or len(c) > 15]
                if title_candidates:
                    name = max(title_candidates, key=len)
                else:
                    name = max(candidates, key=len)
        
        # Validate: If name still looks like garbage, skip this entity
        if name:
            # Clean newlines and extra spaces
            name = re.sub(r'\s+', ' ', name).strip()
            
            # Skip if name is just codes/numbers
            if re.match(r'^[A-Z]{2,4}\s+\d', name):  # "HS 3 0 2 0"
                return None
            if re.match(r'^\d[\d\s\-]+$', name):  # "1 2 3 4"
                return None
            if len(name) < 3:
                return None
        
        if not name:
            name = code  # Fallback
        
        # Build attributes from remaining columns
        attributes = {"code": code}
        
        for header, value in row.items():
            value = value.strip()
            if not value or value == code or value == name:
                continue
            
            header_clean = re.sub(r'\s+', '_', header.lower()).strip('_')
            
            # Skip generic column names
            if header_clean.startswith('column_'):
                continue
            
            # Parse credits
            if 'credit' in header_clean and re.match(r'^\d+$', value):
                attributes['credits'] = int(value)
            elif 'hour' in header_clean and re.match(r'^\d+$', value):
                attributes['hours'] = int(value)
            elif header_clean in ['l', 't', 'p', 'j', 'c']:
                attributes[header_clean] = value
            elif 'prereq' in header_clean:
                attributes['prerequisites'] = value
            elif 'category' in header_clean or 'cat' in header_clean:
                attributes['category'] = value
            else:
                # Store other attributes
                attributes[header_clean] = value
        
        return Entity(
            type=EntityTypes.COURSE,
            canonical_name=name,
            attributes=attributes,
            source_id=source_id,
            page=page,
            scope=scope,
            confidence=0.9,
        )
    
    return None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python table_parser.py <pdf_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    tables = extract_tables(file_path)
    
    print(f"\nExtracted {len(tables)} tables from: {file_path}")
    print("=" * 60)
    
    for table in tables:
        print(f"\nPage {table.page}, Table {table.table_index}")
        print(f"Headers: {table.headers}")
        print(f"Rows: {len(table.rows)}")
        print(f"Scope: {table.scope or 'N/A'}")
        
        if table.rows:
            print("First row:", table.rows[0])
    
    # Extract entities
    entities = tables_to_entities(tables, source_id=file_path)
    print(f"\n{'='*60}")
    print(f"Extracted {len(entities)} entities")
    
    for entity in entities[:10]:
        print(f"  - {entity}")
