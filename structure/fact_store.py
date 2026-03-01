"""
Entity Store
=============
SQLite-backed storage for entities and relations.

Supports both new Entity/Relation schema and legacy Fact compatibility.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime

from .entity_schema import Entity, Relation, Fact


class EntityStore:
    """
    SQLite-backed store for entities and relations.
    
    Features:
    - Full-text search on name, aliases, attributes
    - Scope-based filtering
    - Relation graph queries
    - Legacy Fact compatibility
    """
    
    def __init__(self, db_path: str = "./fact_store/entities.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Entities table
        c.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                aliases JSON,
                attributes JSON,
                source_id TEXT,
                page INTEGER,
                scope TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Relations table
        c.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                evidence TEXT,
                source_id TEXT,
                page INTEGER,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(id)
            )
        ''')
        
        # Indexes
        c.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(canonical_name)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_entity_scope ON entities(scope)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_entity_source ON entities(source_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(type)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_relation_source ON relations(source_entity_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_relation_target ON relations(target_entity_id)')
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # ENTITY OPERATIONS
    # =========================================================================
    
    def add_entity(self, entity: Entity) -> str:
        """Add a single entity. Returns entity ID."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO entities 
            (id, type, canonical_name, aliases, attributes, source_id, page, scope, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entity.id,
            entity.type,
            entity.canonical_name,
            json.dumps(entity.aliases),
            json.dumps(entity.attributes),
            entity.source_id,
            entity.page,
            entity.scope,
            entity.confidence,
            entity.created_at.isoformat(),
        ))
        
        conn.commit()
        conn.close()
        return entity.id
    
    def add_entities(self, entities: List[Entity]) -> int:
        """Add multiple entities. Returns count added."""
        if not entities:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        data = [
            (
                e.id, e.type, e.canonical_name, json.dumps(e.aliases),
                json.dumps(e.attributes), e.source_id, e.page, e.scope,
                e.confidence, e.created_at.isoformat()
            )
            for e in entities
        ]
        
        c.executemany('''
            INSERT OR REPLACE INTO entities 
            (id, type, canonical_name, aliases, attributes, source_id, page, scope, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        
        conn.commit()
        conn.close()
        return len(entities)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT * FROM entities WHERE id = ?', (entity_id,))
        row = c.fetchone()
        conn.close()
        
        return self._row_to_entity(row) if row else None
    
    def search_entities(
        self,
        query: str = None,
        entity_type: str = None,
        scope: str = None,
        source_id: str = None,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Search entities with filters.
        
        Args:
            query: Text search in name, aliases, and attributes
            entity_type: Filter by entity type
            scope: Filter by scope (e.g., "Semester IV")
            source_id: Filter by source document
            limit: Maximum results
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        sql = "SELECT * FROM entities WHERE 1=1"
        params = []
        
        if entity_type:
            sql += " AND type = ?"
            params.append(entity_type)
        
        if scope:
            sql += " AND scope LIKE ?"
            params.append(f"%{scope}%")
        
        if source_id:
            sql += " AND source_id = ?"
            params.append(source_id)
        
        if query:
            sql += """ AND (
                canonical_name LIKE ? 
                OR aliases LIKE ? 
                OR attributes LIKE ?
            )"""
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern])
        
        sql += f" ORDER BY confidence DESC LIMIT {limit}"
        
        c.execute(sql, params)
        rows = c.fetchall()
        conn.close()
        
        return [self._row_to_entity(row) for row in rows]
    
    def delete_by_source(self, source_id: str) -> int:
        """Delete all entities from a source. Returns count deleted."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Delete relations first
        c.execute('''
            DELETE FROM relations 
            WHERE source_entity_id IN (SELECT id FROM entities WHERE source_id = ?)
            OR target_entity_id IN (SELECT id FROM entities WHERE source_id = ?)
        ''', (source_id, source_id))
        
        # Delete entities
        c.execute('DELETE FROM entities WHERE source_id = ?', (source_id,))
        count = c.rowcount
        
        conn.commit()
        conn.close()
        return count
    
    # =========================================================================
    # RELATION OPERATIONS
    # =========================================================================
    
    def add_relation(self, relation: Relation) -> str:
        """Add a relation. Returns relation ID."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO relations
            (id, type, source_entity_id, target_entity_id, evidence, source_id, page, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            relation.id,
            relation.type,
            relation.source_entity_id,
            relation.target_entity_id,
            relation.evidence,
            relation.source_id,
            relation.page,
            relation.confidence,
            relation.created_at.isoformat(),
        ))
        
        conn.commit()
        conn.close()
        return relation.id
    
    def add_relations(self, relations: List[Relation]) -> int:
        """Add multiple relations. Returns count added."""
        if not relations:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        data = [
            (
                r.id, r.type, r.source_entity_id, r.target_entity_id,
                r.evidence, r.source_id, r.page, r.confidence,
                r.created_at.isoformat()
            )
            for r in relations
        ]
        
        c.executemany('''
            INSERT OR REPLACE INTO relations
            (id, type, source_entity_id, target_entity_id, evidence, source_id, page, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        
        conn.commit()
        conn.close()
        return len(relations)
    
    def get_relations(
        self,
        entity_id: str = None,
        relation_type: str = None,
        direction: str = "both",
    ) -> List[Relation]:
        """
        Get relations for an entity.
        
        Args:
            entity_id: Entity to get relations for
            relation_type: Filter by relation type
            direction: "outgoing", "incoming", or "both"
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        sql = "SELECT * FROM relations WHERE 1=1"
        params = []
        
        if entity_id:
            if direction == "outgoing":
                sql += " AND source_entity_id = ?"
                params.append(entity_id)
            elif direction == "incoming":
                sql += " AND target_entity_id = ?"
                params.append(entity_id)
            else:  # both
                sql += " AND (source_entity_id = ? OR target_entity_id = ?)"
                params.extend([entity_id, entity_id])
        
        if relation_type:
            sql += " AND type = ?"
            params.append(relation_type)
        
        c.execute(sql, params)
        rows = c.fetchall()
        conn.close()
        
        return [self._row_to_relation(row) for row in rows]
    
    # =========================================================================
    # STATS AND UTILITIES
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get store statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM entities")
        total_entities = c.fetchone()[0]
        
        c.execute("SELECT type, COUNT(*) FROM entities GROUP BY type")
        entities_by_type = dict(c.fetchall())
        
        c.execute("SELECT COUNT(*) FROM relations")
        total_relations = c.fetchone()[0]
        
        c.execute("SELECT type, COUNT(*) FROM relations GROUP BY type")
        relations_by_type = dict(c.fetchall())
        
        c.execute("SELECT scope, COUNT(*) FROM entities WHERE scope != '' GROUP BY scope")
        by_scope = dict(c.fetchall())
        
        conn.close()
        
        return {
            "total_entities": total_entities,
            "entities_by_type": entities_by_type,
            "total_relations": total_relations,
            "relations_by_type": relations_by_type,
            "by_scope": by_scope,
        }
    
    def clear(self):
        """Clear all data."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM relations")
        c.execute("DELETE FROM entities")
        conn.commit()
        conn.close()
    
    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    
    def _row_to_entity(self, row) -> Entity:
        """Convert database row to Entity."""
        created_at = row['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return Entity(
            id=row['id'],
            type=row['type'],
            canonical_name=row['canonical_name'],
            aliases=json.loads(row['aliases']) if row['aliases'] else [],
            attributes=json.loads(row['attributes']) if row['attributes'] else {},
            source_id=row['source_id'] or "",
            page=row['page'] or 0,
            scope=row['scope'] or "",
            confidence=row['confidence'] or 1.0,
            created_at=created_at,
        )
    
    def _row_to_relation(self, row) -> Relation:
        """Convert database row to Relation."""
        created_at = row['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return Relation(
            id=row['id'],
            type=row['type'],
            source_entity_id=row['source_entity_id'],
            target_entity_id=row['target_entity_id'],
            evidence=row['evidence'] or "",
            source_id=row['source_id'] or "",
            page=row['page'] or 0,
            confidence=row['confidence'] or 1.0,
            created_at=created_at,
        )


# =============================================================================
# LEGACY COMPATIBILITY: FactStore alias
# =============================================================================

class FactStore:
    """
    Legacy FactStore for backwards compatibility.
    
    Wraps EntityStore with Fact interface.
    """
    
    def __init__(self, db_path: str = "./fact_store/facts.db"):
        self._store = EntityStore(db_path)
    
    def add_fact(self, fact: Fact):
        """Add a legacy Fact."""
        entity = fact.to_entity()
        self._store.add_entity(entity)
    
    def add_facts(self, facts: List[Fact]):
        """Add multiple Facts."""
        entities = [f.to_entity() for f in facts]
        self._store.add_entities(entities)
    
    def search(
        self,
        query: str = None,
        entity_type: str = None,
        key: str = None,
        value: str = None,
    ) -> List[Fact]:
        """Search with legacy interface."""
        entities = self._store.search_entities(
            query=query or value,
            entity_type=entity_type,
        )
        return [Fact.from_entity(e) for e in entities]
    
    def get_stats(self) -> dict:
        """Get stats in legacy format."""
        stats = self._store.get_stats()
        return {
            "total_facts": stats["total_entities"],
            "by_type": stats["entities_by_type"],
        }
