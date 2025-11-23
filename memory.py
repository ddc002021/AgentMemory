import sqlite3
import json
from datetime import datetime
import tiktoken
from config import LTM_DB_PATH, STM_TOKEN_BUDGET
import os
import re

class ShortTermMemory:
    def __init__(self, token_budget=STM_TOKEN_BUDGET):
        self.token_budget = token_budget
        self.messages = []
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_to_budget()
    
    def get_messages(self):
        return self.messages
    
    def _count_tokens(self, messages):
        total = 0
        for msg in messages:
            total += len(self.encoding.encode(msg["content"]))
        return total
    
    def _trim_to_budget(self):
        while len(self.messages) > 1 and self._count_tokens(self.messages) > self.token_budget:
            self.messages.pop(0)
    
    def clear(self):
        self.messages = []
    
    def get_context_string(self):
        context = []
        for msg in self.messages:
            context.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(context)

class LongTermMemory:
    def __init__(self, db_path=LTM_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT,
                salience REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL,
                success_outcome INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                attributes TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1_id INTEGER,
                entity2_id INTEGER,
                relation_type TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (entity1_id) REFERENCES entities(id),
                FOREIGN KEY (entity2_id) REFERENCES entities(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_fact(self, content, source=None, salience=0.5, success_outcome=False):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO facts (content, source, salience, timestamp, success_outcome)
            VALUES (?, ?, ?, ?, ?)
        """, (content, source, salience, datetime.now().isoformat(), int(success_outcome)))
        
        conn.commit()
        fact_id = cursor.lastrowid
        conn.close()
        
        return fact_id
    
    def get_facts(self, limit=10, min_salience=0.0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, source, salience, timestamp, success_outcome
            FROM facts
            WHERE salience >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (min_salience, limit))
        
        facts = []
        for row in cursor.fetchall():
            facts.append({
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "salience": row[3],
                "timestamp": row[4],
                "success_outcome": bool(row[5])
            })
        
        conn.close()
        return facts
    
    def save_entity(self, name, entity_type, attributes=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        attributes_json = json.dumps(attributes) if attributes else None
        
        try:
            cursor.execute("""
                INSERT INTO entities (name, type, attributes, timestamp)
                VALUES (?, ?, ?, ?)
            """, (name, entity_type, attributes_json, datetime.now().isoformat()))
            entity_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("""
                UPDATE entities
                SET type = ?, attributes = ?, timestamp = ?
                WHERE name = ?
            """, (entity_type, attributes_json, datetime.now().isoformat(), name))
            
            cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
            entity_id = cursor.fetchone()[0]
        
        conn.commit()
        conn.close()
        
        return entity_id
    
    def get_entity(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, type, attributes, timestamp
            FROM entities
            WHERE name = ?
        """, (name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "attributes": json.loads(row[3]) if row[3] else None,
                "timestamp": row[4]
            }
        return None
    
    def get_all_entities(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, type, attributes, timestamp
            FROM entities
            ORDER BY timestamp DESC
        """)
        
        entities = []
        for row in cursor.fetchall():
            entities.append({
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "attributes": json.loads(row[3]) if row[3] else None,
                "timestamp": row[4]
            })
        
        conn.close()
        return entities
    
    def save_relation(self, entity1_name, entity2_name, relation_type):
        entity1 = self.get_entity(entity1_name)
        entity2 = self.get_entity(entity2_name)
        
        if not entity1 or not entity2:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO entity_relations (entity1_id, entity2_id, relation_type, timestamp)
            VALUES (?, ?, ?, ?)
        """, (entity1["id"], entity2["id"], relation_type, datetime.now().isoformat()))
        
        conn.commit()
        relation_id = cursor.lastrowid
        conn.close()
        
        return relation_id
    
    def get_entity_relations(self, limit=20):
        """
        Retrieve entity relationships with entity names.
        Returns list of dicts with entity names and relation types.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                e1.name as entity1_name,
                e1.type as entity1_type,
                er.relation_type,
                e2.name as entity2_name,
                e2.type as entity2_type,
                er.timestamp
            FROM entity_relations er
            JOIN entities e1 ON er.entity1_id = e1.id
            JOIN entities e2 ON er.entity2_id = e2.id
            ORDER BY er.timestamp DESC
            LIMIT ?
        """, (limit,))
        
        relations = []
        for row in cursor.fetchall():
            relations.append({
                "entity1": row[0],
                "entity1_type": row[1],
                "relation_type": row[2],
                "entity2": row[3],
                "entity2_type": row[4],
                "timestamp": row[5]
            })
        
        conn.close()
        return relations
    
    def clear(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM entity_relations")
        cursor.execute("DELETE FROM entities")
        cursor.execute("DELETE FROM facts")
        
        conn.commit()
        conn.close()

class EntityExtractor:
    def __init__(self):
        self.phenomenon_pattern = re.compile(
            r'\b(green flash|fata morgana|brocken spectre|circumzenithal arc|'
            r'blood falls|blood rain|lake nyos|gravity hill|skyquake|'
            r'morning glory cloud|naga fireball|catatumbo lightning|brinicle|sailing stones)\b',
            re.IGNORECASE
        )
    
    def extract_from_text(self, text):
        entities = []
        
        phenomena = self.phenomenon_pattern.findall(text)
        for phenomenon in phenomena:
            entities.append({
                "name": phenomenon.title(),
                "type": "phenomenon",
                "attributes": {"mentioned_in": "conversation"}
            })
        
        location_indicators = re.findall(
            r'\b(?:in|at|near|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            text
        )
        for location in location_indicators:
            if len(location.split()) <= 3:
                entities.append({
                    "name": location,
                    "type": "location",
                    "attributes": {}
                })
        
        scientist_pattern = re.compile(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b')
        potential_scientists = scientist_pattern.findall(text)
        for name in potential_scientists:
            if name not in [e["name"] for e in entities]:
                entities.append({
                    "name": name,
                    "type": "person",
                    "attributes": {}
                })
        
        return entities
    
    def extract_relationships(self, text, entities):
        """
        Extract relationships between entities from text.
        Returns list of tuples: (entity1_name, entity2_name, relation_type)
        """
        relationships = []
        text_lower = text.lower()
        
        # Create lookup dictionaries for fast matching
        phenomena = [e["name"] for e in entities if e["type"] == "phenomenon"]
        locations = [e["name"] for e in entities if e["type"] == "location"]
        persons = [e["name"] for e in entities if e["type"] == "person"]
        
        # Pattern 1: Phenomenon occurs/found/seen in/at/near location
        location_relation_patterns = [
            r'(occurs?|found|seen|visible|observed|appears?|happens?) (?:in|at|near|over) ',
            r'(?:in|at|near|from) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for phenomenon in phenomena:
            phenomenon_lower = phenomenon.lower()
            
            # Find location relationships
            for location in locations:
                location_lower = location.lower()
                
                # Check various patterns
                patterns_to_check = [
                    f'{phenomenon_lower}.*(?:occurs?|found|seen|visible|observed|appears?|happens?).*(?:in|at|near|over).*{location_lower}',
                    f'{phenomenon_lower}.*(?:in|at|near|from).*{location_lower}',
                    f'(?:in|at|near).*{location_lower}.*{phenomenon_lower}',
                ]
                
                for pattern in patterns_to_check:
                    if re.search(pattern, text_lower):
                        relationships.append((phenomenon, location, "occurs_in"))
                        break
            
            # Find person relationships (studied, discovered, researched)
            for person in persons:
                person_lower = person.lower()
                
                patterns_to_check = [
                    f'{person_lower}.*(?:studied|discovered|researched|investigated|observed|documented).*{phenomenon_lower}',
                    f'{phenomenon_lower}.*(?:studied|discovered|researched) by.*{person_lower}',
                    f'{person_lower}.*(?:studies|discovers|researches).*{phenomenon_lower}',
                ]
                
                for pattern in patterns_to_check:
                    if re.search(pattern, text_lower):
                        relationships.append((person, phenomenon, "studied"))
                        break
        
        # Remove duplicates
        relationships = list(set(relationships))
        
        return relationships