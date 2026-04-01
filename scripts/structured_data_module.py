"""
structured_data_module.py - ADD-ON module, doesn't touch existing code
"""
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

class StructuredDataProcessor:
    """Processes tabular data - works alongside existing RAG"""
    
    def __init__(self, db_path: str = "data/structured_data.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database if not exists"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS table_registry (
                id INTEGER PRIMARY KEY,
                table_name TEXT UNIQUE,
                source_pdf TEXT,
                page_number INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def process_tables_from_pdf(self, pdf_path: str) -> Dict:
        """Extract and store tables - doesn't affect your existing indexing"""
        import pdfplumber
        import camelot
        
        results = {
            'pdf': pdf_path,
            'tables_found': 0,
            'tables_stored': []
        }
        
        # Extract tables (this is NEW functionality)
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:
                        # Convert to DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                        
                        # Auto-generate table name
                        table_name = f"table_p{page_num}_{table_idx}"
                        
                        # Store in SQLite
                        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
                        
                        # Register table
                        self.conn.execute(
                            "INSERT OR REPLACE INTO table_registry (table_name, source_pdf, page_number) VALUES (?, ?, ?)",
                            (table_name, Path(pdf_path).name, page_num)
                        )
                        
                        results['tables_found'] += 1
                        results['tables_stored'].append({
                            'name': table_name,
                            'page': page_num,
                            'rows': len(df),
                            'columns': list(df.columns)
                        })
        
        self.conn.commit()
        return results
    
    def query_table(self, table_name: str, conditions: Dict = None) -> List[Dict]:
        """Query a specific table"""
        cursor = self.conn.cursor()
        query = f"SELECT * FROM {table_name}"
        params = []
        
        if conditions:
            where_clauses = []
            for col, val in conditions.items():
                where_clauses.append(f"{col} = ?")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def find_relevant_tables(self, query: str) -> List[str]:
        """Find which tables might be relevant to a query"""
        query_lower = query.lower()
        relevant_tables = []
        
        # Get all tables
        cursor = self.conn.execute("SELECT table_name, description FROM table_registry")
        
        for row in cursor.fetchall():
            table_name = row['table_name']
            # Sample first few rows to understand content
            try:
                sample = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchone()
                if sample:
                    # Convert to string and check relevance
                    sample_str = str(dict(sample)).lower()
                    if any(word in sample_str for word in query_lower.split()):
                        relevant_tables.append(table_name)
            except:
                continue
        
        return relevant_tables