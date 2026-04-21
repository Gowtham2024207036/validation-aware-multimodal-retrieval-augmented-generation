"""
app5.py - Fixed version with proper image display and sources
"""

import os
import sys
import threading
from pathlib import Path
import sqlite3
import pandas as pd
import re
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

# ==================== YOUR EXISTING IMPORTS ====================
from base_retriever import SharedModels
from lmstudio_client import check_connection, generate_text_only, generate_with_images, LM_STUDIO_BASE_URL
from prompt_builder2 import SYSTEM_PROMPT, build_text_only_prompt, build_multimodal_prompt
import arch6_full_proposed as best_arch

# Import ingest functions
try:
    from document_ingestor import ingest_pdf, ingest_image
except ImportError:
    def ingest_pdf(*args, **kwargs): return False, "Document ingestor not available"
    def ingest_image(*args, **kwargs): return {"success": False, "message": "Not available"}

import config

IMAGES_ROOT = Path(config.RAW_DATA_DIR)

# ==================== STRUCTURED DATA MODULE ====================
class StructuredDataProcessor:
    """Handles table extraction and SQL storage"""
    
    def __init__(self, db_path: str = "data/structured_data.db"):
        os.makedirs("data", exist_ok=True)
        self.db_path = db_path
        self.conn = None
        self._init_db()
        print(f"✅ Structured Data Processor initialized")
    
    def _init_db(self):
        """Initialize database with registry"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS table_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT UNIQUE,
                source_pdf TEXT,
                page_number INTEGER,
                row_count INTEGER,
                col_count INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract all tables from PDF"""
        import pdfplumber
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        
                        headers = table[0] if table[0] else [f"Col_{i}" for i in range(len(table[1]))]
                        headers = [str(h).strip() if h else f"Col_{i}" for i, h in enumerate(headers)]
                        
                        df = pd.DataFrame(table[1:], columns=headers)
                        df = df.replace('', None).replace(' ', None)
                        
                        tables_data.append({
                            'page': page_num,
                            'table_idx': table_idx,
                            'dataframe': df,
                            'headers': headers,
                            'shape': df.shape
                        })
            
        except Exception as e:
            print(f"⚠️ Table extraction warning: {e}")
        
        return tables_data
    
    def process_pdf(self, pdf_path: str, doc_name: str = None) -> Dict:
        """Process PDF for structured data"""
        results = {
            'pdf': pdf_path,
            'doc_name': doc_name or Path(pdf_path).stem,
            'tables_found': 0,
            'tables_stored': [],
            'status': 'success'
        }
        
        try:
            tables = self.extract_tables_from_pdf(pdf_path)
            results['tables_found'] = len(tables)
            
            for i, table_info in enumerate(tables):
                df = table_info['dataframe']
                table_name = f"{results['doc_name']}_p{table_info['page']}_{i}"
                table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
                
                df.to_sql(table_name, self.conn, if_exists='replace', index=False)
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO table_registry 
                    (table_name, source_pdf, page_number, row_count, col_count, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    table_name,
                    Path(pdf_path).name,
                    table_info['page'],
                    df.shape[0],
                    df.shape[1],
                    f"Table with columns: {', '.join(df.columns[:5])}"
                ))
                
                results['tables_stored'].append({
                    'name': table_name,
                    'page': table_info['page'],
                    'rows': df.shape[0],
                    'cols': df.shape[1],
                    'columns': list(df.columns)
                })
            
            self.conn.commit()
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def query_table(self, table_name: str, conditions: Dict = None, limit: int = 10) -> List[Dict]:
        """Query a specific table"""
        try:
            cursor = self.conn.cursor()
            query = f"SELECT * FROM {table_name}"
            params = []
            
            if conditions:
                where_clauses = []
                for col, val in conditions.items():
                    where_clauses.append(f'"{col}" = ?')
                    params.append(val)
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            return []
    
    def find_relevant_tables(self, query: str) -> List[str]:
        """Find tables relevant to query"""
        query_lower = query.lower()
        relevant = []
        
        cursor = self.conn.execute("SELECT table_name FROM table_registry ORDER BY created_at DESC")
        
        for row in cursor.fetchall():
            table_name = row['table_name']
            if any(word in table_name.lower() for word in query_lower.split() if len(word) > 3):
                relevant.append(table_name)
                continue
            
            try:
                sample = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchone()
                if sample:
                    sample_str = str(dict(sample)).lower()
                    if any(word in sample_str for word in query_lower.split() if len(word) > 3):
                        relevant.append(table_name)
            except:
                continue
        
        return relevant[:5]


# ==================== HYBRID ROUTER ====================
class HybridQueryRouter:
    """Routes queries to RAG or Structured Data"""
    
    def __init__(self, rag_system, structured_processor):
        self.rag = rag_system
        self.structured = structured_processor
        self.stats = {'rag': 0, 'structured': 0, 'hybrid': 0, 'total': 0}
    
    def classify_query(self, query: str) -> Dict:
        """Determine if query needs structured data"""
        query_lower = query.lower()
        
        structured_patterns = [
            r'\b(min|max|avg|average|sum|total|count)\b',
            r'\b(rank|score|mark|cutoff|percentage)\b',
            r'\b(compare|vs|versus|difference)\b',
            r'\b(trend|increase|decrease|change)\b',
            r'\b(highest|lowest|least|most|top)\b',
            r'\d{4}',
            r'\b(between|from|to)\b \d{4}',
        ]
        
        rag_patterns = [
            r'\bwhat is\b',
            r'\bexplain\b',
            r'\bhow does\b',
            r'\bdescribe\b',
            r'\bdefine\b',
            r'\bprocess\b',
            r'\bpolicy\b',
        ]
        
        structured_score = sum(1 for p in structured_patterns if re.search(p, query_lower))
        rag_score = sum(1 for p in rag_patterns if re.search(p, query_lower))
        
        years = re.findall(r'\b(20\d{2})\b', query)
        colleges = re.findall(r'\b(CEG|MIT|PSG|GCE|TCE|NIT)\b', query.upper())
        
        return {
            'use_structured': structured_score > rag_score,
            'structured_score': structured_score,
            'rag_score': rag_score,
            'years': years,
            'colleges': colleges,
            'needs_numbers': bool(re.search(r'\d+', query))
        }
    
    def route(self, query: str, context: Dict = None) -> Dict:
        """Route query and return answer with sources"""
        self.stats['total'] += 1
        decision = self.classify_query(query)
        
        # Try structured data first if needed
        if decision['use_structured']:
            self.stats['structured'] += 1
            tables = self.structured.find_relevant_tables(query)
            if tables:
                data = []
                for table in tables[:2]:
                    data.extend(self.structured.query_table(table, limit=10))
                if data:
                    return {
                        'answer': self._format_structured_answer(query, data),
                        'source_type': 'structured',
                        'data': data[:5],
                        'tables': tables
                    }
            self.stats['hybrid'] += 1
        
        # Fallback to RAG
        self.stats['rag'] += 1
        rag_result = self.rag.answer_with_sources(query)
        return rag_result
    
    def _format_structured_answer(self, query: str, data: List[Dict]) -> str:
        """Format structured data answer"""
        if not data:
            return "No data found for your query."
        
        lines = [f"Based on the structured data:"]
        
        for i, row in enumerate(data[:3], 1):
            lines.append(f"\nRecord {i}:")
            for key, value in row.items():
                if value is not None and str(value).strip():
                    lines.append(f"  • {key}: {value}")
        
        return "\n".join(lines)


# ==================== RAG SYSTEM ====================
class RAGSystem:
    """Wrapper for existing RAG functionality"""
    
    def __init__(self):
        self.models = None
        self.images_root = Path(config.RAW_DATA_DIR)
        self._lock = threading.Lock()
    
    def load_models(self):
        with self._lock:
            if self.models is None:
                try:
                    self.models = SharedModels()
                except Exception as e:
                    print(f"Error loading models: {e}")
                    return None
        return self.models
    
    def index_document(self, pdf_path: str, doc_name: str = None):
        success, message = ingest_pdf(pdf_path, doc_name)
        return success, message
    
    def answer_with_sources(self, query: str) -> Dict:
        """Answer query and return with sources"""
        if not query.strip():
            return {
                'answer': "Please enter a question.",
                'source_type': 'error',
                'text_hits': [],
                'image_hits': [],
                'gallery': []
            }
        
        models = self.load_models()
        if models is None:
            return {
                'answer': "System not ready",
                'source_type': 'error',
                'text_hits': [],
                'image_hits': [],
                'gallery': []
            }
        
        try:
            result = best_arch.retrieve(query, models)
            text_hits = result.get("text", [])
            image_hits = result.get("image", [])
            
            # Create gallery
            gallery = []
            for h in image_hits:
                path = h.get("image_path", "")
                if path:
                    full_path = self.images_root / path
                    if full_path.exists():
                        gallery.append(str(full_path))
            
            # Build context
            context = []
            if text_hits:
                context.append("=== RETRIEVED DOCUMENTS ===")
                for i, h in enumerate(text_hits[:3], 1):
                    doc = h.get("doc_name", "Unknown").replace("_", " ").title()
                    text = h.get("text", "").strip()[:800]
                    page = h.get("page_id", "")
                    page_str = f" [Page {page}]" if page else ""
                    context.append(f"\n[{i}] From: {doc}{page_str}\n{text}")
            
            if image_hits:
                context.append("\n=== IMAGES ===")
                for i, h in enumerate(image_hits[:2], 1):
                    doc = h.get("doc_name", "Unknown").replace("_", " ").title()
                    desc = h.get("text", "").strip()
                    context.append(f"\n[Image {i}] {doc}: {desc[:300]}")
            
            context.append(f"\n=== QUESTION ===\n{query}")
            
            system = "You are a helpful assistant. Answer based ONLY on the documents above."
            answer = generate_text_only(system, "\n\n".join(context), max_tokens=512, temperature=0.1)
            
            return {
                'answer': answer,
                'source_type': 'rag',
                'text_hits': text_hits[:3],
                'image_hits': image_hits[:3],
                'gallery': gallery
            }
            
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'source_type': 'error',
                'text_hits': [],
                'image_hits': [],
                'gallery': []
            }


# ==================== MAIN APPLICATION ====================
class HybridDocumentQA:
    """Main application combining everything"""
    
    def __init__(self):
        print("=" * 60)
        print("Initializing Validation Aware Document QA System")
        print("=" * 60)
        
        self.rag = RAGSystem()
        self.structured = StructuredDataProcessor()
        self.router = HybridQueryRouter(self.rag, self.structured)
        
        print("✅ System ready")
    
    def process_document(self, pdf_path: str, doc_name: str = None) -> str:
        """Process document with BOTH systems"""
        print(f"\n📄 Processing: {pdf_path}")
        
        results = []
        
        # RAG indexing
        success, message = self.rag.index_document(pdf_path, doc_name)
        if success:
            results.append(f"✅ RAG: {message}")
        else:
            results.append(f"❌ RAG: {message}")
        
        # Structured data processing
        doc_name = doc_name or Path(pdf_path).stem
        struct_results = self.structured.process_pdf(pdf_path, doc_name)
        
        if struct_results['tables_found'] > 0:
            results.append(f"✅ Structured: Found {struct_results['tables_found']} tables")
        else:
            results.append("ℹ️ Structured: No tables found")
        
        return "\n".join(results)
    
    def answer(self, query: str) -> tuple:
        """Answer and return (answer, gallery, sources_text)"""
        result = self.router.route(query)
        
        if result['source_type'] == 'structured':
            # Format sources for structured data
            sources = []
            if 'tables' in result:
                sources.append(f"📊 Tables: {', '.join(result['tables'])}")
            return result['answer'], [], "\n".join(sources)
        
        elif result['source_type'] == 'rag':
            # Format sources for RAG
            sources = []
            for h in result.get('text_hits', []):
                doc = h.get('doc_name', 'Unknown').replace('_', ' ').title()
                page = h.get('page_id', '')
                page_str = f" (Page {page})" if page else ""
                sources.append(f"📄 {doc}{page_str}")
            
            for h in result.get('image_hits', []):
                doc = h.get('doc_name', 'Unknown').replace('_', ' ').title()
                page = h.get('page_id', '')
                page_str = f" (Page {page})" if page else ""
                sources.append(f"🖼️ {doc}{page_str}")
            
            return result['answer'], result.get('gallery', []), "\n".join(sources)
        
        else:
            return result['answer'], [], "No sources available"


# ==================== GRADIO UI ====================
system = HybridDocumentQA()

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.header { text-align: center; margin: 20px 0; }
.source-box { background: #f0f2f6; padding: 10px; border-radius: 8px; margin: 10px 0; font-size: 14px; }
.gallery { min-height: 150px; }
.answer-box { font-size: 16px; line-height: 1.6; }
"""

with gr.Blocks(title="Hybrid Document QA System", css=CUSTOM_CSS) as app:
    
    gr.Markdown("""
    # Validation Aware Document Question Answering
    ### 
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📤 Upload Document")
                pdf_file = gr.File(label="Select PDF", file_types=[".pdf"])
                pdf_name = gr.Textbox(label="Document Name (optional)", placeholder="Auto-generated")
                upload_btn = gr.Button("Process Document", variant="primary")
                upload_status = gr.Textbox(label="Status", lines=3, interactive=False)
    
    gr.Markdown("---")
    
    with gr.Group():
        gr.Markdown("### ❓ Ask Question")
        query = gr.Textbox(label="", placeholder="e.g. What is the combined percentage of Album Sales and Song Sales for Country genre?", lines=2)
        ask_btn = gr.Button("Ask Question", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer = gr.Textbox(label="Answer", lines=8, interactive=False, elem_classes="answer-box")
        with gr.Column(scale=1):
            sources = gr.Textbox(label="Sources", lines=6, interactive=False, elem_classes="source-box")
    
    gallery = gr.Gallery(label="Retrieved Images", columns=4, rows=1, height=200, elem_classes="gallery")
    
    # Example queries
    gr.Examples(
        examples=[
            ["What is the combined percentage of Album Sales and Song Sales for Country genre?"],
            ["What is the least cutoff for CEG in 2024?"],
            ["Explain TNEA counselling process"],
        ],
        inputs=query
    )
    
    # Event Handlers
    upload_btn.click(
        fn=system.process_document,
        inputs=[pdf_file, pdf_name],
        outputs=[upload_status]
    ).then(
        fn=lambda: gr.update(value=None),
        inputs=None,
        outputs=[pdf_file]
    )
    
    ask_btn.click(
        fn=system.answer,
        inputs=[query],
        outputs=[answer, gallery, sources]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[query]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)