"""
hybrid_router.py - Routes queries to either RAG or Structured Data
This sits BETWEEN your app and the existing systems
"""

import re
from typing import Dict, Any

class HybridQueryRouter:
    """Routes queries to appropriate backend without changing either"""
    
    def __init__(self, rag_system, structured_processor):
        self.rag = rag_system  # Your existing RAG system
        self.structured = structured_processor  # New structured data module
        self.stats = {'rag': 0, 'structured': 0, 'hybrid': 0}
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Determine if query needs structured data or RAG"""
        query_lower = query.lower()
        
        # Patterns that indicate need for structured/tabular data
        structured_patterns = [
            # Numerical operations
            r'\b(min|max|avg|average|sum|total|count|percentage)\b',
            r'\b(rank|score|mark|cutoff|percentage)\b',
            r'\b(compare|vs|versus|difference)\b',
            r'\b(trend|increase|decrease|change|over time)\b',
            r'\b(highest|lowest|least|most|top)\b',
            
            # Year ranges
            r'\d{4}',  # Any 4-digit year
            r'between \d{4} and \d{4}',
            r'from \d{4} to \d{4}',
            
            # Table-specific
            r'\btable\b',
            r'\bdata\b',
            r'\bfigure\b',
            r'\bchart\b',
        ]
        
        # Patterns that indicate need for RAG (text understanding)
        rag_patterns = [
            r'\bwhat is\b',
            r'\bexplain\b',
            r'\bhow does\b',
            r'\bdescribe\b',
            r'\bdefine\b',
            r'\bprocess\b',
            r'\bpolicy\b',
            r'\bguideline\b',
            r'\bprocedure\b',
        ]
        
        # Score the query
        structured_score = 0
        rag_score = 0
        
        for pattern in structured_patterns:
            if re.search(pattern, query_lower):
                structured_score += 1
        
        for pattern in rag_patterns:
            if re.search(pattern, query_lower):
                rag_score += 1
        
        # Extract entities
        years = re.findall(r'\b(20\d{2})\b', query)
        colleges = re.findall(r'\b(CEG|MIT|PSG|GCE|TCE|NIT)\b', query.upper())
        
        decision = {
            'primary': 'structured' if structured_score > rag_score else 'rag',
            'structured_score': structured_score,
            'rag_score': rag_score,
            'years': years,
            'colleges': colleges,
            'needs_numbers': structured_score > 0,
            'needs_explanation': rag_score > 0,
        }
        
        # Track for stats
        if decision['primary'] == 'structured':
            self.stats['structured'] += 1
        else:
            self.stats['rag'] += 1
        
        return decision
    
    def route(self, query: str, context: Dict = None) -> str:
        """Route query to appropriate system"""
        decision = self.classify_query(query)
        
        print(f"\n🔀 Router Decision:")
        print(f"  Primary: {decision['primary']}")
        print(f"  Years found: {decision['years']}")
        print(f"  Colleges: {decision['colleges']}")
        
        # If query needs structured data
        if decision['primary'] == 'structured':
            # Find relevant tables
            tables = self.structured.find_relevant_tables(query)
            
            if tables:
                # Query structured data first
                results = []
                for table in tables[:2]:  # Check top 2 tables
                    data = self.structured.query_table(table)
                    if data:
                        results.extend(data[:5])  # Get first 5 rows
                
                if results:
                    # Format structured results
                    return self._format_structured_response(query, results, decision)
            
            # If no structured data found, fallback to RAG
            print("  ⚠ No structured data found, falling back to RAG")
            self.stats['hybrid'] += 1
        
        # Default to RAG (your existing system)
        return self.rag.answer(query)  # Call your existing RAG
    
    def _format_structured_response(self, query: str, data: List[Dict], decision: Dict) -> str:
        """Format structured data response"""
        if not data:
            return "No data found for your query."
        
        lines = [f"📊 **Query:** {query}"]
        lines.append(f"\n**Results from structured data:**\n")
        
        for i, row in enumerate(data[:3], 1):
            lines.append(f"**Record {i}:**")
            for key, value in row.items():
                if value is not None and str(value).strip():
                    lines.append(f"  • {key}: {value}")
            lines.append("")
        
        return "\n".join(lines)