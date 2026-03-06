from typing import List, Dict, Any
import json
import uuid

class DataTransformer:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # We assume crude word-based tokenization for simplicity, 
        # normally you would use tiktoken or similar based on embedding model.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_text_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks the extracted text pages while preserving metadata.
        """
        chunks = []
        for page in pages:
            text = page.get('text', '')
            page_num = page.get('page')
            
            # Simple word-based chunking
            words = text.split()
            
            if not words:
                continue

            i = 0
            while i < len(words):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "page": page_num,
                    "type": "text",
                    "section_type": "body"
                })
                
                i += (self.chunk_size - self.chunk_overlap)

        return chunks

    def process_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms tables into standalone structured JSON blocks.
        """
        processed_tables = []
        for table in tables:
            processed_tables.append({
                "chunk_id": str(uuid.uuid4()),
                "original_id": table.get('table_id'),
                # Serialize the rows to JSON string so it can be embedded or retrieved easily
                "text": json.dumps(table.get('rows', [])),
                "page": table.get('page'),
                "type": "table",
                "section_type": "table",
                "raw_rows": table.get('rows', [])
            })
        return processed_tables
