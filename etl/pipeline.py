import os
import argparse
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging

# Add project root to path so we can import etl.* and services.*
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from etl.pdf_extractor import PDFTextExtractor
from etl.table_extractor import PDFTableExtractor
from etl.transformer import DataTransformer
from services.vector_storage import VectorStorage
from services.embeddings import EmbeddingsService

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> bool:
    """
    Main ETL pipeline execution function.
    Reads a PDF, extracts text and tables, chunks the text, embeds, and loads into Qdrant.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False
        
    try:
        logger.info(f"Starting ETL pipeline for: {filepath}")
        
        # 1. Extraction
        logger.info("Extracting text from PDF...")
        text_extractor = PDFTextExtractor(filepath)
        text_pages = text_extractor.extract_text()
        logger.info(f"Extracted {len(text_pages)} pages of text.")

        logger.info("Extracting tables from PDF...")
        table_extractor = PDFTableExtractor(filepath)
        extracted_tables = table_extractor.extract_tables()
        logger.info(f"Extracted {len(extracted_tables)} tables.")

        # 2. Transformation (Chunking)
        logger.info("Transforming and chunking data...")
        transformer = DataTransformer(chunk_size=800, chunk_overlap=150)
        
        text_chunks = transformer.process_text_pages(text_pages)
        logger.info(f"Created {len(text_chunks)} text chunks.")
        
        table_chunks = transformer.process_tables(extracted_tables)
        logger.info(f"Created {len(table_chunks)} table chunks.")

        all_documents = text_chunks + table_chunks
        
        if not all_documents:
            logger.warning("No data extracted to index.")
            return False

        # 3. Embedding
        logger.info("Generating embeddings...")
        embedder = EmbeddingsService()
        
        # Prepare texts for embedding
        texts_to_embed = [doc['text'] for doc in all_documents]
        
        # Batch embedding (depending on the provider limits, might need to chunk this list)
        # Using a simple batch logic to prevent payload too large errors
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1}...")
            batch_embeddings = embedder.embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)

        logger.info(f"Generated {len(all_embeddings)} embeddings.")

        if len(all_documents) != len(all_embeddings):
             logger.error("Mismatch between documents and embeddings length.")
             return False

        # 4. Loading (Vector DB Storage)
        logger.info("Loading data into Vector Database (Qdrant)...")
        vector_storage = VectorStorage(vector_size=embedder.get_vector_size())
        vector_storage.init_collection()
        
        # Batch upsert to Qdrant
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i + batch_size]
            batch_embs = all_embeddings[i:i + batch_size]
            logger.info(f"Upserting batch {i // batch_size + 1} to Qdrant...")
            vector_storage.index_documents(batch_docs, batch_embs)

        logger.info("ETL pipeline completed successfully!")
        return True

    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run the ETL pipeline to ingest a PDF into the vector database.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to ingest.")
    
    args = parser.parse_args()
    
    success = load_data(args.pdf_path)
    if not success:
        sys.exit(1)
    sys.exit(0)
