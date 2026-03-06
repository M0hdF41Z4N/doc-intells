from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class VectorStorage:
    def __init__(self, collection_name: str = "documents", vector_size: int = 1536):
        # Configure client. In production this uses Qdrant Cloud or a cluster.
        # For simplicity, fallback to an in-memory DB if url/key are missing.
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if self.qdrant_url and self.qdrant_api_key:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        elif self.qdrant_url:
            self.client = QdrantClient(url=self.qdrant_url)
        else:
            self.client = QdrantClient(":memory:")
            
        self.collection_name = collection_name
        self.vector_size = vector_size

    def init_collection(self):
        """Creates the Qdrant collection if it does not exist."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def index_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Indexes a list of documents alongside their embedding vectors.
        """
        if not documents or not embeddings or len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must be non-empty and of equal length")
        
        points = []
        for doc, emb in zip(documents, embeddings):
            points.append(models.PointStruct(
                id=doc.get('chunk_id'), # Important: chunk_id MUST be a valid UUID
                vector=emb,
                payload={
                    "text": doc.get('text'),
                    "page": doc.get('page'),
                    "type": doc.get('type'),
                    "section_type": doc.get('section_type')
                }
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], top_k: int = 5, filter_type: str = None) -> List[Dict[str, Any]]:
        """
        Executes a similarity search against the indexed documents.
        """
        search_filter = None
        if filter_type:
            search_filter = models.Filter(
                must=[models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=filter_type)
                )]
            )
            
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } 
            for hit in results
        ]
