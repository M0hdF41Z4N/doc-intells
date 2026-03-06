import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

class EmbeddingsService:
    def __init__(self):
        self.provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
        
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002")
            self.vector_size = 1536
            
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = os.getenv("EMBEDDINGS_MODEL", "models/embedding-001")
            self.vector_size = 768
            
        elif self.provider == "mistral":
            from mistralai import Mistral
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
            self.model = os.getenv("EMBEDDINGS_MODEL", "mistral-embed")
            self.vector_size = 1024
            
        else:
            raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {self.provider}")

    def get_vector_size(self) -> int:
        return self.vector_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings.
        """
        if not texts:
            return []
            
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            # Ensure ordered matching
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
            
        elif self.provider == "google":
            import google.generativeai as genai
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
            
        elif self.provider == "mistral":
            embeddings_batch_response = self.client.embeddings.create(
                model=self.model,
                inputs=texts
            )
            return [data.embedding for data in embeddings_batch_response.data]

    def embed_query(self, text: str) -> List[float]:
        """
        Generates embedding for a single query string.
        """
        if self.provider == "google":
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
            
        return self.embed_texts([text])[0]

