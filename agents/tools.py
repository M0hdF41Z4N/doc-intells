from langchain.tools import BaseTool
from typing import ClassVar, Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field
import json
from services.vector_storage import VectorStorage
from services.embeddings import EmbeddingsService

class RetrievalInput(BaseModel):
    query: str = Field(description="The question or keywords to search the document text for.")

class RetrievalTool(BaseTool):
    name: ClassVar[str] = "document_retrieval"
    description: ClassVar[str] = "Useful for finding factual text information, paragraphs, and general document content."
    args_schema: Type[BaseModel] = RetrievalInput
    
    # Needs instances of storage and embeddings to function
    vector_storage: VectorStorage = None
    embeddings_service: EmbeddingsService = None

    def _run(self, query: str) -> str:
        if not self.vector_storage or not self.embeddings_service:
            return "Error: Storage services not initialized."
            
        try:
            query_vector = self.embeddings_service.embed_query(query)
            results = self.vector_storage.search(
                query_vector=query_vector, 
                top_k=5, 
                filter_type="text"
            )
            
            if not results:
                return "No relevant text documents found."
                
            formatted_results = []
            for hit in results:
                payload = hit['payload']
                page = payload.get('page', 'Unknown')
                text = payload.get('text', '')
                formatted_results.append(f"[Page {page}]: {text}")
                
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Retrieval failed: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")


class TableQueryInput(BaseModel):
    query: str = Field(description="The keyword or topic of the table to search for.")

class TableTool(BaseTool):
    name: ClassVar[str] = "table_retrieval"
    description: ClassVar[str] = "Useful for finding structured, tabular data like statistics, metrics, and comparisons."
    args_schema: Type[BaseModel] = TableQueryInput
    
    vector_storage: VectorStorage = None
    embeddings_service: EmbeddingsService = None

    def _run(self, query: str) -> str:
        if not self.vector_storage or not self.embeddings_service:
            return "Error: Storage services not initialized."
            
        try:
            query_vector = self.embeddings_service.embed_query(query)
            results = self.vector_storage.search(
                query_vector=query_vector, 
                top_k=3, 
                filter_type="table"
            )
            
            if not results:
                return "No relevant tables found."
                
            formatted_results = []
            for hit in results:
                payload = hit['payload']
                page = payload.get('page', 'Unknown')
                text_json = payload.get('text', '[]')
                
                try:
                    table_rows = json.loads(text_json)
                    table_str = "\n".join([str(row) for row in table_rows])
                    formatted_results.append(f"Table on Page {page}:\n{table_str}")
                except json.JSONDecodeError:
                    formatted_results.append(f"Table on Page {page} (Parse Error): {text_json}")
                    
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Table retrieval failed: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not implemented")


class MathInput(BaseModel):
    operation: str = Field(description="The math operation to perform. Supported: 'cagr', 'percentage', 'average'")
    values: List[float] = Field(description="The numerical values to compute on.")
    years: Optional[float] = Field(None, description="Number of years, required only for CAGR.")

class MathTool(BaseTool):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = "Useful for deterministic math calculations. Can compute 'cagr', 'percentage' (v1 / v2 * 100), or 'average'. Input values as a list."
    args_schema: Type[BaseModel] = MathInput

    def _run(self, operation: str, values: List[float], years: Optional[float] = None) -> str:
        try:
            op = operation.lower().strip()
            
            if not values:
                return "Error: No values provided for calculation."
                
            if op == "cagr":
                if len(values) < 2:
                    return "Error: CAGR requires [start_value, end_value]."
                if not years or years <= 0:
                    return "Error: CAGR requires a positive number of years."
                
                start_val = values[0]
                end_val = values[-1] # if they provide a list, take first and last
                
                if start_val == 0:
                    return "Error: Division by zero (start value is 0)."
                    
                cagr = ((end_val / start_val) ** (1 / years)) - 1
                return f"CAGR: {cagr:.4%} ({(cagr*100):.2f}%)"
                
            elif op == "percentage":
                if len(values) < 2:
                    return "Error: Percentage requires [numerator, denominator]."
                if values[1] == 0:
                    return "Error: Division by zero."
                
                pct = (values[0] / values[1]) * 100
                return f"Percentage: {pct:.2f}%"
                
            elif op == "average":
                avg = sum(values) / len(values)
                return f"Average: {avg:.2f}"
                
            else:
                return f"Error: Unsupported operation '{operation}'. Use 'cagr', 'percentage', or 'average'."
                
        except Exception as e:
            return f"Math error: {str(e)}"

    def _arun(self, operation: str, values: List[float], years: Optional[float] = None):
        raise NotImplementedError("Async not implemented")
