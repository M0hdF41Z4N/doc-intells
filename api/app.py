from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from agents.graph import DocumentAgent
import logging

# Configure basic logging for the API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_api")

app = FastAPI(
    title="Autonomous Document Intelligence API",
    description="Query complex PDF documents using an autonomous agent.",
    version="1.0.0"
)

# Initialize Agent Graph Manually Once at Startup
agent_instance = None

@app.on_event("startup")
async def startup_event():
    global agent_instance
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        agent_builder = DocumentAgent()
        agent_instance = agent_builder.build()
        logger.info("Agent Graph Initialized Successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")

# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The query to ask against the document base.", min_length=1)

class Citation(BaseModel):
    page: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    trace: List[str]

@app.post("/query", response_model=QueryResponse)
async def execute_query(req: QueryRequest):
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent service is currently unavailable")
        
    initial_state = {
        "messages": [],
        "original_query": req.question,
        "sub_queries": [],
        "execution_plan": [],
        "evidence": [],
        "citations": [],
        "final_answer": "",
        "trace_log": [f"Received query: {req.question}"]
    }
    
    try:
        # Note: Depending on LangGraph version, you might invoke it directly
        final_state = agent_instance.invoke(initial_state)
        
        return QueryResponse(
            answer=final_state.get("final_answer", "No answer generated."),
            citations=final_state.get("citations", []),
            trace=final_state.get("trace_log", [])
        )
        
    except Exception as e:
        logger.error(f"Error executing agent workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "agent_loaded": agent_instance is not None}
