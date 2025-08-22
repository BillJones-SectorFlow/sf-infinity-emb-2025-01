"""
Main RunPod handler with native /runsync endpoint support
Migration from OpenAI-compatible wrapper to native RunPod endpoints
"""
import os
import logging
import traceback
from typing import Dict, Any, Optional

from .config import config
from .client import RunPodClient, RunPodAPIError
from .handlers.embedding import EmbeddingHandler
from .handlers.reranking import RerankHandler
from .models.response import ErrorResponse

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize clients and handlers
runpod_client = RunPodClient(config.runpod)
embedding_handler = EmbeddingHandler(runpod_client)
rerank_handler = RerankHandler(runpod_client)

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler
    Processes embedding and reranking requests using native /runsync endpoint
    
    Expected input format (already unwrapped from RunPod 'input'):
    {
        "task": "embedding" | "rerank",
        "model": "model-name",
        "input": "text" | ["text1", "text2"] | {"query": "...", "docs": [...]}
        ... other parameters
    }
    """
    try:
        # Extract input data (RunPod already unwraps from 'input')
        job_input = job.get("input", {}) if "input" in job else job
        
        if not job_input:
            return _error_response("Missing input data", "INVALID_INPUT")
        
        # Determine task type
        task = job_input.get("task", "embedding")  # Default to embedding
        
        logger.info(f"Processing {task} request")
        logger.debug(f"Input data: {job_input}")
        
        # Validate required parameters
        model = job_input.get("model")
        if not model:
            return _error_response("Missing 'model' parameter", "MISSING_MODEL")
        
        # Route to appropriate handler
        if task == "embedding":
            return _handle_embedding(job_input)
        elif task == "rerank":
            return _handle_rerank(job_input)
        else:
            return _error_response(f"Unknown task type: {task}", "INVALID_TASK")
            
    except RunPodAPIError as e:
        logger.error(f"RunPod API error: {e}")
        return _error_response(
            str(e), 
            "API_ERROR", 
            {"status_code": e.status_code, "response_data": e.response_data}
        )
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return _error_response(str(e), "VALIDATION_ERROR")
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        return _error_response(
            "Out of memory processing request", 
            "MEMORY_ERROR",
            refresh_worker=True
        )
    except Exception as e:
        logger.exception(f"Unexpected error in handler: {e}")
        return _error_response(
            f"Internal server error: {str(e)}", 
            "INTERNAL_ERROR",
            {"traceback": traceback.format_exc()}
        )

def _handle_embedding(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle embedding request"""
    model = job_input["model"]
    input_texts = job_input.get("input")
    
    if not input_texts:
        return _error_response("Missing 'input' parameter for embedding", "MISSING_INPUT")
    
    # Extract optional parameters
    encoding_format = job_input.get("encoding_format", "float")
    dimensions = job_input.get("dimensions")
    
    # Filter out known parameters to pass others as kwargs
    known_params = {"task", "model", "input", "encoding_format", "dimensions"}
    kwargs = {k: v for k, v in job_input.items() if k not in known_params}
    
    try:
        response = embedding_handler.create_embeddings(
            model=model,
            input_texts=input_texts,
            encoding_format=encoding_format,
            dimensions=dimensions,
            **kwargs
        )
        
        # Return in format expected by RunPod (will be wrapped in 'output')
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Embedding handler error: {e}")
        raise

def _handle_rerank(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle reranking request"""
    model = job_input["model"]
    
    # Handle different input formats
    if "query" in job_input and ("documents" in job_input or "docs" in job_input):
        # Direct query and documents
        query = job_input["query"]
        documents = job_input.get("documents", job_input.get("docs", []))
    elif "input" in job_input:
        # Input object format
        input_data = job_input["input"]
        if isinstance(input_data, dict):
            query = input_data.get("query")
            documents = input_data.get("documents", input_data.get("docs", []))
        else:
            return _error_response("Invalid input format for reranking", "INVALID_FORMAT")
    else:
        return _error_response("Missing 'query' and 'documents' for reranking", "MISSING_PARAMS")
    
    if not query:
        return _error_response("Missing or empty 'query' parameter", "MISSING_QUERY")
    if not documents:
        return _error_response("Missing or empty 'documents' parameter", "MISSING_DOCUMENTS")
    
    # Extract optional parameters
    top_k = job_input.get("top_k")
    return_documents = job_input.get("return_documents", False)
    max_chunks_per_doc = job_input.get("max_chunks_per_doc")
    
    # Filter out known parameters
    known_params = {
        "task", "model", "query", "documents", "docs", "input", "top_k", 
        "return_documents", "max_chunks_per_doc"
    }
    kwargs = {k: v for k, v in job_input.items() if k not in known_params}
    
    try:
        response = rerank_handler.rerank(
            model=model,
            query=query,
            documents=documents,
            top_k=top_k,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            **kwargs
        )
        
        # Return in format expected by RunPod
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Rerank handler error: {e}")
        raise

def _error_response(
    error_message: str, 
    error_code: str, 
    details: Optional[Dict[str, Any]] = None,
    refresh_worker: bool = False
) -> Dict[str, Any]:
    """Create standardized error response"""
    error_resp = ErrorResponse(
        error=error_message,
        code=error_code,
        details=details
    ).to_dict()
    
    if refresh_worker:
        error_resp["refresh_worker"] = True
    
    return error_resp

# Health check function for testing
def health_check() -> Dict[str, Any]:
    """Health check function"""
    try:
        is_healthy = runpod_client.health_check()
        endpoint_info = runpod_client.get_endpoint_info()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "config": {
                "models": config.embedding.model_names,
                "use_native_endpoints": config.runpod.use_native_endpoints,
                "endpoint_id": config.runpod.endpoint_id[-8:]  # Only show last 8 chars
            },
            "endpoint_info": endpoint_info
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# For local testing
if __name__ == "__main__":
    # Test embedding
    test_embedding_input = {
        "task": "embedding",
        "model": config.embedding.model_names[0],
        "input": ["Hello world", "Test embedding"]
    }
    
    print("Testing embedding...")
    result = handler({"input": test_embedding_input})
    print(f"Embedding result: {result}")
    
    # Test reranking
    test_rerank_input = {
        "task": "rerank", 
        "model": config.embedding.model_names[0],
        "query": "What is artificial intelligence?",
        "documents": [
            "AI is a branch of computer science.",
            "Machine learning is a subset of AI.",
            "Pizza is a popular Italian food."
        ],
        "return_documents": True
    }
    
    print("\nTesting reranking...")
    result = handler({"input": test_rerank_input})
    print(f"Rerank result: {result}")
