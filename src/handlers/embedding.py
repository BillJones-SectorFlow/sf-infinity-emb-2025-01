"""
Embedding handler with RunPod native endpoint support
Migrated from OpenAI-compatible wrapper to /runsync endpoint
"""
import logging
from typing import List, Union, Dict, Any, Optional

from ..client import RunPodClient, RunPodAPIError
from ..models.response import EmbeddingResponse, Usage

logger = logging.getLogger(__name__)

class EmbeddingHandler:
    """Handle embedding requests using RunPod native /runsync endpoint"""
    
    def __init__(self, client: RunPodClient):
        self.client = client
    
    def create_embeddings(
        self,
        model: str,
        input_texts: Union[str, List[str]],
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Create embeddings using RunPod native endpoint
        
        Args:
            model: Model name/ID
            input_texts: Text(s) to embed
            encoding_format: Format for embeddings (float, base64)
            dimensions: Optional dimensionality for embeddings
            **kwargs: Additional model parameters
        
        Returns:
            EmbeddingResponse: OpenAI-compatible response format
        """
        # Normalize input to list
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        if not input_texts:
            raise ValueError("Input texts cannot be empty")
        
        # Prepare native RunPod request format
        runpod_input = {
            "model": model,
            "input": input_texts,
            "encoding_format": encoding_format
        }
        
        # Add optional parameters
        if dimensions:
            runpod_input["dimensions"] = dimensions
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in runpod_input:
                runpod_input[key] = value
        
        try:
            # Call native RunPod /runsync endpoint
            logger.info(f"Creating embeddings for {len(input_texts)} texts using model {model}")
            result = self.client.run_sync(runpod_input)
            
            # Transform RunPod response to OpenAI-compatible format
            return self._transform_embedding_response(result, model, len(input_texts))
            
        except RunPodAPIError as e:
            logger.error(f"RunPod API error during embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            raise RunPodAPIError(f"Embedding request failed: {str(e)}")
    
    def _transform_embedding_response(
        self, 
        runpod_result: Dict[str, Any], 
        model: str,
        input_count: int
    ) -> EmbeddingResponse:
        """
        Transform RunPod native response to OpenAI-compatible format
        
        RunPod /runsync wraps responses in 'output' array with metadata
        """
        # Extract output from RunPod response wrapper
        if "output" not in runpod_result:
            raise RunPodAPIError("Missing 'output' in RunPod response")
        
        output = runpod_result["output"]
        
        # Handle different output formats
        if isinstance(output, list) and len(output) > 0:
            # Output is wrapped in array, take first element
            embeddings_data = output[0]
        else:
            # Output is direct dict
            embeddings_data = output
        
        # Case 1: Already in OpenAI-compatible format
        if "data" in embeddings_data and "object" in embeddings_data:
            response_data = embeddings_data
        
        # Case 2: Custom embeddings format - need to transform
        elif "embeddings" in embeddings_data:
            embeddings = embeddings_data["embeddings"]
            response_data = {
                "object": "list",
                "model": model,
                "data": [
                    {
                        "object": "embedding",
                        "embedding": emb,
                        "index": i
                    }
                    for i, emb in enumerate(embeddings)
                ],
                "usage": {
                    "prompt_tokens": embeddings_data.get("prompt_tokens", input_count),
                    "total_tokens": embeddings_data.get("total_tokens", input_count)
                }
            }
        
        # Case 3: Direct embeddings array
        elif isinstance(embeddings_data, list):
            response_data = {
                "object": "list", 
                "model": model,
                "data": [
                    {
                        "object": "embedding",
                        "embedding": emb,
                        "index": i  
                    }
                    for i, emb in enumerate(embeddings_data)
                ],
                "usage": {
                    "prompt_tokens": input_count,
                    "total_tokens": input_count
                }
            }
        
        # Case 4: Unexpected format
        else:
            logger.warning(f"Unexpected embedding response format: {embeddings_data}")
            raise RunPodAPIError(f"Unexpected response format from RunPod: {type(embeddings_data)}")
        
        # Validate response structure
        if not isinstance(response_data.get("data"), list):
            raise RunPodAPIError("Invalid embeddings data format")
        
        if len(response_data["data"]) != input_count:
            logger.warning(f"Expected {input_count} embeddings, got {len(response_data['data'])}")
        
        # Create response object
        return EmbeddingResponse(
            object=response_data["object"],
            model=response_data["model"],
            data=response_data["data"],
            usage=Usage(**response_data.get("usage", {"prompt_tokens": 0, "total_tokens": 0}))
        )
    
    def batch_create_embeddings(
        self,
        model: str,
        input_batches: List[List[str]],
        **kwargs
    ) -> List[EmbeddingResponse]:
        """
        Create embeddings in batches for better throughput
        
        Args:
            model: Model name/ID
            input_batches: List of text batches
            **kwargs: Additional parameters
        
        Returns:
            List of embedding responses for each batch
        """
        results = []
        
        for i, batch in enumerate(input_batches):
            logger.info(f"Processing batch {i+1}/{len(input_batches)} with {len(batch)} texts")
            
            try:
                result = self.create_embeddings(model, batch, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch {i+1} failed: {e}")
                raise
        
        return results
