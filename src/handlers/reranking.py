"""
Reranking handler with RunPod native endpoint support
"""
import logging
from typing import List, Dict, Any, Optional, Union

from ..client import RunPodClient, RunPodAPIError
from ..models.response import RerankResponse, RerankResult

logger = logging.getLogger(__name__)

class RerankHandler:
    """Handle reranking requests using RunPod native /runsync endpoint"""
    
    def __init__(self, client: RunPodClient):
        self.client = client
    
    def rerank(
        self,
        model: str,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
        max_chunks_per_doc: Optional[int] = None,
        **kwargs
    ) -> RerankResponse:
        """
        Rerank documents using RunPod native endpoint
        
        Args:
            model: Reranking model name/ID
            query: Query text
            documents: List of documents to rerank
            top_k: Number of top results to return
            return_documents: Whether to return document texts
            max_chunks_per_doc: Max chunks per document
            **kwargs: Additional model parameters
        
        Returns:
            RerankResponse: Reranking results with relevance scores
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Prepare native RunPod request format
        runpod_input = {
            "model": model,
            "query": query,
            "docs": documents,  # RunPod uses 'docs' instead of 'documents'
            "return_docs": return_documents
        }
        
        # Add optional parameters
        if top_k is not None:
            runpod_input["top_k"] = top_k
        if max_chunks_per_doc is not None:
            runpod_input["max_chunks_per_doc"] = max_chunks_per_doc
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in runpod_input:
                runpod_input[key] = value
        
        try:
            logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")
            result = self.client.run_sync(runpod_input)
            
            # Transform RunPod response to standard format
            return self._transform_rerank_response(
                result, 
                model, 
                query,
                documents, 
                return_documents
            )
            
        except RunPodAPIError as e:
            logger.error(f"RunPod API error during reranking: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during reranking: {e}")
            raise RunPodAPIError(f"Reranking request failed: {str(e)}")
    
    def _transform_rerank_response(
        self,
        runpod_result: Dict[str, Any],
        model: str,
        query: str,
        original_documents: List[str],
        return_documents: bool
    ) -> RerankResponse:
        """Transform RunPod rerank response to standard format"""
        
        # Extract output from RunPod response wrapper  
        if "output" not in runpod_result:
            raise RunPodAPIError("Missing 'output' in RunPod rerank response")
        
        output = runpod_result["output"]
        
        # Handle array-wrapped output
        if isinstance(output, list) and len(output) > 0:
            rerank_data = output[0]
        else:
            rerank_data = output
        
        # Extract scores and optional documents
        scores = rerank_data.get("scores", [])
        if not scores:
            raise RunPodAPIError("Missing scores in rerank response")
        
        returned_docs = rerank_data.get("docs", rerank_data.get("documents", []))
        
        # Build results
        results = []
        for i, score in enumerate(scores):
            result = RerankResult(
                index=i,
                relevance_score=float(score)
            )
            
            # Add document text if requested and available
            if return_documents:
                if returned_docs and i < len(returned_docs):
                    result.document = returned_docs[i]
                elif i < len(original_documents):
                    result.document = original_documents[i]
            
            results.append(result)
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return RerankResponse(
            model=model,
            query=query,
            results=results,
            usage={
                "total_tokens": len(query.split()) + sum(len(doc.split()) for doc in original_documents)
            }
        )
