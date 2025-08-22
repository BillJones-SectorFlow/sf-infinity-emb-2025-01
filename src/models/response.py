"""
Response models for RunPod embedding service
Compatible with OpenAI format while supporting RunPod-specific features
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Usage:
    """Token usage information"""
    prompt_tokens: int
    total_tokens: int
    completion_tokens: Optional[int] = None

@dataclass
class EmbeddingData:
    """Individual embedding data"""
    object: str = "embedding"
    embedding: List[float] = None
    index: int = 0
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = []

@dataclass
class EmbeddingResponse:
    """OpenAI-compatible embedding response"""
    object: str = "list"
    model: str = ""
    data: List[EmbeddingData] = None
    usage: Usage = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = []
        if self.usage is None:
            self.usage = Usage(prompt_tokens=0, total_tokens=0)
        
        # Convert dict data to EmbeddingData objects if needed
        if self.data and isinstance(self.data[0], dict):
            self.data = [EmbeddingData(**item) for item in self.data]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class RerankResult:
    """Individual rerank result"""
    index: int
    relevance_score: float
    document: Optional[str] = None

@dataclass 
class RerankResponse:
    """Reranking response"""
    model: str
    query: str
    results: List[RerankResult]
    usage: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
        
        # Convert dict results to RerankResult objects if needed
        if self.results and isinstance(self.results[0], dict):
            self.results = [RerankResult(**item) for item in self.results]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class ErrorResponse:
    """Error response format"""
    error: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
