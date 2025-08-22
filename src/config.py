"""
Configuration management for RunPod embedding service
Handles migration from OpenAI-compatible to native /runsync endpoints
"""
import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RunPodConfig:
    """RunPod API configuration with native endpoint support"""
    api_key: str
    endpoint_id: str
    base_url: str = "https://api.runpod.ai/v2"
    timeout: int = 300
    max_retries: int = 3
    use_native_endpoints: bool = True  # Flag for endpoint type
    
    @classmethod
    def from_env(cls) -> 'RunPodConfig':
        """Create config from environment variables"""
        api_key = os.getenv("RUNPOD_API_KEY", "")
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")
        
        if not api_key:
            raise ValueError("RUNPOD_API_KEY environment variable is required")
        if not endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID environment variable is required")
            
        return cls(
            api_key=api_key,
            endpoint_id=endpoint_id,
            base_url=os.getenv("RUNPOD_BASE_URL", "https://api.runpod.ai/v2"),
            timeout=int(os.getenv("RUNPOD_TIMEOUT", "300")),
            max_retries=int(os.getenv("RUNPOD_MAX_RETRIES", "3")),
            use_native_endpoints=os.getenv("USE_NATIVE_ENDPOINTS", "true").lower() == "true"
        )
    
    @property
    def embedding_endpoint(self) -> str:
        """Get the appropriate embedding endpoint URL"""
        if self.use_native_endpoints:
            return f"{self.base_url}/{self.endpoint_id}/runsync"
        else:
            return f"{self.base_url}/{self.endpoint_id}/openai/v1/embeddings"
    
    @property
    def rerank_endpoint(self) -> str:
        """Get the appropriate reranking endpoint URL"""
        if self.use_native_endpoints:
            return f"{self.base_url}/{self.endpoint_id}/runsync"
        else:
            return f"{self.base_url}/{self.endpoint_id}/openai/v1/chat/completions"

@dataclass
class EmbeddingConfig:
    """Embedding service configuration"""
    model_names: List[str]
    batch_sizes: List[int]
    max_tokens: int = 8192
    backend: str = "torch"
    dtypes: List[str] = None
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        model_names_str = os.getenv("MODEL_NAMES", "")
        if not model_names_str:
            raise ValueError("MODEL_NAMES environment variable is required")
            
        model_names = [name.strip() for name in model_names_str.split(";") if name.strip()]
        batch_sizes = [int(x.strip()) for x in os.getenv("BATCH_SIZES", "32").split(";") if x.strip()]
        dtypes = [x.strip() for x in os.getenv("DTYPES", "auto").split(";") if x.strip()] if os.getenv("DTYPES") else ["auto"]
        
        return cls(
            model_names=model_names,
            batch_sizes=batch_sizes,
            max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
            backend=os.getenv("BACKEND", "torch"),
            dtypes=dtypes
        )

class Config:
    """Main application configuration"""
    def __init__(self):
        self.runpod = RunPodConfig.from_env()
        self.embedding = EmbeddingConfig.from_env()
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def validate(self) -> None:
        """Validate configuration"""
        if not self.runpod.api_key:
            raise ValueError("RUNPOD_API_KEY is required")
        if not self.runpod.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID is required")
        if not self.embedding.model_names or not self.embedding.model_names[0]:
            raise ValueError("MODEL_NAMES is required")

# Global config instance
config = Config()
config.validate()
