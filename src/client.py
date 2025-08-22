"""
RunPod API client with native /runsync endpoint support
Handles migration from OpenAI-compatible endpoints to native RunPod endpoints
"""
import time
import logging
import json
from typing import Dict, Any, Optional, Union, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RunPodConfig

logger = logging.getLogger(__name__)

class RunPodAPIError(Exception):
    """Custom exception for RunPod API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}

class RunPodClient:
    """RunPod API client with support for both OpenAI-compatible and native endpoints"""
    
    def __init__(self, config: RunPodConfig):
        self.config = config
        self.session = self._create_session()
        self.logger = logging.getLogger(__name__)
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy and proper headers"""
        session = requests.Session()
        
        # Configure retry strategy for resilient connections
        retry_strategy = Retry(
            total=self.config.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["POST", "GET"],
            backoff_factor=1,
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "sf-infinity-emb-client/1.0.0"
        })
        
        return session
    
    def _make_request(self, url: str, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        timeout = timeout or self.config.timeout
        
        try:
            self.logger.debug(f"Making request to {url} with payload: {json.dumps(payload, indent=2)}")
            
            response = self.session.post(url, json=payload, timeout=timeout)
            
            # Log response for debugging
            self.logger.debug(f"Response status: {response.status_code}")
            self.logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Parse response
            try:
                result = response.json()
                self.logger.debug(f"Response body: {json.dumps(result, indent=2)}")
            except ValueError as e:
                raise RunPodAPIError(
                    f"Invalid JSON response: {response.text[:500]}", 
                    response.status_code, 
                    {"raw_response": response.text}
                )
            
            # Handle HTTP errors
            if response.status_code >= 400:
                error_msg = result.get("error", {}).get("message") if isinstance(result.get("error"), dict) else result.get("error", f"HTTP {response.status_code}")
                raise RunPodAPIError(error_msg, response.status_code, result)
            
            # Handle RunPod-specific errors
            if "error" in result:
                error_msg = result["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                raise RunPodAPIError(str(error_msg), response.status_code, result)
                
            return result
            
        except requests.exceptions.Timeout:
            raise RunPodAPIError(f"Request timeout after {timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise RunPodAPIError(f"Connection error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise RunPodAPIError(f"Request failed: {str(e)}")
    
    def run_sync(self, input_data: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute synchronous request to RunPod /runsync endpoint
        Automatically wraps input data in required format
        """
        url = f"{self.config.base_url}/{self.config.endpoint_id}/runsync"
        
        # Wrap input data for native endpoint
        payload = {"input": input_data}
        
        result = self._make_request(url, payload, timeout)
        
        # Extract and validate output from RunPod response
        if "output" not in result:
            raise RunPodAPIError("Missing 'output' in RunPod response", response_data=result)
        
        # Log execution metrics if available
        if "executionTime" in result:
            self.logger.info(f"RunPod execution time: {result['executionTime']}ms")
        if "delayTime" in result:
            self.logger.info(f"RunPod delay time: {result['delayTime']}ms")
            
        return result
    
    def run_openai_compatible(self, payload: Dict[str, Any], endpoint_type: str = "embeddings") -> Dict[str, Any]:
        """
        Execute request to OpenAI-compatible endpoint (for backward compatibility)
        """
        if endpoint_type == "embeddings":
            url = f"{self.config.base_url}/{self.config.endpoint_id}/openai/v1/embeddings"
        elif endpoint_type == "chat":
            url = f"{self.config.base_url}/{self.config.endpoint_id}/openai/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported endpoint type: {endpoint_type}")
        
        return self._make_request(url, payload)
    
    def health_check(self) -> bool:
        """Check if the RunPod endpoint is healthy"""
        try:
            # Simple health check with minimal input
            test_input = {"test": "health_check"}
            result = self.run_sync(test_input, timeout=30)
            
            # If we get any response without error, consider it healthy
            return "output" in result or "error" not in result
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get information about the RunPod endpoint"""
        url = f"{self.config.base_url}/{self.config.endpoint_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            return response.json()
        except Exception as e:
            self.logger.warning(f"Could not get endpoint info: {e}")
            return {}
