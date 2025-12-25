"""Layer result model for tracking defense layer decisions."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class LayerResult(BaseModel):
    """
    Result from a single defense layer execution.
    
    Captures the decision, confidence, and reasoning from each layer,
    enabling traceability and debugging of defense decisions.
    """
    
    layer_name: str = Field(
        ...,
        description="Name of the layer that produced this result"
    )
    
    passed: bool = Field(
        ...,
        description="Whether the request passed this layer's checks"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this layer's decision (0.0 to 1.0)"
    )
    
    flags: List[str] = Field(
        default_factory=list,
        description="List of security flags raised by this layer"
    )
    
    annotations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual information from this layer"
    )
    
    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional risk score (0.0 = safe, 1.0 = high risk)"
    )
    
    latency_ms: Optional[float] = Field(
        default=None,
        description="Processing time for this layer in milliseconds"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "layer_name": "Layer2_SemanticAnalysis",
                "passed": True,
                "confidence": 0.85,
                "flags": [],
                "annotations": {
                    "semantic_similarity": 0.15,
                    "detected_patterns": []
                },
                "risk_score": 0.15,
                "latency_ms": 45.2
            }
        }
