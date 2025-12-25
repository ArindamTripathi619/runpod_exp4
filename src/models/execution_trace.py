"""Execution trace model for complete request processing history."""

from typing import Dict, Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime

from .layer_result import LayerResult


class ExecutionTrace(BaseModel):
    """
    Complete execution trace for a request through all defense layers.
    
    This is the primary artifact for experimental analysis, containing
    all decisions, timing, and outcomes for reproducibility.
    """
    
    request_id: str = Field(
        ...,
        description="Reference to the original request"
    )
    
    session_id: str = Field(
        default="default",
        description="Session this request belongs to"
    )
    
    layer_results: List[LayerResult] = Field(
        default_factory=list,
        description="Results from each layer in execution order"
    )
    
    final_output: Optional[str] = Field(
        default=None,
        description="Final LLM output (if request passed all layers)"
    )
    
    violation_detected: bool = Field(
        default=False,
        description="Whether any layer detected a violation"
    )
    
    blocked_at_layer: Optional[str] = Field(
        default=None,
        description="Name of the layer that blocked the request (if any)"
    )
    
    total_latency_ms: Optional[float] = Field(
        default=None,
        description="Total end-to-end processing time in milliseconds"
    )
    
    attack_label: Optional[str] = Field(
        default=None,
        description="Ground truth attack type (for experiments)"
    )
    
    attack_successful: Optional[bool] = Field(
        default=None,
        description="Whether the attack succeeded (experimental validation)"
    )
    
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration used for this request"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Execution timestamp"
    )
    
    experiment_id: Optional[str] = Field(
        default=None,
        description="Experiment batch identifier"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "session_id": "session_001",
                "layer_results": {
                    "Layer1_Boundary": {
                        "layer_name": "Layer1_Boundary",
                        "passed": True,
                        "confidence": 1.0,
                        "flags": [],
                        "annotations": {},
                        "risk_score": 0.0,
                        "latency_ms": 2.1
                    }
                },
                "final_output": "Here is the summary...",
                "violation_detected": False,
                "blocked_at_layer": None,
                "total_latency_ms": 523.4,
                "attack_label": "direct_injection",
                "attack_successful": False,
                "configuration": {
                    "layer1_enabled": True,
                    "layer2_enabled": True,
                    "layer3_enabled": True,
                    "layer4_enabled": True,
                    "layer5_enabled": True
                },
                "experiment_id": "exp_001"
            }
        }
