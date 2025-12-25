"""Request envelope model for tracking user inputs through the system."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class RequestEnvelope(BaseModel):
    """
    Envelope containing user request and metadata for tracking through layers.
    
    This model represents a single user request as it flows through the 
    defense workflow, maintaining traceability and security context.
    """
    
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request"
    )
    
    user_input: str = Field(
        ...,
        description="The raw user input text",
        min_length=1,
        max_length=10000
    )
    
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session identifier for tracking multi-turn interactions"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for request tracking"
    )
    
    attack_label: Optional[str] = Field(
        default=None,
        description="Ground truth attack type (for experimental validation only)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request timestamp"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_input": "Summarize this document...",
                "session_id": "session_001",
                "metadata": {"source": "web_ui"},
                "attack_label": None,
                "timestamp": "2024-01-01T00:00:00"
            }
        }
