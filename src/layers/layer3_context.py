"""Layer 3: Context Isolation.

This layer enforces separation between system instructions and user input
to prevent user content from being interpreted as system commands.
"""

import logging
import time
from typing import Optional, Dict, List, Literal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RequestEnvelope, LayerResult
from config import Config

logger = logging.getLogger(__name__)


class Layer3ContextIsolation:
    """
    Layer 3: Context Isolation.
    
    Enforces architectural separation between system prompts and user input.
    This is the MOST IMPORTANT layer according to the technical plan, as it
    prevents user input from contaminating system instructions through proper
    role-based message structuring.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize context isolation layer."""
        self.config = config or Config.get()
        self.use_role_separation = getattr(
            self.config.layers, 'use_role_separation', True
        )
        self.use_metadata_tags = getattr(
            self.config.layers, 'use_metadata_tags', True
        )
        
    def isolate(
        self, 
        request: RequestEnvelope,
        system_prompt: str,
        mode: Literal["bad", "good", "metadata", "strict"] = "good"
    ) -> tuple[LayerResult, Dict[str, any]]:
        """
        Create isolated context for LLM interaction.
        
        Args:
            request: The incoming request envelope
            system_prompt: The system prompt to use
            mode: Isolation mode:
                - "bad": Concatenate user input with system prompt (vulnerable)
                - "good": Separate system/user messages with roles
                - "metadata": Add XML-style tags for separation
                - "strict": Hard isolation with no shared context
            
        Returns:
            Tuple of (LayerResult, isolated_context)
        """
        start_time = time.time()
        flags = []
        annotations = {}
        passed = True
        confidence = 1.0
        risk_score = 0.0
        
        annotations["isolation_mode"] = mode
        
        # Create isolated context based on mode
        if mode == "bad":
            # BAD: Simple concatenation (for baseline experiments)
            isolated_context = {
                "type": "concatenated",
                "prompt": f"{system_prompt}\n\nUser: {request.user_input}",
                "origin": "mixed",  # User and system content mixed
                "privilege": "user",  # Treated at user privilege level
            }
            flags.append("no_isolation")
            risk_score = 0.8
            annotations["warning"] = "No context isolation - vulnerable to injection"
            logger.warning(
                f"Request {request.request_id}: Using BAD isolation mode (concatenated)"
            )
            
        elif mode == "good":
            # GOOD: Role-based separation (recommended)
            isolated_context = {
                "type": "role_separated",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                        "origin": "system",
                        "privilege": "system"
                    },
                    {
                        "role": "user",
                        "content": request.user_input,
                        "origin": "user",
                        "privilege": "user"
                    }
                ]
            }
            flags.append("role_separation")
            annotations["separation_method"] = "role_based"
            
        elif mode == "metadata":
            # METADATA: XML-style tags for additional separation
            isolated_context = {
                "type": "metadata_tagged",
                "messages": [
                    {
                        "role": "system",
                        "content": f"<SYSTEM_INSTRUCTION>\n{system_prompt}\n</SYSTEM_INSTRUCTION>",
                        "origin": "system",
                        "privilege": "system"
                    },
                    {
                        "role": "user",
                        "content": f"<USER_INPUT>\n{request.user_input}\n</USER_INPUT>",
                        "origin": "user",
                        "privilege": "user"
                    }
                ]
            }
            flags.append("metadata_tags")
            flags.append("role_separation")
            annotations["separation_method"] = "metadata_tagged"
            
        elif mode == "strict":
            # STRICT: Maximum isolation (for high-security experiments)
            isolated_context = {
                "type": "strict_isolation",
                "system_context": {
                    "content": system_prompt,
                    "hash": hash(system_prompt),
                    "origin": "system",
                    "privilege": "system",
                    "immutable": True
                },
                "user_context": {
                    "content": request.user_input,
                    "origin": "user",
                    "privilege": "user",
                    "sanitized": True
                },
                "allow_mixing": False
            }
            flags.append("strict_isolation")
            annotations["separation_method"] = "strict"
            confidence = 1.0
            
        else:
            raise ValueError(f"Unknown isolation mode: {mode}")
        
        # Check for potential boundary violations in user input
        violations = self._check_boundary_violations(request.user_input)
        if violations:
            flags.extend([f"violation_{v}" for v in violations])
            annotations["boundary_violations"] = violations
            risk_score = max(risk_score, 0.5)
            logger.warning(
                f"Request {request.request_id}: Boundary violation indicators detected - {violations}"
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = LayerResult(
            layer_name="Layer3_Context",
            passed=passed,
            confidence=confidence,
            flags=flags,
            annotations=annotations,
            risk_score=risk_score,
            latency_ms=latency_ms
        )
        
        logger.info(
            f"Layer3 [{request.request_id}]: mode={mode}, "
            f"flags={flags}, latency={latency_ms:.2f}ms"
        )
        
        return result, isolated_context
    
    def _check_boundary_violations(self, user_input: str) -> List[str]:
        """
        Check for indicators of attempts to violate context boundaries.
        
        Args:
            user_input: The user's input text
            
        Returns:
            List of violation types detected
        """
        violations = []
        input_lower = user_input.lower()
        
        # Check for role impersonation attempts
        role_keywords = ["system:", "assistant:", "[system]", "<system>", "role:"]
        if any(kw in input_lower for kw in role_keywords):
            violations.append("role_impersonation")
        
        # Check for context override attempts
        override_keywords = [
            "new instructions:", "system message:", "override:",
            "admin mode:", "debug mode:", "you are now"
        ]
        if any(kw in input_lower for kw in override_keywords):
            violations.append("context_override")
        
        # Check for XML/tag injection attempts
        if any(tag in user_input for tag in ["<system>", "</system>", "<instruction>", "</instruction>"]):
            violations.append("tag_injection")
        
        # Check for attempts to escape context
        escape_sequences = ["```", "---", "===", "###"]
        if any(seq in user_input for seq in escape_sequences):
            violations.append("escape_sequence")
        
        return violations
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Layer3ContextIsolation("
            f"role_separation={self.use_role_separation}, "
            f"metadata_tags={self.use_metadata_tags})"
        )
