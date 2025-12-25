"""Layer 5: Output Validation.

This layer validates LLM outputs to detect system prompt leakage,
policy violations, and semantic inconsistencies.
"""

import logging
import time
import re
from typing import Optional, Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RequestEnvelope, LayerResult
from config import Config

logger = logging.getLogger(__name__)


class Layer5OutputValidation:
    """
    Layer 5: Output Validation.
    
    Validates LLM outputs to prevent leakage of system instructions
    and ensure policy compliance. This is the final defense layer.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize output validation layer."""
        self.config = config or Config.get()
        self.check_system_leakage = getattr(
            self.config.layers, 'check_system_leakage', True
        )
        self.check_policy_violations = getattr(
            self.config.layers, 'check_policy_violations', True
        )
        self.check_semantic_consistency = getattr(
            self.config.layers, 'check_semantic_consistency', False
        )
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Load semantic model if needed (lazy)
        self._semantic_model = None
    
    def _compile_patterns(self):
        """Compile regex patterns for output validation."""
        # System prompt leakage indicators
        self.system_leakage_patterns = [
            re.compile(r'system\s*prompt', re.IGNORECASE),
            re.compile(r'system\s*instructions?', re.IGNORECASE),
            re.compile(r'you\s*are\s*an?\s*(ai|assistant|chatbot)', re.IGNORECASE),
            re.compile(r'your\s*role\s*is', re.IGNORECASE),
            re.compile(r'<system>', re.IGNORECASE),
            re.compile(r'<instruction>', re.IGNORECASE),
            re.compile(r'\[system\]', re.IGNORECASE),
            re.compile(r'###\s*system', re.IGNORECASE),
        ]
        
        # Policy violation keywords
        self.policy_keywords = [
            "hack", "exploit", "bypass", "jailbreak", "ignore",
            "override", "admin", "root", "privileged", "unauthorized",
            "steal", "leak", "expose", "credential", "password"
        ]
        
        # Encoding that might indicate obfuscation
        self.encoding_patterns = [
            re.compile(r'\\x[0-9a-f]{2}', re.IGNORECASE),  # Hex encoding
            re.compile(r'\\u[0-9a-f]{4}', re.IGNORECASE),  # Unicode
            re.compile(r'base64:', re.IGNORECASE),
            re.compile(r'%[0-9a-f]{2}', re.IGNORECASE),    # URL encoding
        ]
    
    def validate(
        self, 
        request: RequestEnvelope,
        llm_output: str,
        system_prompt: Optional[str] = None
    ) -> LayerResult:
        """
        Validate LLM output for safety and policy compliance.
        
        Args:
            request: The incoming request envelope
            llm_output: The LLM's generated response
            system_prompt: Original system prompt to check for leakage
            
        Returns:
            LayerResult with validation outcome
        """
        start_time = time.time()
        flags = []
        annotations = {}
        passed = True
        confidence = 1.0
        risk_score = 0.0
        
        # Check for system prompt leakage
        if self.check_system_leakage:
            leakage_detected, leakage_details = self._check_leakage(
                llm_output, system_prompt
            )
            
            if leakage_detected:
                passed = False
                flags.append("system_leakage")
                annotations["leakage_details"] = leakage_details
                risk_score = max(risk_score, 0.9)
                logger.warning(
                    f"Request {request.request_id}: System prompt leakage detected - {leakage_details}"
                )
        
        # Check for policy violations
        if self.check_policy_violations:
            policy_violations, violation_details = self._check_policy_violations(
                llm_output
            )
            
            if policy_violations:
                flags.append("policy_violation")
                annotations["policy_violations"] = violation_details
                risk_score = max(risk_score, 0.7)
                logger.warning(
                    f"Request {request.request_id}: Policy violations - {violation_details}"
                )
        
        # Check for suspicious encoding/obfuscation
        encoding_detected, encoding_types = self._check_encoding(llm_output)
        if encoding_detected:
            flags.append("suspicious_encoding")
            annotations["encoding_types"] = encoding_types
            risk_score = max(risk_score, 0.5)
            logger.info(
                f"Request {request.request_id}: Suspicious encoding detected - {encoding_types}"
            )
        
        # Check output length
        output_length = len(llm_output)
        annotations["output_length"] = output_length
        
        if output_length > 5000:
            flags.append("excessive_length")
            annotations["warning"] = "Output exceeds normal length"
        
        # Check semantic consistency if enabled
        if self.check_semantic_consistency:
            consistent, consistency_score = self._check_semantic_consistency(
                request.user_input, llm_output
            )
            annotations["semantic_consistency"] = consistency_score
            
            if not consistent:
                flags.append("semantic_inconsistency")
                risk_score = max(risk_score, 0.4)
                logger.info(
                    f"Request {request.request_id}: Semantic inconsistency detected "
                    f"(score={consistency_score:.2f})"
                )
        
        # Set confidence based on checks performed
        checks_performed = sum([
            self.check_system_leakage,
            self.check_policy_violations,
            self.check_semantic_consistency
        ])
        confidence = 1.0 if checks_performed > 0 else 0.5
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = LayerResult(
            layer_name="Layer5_Output",
            passed=passed,
            confidence=confidence,
            flags=flags,
            annotations=annotations,
            risk_score=risk_score,
            latency_ms=latency_ms
        )
        
        logger.info(
            f"Layer5 [{request.request_id}]: passed={passed}, "
            f"risk={risk_score:.2f}, flags={flags}, latency={latency_ms:.2f}ms"
        )
        
        return result
    
    def _check_leakage(
        self, 
        output: str,
        system_prompt: Optional[str]
    ) -> tuple[bool, List[str]]:
        """
        Check if output leaks system prompt information.
        
        Args:
            output: LLM output to check
            system_prompt: Original system prompt
            
        Returns:
            Tuple of (leakage_detected, details)
        """
        details = []
        
        # Check regex patterns
        for pattern in self.system_leakage_patterns:
            if pattern.search(output):
                details.append(f"Pattern match: {pattern.pattern}")
        
        # Check for direct system prompt content if provided
        if system_prompt:
            # Split system prompt into meaningful chunks (5+ words)
            words = system_prompt.split()
            for i in range(len(words) - 4):
                chunk = " ".join(words[i:i+5])
                if len(chunk) > 20 and chunk.lower() in output.lower():
                    details.append(f"Direct leakage: '{chunk}'")
                    break  # One detection is enough
        
        return len(details) > 0, details
    
    def _check_policy_violations(self, output: str) -> tuple[bool, List[str]]:
        """
        Check for policy violation keywords in output.
        
        Args:
            output: LLM output to check
            
        Returns:
            Tuple of (violations_detected, details)
        """
        output_lower = output.lower()
        violations = []
        
        for keyword in self.policy_keywords:
            if keyword in output_lower:
                violations.append(keyword)
        
        # More than 2 policy keywords is suspicious
        return len(violations) >= 3, violations
    
    def _check_encoding(self, output: str) -> tuple[bool, List[str]]:
        """
        Check for suspicious encoding that might indicate obfuscation.
        
        Args:
            output: LLM output to check
            
        Returns:
            Tuple of (encoding_detected, encoding_types)
        """
        encoding_types = []
        
        for pattern in self.encoding_patterns:
            matches = pattern.findall(output)
            if len(matches) >= 3:  # More than 2 matches indicates intentional use
                encoding_types.append(pattern.pattern)
        
        return len(encoding_types) > 0, encoding_types
    
    def _check_semantic_consistency(
        self, 
        user_input: str,
        llm_output: str
    ) -> tuple[bool, float]:
        """
        Check if output is semantically consistent with input.
        
        This uses embedding similarity to detect if the LLM is responding
        to something other than the user's actual query (possible injection).
        
        Args:
            user_input: Original user input
            llm_output: LLM's response
            
        Returns:
            Tuple of (is_consistent, consistency_score)
        """
        try:
            # Lazy load semantic model
            if self._semantic_model is None:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence-transformers model for semantic validation")
            
            # Compute embeddings
            input_embedding = self._semantic_model.encode(user_input)
            output_embedding = self._semantic_model.encode(llm_output)
            
            # Compute cosine similarity
            import numpy as np
            similarity = np.dot(input_embedding, output_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(output_embedding)
            )
            
            # Threshold: outputs should have some semantic relationship to inputs
            # But not too high (which might indicate parroting)
            is_consistent = 0.2 <= similarity <= 0.85
            
            return is_consistent, float(similarity)
            
        except Exception as e:
            logger.error(f"Semantic consistency check failed: {e}")
            # Fail open - assume consistent if check fails
            return True, 0.5
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Layer5OutputValidation("
            f"system_leakage={self.check_system_leakage}, "
            f"policy={self.check_policy_violations}, "
            f"semantic={self.check_semantic_consistency})"
        )
