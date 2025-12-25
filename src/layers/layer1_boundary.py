"""Layer 1: Request Boundary Validation.

This layer performs basic input validation to reject malformed or extreme inputs
before they reach more expensive processing layers.
"""

import logging
import time
import unicodedata
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RequestEnvelope, LayerResult
from config import Config

logger = logging.getLogger(__name__)


class Layer1BoundaryValidation:
    """
    Layer 1: Request Boundary Validation.
    
    Performs fast, inexpensive checks on incoming requests:
    - Input length validation
    - Character encoding validation
    - Basic format checks
    
    This layer acts as a first line of defense, rejecting obviously
    malformed inputs before expensive processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize boundary validation layer."""
        self.config = config or Config.get()
        self.max_length = self.config.layers.max_input_length
        
    def validate(self, request: RequestEnvelope) -> LayerResult:
        """
        Validate request boundaries.
        
        Args:
            request: The incoming request envelope
            
        Returns:
            LayerResult indicating pass/fail with metadata
        """
        start_time = time.time()
        flags = []
        annotations = {}
        passed = True
        confidence = 1.0
        risk_score = 0.0
        
        user_input = request.user_input
        
        # Check 1: Input length
        input_length = len(user_input)
        annotations["input_length"] = input_length
        
        if input_length == 0:
            flags.append("empty_input")
            passed = False
            confidence = 1.0
            risk_score = 0.0
            logger.warning(f"Request {request.request_id}: Empty input")
            
        elif input_length > self.max_length:
            flags.append("length_exceeded")
            passed = False
            confidence = 1.0
            risk_score = 0.8
            annotations["max_length"] = self.max_length
            logger.warning(
                f"Request {request.request_id}: Input length {input_length} "
                f"exceeds maximum {self.max_length}"
            )
        
        # Check 2: Unicode encoding validation
        if passed:
            encoding_issues = self._check_encoding(user_input)
            if encoding_issues:
                flags.extend(encoding_issues)
                passed = False
                confidence = 0.95
                risk_score = 0.6
                annotations["encoding_issues"] = encoding_issues
                logger.warning(
                    f"Request {request.request_id}: Encoding issues detected - {encoding_issues}"
                )
        
        # Check 3: Control character detection
        if passed:
            control_chars = self._check_control_characters(user_input)
            if control_chars:
                flags.append("suspicious_control_chars")
                annotations["control_char_count"] = len(control_chars)
                annotations["control_char_positions"] = control_chars[:10]  # First 10
                # Don't fail on control chars, just flag them
                risk_score = min(0.3, len(control_chars) / 100)
                logger.info(
                    f"Request {request.request_id}: Found {len(control_chars)} control characters"
                )
        
        # Check 4: Null byte detection (common in injection attempts)
        if passed and '\x00' in user_input:
            flags.append("null_byte_detected")
            passed = False
            confidence = 1.0
            risk_score = 0.9
            logger.warning(f"Request {request.request_id}: Null byte detected in input")
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = LayerResult(
            layer_name="Layer1_Boundary",
            passed=passed,
            confidence=confidence,
            flags=flags,
            annotations=annotations,
            risk_score=risk_score,
            latency_ms=latency_ms
        )
        
        logger.info(
            f"Layer1 [{request.request_id}]: passed={passed}, "
            f"flags={flags}, latency={latency_ms:.2f}ms"
        )
        
        return result
    
    def _check_encoding(self, text: str) -> list[str]:
        """
        Check for problematic Unicode encoding issues.
        
        Args:
            text: Input text to check
            
        Returns:
            List of encoding issue descriptions
        """
        issues = []
        
        try:
            # Try encoding to UTF-8 and back
            text.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            issues.append(f"unicode_error_{type(e).__name__}")
        
        # Check for invalid Unicode characters
        try:
            for char in text:
                if unicodedata.category(char) == 'Cn':  # Undefined character
                    issues.append("undefined_unicode_char")
                    break
        except Exception as e:
            issues.append(f"unicode_check_error_{type(e).__name__}")
        
        return issues
    
    def _check_control_characters(self, text: str) -> list[int]:
        """
        Find positions of control characters in text.
        
        Args:
            text: Input text to check
            
        Returns:
            List of positions where control characters were found
        """
        control_positions = []
        
        for i, char in enumerate(text):
            # Check for control characters (excluding common whitespace)
            if unicodedata.category(char).startswith('C') and char not in '\n\r\t':
                control_positions.append(i)
        
        return control_positions
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Layer1BoundaryValidation(max_length={self.max_length})"
