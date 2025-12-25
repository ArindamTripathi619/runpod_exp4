"""Defense pipeline orchestrator.

This module orchestrates the execution of all 5 defense layers
and creates complete execution traces for analysis.
"""

import logging
import time
from typing import Optional, Literal, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import RequestEnvelope, LayerResult, ExecutionTrace
from config import Config
from layers import (
    Layer1BoundaryValidation,
    Layer2SemanticAnalysis,
    Layer3ContextIsolation,
    Layer4LLMInteraction,
    Layer5OutputValidation,
)

logger = logging.getLogger(__name__)


class DefensePipeline:
    """
    Orchestrates the 5-layer defense architecture.
    
    The pipeline can be configured to enable/disable specific layers
    for different experimental conditions.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize defense pipeline with configuration."""
        self.config = config or Config.get()
        
        # Initialize all layers
        self.layer1 = Layer1BoundaryValidation(config)
        self.layer2 = Layer2SemanticAnalysis(config)
        self.layer3 = Layer3ContextIsolation(config)
        self.layer4 = Layer4LLMInteraction(config)
        self.layer5 = Layer5OutputValidation(config)
        
        # Get layer enable/disable configuration
        self.layers_enabled = {
            "layer1": self.config.layers.layer1_enabled,
            "layer2": self.config.layers.layer2_enabled,
            "layer3": self.config.layers.layer3_enabled,
            "layer4": self.config.layers.layer4_enabled,
            "layer5": self.config.layers.layer5_enabled,
        }
        
        # Get system prompt
        self.system_prompt = self.config.system_prompt.content
        
        logger.info(
            f"DefensePipeline initialized: layers_enabled={self.layers_enabled}"
        )
    
    def process(
        self, 
        request: RequestEnvelope,
        isolation_mode: Literal["bad", "good", "metadata", "strict"] = "good",
        experiment_id: Optional[str] = None
    ) -> ExecutionTrace:
        """
        Process a request through the defense pipeline.
        
        Args:
            request: The incoming request to process
            isolation_mode: Context isolation mode for Layer 3
            experiment_id: Optional experiment identifier
            
        Returns:
            ExecutionTrace with complete execution history
        """
        start_time = time.time()
        layer_results: List[LayerResult] = []
        final_output = ""
        violation_detected = False
        blocked_at_layer = None
        
        logger.info(f"Processing request {request.request_id} with mode={isolation_mode}")
        
        # Layer 1: Boundary Validation
        if self.layers_enabled["layer1"]:
            try:
                layer1_result = self.layer1.validate(request)
                layer_results.append(layer1_result)
                
                if not layer1_result.passed:
                    violation_detected = True
                    blocked_at_layer = "Layer1_Boundary"
                    final_output = "BLOCKED: Input validation failed"
                    logger.warning(
                        f"Request {request.request_id} blocked at Layer 1: "
                        f"flags={layer1_result.flags}"
                    )
                    return self._create_trace(
                        request, layer_results, final_output, 
                        violation_detected, blocked_at_layer, 
                        start_time, isolation_mode, experiment_id
                    )
            except Exception as e:
                logger.error(f"Layer 1 error: {e}")
                layer_results.append(self._error_result("Layer1_Boundary", str(e)))
        
        # Layer 2: Semantic Analysis
        if self.layers_enabled["layer2"]:
            try:
                layer2_result = self.layer2.analyze(request)
                layer_results.append(layer2_result)
                
                if not layer2_result.passed:
                    violation_detected = True
                    blocked_at_layer = "Layer2_Semantic"
                    final_output = "BLOCKED: Semantic attack detected"
                    logger.warning(
                        f"Request {request.request_id} blocked at Layer 2: "
                        f"risk_score={layer2_result.risk_score:.2f}"
                    )
                    return self._create_trace(
                        request, layer_results, final_output, 
                        violation_detected, blocked_at_layer, 
                        start_time, isolation_mode, experiment_id
                    )
            except Exception as e:
                logger.error(f"Layer 2 error: {e}")
                layer_results.append(self._error_result("Layer2_Semantic", str(e)))
        
        # Layer 3: Context Isolation
        isolated_context = None
        if self.layers_enabled["layer3"]:
            try:
                layer3_result, isolated_context = self.layer3.isolate(
                    request, self.system_prompt, mode=isolation_mode
                )
                layer_results.append(layer3_result)
                
                # Layer 3 warnings don't block (just flag risks)
                if layer3_result.risk_score > 0.8:
                    violation_detected = True
                    logger.warning(
                        f"Request {request.request_id}: High-risk context detected "
                        f"(score={layer3_result.risk_score:.2f})"
                    )
            except Exception as e:
                logger.error(f"Layer 3 error: {e}")
                layer_results.append(self._error_result("Layer3_Context", str(e)))
                # Create fallback context
                isolated_context = {
                    "type": "concatenated",
                    "prompt": f"{self.system_prompt}\n\nUser: {request.user_input}"
                }
        else:
            # No isolation - create concatenated context
            isolated_context = {
                "type": "concatenated",
                "prompt": f"{self.system_prompt}\n\nUser: {request.user_input}"
            }
        
        # Layer 4: LLM Interaction
        llm_response = ""
        if self.layers_enabled["layer4"]:
            try:
                layer4_result, llm_response = self.layer4.interact(
                    request, isolated_context
                )
                layer_results.append(layer4_result)
                
                if not layer4_result.passed:
                    violation_detected = True
                    blocked_at_layer = "Layer4_LLM"
                    final_output = llm_response  # Contains block message
                    logger.warning(
                        f"Request {request.request_id} blocked at Layer 4: "
                        f"flags={layer4_result.flags}"
                    )
                    return self._create_trace(
                        request, layer_results, final_output, 
                        violation_detected, blocked_at_layer, 
                        start_time, isolation_mode, experiment_id
                    )
            except Exception as e:
                logger.error(f"Layer 4 error: {e}")
                layer_results.append(self._error_result("Layer4_LLM", str(e)))
                llm_response = f"ERROR: LLM interaction failed - {e}"
        else:
            llm_response = "LLM layer disabled"
        
        # Layer 5: Output Validation
        if self.layers_enabled["layer5"]:
            try:
                layer5_result = self.layer5.validate(
                    request, llm_response, self.system_prompt
                )
                layer_results.append(layer5_result)
                
                if not layer5_result.passed:
                    violation_detected = True
                    blocked_at_layer = "Layer5_Output"
                    final_output = "BLOCKED: Output validation failed"
                    logger.warning(
                        f"Request {request.request_id} blocked at Layer 5: "
                        f"flags={layer5_result.flags}"
                    )
                    return self._create_trace(
                        request, layer_results, final_output, 
                        violation_detected, blocked_at_layer, 
                        start_time, isolation_mode, experiment_id
                    )
            except Exception as e:
                logger.error(f"Layer 5 error: {e}")
                layer_results.append(self._error_result("Layer5_Output", str(e)))
        
        # All layers passed
        final_output = llm_response
        
        logger.info(
            f"Request {request.request_id} completed successfully "
            f"({len(layer_results)} layers)"
        )
        
        return self._create_trace(
            request, layer_results, final_output, 
            violation_detected, blocked_at_layer, 
            start_time, isolation_mode, experiment_id
        )
    
    def _create_trace(
        self,
        request: RequestEnvelope,
        layer_results: List[LayerResult],
        final_output: str,
        violation_detected: bool,
        blocked_at_layer: Optional[str],
        start_time: float,
        isolation_mode: str,
        experiment_id: Optional[str]
    ) -> ExecutionTrace:
        """Create execution trace from processing results."""
        total_latency_ms = (time.time() - start_time) * 1000
        
        # Determine if attack was successful
        # Success = (attack labeled AND not blocked) OR (no violation detected for attack)
        attack_successful = False
        if request.attack_label:
            attack_successful = not violation_detected
        
        trace = ExecutionTrace(
            request_id=request.request_id,
            layer_results=layer_results,
            final_output=final_output,
            violation_detected=violation_detected,
            blocked_at_layer=blocked_at_layer,
            total_latency_ms=total_latency_ms,
            attack_label=request.attack_label,
            attack_successful=attack_successful,
            configuration={
                "layers_enabled": self.layers_enabled,
                "isolation_mode": isolation_mode,
                "user_input": request.user_input,  # Store for database
            },
            experiment_id=experiment_id
        )
        
        return trace
    
    def _error_result(self, layer_name: str, error_msg: str) -> LayerResult:
        """Create error result for layer failure."""
        return LayerResult(
            layer_name=layer_name,
            passed=False,
            confidence=0.0,
            flags=["layer_error"],
            annotations={"error": error_msg},
            risk_score=1.0,
            latency_ms=0.0
        )
    
    def configure_layers(self, **layer_flags):
        """
        Dynamically configure which layers are enabled.
        
        Args:
            **layer_flags: Keyword arguments like enable_layer1=True, enable_layer2=False
        """
        for key, value in layer_flags.items():
            if key.startswith("enable_layer"):
                layer_num = key.replace("enable_layer", "")
                layer_key = f"layer{layer_num}"
                if layer_key in self.layers_enabled:
                    self.layers_enabled[layer_key] = value
                    logger.info(f"Layer {layer_num} {'enabled' if value else 'disabled'}")
    
    def get_enabled_layers(self) -> List[str]:
        """Get list of currently enabled layers."""
        return [
            layer for layer, enabled in self.layers_enabled.items() 
            if enabled
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        enabled = ", ".join(self.get_enabled_layers())
        return f"DefensePipeline(enabled=[{enabled}])"
