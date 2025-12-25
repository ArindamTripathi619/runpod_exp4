"""Statistical analysis utilities for experiment results.

Implements:
- Wilson Score confidence intervals
- McNemar's test for paired proportions
- Effect size calculations (Cohen's h, ARR)
- Weighted ASR for partial success metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
from dataclasses import dataclass
import math


@dataclass
class ConfidenceInterval:
    """Confidence interval for a proportion."""
    lower: float
    upper: float
    confidence_level: float = 0.95
    
    def __repr__(self):
        return f"[{self.lower:.1%}, {self.upper:.1%}]"


@dataclass
class StatisticalResult:
    """Result of statistical test."""
    statistic: float
    p_value: float
    significant: bool
    test_name: str
    
    def __repr__(self):
        sig_str = "✓ significant" if self.significant else "✗ not significant"
        return f"{self.test_name}: p={self.p_value:.4f} ({sig_str})"


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95
) -> ConfidenceInterval:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    This is the recommended method for small sample sizes (better than
    normal approximation). Used for binomial proportions like ASR.
    
    Args:
        successes: Number of successful attacks
        trials: Total number of attacks
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        ConfidenceInterval with lower and upper bounds
        
    References:
        Wilson, E. B. (1927). Probable inference, the law of succession,
        and statistical inference. Journal of the American Statistical
        Association, 22(158), 209-212.
    """
    if trials == 0:
        return ConfidenceInterval(0.0, 0.0, confidence)
    
    p = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    z2 = z * z
    
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) / trials + z2 / (4 * trials * trials))) / denominator
    
    lower = float(max(0.0, center - margin))
    upper = float(min(1.0, center + margin))
    
    return ConfidenceInterval(lower, upper, confidence)


def mcnemar_test(
    contingency_table: List[List[int]],
    alpha: float = 0.05
) -> StatisticalResult:
    """
    McNemar's test for paired nominal data.
    
    Used to test if two defenses have significantly different success rates
    on the same set of attacks.
    
    Args:
        contingency_table: 2x2 table where:
            [[both_fail, defense1_succeeds_only],
             [defense2_succeeds_only, both_succeed]]
        alpha: Significance level (default 0.05)
    
    Returns:
        StatisticalResult with p-value and significance
        
    Example:
        Compare "No Defense" vs "Full Stack":
        - Both fail (attack succeeds in both): a
        - No Defense fails, Full Stack succeeds: b
        - No Defense succeeds, Full Stack fails: c (should be 0)
        - Both succeed (attack fails in both): d
        
        We care about b and c (discordant pairs).
    """
    # Extract discordant pairs
    b = contingency_table[0][1]  # Defense 1 succeeds, defense 2 fails
    c = contingency_table[1][0]  # Defense 1 fails, defense 2 succeeds
    
    # McNemar's test statistic with continuity correction
    if b + c == 0:
        # No discordant pairs - defenses are identical
        return StatisticalResult(0.0, 1.0, False, "McNemar")
    
    statistic = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = float(1 - stats.chi2.cdf(statistic, df=1))
    
    significant = bool(p_value < alpha)
    
    return StatisticalResult(statistic, p_value, significant, "McNemar")


def mcnemar_test_from_results(
    attacks1: List[bool],
    attacks2: List[bool],
    alpha: float = 0.05
) -> StatisticalResult:
    """
    Convenience function to run McNemar test from attack result lists.
    
    Args:
        attacks1: List of True (attack succeeded) / False (blocked) for defense 1
        attacks2: List of True/False for defense 2
        alpha: Significance level
        
    Returns:
        StatisticalResult
    """
    assert len(attacks1) == len(attacks2), "Attack lists must be same length"
    
    # Build contingency table
    both_succeed = sum(1 for a1, a2 in zip(attacks1, attacks2) if a1 and a2)
    only1_succeeds = sum(1 for a1, a2 in zip(attacks1, attacks2) if a1 and not a2)
    only2_succeeds = sum(1 for a1, a2 in zip(attacks1, attacks2) if not a1 and a2)
    both_fail = sum(1 for a1, a2 in zip(attacks1, attacks2) if not a1 and not a2)
    
    contingency = [
        [both_fail, only1_succeeds],
        [only2_succeeds, both_succeed]
    ]
    
    return mcnemar_test(contingency, alpha)


def cohen_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for difference between two proportions.
    
    Args:
        p1: Proportion 1 (e.g., ASR for defense 1)
        p2: Proportion 2 (e.g., ASR for defense 2)
        
    Returns:
        Effect size h where:
        - |h| < 0.2: small effect
        - 0.2 ≤ |h| < 0.5: small effect
        - 0.5 ≤ |h| < 0.8: medium effect
        - |h| ≥ 0.8: large effect
        
    Reference:
        Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
    """
    # Arcsine transformation
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    
    return abs(phi1 - phi2)


def interpret_cohen_h(h: float) -> str:
    """Interpret Cohen's h effect size."""
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    else:
        return "large"


def absolute_risk_reduction(p_baseline: float, p_treatment: float) -> float:
    """
    Calculate Absolute Risk Reduction (ARR).
    
    This is the simple difference in proportions, expressed in percentage points.
    More interpretable than relative risk reduction.
    
    Args:
        p_baseline: Baseline proportion (e.g., ASR with no defense)
        p_treatment: Treatment proportion (e.g., ASR with defense)
        
    Returns:
        ARR in decimal form (e.g., 0.81 for 81 percentage points)
    """
    return p_baseline - p_treatment


def relative_risk_reduction(p_baseline: float, p_treatment: float) -> float:
    """
    Calculate Relative Risk Reduction (RRR).
    
    Args:
        p_baseline: Baseline proportion
        p_treatment: Treatment proportion
        
    Returns:
        RRR as proportion (e.g., 0.50 for 50% reduction)
    """
    if p_baseline == 0:
        return 0.0
    return (p_baseline - p_treatment) / p_baseline


def aggregate_trial_results(
    trial_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate results from multiple trials.
    
    Args:
        trial_results: List of result dicts, each containing:
            - total_attacks: int
            - successful_attacks: int
            - attack_success_rate: float
            - (other metrics)
            
    Returns:
        Aggregated statistics with mean, std, CI, etc.
    """
    n_trials = len(trial_results)
    
    if n_trials == 0:
        return {}
    
    # Extract metrics from nested structure
    # Handle both flat and nested trial results
    def get_metric(trial, key):
        if "metrics" in trial:
            return trial["metrics"].get(key, 0)
        return trial.get(key, 0)
    
    # Extract ASRs
    asrs = [get_metric(r, "attack_success_rate") for r in trial_results]
    
    # Aggregate attack counts across all trials
    total_attacks = sum(get_metric(r, "total_attacks") for r in trial_results)
    
    # For successes, compute from attack_results
    total_successes = sum(
        sum(1 for a in r.get("attack_results", []) if a.get("attack_successful", False))
        for r in trial_results
    )
    
    # Calculate pooled ASR and CI
    pooled_asr = total_successes / total_attacks if total_attacks > 0 else 0.0
    ci = wilson_score_interval(total_successes, total_attacks)
    
    # Calculate trial-level statistics
    mean_asr = np.mean(asrs)
    std_asr = np.std(asrs, ddof=1) if n_trials > 1 else 0.0
    min_asr = np.min(asrs)
    max_asr = np.max(asrs)
    
    return {
        "n_trials": n_trials,
        "mean_asr": mean_asr,
        "std_asr": std_asr,
        "min_asr": min_asr,
        "max_asr": max_asr,
        "pooled_asr": pooled_asr,
        "total_attacks": total_attacks,
        "total_successes": total_successes,
        "confidence_interval_95": {
            "lower": ci.lower,
            "upper": ci.upper
        },
        "trials": trial_results
    }


def calculate_weighted_asr(
    outcomes: List[float]
) -> float:
    """
    Calculate weighted ASR for partial success scoring.
    
    Args:
        outcomes: List of scores where:
            - 0.0 = fully blocked
            - 0.5 = partial leakage
            - 1.0 = full policy violation
            
    Returns:
        Weighted ASR (average of scores)
    """
    if len(outcomes) == 0:
        return 0.0
    
    return float(np.mean(outcomes))


def format_ci_string(ci: ConfidenceInterval) -> str:
    """Format confidence interval for display."""
    return f"95% CI: [{ci.lower:.1%}, {ci.upper:.1%}]"


def format_asr_with_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95
) -> str:
    """
    Format ASR with confidence interval for reporting.
    
    Example output: "19.0% (95% CI: 14.2%-24.8%)"
    """
    if trials == 0:
        return "N/A"
    
    asr = successes / trials
    ci = wilson_score_interval(successes, trials, confidence)
    
    return f"{asr:.1%} (95% CI: {ci.lower:.1%}–{ci.upper:.1%})"


def compare_configurations(
    config1_results: Dict[str, Any],
    config2_results: Dict[str, Any],
    config1_name: str,
    config2_name: str
) -> Dict[str, Any]:
    """
    Statistical comparison between two configurations.
    
    Returns:
        Dictionary with:
        - absolute_reduction: ARR
        - relative_reduction: RRR
        - effect_size_cohen_h: Cohen's h
        - mcnemar_test: Statistical test result (if raw data available)
    """
    asr1 = config1_results.get("pooled_asr") or config1_results.get("attack_success_rate", 0)
    asr2 = config2_results.get("pooled_asr") or config2_results.get("attack_success_rate", 0)
    
    arr = absolute_risk_reduction(asr1, asr2)
    rrr = relative_risk_reduction(asr1, asr2)
    h = cohen_h(asr1, asr2)
    
    comparison = {
        "config1": config1_name,
        "config2": config2_name,
        "config1_asr": asr1,
        "config2_asr": asr2,
        "absolute_risk_reduction": arr,
        "relative_risk_reduction": rrr,
        "effect_size_cohen_h": h,
        "effect_interpretation": interpret_cohen_h(h)
    }
    
    return comparison


if __name__ == "__main__":
    # Example usage and tests
    print("Statistical Analysis Utilities\n")
    
    # Test Wilson CI
    print("1. Wilson Score Confidence Interval")
    print("   8 successes out of 42 attacks:")
    ci = wilson_score_interval(8, 42)
    print(f"   ASR = 19.0%, {format_ci_string(ci)}")
    print()
    
    # Test McNemar
    print("2. McNemar's Test")
    print("   Comparing No Defense (42/42) vs Full Stack (8/42):")
    # All 42 attacks: No Defense succeeds on all, Full Stack blocks 34
    attacks_no_defense = [True] * 42
    attacks_full_stack = [True] * 8 + [False] * 34
    result = mcnemar_test_from_results(attacks_no_defense, attacks_full_stack)
    print(f"   {result}")
    print()
    
    # Test effect sizes
    print("3. Effect Sizes")
    h = cohen_h(1.0, 0.19)
    arr = absolute_risk_reduction(1.0, 0.19)
    rrr = relative_risk_reduction(1.0, 0.19)
    print(f"   No Defense (100%) vs Full Stack (19%):")
    print(f"   - Cohen's h = {h:.2f} ({interpret_cohen_h(h)})")
    print(f"   - ARR = {arr:.1%} (81 percentage points)")
    print(f"   - RRR = {rrr:.1%} (81% relative reduction)")
