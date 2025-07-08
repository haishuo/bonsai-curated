#!/usr/bin/env python3
"""
Statistical Testing Module for Bonsai Neural Pruning
===================================================

PURPOSE: Modular statistical testing framework following the mathematical methodology
PHILOSOPHY: Production-ready statistical rigor with configurable risk profiles

Core Components:
1. Wilcoxon signed-rank testing
2. Multiple comparison correction methods (Bonferroni, Holm-BH, BH, BY)
3. Factory pattern for risk profile selection
4. Temporal direction support (FANIM/BANIM)

Reference: Bonsai Neural Pruning Mathematical Framework & Statistical Methodology.md
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import wilcoxon
from dataclasses import dataclass
from enum import Enum

class RiskProfile(Enum):
    """Risk profile mapping to correction methods per the mathematical framework."""
    MEDICAL_DEVICE = "bonferroni"           # Maximum caution - never prune helpful neurons
    AUTONOMOUS_VEHICLE = "bonferroni"       # Safety critical - false positives dangerous  
    PRODUCTION_ML = "holm_bonferroni"       # Balanced - slight efficiency gain acceptable
    RESEARCH_EXPERIMENT = "benjamini_hochberg"  # Power over caution - explore aggressively
    NEURAL_ARCHITECTURE = "benjamini_yekutieli"  # Accounts for neuron correlations

@dataclass
class StatisticalResult:
    """Results from statistical testing of a single neuron."""
    neuron_id: int
    is_significant: bool
    p_value: float
    adjusted_p_value: float
    median_value: float
    decision: str  # 'PRUNE', 'KEEP'
    sample_size: int

@dataclass
class LayerStatistics:
    """Aggregated statistics for a layer."""
    layer_name: str
    total_neurons: int
    significant_neurons: int
    pruned_neurons: int
    neuron_results: List[StatisticalResult]
    significance_rate: float
    pruning_rate: float

@dataclass
class PruningStatistics:
    """Complete statistical analysis results."""
    alpha: float
    correction_method: str
    temporal_direction: str  # 'FANIM' or 'BANIM'
    layer_stats: Dict[str, LayerStatistics]
    total_neurons: int
    total_significant: int
    total_pruned: int
    overall_significance_rate: float
    overall_pruning_rate: float

# ===== MULTIPLE COMPARISON CORRECTION METHODS =====

class CorrectionMethod(ABC):
    """Abstract base class for multiple comparison correction methods."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    @abstractmethod
    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        """Apply correction to p-values."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return method name."""
        pass

class BonferroniCorrection(CorrectionMethod):
    """Bonferroni correction: Controls Family-Wise Error Rate (FWER)."""
    
    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Bonferroni correction: p_adj = p * n"""
        n = len(p_values)
        return np.minimum(p_values * n, 1.0)
    
    def get_name(self) -> str:
        return "bonferroni"

class HolmBonferroniCorrection(CorrectionMethod):
    """Holm-Bonferroni correction: Step-down FWER control, more powerful than Bonferroni."""
    
    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Holm-Bonferroni step-down correction."""
        n = len(p_values)
        order = np.argsort(p_values)
        adjusted = np.zeros_like(p_values)
        
        for i, idx in enumerate(order):
            factor = n - i
            adjusted[idx] = min(p_values[idx] * factor, 1.0)
            
            # Ensure monotonicity (later p-values can't be smaller)
            if i > 0:
                prev_idx = order[i-1]
                adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
        
        return adjusted
    
    def get_name(self) -> str:
        return "holm_bonferroni"

class BenjaminiHochbergCorrection(CorrectionMethod):
    """Benjamini-Hochberg correction: Controls False Discovery Rate (FDR)."""
    
    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Hochberg step-up correction."""
        n = len(p_values)
        order = np.argsort(p_values)[::-1]  # Descending order
        adjusted = np.zeros_like(p_values)
        
        for i, idx in enumerate(order):
            rank = n - i  # Rank from largest to smallest
            adjusted[idx] = min(p_values[idx] * n / rank, 1.0)
            
            # Ensure monotonicity (earlier p-values can't be larger)
            if i > 0:
                prev_idx = order[i-1]
                adjusted[idx] = min(adjusted[idx], adjusted[prev_idx])
        
        return adjusted
    
    def get_name(self) -> str:
        return "benjamini_hochberg"

class BenjaminiYekuteliCorrection(CorrectionMethod):
    """Benjamini-Yekutieli correction: FDR control under arbitrary dependence."""
    
    def adjust(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Yekutieli correction for dependent tests."""
        n = len(p_values)
        # Calculate correction factor for dependent tests
        c_n = sum(1.0 / i for i in range(1, n + 1))  # Harmonic number
        
        order = np.argsort(p_values)[::-1]  # Descending order
        adjusted = np.zeros_like(p_values)
        
        for i, idx in enumerate(order):
            rank = n - i  # Rank from largest to smallest
            adjusted[idx] = min(p_values[idx] * n * c_n / rank, 1.0)
            
            # Ensure monotonicity
            if i > 0:
                prev_idx = order[i-1]
                adjusted[idx] = min(adjusted[idx], adjusted[prev_idx])
        
        return adjusted
    
    def get_name(self) -> str:
        return "benjamini_yekutieli"

# ===== FACTORY PATTERN FOR CORRECTION METHODS =====

def get_correction_method(method_name: str, alpha: float = 0.05) -> CorrectionMethod:
    """Factory function for multiple comparison correction methods."""
    methods = {
        'bonferroni': BonferroniCorrection(alpha),
        'holm_bonferroni': HolmBonferroniCorrection(alpha), 
        'benjamini_hochberg': BenjaminiHochbergCorrection(alpha),
        'benjamini_yekutieli': BenjaminiYekuteliCorrection(alpha)
    }
    
    if method_name not in methods:
        available = list(methods.keys())
        raise ValueError(f"Unknown correction method: {method_name}. Available: {available}")
    
    return methods[method_name]

def get_correction_method_by_risk_profile(risk_profile: RiskProfile, alpha: float = 0.05) -> CorrectionMethod:
    """Get correction method based on risk profile."""
    return get_correction_method(risk_profile.value, alpha)

# ===== CORE STATISTICAL TESTING =====

class WilcoxonTester:
    """Wilcoxon signed-rank test implementation for neural pruning."""
    
    def __init__(self, alpha: float = 0.05, min_sample_size: int = 10):
        self.alpha = alpha
        self.min_sample_size = min_sample_size
    
    def test_single_neuron(self, scores: np.ndarray, neuron_id: int) -> Tuple[bool, float, float, int]:
        """
        Perform Wilcoxon signed-rank test on scores for one neuron.
        
        Args:
            scores: Impact scores (FANIM or BANIM) for one neuron across batches
            neuron_id: Identifier for the neuron
            
        Returns:
            Tuple of (is_significant, p_value, median_value, effective_sample_size)
        """
        # Remove zeros for valid Wilcoxon test
        non_zero_scores = scores[scores != 0]
        effective_sample_size = len(non_zero_scores)
        
        if effective_sample_size < self.min_sample_size:
            return False, 1.0, np.median(scores), effective_sample_size
        
        try:
            # Wilcoxon signed-rank test against H₀: median = 0
            statistic, p_value = wilcoxon(non_zero_scores, alternative='two-sided')
            median_val = np.median(scores)
            is_significant = p_value < self.alpha
            return is_significant, p_value, median_val, effective_sample_size
        except ValueError:
            # Handle edge cases (e.g., all values same sign)
            return False, 1.0, np.median(scores), effective_sample_size

    def test_layer(self, layer_scores: np.ndarray, layer_name: str, 
                   correction_method: CorrectionMethod) -> LayerStatistics:
        """
        Test all neurons in a layer with multiple comparison correction.
        
        Args:
            layer_scores: Scores array [num_batches, num_neurons]
            layer_name: Name of the layer
            correction_method: Correction method to apply
            
        Returns:
            LayerStatistics object with all results
        """
        num_batches, num_neurons = layer_scores.shape
        
        # Test each neuron individually
        raw_p_values = []
        neuron_results = []
        
        for neuron_idx in range(num_neurons):
            neuron_scores = layer_scores[:, neuron_idx]
            is_sig, p_val, median_val, sample_size = self.test_single_neuron(neuron_scores, neuron_idx)
            
            raw_p_values.append(p_val)
            neuron_results.append({
                'neuron_id': neuron_idx,
                'raw_p_value': p_val,
                'median_value': median_val,
                'sample_size': sample_size,
                'raw_significant': is_sig
            })
        
        # Apply multiple comparison correction
        raw_p_values = np.array(raw_p_values)
        adjusted_p_values = correction_method.adjust(raw_p_values)
        
        # Make final decisions and create results
        final_results = []
        significant_count = 0
        pruned_count = 0
        
        for i, result in enumerate(neuron_results):
            adjusted_p = adjusted_p_values[i]
            is_significant = adjusted_p < self.alpha
            
            # Decision rule: significant AND median > 0 → PRUNE
            should_prune = is_significant and result['median_value'] > 0
            decision = 'PRUNE' if should_prune else 'KEEP'
            
            final_result = StatisticalResult(
                neuron_id=result['neuron_id'],
                is_significant=is_significant,
                p_value=result['raw_p_value'],
                adjusted_p_value=adjusted_p,
                median_value=result['median_value'],
                decision=decision,
                sample_size=result['sample_size']
            )
            
            final_results.append(final_result)
            
            if is_significant:
                significant_count += 1
            if should_prune:
                pruned_count += 1
        
        return LayerStatistics(
            layer_name=layer_name,
            total_neurons=num_neurons,
            significant_neurons=significant_count,
            pruned_neurons=pruned_count,
            neuron_results=final_results,
            significance_rate=significant_count / num_neurons if num_neurons > 0 else 0,
            pruning_rate=pruned_count / num_neurons if num_neurons > 0 else 0
        )

# ===== HIGH-LEVEL STATISTICAL ANALYSIS =====

def analyze_neural_impact_statistics(impact_data: Dict[str, np.ndarray], 
                                    temporal_direction: str = "FANIM",
                                    risk_profile: RiskProfile = RiskProfile.PRODUCTION_ML,
                                    alpha: float = 0.05,
                                    min_sample_size: int = 10) -> PruningStatistics:
    """
    Complete statistical analysis pipeline for neural impact data.
    
    Args:
        impact_data: Dict mapping layer names to impact scores [num_batches, num_neurons]
        temporal_direction: "FANIM" or "BANIM"
        risk_profile: Risk profile determining correction method
        alpha: Significance level
        min_sample_size: Minimum samples required for testing
        
    Returns:
        PruningStatistics with complete analysis
    """
    # Get correction method based on risk profile
    correction_method = get_correction_method_by_risk_profile(risk_profile, alpha)
    
    # Initialize tester
    tester = WilcoxonTester(alpha=alpha, min_sample_size=min_sample_size)
    
    # Analyze each layer
    layer_stats = {}
    total_neurons = 0
    total_significant = 0
    total_pruned = 0
    
    print(f"🧪 STATISTICAL ANALYSIS: {temporal_direction}")
    print(f"📊 Risk Profile: {risk_profile.name} → {correction_method.get_name()}")
    print(f"📊 Significance Level: α = {alpha}")
    print("=" * 60)
    
    for layer_name, layer_scores in impact_data.items():
        layer_stat = tester.test_layer(layer_scores, layer_name, correction_method)
        layer_stats[layer_name] = layer_stat
        
        total_neurons += layer_stat.total_neurons
        total_significant += layer_stat.significant_neurons
        total_pruned += layer_stat.pruned_neurons
        
        print(f"{layer_name}: {layer_stat.total_neurons} neurons")
        print(f"  Significant: {layer_stat.significant_neurons} ({layer_stat.significance_rate:.1%})")
        print(f"  To Prune: {layer_stat.pruned_neurons} ({layer_stat.pruning_rate:.1%})")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Significant: {total_significant} ({total_significant/total_neurons:.1%})")
    print(f"  To prune: {total_pruned} ({total_pruned/total_neurons:.1%})")
    
    return PruningStatistics(
        alpha=alpha,
        correction_method=correction_method.get_name(),
        temporal_direction=temporal_direction,
        layer_stats=layer_stats,
        total_neurons=total_neurons,
        total_significant=total_significant,
        total_pruned=total_pruned,
        overall_significance_rate=total_significant / total_neurons if total_neurons > 0 else 0,
        overall_pruning_rate=total_pruned / total_neurons if total_neurons > 0 else 0
    )

def get_pruning_decisions(statistics: PruningStatistics) -> Dict[str, List[int]]:
    """Extract pruning decisions from statistical analysis."""
    prune_decisions = {}
    
    for layer_name, layer_stat in statistics.layer_stats.items():
        prune_indices = [
            result.neuron_id for result in layer_stat.neuron_results 
            if result.decision == 'PRUNE'
        ]
        prune_decisions[layer_name] = prune_indices
    
    return prune_decisions