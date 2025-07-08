#!/usr/bin/env python3
"""
Wine Quality Enhanced Iterative Wilcoxon Pruning - EXACT COPY FROM BONSAI-ARCHIVE
=================================================================================

PURPOSE: Production-ready iterative Wilcoxon significance testing for neural pruning
PHILOSOPHY: Statistical rigor with adaptive strategies and robust error handling
METHOD: Progressive Wilcoxon testing with adaptive alpha and rollback protection

Features:
- Adaptive alpha scheduling (liberal → conservative → strict)
- Checkpoint management and rollback system  
- Fresh FANIM computation each iteration
- Comprehensive convergence detection
- Production-grade error handling

EXACT COPY: No modifications from bonsai-archive except file paths for forge
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import wilcoxon

# FORGE-SPECIFIC IMPORTS (only allowed changes)
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shared.forge_config import get_bonsai_config

# Import base functions - USING ARCHIVE VERSION
from wq_base_prune import (
    compute_fanim_from_model,
    load_wine_quality_data,
    load_trained_model,
    apply_structural_pruning,
    fine_tune_model,
    evaluate_model_performance,
    count_parameters,
    get_model_sparsity,
    measure_inference_time
)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FORGE-SPECIFIC PATHS (only allowed changes)
forge_config = get_bonsai_config()
RESULTS_FILE = forge_config.results_dir / 'wq_wilcoxon_iterative_enhanced_results.json'
DETAILED_LOG_FILE = forge_config.results_dir / 'wq_wilcoxon_iterative_detailed_log.txt'

# Configuration class from archive
class WilcoxonConfig:
    """Configuration for Enhanced Iterative Wilcoxon pruning."""
    
    def __init__(self):
        # Core iteration parameters
        self.max_iterations = 15
        self.epochs_per_iteration = 5
        
        # Alpha scheduling (adaptive significance levels)
        self.alpha_schedule = {
            'liberal_phase': (1, 10, 0.1),     # Iterations 1-10: α = 0.1
            'conservative_phase': (11, 15, 0.05), # Iterations 11-15: α = 0.05
            'strict_phase': (16, 20, 0.01)     # Iterations 16-20: α = 0.01
        }
        
        # Rollback and safety thresholds
        self.rollback_threshold = 1.5  # % accuracy drop that triggers rollback
        self.min_accuracy_threshold = 2.0  # % drop that stops experiment
        self.patience = 3  # Iterations with no pruning before stopping
        self.max_rollbacks = 3  # Maximum rollbacks allowed
        
        # Statistical testing
        self.min_sample_size = 10  # Minimum FANIM samples for valid Wilcoxon test

class AdaptiveAlphaScheduler:
    """Adaptive alpha scheduling for progressive statistical rigor."""
    
    def __init__(self, config: WilcoxonConfig):
        self.config = config
    
    def get_alpha(self, iteration: int) -> float:
        """Get alpha value for current iteration."""
        for phase_name, (start, end, alpha) in self.config.alpha_schedule.items():
            if start <= iteration <= end:
                return alpha
        
        # Default to most conservative if beyond schedule
        return 0.01
    
    def get_phase_description(self, iteration: int) -> str:
        """Get human-readable phase description."""
        for phase_name, (start, end, alpha) in self.config.alpha_schedule.items():
            if start <= iteration <= end:
                return f"{phase_name.replace('_', ' ').title()}"
        return "Ultra-Conservative Phase"

class CheckpointManager:
    """Manage model checkpoints and rollback functionality."""
    
    def __init__(self, baseline_accuracy: float, config: WilcoxonConfig):
        self.baseline_accuracy = baseline_accuracy
        self.config = config
        self.checkpoints = {}  # iteration -> (model, accuracy)
        self.rollback_count = 0
    
    def add_checkpoint(self, model: nn.Module, accuracy: float, iteration: int):
        """Add a checkpoint for potential rollback."""
        self.checkpoints[iteration] = (copy.deepcopy(model), accuracy)
        print(f"💾 Checkpoint saved: Iteration {iteration}, Accuracy {accuracy:.2f}%")
    
    def should_rollback(self, current_accuracy: float, baseline_accuracy: float) -> bool:
        """Check if current accuracy warrants a rollback."""
        accuracy_drop = baseline_accuracy - current_accuracy
        return accuracy_drop > self.config.rollback_threshold
    
    def execute_rollback(self) -> Tuple[nn.Module, Dict]:
        """Execute rollback to best previous checkpoint."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available for rollback")
        
        if self.rollback_count >= self.config.max_rollbacks:
            raise ValueError(f"Maximum rollbacks ({self.config.max_rollbacks}) exceeded")
        
        # Find best checkpoint
        best_iteration = max(self.checkpoints.keys(), 
                           key=lambda k: self.checkpoints[k][1])
        best_model, best_accuracy = self.checkpoints[best_iteration]
        
        self.rollback_count += 1
        
        rollback_info = {
            'rollback_to_iteration': best_iteration,
            'rollback_accuracy': best_accuracy,
            'rollback_count': self.rollback_count
        }
        
        print(f"🔄 Rollback #{self.rollback_count}: Restored to iteration {best_iteration} ({best_accuracy:.2f}%)")
        
        return copy.deepcopy(best_model), rollback_info

class ConvergenceDetector:
    """Detect convergence conditions for iterative pruning."""
    
    def __init__(self, config: WilcoxonConfig):
        self.config = config
        self.no_pruning_count = 0
    
    def check_convergence(self, neurons_pruned: int) -> Tuple[bool, str]:
        """Check if convergence conditions are met."""
        if neurons_pruned == 0:
            self.no_pruning_count += 1
            if self.no_pruning_count >= self.config.patience:
                return True, f"No pruning for {self.config.patience} consecutive iterations"
        else:
            self.no_pruning_count = 0
        
        return False, "Not converged"

def make_wilcoxon_pruning_decisions(fanim_scores: Dict[str, np.ndarray], 
                                  alpha: float = 0.1) -> Tuple[Dict[str, List[str]], Dict]:
    """
    Make pruning decisions using Wilcoxon statistical significance testing.
    
    Decision Logic:
    1. Test H₀: median FANIM = 0 for each neuron
    2. If significant (p < α) AND median > 0 → PRUNE (harmful)
    3. If significant (p < α) AND median < 0 → KEEP (helpful)
    4. If not significant → KEEP (conservative default)
    
    Args:
        fanim_scores: Dict[layer_name, scores_array]
        alpha: Significance level (e.g., 0.1, 0.05, 0.01)
    
    Returns:
        Tuple of (pruning_decisions, statistics)
    """
    print(f"🧪 WILCOXON STATISTICAL PRUNING (α = {alpha})")
    print("=" * 50)
    
    pruning_decisions = {}
    detailed_stats = {}
    
    total_neurons = 0
    total_significant = 0
    total_pruned = 0
    
    for layer_name, scores in fanim_scores.items():
        print(f"\n🔬 Analyzing {layer_name}...")
        num_batches, num_neurons = scores.shape
        layer_decisions = []
        
        layer_significant = 0
        layer_pruned = 0
        
        for neuron_idx in range(num_neurons):
            neuron_scores = scores[:, neuron_idx]
            
            # Remove exact zeros for Wilcoxon test
            non_zero_scores = neuron_scores[neuron_scores != 0]
            
            if len(non_zero_scores) < 10:  # Minimum sample size
                decision = 'KEEP'
                p_value = 1.0
                significant = False
                median_score = np.median(neuron_scores)
            else:
                try:
                    # Wilcoxon signed-rank test
                    statistic, p_value = wilcoxon(non_zero_scores, alternative='two-sided')
                    significant = p_value < alpha
                    median_score = np.median(neuron_scores)
                    
                    if significant:
                        # Statistically significant effect detected
                        decision = 'PRUNE' if median_score > 0 else 'KEEP'
                        layer_significant += 1
                        if decision == 'PRUNE':
                            layer_pruned += 1
                    else:
                        # No significant effect - conservative default
                        decision = 'KEEP'
                        
                except ValueError:
                    # Handle edge cases
                    decision = 'KEEP'
                    p_value = 1.0
                    significant = False
                    median_score = np.median(neuron_scores)
            
            layer_decisions.append(decision)
        
        pruning_decisions[layer_name] = layer_decisions
        
        # Layer summary
        print(f"   Neurons analyzed: {num_neurons}")
        print(f"   Statistically significant: {layer_significant} ({layer_significant/num_neurons:.1%})")
        print(f"   Selected for pruning: {layer_pruned} ({layer_pruned/num_neurons:.1%})")
        
        total_neurons += num_neurons
        total_significant += layer_significant
        total_pruned += layer_pruned
    
    # Overall summary
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Total neurons: {total_neurons}")
    print(f"   Statistically significant: {total_significant} ({total_significant/total_neurons:.1%})")
    print(f"   Selected for pruning: {total_pruned} ({total_pruned/total_neurons:.1%})")
    
    statistics = {
        'alpha': alpha,
        'total_neurons': total_neurons,
        'significant_neurons': total_significant,
        'pruned_neurons': total_pruned,
        'significance_rate': total_significant / total_neurons,
        'pruning_rate': total_pruned / total_neurons
    }
    
    return pruning_decisions, statistics

class EnhancedIterativeWilcoxon:
    """
    Enhanced iterative Wilcoxon pruning with adaptive alpha and rollbacks.
    
    This class encapsulates the complete iterative pruning workflow with
    sophisticated error handling, checkpointing, and adaptive strategies.
    """
    
    def __init__(self, config: Optional[WilcoxonConfig] = None):
        self.config = config or WilcoxonConfig()
        self.alpha_scheduler = AdaptiveAlphaScheduler(self.config)
        self.checkpoint_manager = None  # Initialized with baseline accuracy
        self.convergence_detector = ConvergenceDetector(self.config)
        
        # Tracking
        self.iteration_history = []
        self.detailed_log = []
        self.total_neurons_pruned = 0
        self.start_time = None
        
    def log_message(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.detailed_log.append(log_entry)
        print(log_entry)
    
    def run_single_iteration(self, current_model: nn.Module, train_loader, test_loader,
                           iteration: int, baseline_accuracy: float) -> Dict:
        """
        Execute a single iteration of enhanced Wilcoxon pruning.
        
        Returns:
            Dictionary with comprehensive iteration results
        """
        self.log_message(f"Starting iteration {iteration}", "INFO")
        
        # Get adaptive alpha
        current_alpha = self.alpha_scheduler.get_alpha(iteration)
        phase_desc = self.alpha_scheduler.get_phase_description(iteration)
        
        self.log_message(f"Adaptive α = {current_alpha} ({phase_desc})", "INFO")
        
        try:
            # Step 1: Compute fresh FANIM scores
            self.log_message("Computing fresh FANIM scores...", "INFO")
            current_fanim = compute_fanim_from_model(
                current_model, 
                train_loader, 
                num_batches=50,
                disable_dropout=True
            )
            
            # Step 2: Make Wilcoxon decisions
            self.log_message("Performing Wilcoxon significance testing...", "INFO")
            pruning_decisions, statistics = make_wilcoxon_pruning_decisions(
                current_fanim, current_alpha
            )
            
            # Step 3: Count neurons to prune
            neurons_to_prune = sum(
                sum(1 for decision in layer_decisions if decision == 'PRUNE')
                for layer_decisions in pruning_decisions.values()
            )
            
            self.log_message(f"Neurons selected for pruning: {neurons_to_prune}", "INFO")
            
            # Step 4: Handle convergence case
            if neurons_to_prune == 0:
                current_accuracy, _ = evaluate_model_performance(current_model, test_loader)
                self.checkpoint_manager.add_checkpoint(current_model, current_accuracy, iteration)
                
                return {
                    'iteration': iteration,
                    'alpha': current_alpha,
                    'neurons_pruned': 0,
                    'final_accuracy': current_accuracy,
                    'vs_baseline': current_accuracy - baseline_accuracy,
                    'status': 'convergence_achieved',
                    'pruned_model': current_model,
                    'statistics': statistics
                }
            
            # Step 5: Apply pruning
            self.log_message("Applying structural pruning...", "INFO")
            pruned_model = apply_structural_pruning(current_model, pruning_decisions)
            
            # Step 6: Evaluate after pruning
            pruned_accuracy, _ = evaluate_model_performance(pruned_model, test_loader)
            self.log_message(f"Accuracy after pruning: {pruned_accuracy:.2f}%", "INFO")
            
            # Step 7: Fine-tune
            self.log_message(f"Fine-tuning for {self.config.epochs_per_iteration} epochs...", "INFO")
            fine_tuned_model = fine_tune_model(pruned_model, train_loader, self.config.epochs_per_iteration)
            
            # Step 8: Final evaluation
            final_accuracy, _ = evaluate_model_performance(fine_tuned_model, test_loader)
            self.log_message(f"Final accuracy: {final_accuracy:.2f}%", "INFO")
            
            # Step 9: Check for rollback condition
            if self.checkpoint_manager.should_rollback(final_accuracy, baseline_accuracy):
                self.log_message(f"Accuracy drop too large: {baseline_accuracy - final_accuracy:.2f}%", "WARN")
                return {
                    'iteration': iteration,
                    'alpha': current_alpha,
                    'neurons_pruned': neurons_to_prune,
                    'pruned_accuracy': pruned_accuracy,
                    'final_accuracy': final_accuracy,
                    'vs_baseline': final_accuracy - baseline_accuracy,
                    'status': 'needs_rollback',
                    'statistics': statistics
                }
            
            # Step 10: Success - add checkpoint
            self.checkpoint_manager.add_checkpoint(fine_tuned_model, final_accuracy, iteration)
            self.total_neurons_pruned += neurons_to_prune
            
            return {
                'iteration': iteration,
                'alpha': current_alpha,
                'neurons_pruned': neurons_to_prune,
                'pruned_accuracy': pruned_accuracy,
                'final_accuracy': final_accuracy,
                'vs_baseline': final_accuracy - baseline_accuracy,
                'status': 'success',
                'pruned_model': fine_tuned_model,
                'statistics': statistics
            }
            
        except Exception as e:
            self.log_message(f"Error in iteration {iteration}: {str(e)}", "ERROR")
            return {
                'iteration': iteration,
                'alpha': current_alpha,
                'status': 'error',
                'error': str(e)
            }
    
    def run_complete_experiment(self) -> Dict:
        """
        Run complete enhanced iterative Wilcoxon experiment.
        
        Returns:
            Comprehensive experiment results
        """
        self.start_time = time.time()
        self.log_message("Starting Enhanced Iterative Wilcoxon Pruning v2.0", "INFO")
        
        # Load data and models
        train_loader, test_loader, scaler = load_wine_quality_data()
        original_model = load_trained_model()
        
        # Baseline evaluation
        baseline_accuracy, baseline_loss = evaluate_model_performance(original_model, test_loader)
        baseline_params = count_parameters(original_model)
        
        self.log_message(f"Baseline: {baseline_accuracy:.2f}% accuracy, {baseline_params:,} parameters", "INFO")
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(baseline_accuracy, self.config)
        self.checkpoint_manager.add_checkpoint(original_model, baseline_accuracy, 0)
        
        # Initialize current model
        current_model = copy.deepcopy(original_model)
        
        # Main iterative loop
        for iteration in range(1, self.config.max_iterations + 1):
            # Run single iteration
            iteration_result = self.run_single_iteration(
                current_model, train_loader, test_loader, iteration, baseline_accuracy
            )
            
            # Handle rollback if needed
            if iteration_result['status'] == 'needs_rollback':
                try:
                    self.log_message("Executing rollback...", "WARN")
                    current_model, rollback_info = self.checkpoint_manager.execute_rollback()
                    iteration_result['rollback_info'] = rollback_info
                    self.iteration_history.append(iteration_result)
                    continue
                except Exception as e:
                    self.log_message(f"Rollback failed: {e}", "ERROR")
                    break
            
            # Add to history
            self.iteration_history.append(iteration_result)
            
            # Check for failure
            if iteration_result['status'] not in ['success', 'convergence_achieved']:
                self.log_message(f"Stopping due to status: {iteration_result['status']}", "ERROR")
                break
            
            # Update current model
            if 'pruned_model' in iteration_result:
                current_model = iteration_result['pruned_model']
            
            # Check convergence
            neurons_pruned = iteration_result.get('neurons_pruned', 0)
            converged, reason = self.convergence_detector.check_convergence(neurons_pruned)
            
            if converged:
                self.log_message(f"Convergence achieved: {reason}", "INFO")
                break
        
        # Final evaluation
        final_accuracy, _ = evaluate_model_performance(current_model, test_loader)
        final_params = count_parameters(current_model)
        final_sparsity = get_model_sparsity(baseline_params, final_params)
        inference_time = measure_inference_time(current_model, test_loader)
        
        total_time = time.time() - self.start_time
        
        # Compile results
        successful_iterations = sum(1 for h in self.iteration_history if h.get('status') == 'success')
        
        results = {
            'experiment_info': {
                'method': 'Enhanced Iterative Wilcoxon v2.0',
                'status': 'completed',
                'iterations_completed': len(self.iteration_history),
                'successful_iterations': successful_iterations,
                'total_neurons_pruned': self.total_neurons_pruned,
                'total_processing_time': total_time
            },
            
            'performance_metrics': {
                'baseline_accuracy': baseline_accuracy,
                'final_accuracy': final_accuracy,
                'accuracy_improvement': final_accuracy - baseline_accuracy,
                'baseline_parameters': baseline_params,
                'final_parameters': final_params,
                'final_sparsity': final_sparsity,
                'compression_ratio': baseline_params / final_params,
                'inference_time_ms': inference_time
            },
            
            'iteration_history': [
                {k: v for k, v in iteration.items() if k != 'pruned_model'}  # Remove non-serializable models
                for iteration in self.iteration_history
            ]
        }
        
        return results

def run_enhanced_iterative_wilcoxon(config: Optional[WilcoxonConfig] = None) -> Dict:
    """Main entry point for enhanced iterative Wilcoxon experiment."""
    
    if config is None:
        config = WilcoxonConfig()
    
    experiment = EnhancedIterativeWilcoxon(config)
    return experiment.run_complete_experiment()

def main():
    """Run enhanced iterative Wilcoxon experiment with production settings."""
    print("🔄 Enhanced Iterative Wilcoxon Pruning v2.0 - Clean Implementation")
    print("=" * 70)
    
    # Create custom configuration if desired
    config = WilcoxonConfig()
    config.max_iterations = 15
    config.epochs_per_iteration = 5
    config.rollback_threshold = 1.5
    config.min_accuracy_threshold = 2.0
    config.max_rollbacks = 3
    
    print(f"Configuration:")
    print(f"  • Max iterations: {config.max_iterations}")
    print(f"  • Epochs per iteration: {config.epochs_per_iteration}")
    print(f"  • Rollback threshold: {config.rollback_threshold}%")
    print(f"  • Alpha schedule: Liberal(0.1) → Conservative(0.05) → Strict(0.01)")
    print()
    
    # Run experiment
    results = run_enhanced_iterative_wilcoxon(config)
    
    # Save results - FORGE SPECIFIC PATH
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved: {RESULTS_FILE}")
    
    return results

if __name__ == "__main__":
    main()