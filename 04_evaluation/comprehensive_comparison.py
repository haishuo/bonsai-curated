#!/usr/bin/env python3
"""
Comprehensive Pruning Method Comparison
=======================================

PURPOSE: Complete evaluation framework comparing all pruning methods
PHILOSOPHY: Rigorous empirical comparison across statistical and baseline methods

Comparison Framework:
1. Bonsai Methods (FANIM & BANIM) across different risk profiles
2. Baseline methods (Random & Magnitude) across different configurations
3. Statistical analysis of results
4. Visualization and reporting

Belongs in 04_evaluation/ as it's pure evaluation, not a pruning method itself.
"""

import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Optional
from pathlib import Path

import sys
from pathlib import Path

# Handle numbered directory imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "03_pruning"))

from shared.pruning_utils import (
    load_trained_model, load_wine_quality_data, save_pruning_results, PruningResults
)
from shared.forge_config import get_bonsai_config

# Import all pruning methods
from bonsai_methods import WilcoxonFANIMPruner, WilcoxonBANIMPruner
from random_pruning import RandomPruner
from magnitude_pruning import MagnitudePruner
from statistical_tests import RiskProfile

# Initialize forge config for results saving
forge_config = get_bonsai_config()

class ComprehensivePruningComparison:
    """Framework for comprehensive comparison of all pruning methods."""
    
    def __init__(self, model, test_loader, train_loader=None):
        """
        Initialize comparison framework.
        
        Args:
            model: Trained model to prune
            test_loader: Test data loader
            train_loader: Training data loader (for fine-tuning)
        """
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.results = {}
        
    def test_bonsai_methods(self, risk_profiles: Optional[List[RiskProfile]] = None,
                          alpha_levels: Optional[List[float]] = None) -> Dict[str, PruningResults]:
        """
        Test all Bonsai methods across risk profiles and significance levels.
        
        Args:
            risk_profiles: Risk profiles to test (default: all)
            alpha_levels: Significance levels to test (default: [0.01, 0.05, 0.1])
            
        Returns:
            Dictionary of results
        """
        if risk_profiles is None:
            risk_profiles = [
                RiskProfile.NO_CORRECTION
            ]
            
        if alpha_levels is None:
            alpha_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Common significance levels
        
        print("🌳 TESTING BONSAI METHODS")
        print("=" * 50)
        
        bonsai_results = {}
        
        for risk_profile in risk_profiles:
            for alpha in alpha_levels:
                print(f"\n🔬 Risk Profile: {risk_profile.name}, α = {alpha}")
                
                # Test FANIM
                try:
                    fanim_pruner = WilcoxonFANIMPruner(
                        self.model, self.test_loader, self.train_loader,
                        risk_profile=risk_profile, alpha=alpha
                    )
                    fanim_result = fanim_pruner.prune()
                    key = f"FANIM_{risk_profile.name}_alpha_{alpha}"
                    bonsai_results[key] = fanim_result
                    
                    print(f"  ✅ FANIM: {fanim_result.final_accuracy:.2f}% accuracy, {fanim_result.sparsity:.1%} sparsity")
                    
                except Exception as e:
                    print(f"  ❌ FANIM failed: {e}")
                
                # Test BANIM
                try:
                    banim_pruner = WilcoxonBANIMPruner(
                        self.model, self.test_loader, self.train_loader,
                        risk_profile=risk_profile, alpha=alpha
                    )
                    banim_result = banim_pruner.prune()
                    key = f"BANIM_{risk_profile.name}_alpha_{alpha}"
                    bonsai_results[key] = banim_result
                    
                    print(f"  ✅ BANIM: {banim_result.final_accuracy:.2f}% accuracy, {banim_result.sparsity:.1%} sparsity")
                    
                except Exception as e:
                    print(f"  ❌ BANIM failed: {e}")
        
        return bonsai_results
    
    def test_baseline_methods(self, sparsity_levels: Optional[List[float]] = None,
                            magnitude_types: Optional[List[str]] = None,
                            random_seeds: Optional[List[int]] = None) -> Dict[str, PruningResults]:
        """
        Test baseline methods across configurations.
        
        Args:
            sparsity_levels: Sparsity levels to test
            magnitude_types: Magnitude calculation types
            random_seeds: Random seeds for random pruning
            
        Returns:
            Dictionary of results
        """
        if sparsity_levels is None:
            sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.9]
            
        if magnitude_types is None:
            magnitude_types = ["L2", "L1"]
            
        if random_seeds is None:
            random_seeds = [42, 123, 456]  # Multiple seeds for statistical robustness
        
        print("\n🎯 TESTING BASELINE METHODS")
        print("=" * 50)
        
        baseline_results = {}
        
        # Test Random Pruning
        for sparsity in sparsity_levels:
            for seed in random_seeds:
                print(f"\n🎲 Random Pruning: {sparsity:.1%} sparsity, seed {seed}")
                
                try:
                    random_pruner = RandomPruner(
                        self.model, self.test_loader, self.train_loader,
                        sparsity_target=sparsity, random_seed=seed
                    )
                    random_result = random_pruner.prune()
                    key = f"Random_sparsity_{sparsity:.1f}_seed_{seed}"
                    baseline_results[key] = random_result
                    
                    print(f"  ✅ Random: {random_result.final_accuracy:.2f}% accuracy, {random_result.sparsity:.1%} sparsity")
                    
                except Exception as e:
                    print(f"  ❌ Random failed: {e}")
        
        # Test Magnitude Pruning
        for sparsity in sparsity_levels:
            for magnitude_type in magnitude_types:
                print(f"\n📏 Magnitude Pruning: {magnitude_type} norm, {sparsity:.1%} sparsity")
                
                try:
                    magnitude_pruner = MagnitudePruner(
                        self.model, self.test_loader, self.train_loader,
                        sparsity_target=sparsity, magnitude_type=magnitude_type
                    )
                    magnitude_result = magnitude_pruner.prune()
                    key = f"Magnitude_{magnitude_type}_sparsity_{sparsity:.1f}"
                    baseline_results[key] = magnitude_result
                    
                    print(f"  ✅ Magnitude: {magnitude_result.final_accuracy:.2f}% accuracy, {magnitude_result.sparsity:.1%} sparsity")
                    
                except Exception as e:
                    print(f"  ❌ Magnitude failed: {e}")
        
        return baseline_results
    
    def run_complete_comparison(self, save_results: bool = True) -> Dict[str, PruningResults]:
        """
        Run complete comparison of all methods.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            Complete results dictionary
        """
        print("🌳 COMPREHENSIVE PRUNING METHOD COMPARISON")
        print("=" * 60)
        print(f"Evaluating model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        start_time = time.time()
        
        # Test all methods
        bonsai_results = self.test_bonsai_methods()
        baseline_results = self.test_baseline_methods()
        
        # Combine results
        all_results = {**bonsai_results, **baseline_results}
        self.results = all_results
        
        total_time = time.time() - start_time
        
        print(f"\n📊 COMPARISON COMPLETE")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🧪 Methods tested: {len(all_results)}")
        
        # Generate summary
        self.print_summary()
        
        if save_results:
            self.save_all_results()
        
        return all_results
    
    def print_summary(self):
        """Print summary table of all results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n📊 RESULTS SUMMARY")
        print("=" * 80)
        
        # Create summary dataframe
        summary_data = []
        for method_name, result in self.results.items():
            summary_data.append({
                'Method': method_name,
                'Final_Accuracy': result.final_accuracy,
                'Accuracy_Change': result.accuracy_change,
                'Sparsity': result.sparsity,
                'Processing_Time': result.processing_time,
                'Neurons_Pruned_Total': sum(result.neurons_pruned.values())
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Final_Accuracy', ascending=False)
        
        # Print top performers
        print("🏆 TOP 10 METHODS BY FINAL ACCURACY:")
        print(df[['Method', 'Final_Accuracy', 'Accuracy_Change', 'Sparsity']].head(10).to_string(index=False, float_format='%.2f'))
        
        print(f"\n📈 BEST METHODS BY CATEGORY:")
        
        # Best Bonsai methods
        bonsai_methods = df[df['Method'].str.contains('FANIM|BANIM')]
        if not bonsai_methods.empty:
            best_bonsai = bonsai_methods.iloc[0]
            print(f"  🌳 Best Bonsai: {best_bonsai['Method']} - {best_bonsai['Final_Accuracy']:.2f}% accuracy")
        
        # Best baseline methods
        baseline_methods = df[df['Method'].str.contains('Random|Magnitude')]
        if not baseline_methods.empty:
            best_baseline = baseline_methods.iloc[0]
            print(f"  📏 Best Baseline: {best_baseline['Method']} - {best_baseline['Final_Accuracy']:.2f}% accuracy")
        
        # Statistical significance (if we have both Bonsai and baseline)
        if not bonsai_methods.empty and not baseline_methods.empty:
            bonsai_best_acc = bonsai_methods['Final_Accuracy'].max()
            baseline_best_acc = baseline_methods['Final_Accuracy'].max()
            improvement = bonsai_best_acc - baseline_best_acc
            print(f"  📊 Bonsai Advantage: +{improvement:.2f}% accuracy improvement")
    
    def save_all_results(self):
        """Save all results to disk using forge config."""
        if not self.results:
            return
        
        # Save individual results
        for method_name, result in self.results.items():
            save_pruning_results(result, f"comparison_{method_name}")
        
        # Save comprehensive summary
        summary_file = forge_config.results_path("comprehensive_comparison_summary")
        
        summary_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_methods_tested': len(self.results),
            'baseline_parameters': list(self.results.values())[0].baseline_params,
            'summary_statistics': {},
            'method_results': {}
        }
        
        # Add method results
        for method_name, result in self.results.items():
            summary_data['method_results'][method_name] = {
                'final_accuracy': result.final_accuracy,
                'accuracy_change': result.accuracy_change,
                'sparsity': result.sparsity,
                'processing_time': result.processing_time,
                'neurons_pruned': result.neurons_pruned
            }
        
        # Add summary statistics
        accuracies = [r.final_accuracy for r in self.results.values()]
        sparsities = [r.sparsity for r in self.results.values()]
        
        summary_data['summary_statistics'] = {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'accuracy_min': float(np.min(accuracies)),
            'accuracy_max': float(np.max(accuracies)),
            'sparsity_mean': float(np.mean(sparsities)),
            'sparsity_std': float(np.std(sparsities))
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Comprehensive results saved: {summary_file}")

def run_comprehensive_comparison():
    """Main function to run comprehensive comparison."""
    # Load model and data
    print("📂 Loading model and data...")
    model = load_trained_model()
    train_loader, test_loader, _ = load_wine_quality_data()
    
    # Run comparison
    comparison = ComprehensivePruningComparison(model, test_loader, train_loader)
    results = comparison.run_complete_comparison(save_results=True)
    
    return results

if __name__ == "__main__":
    run_comprehensive_comparison()