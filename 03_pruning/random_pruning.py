#!/usr/bin/env python3
"""
Random Pruning Baseline
=======================

PURPOSE: Random pruning for statistical comparison with Bonsai methods
PHILOSOPHY: Unbiased baseline - validates that statistical methods outperform chance

Implementation:
- Randomly selects neurons to prune based on target sparsity
- Reproducible via random seed control
- Essential baseline for validating statistical significance of Bonsai methods
"""

import numpy as np
import time
import random
from typing import Optional

import sys
from pathlib import Path

# Ensure we can import from shared
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from shared.pruning_utils import BasePruner, PruningResults

class RandomPruner(BasePruner):
    """Random pruning baseline for statistical comparison."""
    
    def __init__(self, model, test_loader, train_loader=None, 
                 sparsity_target: float = 0.2, random_seed: int = 42):
        """
        Initialize random pruner.
        
        Args:
            model: Model to prune
            test_loader: Test data loader
            train_loader: Training data loader (for fine-tuning)
            sparsity_target: Fraction of neurons to prune (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        """
        super().__init__(model, test_loader, train_loader)
        self.sparsity_target = sparsity_target
        self.random_seed = random_seed
        
    def prune(self) -> PruningResults:
        """Execute random pruning."""
        start_time = time.time()
        
        print(f"🎲 RANDOM PRUNING BASELINE")
        print(f"🎯 Target Sparsity: {self.sparsity_target:.1%}")
        print(f"🔢 Random Seed: {self.random_seed}")
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Get layer sizes from model
        layer_sizes = self.original_model.get_layer_sizes()
        prune_decisions = {}
        
        total_neurons = 0
        total_pruned = 0
        
        for layer_name, size in layer_sizes.items():
            num_to_prune = int(size * self.sparsity_target)
            
            # Randomly select neurons to prune
            all_indices = list(range(size))
            random.shuffle(all_indices)
            prune_indices = all_indices[:num_to_prune]
            prune_decisions[layer_name] = prune_indices
            
            total_neurons += size
            total_pruned += num_to_prune
            
            print(f"  {layer_name}: Pruning {num_to_prune}/{size} neurons randomly")
        
        print(f"\n📊 RANDOM PRUNING SUMMARY:")
        print(f"  Total neurons: {total_neurons}")
        print(f"  Neurons to prune: {total_pruned}")
        print(f"  Actual sparsity: {total_pruned/total_neurons:.1%}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create results using shared infrastructure
        return self._create_pruning_results(
            method_name="Random_Baseline",
            prune_decisions=prune_decisions,
            processing_time=processing_time,
            statistical_info={
                'method': 'random',
                'sparsity_target': self.sparsity_target,
                'random_seed': self.random_seed,
                'total_neurons': total_neurons,
                'total_pruned': total_pruned
            }
        )

if __name__ == "__main__":
    # Example usage
    from shared.pruning_utils import load_trained_model, load_wine_quality_data, save_pruning_results
    
    # Load model and data
    model = load_trained_model()
    train_loader, test_loader, _ = load_wine_quality_data()
    
    # Test different sparsity levels
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for sparsity in sparsity_levels:
        print(f"\n{'='*60}")
        print(f"Testing Random Pruning at {sparsity:.1%} sparsity")
        print(f"{'='*60}")
        
        # Create pruner and run
        pruner = RandomPruner(model, test_loader, train_loader, 
                            sparsity_target=sparsity, random_seed=42)
        results = pruner.prune()
        
        # Print results
        print(f"\n🎯 RESULTS:")
        print(f"   Final accuracy: {results.final_accuracy:.2f}%")
        print(f"   vs Baseline: {results.accuracy_change:+.2f}%")
        print(f"   Actual sparsity: {results.sparsity:.1%}")
        print(f"   Processing time: {results.processing_time:.2f}s")
        
        # Save results
        save_pruning_results(results, f"random_pruning_sparsity_{sparsity:.1f}")