#!/usr/bin/env python3
"""
Magnitude-Based Pruning Baseline
================================

PURPOSE: Traditional magnitude-based pruning for comparison with Bonsai methods
PHILOSOPHY: Industry standard baseline - prune neurons with smallest weight magnitudes

Implementation:
- Calculates L2 norm of weights for each neuron
- Prunes neurons with smallest magnitudes
- Deterministic and widely used in literature
- Essential baseline to validate Bonsai's superiority over traditional methods
"""

import torch
import numpy as np
import time
from typing import Optional

import sys
from pathlib import Path

# Ensure we can import from shared
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from shared.pruning_utils import BasePruner, PruningResults, DEVICE

class MagnitudePruner(BasePruner):
    """Magnitude-based pruning baseline - traditional approach."""
    
    def __init__(self, model, test_loader, train_loader=None, 
                 sparsity_target: float = 0.2, magnitude_type: str = "L2"):
        """
        Initialize magnitude pruner.
        
        Args:
            model: Model to prune
            test_loader: Test data loader
            train_loader: Training data loader (for fine-tuning)
            sparsity_target: Fraction of neurons to prune (0.0 to 1.0)
            magnitude_type: Type of magnitude calculation ("L2", "L1", "max")
        """
        super().__init__(model, test_loader, train_loader)
        self.sparsity_target = sparsity_target
        self.magnitude_type = magnitude_type
        
        if magnitude_type not in ["L2", "L1", "max"]:
            raise ValueError(f"Unknown magnitude type: {magnitude_type}. Use 'L2', 'L1', or 'max'")
        
    def calculate_neuron_magnitudes(self, weights: torch.Tensor) -> np.ndarray:
        """Calculate magnitude for each neuron (output unit)."""
        if self.magnitude_type == "L2":
            # L2 norm across input dimensions
            magnitudes = torch.norm(weights, p=2, dim=1)
        elif self.magnitude_type == "L1":
            # L1 norm across input dimensions
            magnitudes = torch.norm(weights, p=1, dim=1)
        elif self.magnitude_type == "max":
            # Maximum absolute weight
            magnitudes = torch.max(torch.abs(weights), dim=1)[0]
        
        return magnitudes.cpu().numpy()
        
    def prune(self) -> PruningResults:
        """Execute magnitude-based pruning."""
        start_time = time.time()
        
        print(f"📏 MAGNITUDE PRUNING BASELINE")
        print(f"🎯 Target Sparsity: {self.sparsity_target:.1%}")
        print(f"📐 Magnitude Type: {self.magnitude_type} norm")
        
        # Calculate neuron magnitudes for each layer
        prune_decisions = {}
        magnitude_stats = {}
        
        total_neurons = 0
        total_pruned = 0
        
        for layer_name in ['fc1', 'fc2', 'fc3']:
            layer = getattr(self.original_model, layer_name)
            weights = layer.weight.data  # Shape: [out_features, in_features]
            
            # Calculate magnitudes for each output neuron (row)
            neuron_magnitudes = self.calculate_neuron_magnitudes(weights)
            
            # Determine how many to prune
            num_neurons = len(neuron_magnitudes)
            num_to_prune = int(num_neurons * self.sparsity_target)
            
            # Get indices of smallest magnitude neurons
            prune_indices = np.argsort(neuron_magnitudes)[:num_to_prune].tolist()
            prune_decisions[layer_name] = prune_indices
            
            # Store statistics
            magnitude_stats[layer_name] = {
                'mean_magnitude': float(np.mean(neuron_magnitudes)),
                'std_magnitude': float(np.std(neuron_magnitudes)),
                'min_magnitude': float(np.min(neuron_magnitudes)),
                'max_magnitude': float(np.max(neuron_magnitudes)),
                'pruned_magnitude_threshold': float(neuron_magnitudes[np.argsort(neuron_magnitudes)[num_to_prune-1]]) if num_to_prune > 0 else None
            }
            
            total_neurons += num_neurons
            total_pruned += num_to_prune
            
            print(f"  {layer_name}: Pruning {num_to_prune}/{num_neurons} smallest magnitude neurons")
            print(f"    Magnitude range: [{neuron_magnitudes.min():.4f}, {neuron_magnitudes.max():.4f}]")
            if num_to_prune > 0:
                threshold = magnitude_stats[layer_name]['pruned_magnitude_threshold']
                print(f"    Pruning threshold: {threshold:.4f}")
        
        print(f"\n📊 MAGNITUDE PRUNING SUMMARY:")
        print(f"  Total neurons: {total_neurons}")
        print(f"  Neurons to prune: {total_pruned}")
        print(f"  Actual sparsity: {total_pruned/total_neurons:.1%}")
        print(f"  Magnitude criterion: {self.magnitude_type} norm (smallest pruned)")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create results using shared infrastructure
        return self._create_pruning_results(
            method_name=f"Magnitude_{self.magnitude_type}_Baseline",
            prune_decisions=prune_decisions,
            processing_time=processing_time,
            statistical_info={
                'method': 'magnitude',
                'magnitude_type': self.magnitude_type,
                'sparsity_target': self.sparsity_target,
                'criterion': f'{self.magnitude_type}_norm_smallest',
                'magnitude_stats': magnitude_stats,
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
    
    # Test different magnitude types and sparsity levels
    magnitude_types = ["L2", "L1", "max"]
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for magnitude_type in magnitude_types:
        for sparsity in sparsity_levels:
            print(f"\n{'='*60}")
            print(f"Testing {magnitude_type} Magnitude Pruning at {sparsity:.1%} sparsity")
            print(f"{'='*60}")
            
            # Create pruner and run
            pruner = MagnitudePruner(model, test_loader, train_loader, 
                                   sparsity_target=sparsity, magnitude_type=magnitude_type)
            results = pruner.prune()
            
            # Print results
            print(f"\n🎯 RESULTS:")
            print(f"   Final accuracy: {results.final_accuracy:.2f}%")
            print(f"   vs Baseline: {results.accuracy_change:+.2f}%")
            print(f"   Actual sparsity: {results.sparsity:.1%}")
            print(f"   Processing time: {results.processing_time:.2f}s")
            
            # Save results
            save_pruning_results(results, f"magnitude_{magnitude_type.lower()}_pruning_sparsity_{sparsity:.1f}")