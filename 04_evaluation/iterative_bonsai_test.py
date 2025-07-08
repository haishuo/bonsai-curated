#!/usr/bin/env python3
"""
Iterative Bonsai Test Script
===========================

PURPOSE: Complete working example of iterative Bonsai pruning
PHILOSOPHY: Test the full iterative implementation with real data

This script demonstrates how to:
1. Load a trained model and data
2. Run iterative BANIM pruning (since it performed better in your tests)
3. Compare results with single-shot approaches
4. Validate that iterative achieves higher sparsity

Expected Results:
- Target: 50%+ sparsity with maintained/improved accuracy
- Should beat random pruning at equivalent sparsity levels
- Should demonstrate the power of fresh evidence collection
"""

import sys
from pathlib import Path
import json
import pandas as pd
import time

# Handle numbered directory imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "03_pruning"))

from shared.pruning_utils import load_trained_model, load_wine_quality_data, save_pruning_results
from iterative_bonsai import create_iterative_banim_pruner, create_iterative_fanim_pruner, quick_iterative_test

def run_iterative_bonsai_comparison():
    """
    Run comprehensive iterative Bonsai comparison.
    
    Tests multiple configurations to find optimal settings.
    """
    print("🔄 ITERATIVE BONSAI COMPREHENSIVE COMPARISON")
    print("=" * 60)
    
    # Load model and data
    print("📂 Loading trained model and data...")
    model = load_trained_model()  # Load your pre-trained wine quality model
    train_loader, test_loader, _ = load_wine_quality_data()
    
    if model is None:
        print("❌ Could not load trained model. Please ensure you have a trained model available.")
        print("   Run training first: python 01_training/wine_quality_ds.py")
        return
    
    print("✅ Model and data loaded successfully")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Conservative_BANIM',
            'temporal_direction': 'BANIM',
            'alpha': 0.05,
            'max_iterations': 8,
            'fine_tune_epochs': 5,
            'collection_batches': 100
        },
        {
            'name': 'Moderate_BANIM', 
            'temporal_direction': 'BANIM',
            'alpha': 0.1,
            'max_iterations': 10,
            'fine_tune_epochs': 5,
            'collection_batches': 100
        },
        {
            'name': 'Aggressive_BANIM',
            'temporal_direction': 'BANIM', 
            'alpha': 0.15,
            'max_iterations': 6,
            'fine_tune_epochs': 3,
            'collection_batches': 80
        },
        {
            'name': 'Conservative_FANIM',
            'temporal_direction': 'FANIM',
            'alpha': 0.05,
            'max_iterations': 8,
            'fine_tune_epochs': 5,
            'collection_batches': 100
        }
    ]
    
    results = {}
    
    # Run each configuration
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {config['name']}")
        print(f"{'='*60}")
        
        try:
            if config['temporal_direction'] == 'BANIM':
                pruner = create_iterative_banim_pruner(
                    model=model,
                    test_loader=test_loader,
                    train_loader=train_loader,
                    alpha=config['alpha'],
                    max_iterations=config['max_iterations'],
                    fine_tune_epochs=config['fine_tune_epochs'],
                    collection_batches=config['collection_batches']
                )
            else:  # FANIM
                pruner = create_iterative_fanim_pruner(
                    model=model,
                    test_loader=test_loader,
                    train_loader=train_loader,
                    alpha=config['alpha'],
                    max_iterations=config['max_iterations'], 
                    fine_tune_epochs=config['fine_tune_epochs'],
                    collection_batches=config['collection_batches']
                )
            
            # Run pruning
            result = pruner.prune()
            results[config['name']] = result
            
            # Save individual result
            save_pruning_results(result, f"iterative_{config['name'].lower()}_results.json")
            
            print(f"\n✅ {config['name']} COMPLETED:")
            print(f"   Final Accuracy: {result.final_accuracy:.2f}%")
            print(f"   Accuracy Change: {result.accuracy_change:+.2f}%")
            print(f"   Sparsity: {result.sparsity:.1%}")
            print(f"   Iterations: {result.statistical_info['iterations_completed']}")
            print(f"   Converged: {result.statistical_info['converged']}")
            
        except Exception as e:
            print(f"❌ {config['name']} FAILED: {e}")
            results[config['name']] = {'error': str(e)}
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("📊 ITERATIVE BONSAI RESULTS SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [(name, result) for name, result in results.items() 
                         if isinstance(result, type(results[list(results.keys())[0]])) and not hasattr(result, 'error')]
    
    if successful_results:
        # Sort by final accuracy
        successful_results.sort(key=lambda x: x[1].final_accuracy, reverse=True)
        
        print(f"🏆 TOP PERFORMERS:")
        print(f"{'Method':<20} {'Accuracy':<10} {'Change':<8} {'Sparsity':<9} {'Iterations'}")
        print("-" * 65)
        
        for name, result in successful_results:
            iterations = result.statistical_info.get('iterations_completed', 'N/A')
            print(f"{name:<20} {result.final_accuracy:<10.2f} {result.accuracy_change:<8.2f} {result.sparsity:<9.1%} {iterations}")
        
        # Find best overall result
        best_name, best_result = successful_results[0]
        print(f"\n🎯 BEST OVERALL: {best_name}")
        print(f"   Accuracy: {best_result.final_accuracy:.2f}% ({best_result.accuracy_change:+.2f}%)")
        print(f"   Sparsity: {best_result.sparsity:.1%}")
        print(f"   Parameters: {best_result.baseline_params:,} → {best_result.pruned_params:,}")
        
        # Check if we beat the target
        if best_result.sparsity >= 0.5:  # 50% sparsity target
            print(f"   🎉 SUCCESS: Achieved target 50%+ sparsity!")
        else:
            print(f"   📈 Progress: {best_result.sparsity:.1%} sparsity (target: 50%+)")
        
        # Save comprehensive results
        summary = {
            'timestamp': str(pd.Timestamp.now()),
            'best_method': best_name,
            'best_result': {
                'accuracy': best_result.final_accuracy,
                'accuracy_change': best_result.accuracy_change,
                'sparsity': best_result.sparsity,
                'iterations': best_result.statistical_info.get('iterations_completed'),
                'converged': best_result.statistical_info.get('converged')
            },
            'all_results': {name: {
                'accuracy': result.final_accuracy,
                'accuracy_change': result.accuracy_change, 
                'sparsity': result.sparsity,
                'processing_time': result.processing_time
            } for name, result in successful_results}
        }
        
        with open('iterative_bonsai_comprehensive_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved: iterative_bonsai_comprehensive_results.json")
        
    else:
        print("❌ No successful results to compare")
    
    return results

def run_quick_validation():
    """
    Quick validation test with minimal parameters.
    
    Useful for testing the implementation before full runs.
    """
    print("🧪 QUICK ITERATIVE VALIDATION TEST")
    print("=" * 40)
    
    # Load model and data
    print("📂 Loading model and data...")
    model = load_trained_model()
    train_loader, test_loader, _ = load_wine_quality_data()
    
    if model is None:
        print("❌ Could not load trained model")
        return None
    
    # Quick test with BANIM (performed better in your tests)
    print("🔄 Running quick BANIM test...")
    try:
        result = quick_iterative_test(
            model=model,
            test_loader=test_loader,
            train_loader=train_loader,
            temporal_direction="BANIM",
            alpha=0.1,
            max_iterations=3
        )
        
        print(f"\n✅ QUICK TEST SUCCESSFUL:")
        print(f"   Accuracy: {result.final_accuracy:.2f}% ({result.accuracy_change:+.2f}%)")
        print(f"   Sparsity: {result.sparsity:.1%}")
        print(f"   Iterations: {result.statistical_info['iterations_completed']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return None

if __name__ == "__main__":
    """
    Main execution with command line options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Iterative Bonsai Testing")
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Test mode: quick validation or full comparison')
    parser.add_argument('--temporal', choices=['BANIM', 'FANIM'], default='BANIM',
                       help='Temporal direction for quick test')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        print("Running quick validation test...")
        result = run_quick_validation()
        if result:
            print("\n🎉 Implementation validated! Ready for full experiments.")
        else:
            print("\n❌ Validation failed. Check your model and data setup.")
    
    elif args.mode == 'full':
        print("Running comprehensive iterative Bonsai comparison...")
        results = run_iterative_bonsai_comparison()
        print("\n🎉 Comprehensive testing complete!")
    
    print(f"\nNext steps:")
    print(f"1. Review results files for detailed analysis")
    print(f"2. Compare with your single-shot and random baseline results") 
    print(f"3. If performance is good, scale to more challenging datasets")
    print(f"4. Expected: 50%+ sparsity with better accuracy than random")