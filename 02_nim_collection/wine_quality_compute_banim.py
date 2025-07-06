#!/usr/bin/env python3
"""
Wine Quality: Backward Advanced Neuron Impact Metric Processor
=============================================================

PURPOSE: Process raw neuron impact data into Backward Advanced Neuron Impact Metrics (BANIM)
PHILOSOPHY: Engineering efficiency over mathematical purity - pragmatic production approach
WORKFLOW: wine_quality_sensitivity.h5 → GPU processing → banim.h5

INPUT:  Raw batch-level neuron impact data from wine_quality_ds.py training
OUTPUT: Processed Backward Advanced Neuron Impact Metrics ready for all Bonsai methods
BENEFIT: 10-100x speedup + no extra epoch requirement + noise filtering

Backward Advanced Neuron Impact Metric Formula:
BANIM_i = (∂L/∂a_i) × Δa_i × (1/Loss)
Where: Δa_i = a_i(t) - a_i(t-1)  [backward temporal difference]

Engineering Trade-offs:
- Mathematical compromise: Uses current gradient to explain past changes
- Computational efficiency: No need for 501st epoch
- Signal quality: Excludes random initialization noise from first batch
- Production advantage: Computed during normal training workflow

Empirical Question: Does performance match FANIM despite theoretical limitations?
"""

import torch
import numpy as np
import h5py
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import sys

# Import forge configuration
sys.path.append(str(Path(__file__).parent.parent))
from shared.forge_config import get_bonsai_config

# === CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Statistical configuration
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL
K_VALUE = np.sqrt(1.0 / ALPHA)  # Chebyshev k ≈ 4.47

# GPU processing configuration
GPU_CHUNK_SIZE = 5000  # Optimized for wine quality dataset
ENABLE_GPU_ACCELERATION = True

# Wine Quality Architecture (must match 01_training/wine_quality_ds.py)
LAYER_SIZES = {'fc1': 256, 'fc2': 128, 'fc3': 64}

# Initialize forge configuration
forge_config = get_bonsai_config()

# File paths using forge structure
INPUT_FILE = forge_config.nim_data_dir / 'wine_quality_raw_nim_data.h5'  # Actual training output
OUTPUT_FILE = forge_config.nim_data_dir / 'wine_quality_banim.h5'        # BANIM output
METADATA_FILE = forge_config.nim_data_dir / 'wine_quality_banim_metadata.json'


def load_raw_neuron_impact_data(filename: Path) -> Dict:
    """
    Load raw neuron impact data from wine quality training collection.
    
    Note: Input file uses legacy "sensitivity" naming but contains neuron impact metrics.
    
    Returns:
        Dictionary with batch-level gradients (NIMs), activations, and losses
    """
    print(f"📂 Loading raw neuron impact data: {filename}")
    print(f"   Note: This file contains raw NIM data from wine quality training")
    
    if not filename.exists():
        raise FileNotFoundError(f"Raw neuron impact data not found: {filename}. Check if wine quality training completed successfully.")
    
    data = {}
    
    with h5py.File(filename, 'r') as f:
        print(f"📋 Available datasets in {filename.name}:")
        
        # Load data for each layer using actual structure: /raw_data/layer_name/
        if 'raw_data' in f and 'batch_info' in f:
            # Get batch losses from batch_info
            batch_info = f['batch_info']
            if 'train_losses' in batch_info:
                batch_losses = np.array(batch_info['train_losses'])
                print(f"   Found {len(batch_losses)} batches with loss data")
            else:
                print(f"   ⚠️  No train_losses found in batch_info")
                batch_losses = None
            
            for layer_name in ['fc1', 'fc2', 'fc3']:
                if layer_name in f['raw_data']:
                    layer_data = f['raw_data'][layer_name]
                    
                    data[layer_name] = {
                        'nims': [],           # Neuron Impact Metrics (∂L/∂a_i)
                        'activations': [],    # a_i values
                        'losses': []          # Batch losses
                    }
                    
                    if 'gradients' in layer_data and 'activations' in layer_data:
                        # Load gradients (NIMs) and activations
                        gradients = np.array(layer_data['gradients'])  # Shape: (num_batches, num_neurons)
                        activations = np.array(layer_data['activations'])  # Shape: (num_batches + fanim_extra, num_neurons)
                        
                        # For BANIM, we only need training data (no epoch 501 needed)
                        if activations.shape[0] > gradients.shape[0]:
                            print(f"   📊 {layer_name}: Using training data only - {gradients.shape[0]} batches (ignoring epoch 501)")
                            activations = activations[:gradients.shape[0]]  # Use only training data for BANIM
                        elif activations.shape[0] == gradients.shape[0]:
                            print(f"   📊 {layer_name}: Perfect alignment - {gradients.shape[0]} batches")
                        else:
                            print(f"   ⚠️  {layer_name}: Missing activation data - activations={activations.shape[0]}, gradients={gradients.shape[0]}")
                        
                        print(f"   ✅ {layer_name}: {gradients.shape[0]} batches, {gradients.shape[1]} neurons")
                        
                        data[layer_name]['nims'] = gradients
                        data[layer_name]['activations'] = activations
                        
                        # Use batch losses if available, otherwise create dummy losses
                        if batch_losses is not None and len(batch_losses) == gradients.shape[0]:
                            data[layer_name]['losses'] = batch_losses
                        else:
                            print(f"   ⚠️  {layer_name}: Loss data mismatch, using dummy losses")
                            data[layer_name]['losses'] = np.ones(gradients.shape[0])  # Dummy losses
                    else:
                        print(f"   ⚠️  {layer_name}: Missing gradients or activations")
                        data[layer_name] = {'nims': [], 'activations': [], 'losses': []}
                else:
                    print(f"   ⚠️  {layer_name}: Layer not found in raw_data")
                    data[layer_name] = {'nims': [], 'activations': [], 'losses': []}
        else:
            print(f"   ⚠️  Expected 'raw_data' and 'batch_info' structure not found in {filename}")
            for layer_name in ['fc1', 'fc2', 'fc3']:
                data[layer_name] = {'nims': [], 'activations': [], 'losses': []}
    
    return data


def compute_layer_banim_statistics_gpu(raw_data: Dict, layer_name: str) -> Dict:
    """
    Compute Backward Advanced Neuron Impact Metrics and statistical summaries using GPU acceleration.
    
    BANIM Formula: BANIM_i = NIM_i × Δa_i × (1/Loss)
    Where: Δa_i = a_i(t) - a_i(t-1)  [backward temporal difference]
    
    Engineering Note: Uses backward differences for computational efficiency.
    Mathematical compromise: Current gradient explains past activation changes.
    Signal quality benefit: Excludes first batch (random initialization noise).
    """
    print(f"⚡ Processing {layer_name} with Backward Advanced Neuron Impact Metrics...")
    
    layer_data = raw_data[layer_name]
    
    if len(layer_data['nims']) == 0:
        print(f"   ⚠️  No data for {layer_name}, skipping...")
        return {}
    
    start_time = time.time()
    
    # Convert to tensors
    nims = torch.tensor(layer_data['nims'], dtype=torch.float32, device=DEVICE)
    activations = torch.tensor(layer_data['activations'], dtype=torch.float32, device=DEVICE)
    losses = torch.tensor(layer_data['losses'], dtype=torch.float32, device=DEVICE)
    
    num_batches, num_neurons = nims.shape
    print(f"   📊 Processing {num_batches} batches × {num_neurons} neurons")
    
    # Compute backward temporal differences: Δa_i = a_i(t) - a_i(t-1)
    # Note: We lose the first batch since it has no previous activation
    # This is actually beneficial - excludes random initialization noise!
    if num_batches < 2:
        raise ValueError(f"Need at least 2 batches for backward differences, got {num_batches}")
    
    backward_differences = activations[1:] - activations[:-1]  # Shape: (num_batches-1, num_neurons)
    
    # Use the corresponding NIMs and losses (skip first batch)
    banim_nims = nims[1:]  # Shape: (num_batches-1, num_neurons) - current gradient
    banim_losses = losses[1:]  # Shape: (num_batches-1,) - current loss
    
    print(f"   📈 Backward differences: {backward_differences.shape} (excluded first batch - noise filtering)")
    print(f"   🎯 Signal quality benefit: Excluded random initialization from batch 1")
    
    # Compute BANIM scores in chunks for memory efficiency
    banim_scores = []
    chunk_size = GPU_CHUNK_SIZE if ENABLE_GPU_ACCELERATION else 1000
    
    for start_idx in range(0, backward_differences.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, backward_differences.shape[0])
        
        # Get chunk data
        chunk_nims = banim_nims[start_idx:end_idx]
        chunk_diffs = backward_differences[start_idx:end_idx]
        chunk_losses = banim_losses[start_idx:end_idx].unsqueeze(-1)  # Broadcast for neuron dimension
        
        # Compute BANIM: NIM × Δa × (1/Loss)
        chunk_banim = chunk_nims * chunk_diffs / chunk_losses
        banim_scores.append(chunk_banim)
    
    # Combine all chunks
    banim_scores = torch.cat(banim_scores, dim=0)  # Shape: (num_batches-1, num_neurons)
    
    # Compute statistical summaries per neuron
    means = torch.mean(banim_scores, dim=0)
    stds = torch.std(banim_scores, dim=0, unbiased=True)
    
    # Chebyshev confidence intervals
    margin = K_VALUE * stds
    lower_bounds = means - margin
    upper_bounds = means + margin
    intervals = torch.stack([lower_bounds, upper_bounds], dim=1)
    
    # Proportion of positive BANIM scores (indicates harmful neurons)
    proportions_above_zero = torch.mean((banim_scores > 0).float(), dim=0)
    
    processing_time = time.time() - start_time
    
    print(f"   ✅ {layer_name} processed in {processing_time:.1f}s")
    print(f"   📊 BANIM range: [{means.min().item():.6f}, {means.max().item():.6f}]")
    print(f"   🎯 Harmful neuron candidates: {(proportions_above_zero > 0.5).sum().item()}/{num_neurons}")
    
    return {
        'banim_scores': banim_scores.cpu().numpy(),
        'means': means.cpu().numpy(),
        'stds': stds.cpu().numpy(),
        'intervals': intervals.cpu().numpy(),
        'proportions_above_zero': proportions_above_zero.cpu().numpy(),
        'num_neurons': num_neurons,
        'num_batches': banim_scores.shape[0],  # Note: reduced by 1 due to backward differences
        'processing_time': processing_time,
    }


def save_banim_scores(all_results: Dict, filename: Path):
    """
    Save Backward Advanced Neuron Impact Metrics to HDF5 file.
    
    File structure:
    /metadata/
    /fc1/banim_scores, intervals, proportions_above_zero, means, stds
    /fc2/...
    /fc3/...
    """
    print(f"💾 Saving Backward Advanced Neuron Impact Metrics: {filename}")
    
    # Ensure directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filename, 'w') as f:
        # Metadata group
        meta_group = f.create_group('metadata')
        meta_group.create_dataset('computation_method', data='backward_advanced_neuron_impact_metric')
        meta_group.create_dataset('formula', data='BANIM_i = NIM_i × Δa_i × (1/Loss)')
        meta_group.create_dataset('temporal_direction', data='backward: Δa_i = a_i(t) - a_i(t-1)')
        meta_group.create_dataset('mathematical_rigor', data='engineering_approximation')
        meta_group.create_dataset('computational_advantage', data='no_extra_epoch_required')
        meta_group.create_dataset('signal_quality', data='excludes_first_batch_initialization_noise')
        meta_group.create_dataset('terminology_nim', data='Neuron Impact Metric (∂L/∂a_i)')
        meta_group.create_dataset('terminology_anim', data='Advanced Neuron Impact Metric (NIM × activations)')
        meta_group.create_dataset('terminology_banim', data='Backward Advanced Neuron Impact Metric (ANIM × backward × normalization)')
        meta_group.create_dataset('confidence_level', data=CONFIDENCE_LEVEL)
        meta_group.create_dataset('alpha', data=ALPHA)
        meta_group.create_dataset('k_value', data=K_VALUE)
        meta_group.create_dataset('device', data=str(DEVICE))
        meta_group.create_dataset('gpu_acceleration', data=ENABLE_GPU_ACCELERATION)
        meta_group.create_dataset('chunk_size', data=GPU_CHUNK_SIZE)
        meta_group.create_dataset('creation_timestamp', data=datetime.now().isoformat())
        meta_group.create_dataset('file_version', data='wine_quality_banim_v1.0')
        meta_group.create_dataset('dataset', data='wine_quality_red')
        
        # Layer results
        total_neurons = 0
        total_batches = 0
        
        for layer_name, results in all_results.items():
            layer_group = f.create_group(layer_name)
            
            # Save all computed BANIM data
            layer_group.create_dataset('banim_scores', data=results['banim_scores'], compression='gzip')
            layer_group.create_dataset('intervals', data=results['intervals'], compression='gzip')
            layer_group.create_dataset('proportions_above_zero', data=results['proportions_above_zero'], compression='gzip')
            layer_group.create_dataset('means', data=results['means'], compression='gzip')
            layer_group.create_dataset('stds', data=results['stds'], compression='gzip')
            
            # Layer metadata
            layer_group.create_dataset('num_neurons', data=results['num_neurons'])
            layer_group.create_dataset('num_batches', data=results['num_batches'])
            layer_group.create_dataset('processing_time', data=results['processing_time'])
            
            total_neurons += results['num_neurons']
            total_batches = results['num_batches']  # Should be same for all layers
        
        # Global statistics
        meta_group.create_dataset('total_neurons', data=total_neurons)
        meta_group.create_dataset('total_batches', data=total_batches)
    
    print(f"✅ BANIM scores saved: {filename}")


def save_processing_metadata(all_results: Dict, total_time: float):
    """Save processing metadata as JSON for easy inspection."""
    metadata = {
        'processing_info': {
            'creation_timestamp': datetime.now().isoformat(),
            'total_processing_time': total_time,
            'device': str(DEVICE),
            'gpu_acceleration': ENABLE_GPU_ACCELERATION,
            'chunk_size': GPU_CHUNK_SIZE if ENABLE_GPU_ACCELERATION else None,
            'method': 'Backward Advanced Neuron Impact Metric (BANIM)',
            'mathematical_rigor': 'Engineering approximation - current gradient explains past changes',
            'computational_advantage': 'No extra epoch required',
            'signal_quality': 'Excludes first batch initialization noise'
        },
        'terminology': {
            'nim': 'Neuron Impact Metric (∂L/∂a_i)',
            'anim': 'Advanced Neuron Impact Metric (NIM × activations)',
            'banim': 'Backward Advanced Neuron Impact Metric (ANIM × backward × normalization)',
            'temporal_direction': 'Backward: Δa_i = a_i(t) - a_i(t-1)',
            'legacy_names': 'Professional NIM terminology replacing older "sensitivity" naming'
        },
        'engineering_tradeoffs': {
            'mathematical_compromise': 'Uses current gradient to explain past activation changes',
            'computational_efficiency': 'Computed during normal training - no 501st epoch needed',
            'signal_quality_benefit': 'Naturally excludes random initialization noise from first batch',
            'empirical_question': 'Does performance match FANIM despite theoretical limitations?'
        },
        'statistical_parameters': {
            'confidence_level': CONFIDENCE_LEVEL,
            'alpha': ALPHA,
            'k_value': K_VALUE,
            'formula': 'BANIM_i = NIM_i × Δa_i × (1/Loss)'
        },
        'layer_summary': {},
        'file_structure': {
            'input_file': str(INPUT_FILE),
            'output_file': str(OUTPUT_FILE),
            'format': 'HDF5 with gzip compression',
            'note': 'Input file contains raw NIM data from wine quality training'
        },
        'forge_config': {
            'nim_data_dir': str(forge_config.nim_data_dir),
            'models_dir': str(forge_config.models_dir)
        }
    }
    
    # Layer summaries
    for layer_name, results in all_results.items():
        metadata['layer_summary'][layer_name] = {
            'num_neurons': int(results['num_neurons']),
            'num_batches': int(results['num_batches']),
            'processing_time': float(results['processing_time']),
            'banim_mean_range': [float(results['means'].min()), float(results['means'].max())],
            'proportion_range': [float(results['proportions_above_zero'].min()), float(results['proportions_above_zero'].max())]
        }
    
    # Ensure directory exists
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ BANIM processing metadata saved: {METADATA_FILE}")


def main():
    """Main BANIM processing pipeline."""
    print("🧮 WINE QUALITY: BACKWARD ADVANCED NEURON IMPACT METRIC PROCESSOR")
    print("=" * 70)
    print("Converting raw neuron impact data → processed BANIM for efficient Bonsai experiments")
    print("")
    print("📊 TERMINOLOGY:")
    print("   NIM   = Neuron Impact Metric (∂L/∂a_i)")
    print("   ANIM  = Advanced Neuron Impact Metric (NIM × activations)")  
    print("   BANIM = Backward Advanced Neuron Impact Metric (ANIM × backward × normalization)")
    print("")
    print("⚖️  ENGINEERING TRADE-OFFS:")
    print("   ✅ Computational efficiency: No extra epoch required")
    print("   ✅ Signal quality: Excludes first batch initialization noise")
    print("   ⚠️  Mathematical compromise: Current gradient explains past changes")
    print("   ❓ Empirical question: Does performance match FANIM?")
    print("")
    print("🔧 TEMPORAL DIRECTION:")
    print("   Backward temporal differences: Δa_i = a_i(t) - a_i(t-1)")
    print("   Engineering approximation for production efficiency")
    print("")
    print(f"Input:  {INPUT_FILE.name}")
    print(f"Output: {OUTPUT_FILE.name}")
    print(f"Device: {DEVICE}")
    
    start_time = time.time()
    
    try:
        # Step 1: Load raw neuron impact data
        print(f"\n📂 STEP 1: Loading raw neuron impact data...")
        raw_data = load_raw_neuron_impact_data(INPUT_FILE)
        
        # Step 2: Process each layer with GPU acceleration
        print(f"\n⚡ STEP 2: Processing Backward Advanced Neuron Impact Metrics...")
        all_results = {}
        
        for layer_name in ['fc1', 'fc2', 'fc3']:
            if layer_name in raw_data and len(raw_data[layer_name]['nims']) > 0:
                layer_results = compute_layer_banim_statistics_gpu(raw_data, layer_name)
                all_results[layer_name] = layer_results
            else:
                print(f"   ⚠️  Skipping {layer_name}: no neuron impact data found")
        
        # Step 3: Save processed BANIM results
        print(f"\n💾 STEP 3: Saving processed Backward Advanced Neuron Impact Metrics...")
        save_banim_scores(all_results, OUTPUT_FILE)
        
        # Step 4: Save metadata
        total_time = time.time() - start_time
        save_processing_metadata(all_results, total_time)
        
        # Summary
        print(f"\n🎉 BACKWARD ADVANCED NEURON IMPACT METRIC PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Total processing time: {total_time:.1f}s")
        print(f"📁 Files created:")
        print(f"   - {OUTPUT_FILE.name} (processed BANIM data)")
        print(f"   - {METADATA_FILE.name} (processing metadata)")
        
        print(f"\n🚀 READY FOR EFFICIENT BONSAI EXPERIMENTS!")
        print("All Bonsai methods can now load pre-computed BANIM scores:")
        print("   - Wilcoxon statistical pruning")
        print("   - Comparative studies vs FANIM")
        print("   - Future Bonsai variants...")
        
        print(f"\n⚖️  BANIM vs FANIM COMPARISON:")
        print("   ✅ BANIM: No extra epoch, excludes noise, production-ready")
        print("   ✅ FANIM: Mathematically rigorous, theoretically correct")
        print("   🔬 Next: Empirical comparison to validate BANIM performance")
        
        # Performance estimate
        total_records = sum(results['num_neurons'] * results['num_batches'] for results in all_results.values())
        print(f"\n⚡ SPEEDUP ESTIMATE:")
        print(f"   Processed {total_records:,} BANIM records in {total_time:.1f}s")
        print(f"   Future Bonsai methods: ~0.1s to load vs {total_time:.1f}s to recompute")
        print(f"   Expected speedup: ~{int(total_time/0.1):,}x faster!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()