#!/usr/bin/env python3
"""
Wine Quality: Forward Advanced Neuron Impact Metric Processor
============================================================

PURPOSE: Process raw neuron impact data into Forward Advanced Neuron Impact Metrics (FANIM)
PHILOSOPHY: Compute once, use many times - avoid recalculating expensive metrics
WORKFLOW: wine_quality_sensitivity.h5 → GPU processing → fanim.h5

INPUT:  Raw batch-level neuron impact data from wine_quality_ds.py training
OUTPUT: Processed Forward Advanced Neuron Impact Metrics ready for all Bonsai methods
BENEFIT: 10-100x speedup for subsequent Bonsai experiments

Forward Advanced Neuron Impact Metric Formula:
FANIM_i = (∂L/∂a_i) × Δa_i × (1/Loss)
Where: Δa_i = a_i(t+1) - a_i(t)  [forward temporal difference]

Terminology:
- NIM: Neuron Impact Metric (basic ∂L/∂a_i)
- ANIM: Advanced Neuron Impact Metric (NIM × activations)  
- FANIM: Forward Advanced Neuron Impact Metric (ANIM × forward direction × loss normalization)

Mathematical Rigor: Uses theoretically correct forward temporal differences
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
OUTPUT_FILE = forge_config.nim_data_dir / 'wine_quality_fanim.h5'        # FANIM output
METADATA_FILE = forge_config.nim_data_dir / 'wine_quality_fanim_metadata.json'


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
                        
                        # Handle FANIM collection: activations should include epoch 501 data for forward differences
                        if activations.shape[0] == gradients.shape[0] + 1:
                            print(f"   ✅ {layer_name}: Perfect FANIM setup - {gradients.shape[0]} gradient batches + 1 FANIM activation batch")
                            print(f"   🎯 Can compute FANIM for all {gradients.shape[0]} training batches including final epoch")
                        elif activations.shape[0] > gradients.shape[0]:
                            print(f"   ✅ {layer_name}: FANIM data available - {gradients.shape[0]} gradient batches, {activations.shape[0]} activation batches")
                            print(f"   🎯 Will use extra activation data for forward differences")
                        else:
                            print(f"   ⚠️  {layer_name}: Missing FANIM data - activations={activations.shape[0]}, gradients={gradients.shape[0]}")
                        
                        print(f"   📊 Data: {gradients.shape[1]} neurons, {gradients.shape[0]} training batches")
                        
                        data[layer_name]['nims'] = gradients
                        data[layer_name]['activations'] = activations  # Keep ALL activations including epoch 501
                        
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


def compute_layer_fanim_statistics_gpu(raw_data: Dict, layer_name: str) -> Dict:
    """
    Compute Forward Advanced Neuron Impact Metrics and statistical summaries using GPU acceleration.
    
    FANIM Formula: FANIM_i = NIM_i × Δa_i × (1/Loss)
    Where: Δa_i = a_i(t+1) - a_i(t)  [forward temporal difference]
    
    Mathematical Note: This uses theoretically correct forward differences.
    """
    print(f"⚡ Processing {layer_name} with Forward Advanced Neuron Impact Metrics...")
    
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
    
    # Compute forward temporal differences: Δa_i = a_i(t+1) - a_i(t)
    # Key insight: We need activations[t+1] for each gradient[t]
    if activations.shape[0] <= nims.shape[0]:
        raise ValueError(f"Need at least {nims.shape[0] + 1} activation batches for forward differences, got {activations.shape[0]}")
    
    # Forward differences: use activations[1:N+1] - activations[0:N] 
    # This gives us differences for batches 0, 1, 2, ..., N-1 (matching our gradients)
    forward_differences = activations[1:nims.shape[0]+1] - activations[:nims.shape[0]]  # Shape: (num_batches, num_neurons)
    
    # Use all gradient and loss data (they already match)
    fanim_nims = nims  # Shape: (num_batches, num_neurons)
    fanim_losses = losses  # Shape: (num_batches,)
    
    print(f"   📈 Forward differences: {forward_differences.shape} (using epoch 501 data)")
    print(f"   🎯 Perfect alignment: {fanim_nims.shape[0]} gradients, {forward_differences.shape[0]} forward diffs")
    
    # Compute FANIM scores in chunks for memory efficiency
    fanim_scores = []
    chunk_size = GPU_CHUNK_SIZE if ENABLE_GPU_ACCELERATION else 1000
    
    for start_idx in range(0, forward_differences.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, forward_differences.shape[0])
        
        # Get chunk data
        chunk_nims = fanim_nims[start_idx:end_idx]
        chunk_diffs = forward_differences[start_idx:end_idx]
        chunk_losses = fanim_losses[start_idx:end_idx].unsqueeze(-1)  # Broadcast for neuron dimension
        
        # Compute FANIM: NIM × Δa × (1/Loss)
        chunk_fanim = chunk_nims * chunk_diffs / chunk_losses
        fanim_scores.append(chunk_fanim)
    
    # Combine all chunks
    fanim_scores = torch.cat(fanim_scores, dim=0)  # Shape: (num_batches-1, num_neurons)
    
    # Compute statistical summaries per neuron
    means = torch.mean(fanim_scores, dim=0)
    stds = torch.std(fanim_scores, dim=0, unbiased=True)
    
    # Chebyshev confidence intervals
    margin = K_VALUE * stds
    lower_bounds = means - margin
    upper_bounds = means + margin
    intervals = torch.stack([lower_bounds, upper_bounds], dim=1)
    
    # Proportion of positive FANIM scores (indicates harmful neurons)
    proportions_above_zero = torch.mean((fanim_scores > 0).float(), dim=0)
    
    processing_time = time.time() - start_time
    
    print(f"   ✅ {layer_name} processed in {processing_time:.1f}s")
    print(f"   📊 FANIM range: [{means.min().item():.6f}, {means.max().item():.6f}]")
    print(f"   🎯 Harmful neuron candidates: {(proportions_above_zero > 0.5).sum().item()}/{num_neurons}")
    
    return {
        'fanim_scores': fanim_scores.cpu().numpy(),
        'means': means.cpu().numpy(),
        'stds': stds.cpu().numpy(),
        'intervals': intervals.cpu().numpy(),
        'proportions_above_zero': proportions_above_zero.cpu().numpy(),
        'num_neurons': num_neurons,
        'num_batches': fanim_scores.shape[0],  # Note: reduced by 1 due to forward differences
        'processing_time': processing_time,
    }


def save_fanim_scores(all_results: Dict, filename: Path):
    """
    Save Forward Advanced Neuron Impact Metrics to HDF5 file.
    
    File structure:
    /metadata/
    /fc1/fanim_scores, intervals, proportions_above_zero, means, stds
    /fc2/...
    /fc3/...
    """
    print(f"💾 Saving Forward Advanced Neuron Impact Metrics: {filename}")
    
    # Ensure directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filename, 'w') as f:
        # Metadata group
        meta_group = f.create_group('metadata')
        meta_group.create_dataset('computation_method', data='forward_advanced_neuron_impact_metric')
        meta_group.create_dataset('formula', data='FANIM_i = NIM_i × Δa_i × (1/Loss)')
        meta_group.create_dataset('temporal_direction', data='forward: Δa_i = a_i(t+1) - a_i(t)')
        meta_group.create_dataset('mathematical_rigor', data='theoretically_correct')
        meta_group.create_dataset('terminology_nim', data='Neuron Impact Metric (∂L/∂a_i)')
        meta_group.create_dataset('terminology_anim', data='Advanced Neuron Impact Metric (NIM × activations)')
        meta_group.create_dataset('terminology_fanim', data='Forward Advanced Neuron Impact Metric (ANIM × forward × normalization)')
        meta_group.create_dataset('confidence_level', data=CONFIDENCE_LEVEL)
        meta_group.create_dataset('alpha', data=ALPHA)
        meta_group.create_dataset('k_value', data=K_VALUE)
        meta_group.create_dataset('device', data=str(DEVICE))
        meta_group.create_dataset('gpu_acceleration', data=ENABLE_GPU_ACCELERATION)
        meta_group.create_dataset('chunk_size', data=GPU_CHUNK_SIZE)
        meta_group.create_dataset('creation_timestamp', data=datetime.now().isoformat())
        meta_group.create_dataset('file_version', data='wine_quality_fanim_v1.0')
        meta_group.create_dataset('dataset', data='wine_quality_red')
        
        # Layer results
        total_neurons = 0
        total_batches = 0
        
        for layer_name, results in all_results.items():
            layer_group = f.create_group(layer_name)
            
            # Save all computed FANIM data
            layer_group.create_dataset('fanim_scores', data=results['fanim_scores'], compression='gzip')
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
    
    print(f"✅ FANIM scores saved: {filename}")


def save_processing_metadata(all_results: Dict, total_time: float):
    """Save processing metadata as JSON for easy inspection."""
    metadata = {
        'processing_info': {
            'creation_timestamp': datetime.now().isoformat(),
            'total_processing_time': total_time,
            'device': str(DEVICE),
            'gpu_acceleration': ENABLE_GPU_ACCELERATION,
            'chunk_size': GPU_CHUNK_SIZE if ENABLE_GPU_ACCELERATION else None,
            'method': 'Forward Advanced Neuron Impact Metric (FANIM)',
            'mathematical_rigor': 'Theoretically correct forward temporal differences'
        },
        'terminology': {
            'nim': 'Neuron Impact Metric (∂L/∂a_i)',
            'anim': 'Advanced Neuron Impact Metric (NIM × activations)',
            'fanim': 'Forward Advanced Neuron Impact Metric (ANIM × forward × normalization)',
            'temporal_direction': 'Forward: Δa_i = a_i(t+1) - a_i(t)',
            'legacy_names': 'Professional NIM terminology replacing older "sensitivity" naming'
        },
        'statistical_parameters': {
            'confidence_level': CONFIDENCE_LEVEL,
            'alpha': ALPHA,
            'k_value': K_VALUE,
            'formula': 'FANIM_i = NIM_i × Δa_i × (1/Loss)'
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
            'fanim_mean_range': [float(results['means'].min()), float(results['means'].max())],
            'proportion_range': [float(results['proportions_above_zero'].min()), float(results['proportions_above_zero'].max())]
        }
    
    # Ensure directory exists
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ FANIM processing metadata saved: {METADATA_FILE}")


def main():
    """Main FANIM processing pipeline."""
    print("🧮 WINE QUALITY: FORWARD ADVANCED NEURON IMPACT METRIC PROCESSOR")
    print("=" * 70)
    print("Converting raw neuron impact data → processed FANIM for fast Bonsai experiments")
    print("")
    print("📊 TERMINOLOGY:")
    print("   NIM   = Neuron Impact Metric (∂L/∂a_i)")
    print("   ANIM  = Advanced Neuron Impact Metric (NIM × activations)")  
    print("   FANIM = Forward Advanced Neuron Impact Metric (ANIM × forward × normalization)")
    print("")
    print("🔬 MATHEMATICAL RIGOR:")
    print("   Forward temporal differences: Δa_i = a_i(t+1) - a_i(t)")
    print("   Theoretically correct Taylor series approximation")
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
        print(f"\n⚡ STEP 2: Processing Forward Advanced Neuron Impact Metrics...")
        all_results = {}
        
        for layer_name in ['fc1', 'fc2', 'fc3']:
            if layer_name in raw_data and len(raw_data[layer_name]['nims']) > 0:
                layer_results = compute_layer_fanim_statistics_gpu(raw_data, layer_name)
                all_results[layer_name] = layer_results
            else:
                print(f"   ⚠️  Skipping {layer_name}: no neuron impact data found")
        
        # Step 3: Save processed FANIM results
        print(f"\n💾 STEP 3: Saving processed Forward Advanced Neuron Impact Metrics...")
        save_fanim_scores(all_results, OUTPUT_FILE)
        
        # Step 4: Save metadata
        total_time = time.time() - start_time
        save_processing_metadata(all_results, total_time)
        
        # Summary
        print(f"\n🎉 FORWARD ADVANCED NEURON IMPACT METRIC PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Total processing time: {total_time:.1f}s")
        print(f"📁 Files created:")
        print(f"   - {OUTPUT_FILE.name} (processed FANIM data)")
        print(f"   - {METADATA_FILE.name} (processing metadata)")
        
        print(f"\n🚀 READY FOR FAST BONSAI EXPERIMENTS!")
        print("All Bonsai methods can now load pre-computed FANIM scores instead of recomputing:")
        print("   - Wilcoxon statistical pruning")
        print("   - Future Bonsai variants...")
        
        # Performance estimate
        total_records = sum(results['num_neurons'] * results['num_batches'] for results in all_results.values())
        print(f"\n⚡ SPEEDUP ESTIMATE:")
        print(f"   Processed {total_records:,} FANIM records in {total_time:.1f}s")
        print(f"   Future Bonsai methods: ~0.1s to load vs {total_time:.1f}s to recompute")
        print(f"   Expected speedup: ~{int(total_time/0.1):,}x faster!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()