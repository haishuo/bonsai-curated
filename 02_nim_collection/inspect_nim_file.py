#!/usr/bin/env python3
"""
HDF5 File Inspector for NIM Data
===============================

Quick diagnostic tool to inspect the structure and contents of the NIM data file.
"""

import h5py
import numpy as np
from pathlib import Path

def inspect_hdf5_file(filename):
    """Inspect the structure and contents of an HDF5 file."""
    print(f"🔍 INSPECTING: {filename}")
    print("=" * 60)
    
    with h5py.File(filename, 'r') as f:
        print("📁 TOP-LEVEL GROUPS:")
        for key in f.keys():
            print(f"   {key}")
        print()
        
        # Inspect metadata
        if 'metadata' in f:
            print("📊 METADATA:")
            meta = f['metadata']
            for attr_name in meta.attrs.keys():
                attr_value = meta.attrs[attr_name]
                print(f"   {attr_name}: {attr_value}")
            print()
        
        # Inspect batch_info
        if 'batch_info' in f:
            print("📈 BATCH_INFO CONTENTS:")
            batch_info = f['batch_info']
            for key in batch_info.keys():
                dataset = batch_info[key]
                print(f"   {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if dataset.size > 0:
                    print(f"      sample values: {dataset[:5] if len(dataset) >= 5 else dataset[:]}")
            print()
        else:
            print("❌ No 'batch_info' group found!")
            print()
        
        # Inspect raw_data structure
        if 'raw_data' in f:
            print("🧠 RAW_DATA STRUCTURE:")
            raw_data = f['raw_data']
            for layer_name in raw_data.keys():
                print(f"   Layer: {layer_name}")
                layer = raw_data[layer_name]
                for dataset_name in layer.keys():
                    dataset = layer[dataset_name]
                    print(f"      {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")
                    
                    # Check for sample values
                    if dataset.size > 0:
                        sample = dataset[0] if len(dataset.shape) > 1 else dataset[:5]
                        print(f"         sample: {sample[:3] if hasattr(sample, '__len__') and len(sample) > 3 else sample}")
            print()
        else:
            print("❌ No 'raw_data' group found!")
            print()
        
        # Look for any loss-related data
        print("🔍 SEARCHING FOR LOSS DATA:")
        def find_loss_data(group, path=""):
            for key in group.keys():
                current_path = f"{path}/{key}" if path else key
                if 'loss' in key.lower():
                    item = group[key]
                    if hasattr(item, 'shape'):  # It's a dataset
                        print(f"   FOUND: {current_path} - shape={item.shape}, dtype={item.dtype}")
                        if item.size > 0:
                            print(f"      sample: {item[:5] if len(item) >= 5 else item[:]}")
                    else:  # It's a group
                        print(f"   FOUND GROUP: {current_path}")
                elif hasattr(group[key], 'keys'):  # It's a subgroup
                    find_loss_data(group[key], current_path)
        
        find_loss_data(f)
        print()
        
        # File size and summary
        import os
        file_size = os.path.getsize(filename) / (1024**3)  # GB
        print(f"📊 FILE SUMMARY:")
        print(f"   Size: {file_size:.2f} GB")
        
        if 'raw_data' in f:
            total_batches = 0
            total_neurons = 0
            for layer_name in f['raw_data'].keys():
                layer = f['raw_data'][layer_name]
                if 'gradients' in layer:
                    shape = layer['gradients'].shape
                    total_batches = max(total_batches, shape[0])
                    total_neurons += shape[1]
            print(f"   Total batches: {total_batches:,}")
            print(f"   Total neurons: {total_neurons:,}")

if __name__ == "__main__":
    # Adjust this path to your actual file
    nim_file = Path("/mnt/data/bonsai/quick/nim_data/wine_quality_raw_nim_data.h5")
    
    if nim_file.exists():
        inspect_hdf5_file(nim_file)
    else:
        print(f"❌ File not found: {nim_file}")
        print("Available files in nim_data directory:")
        nim_dir = nim_file.parent
        if nim_dir.exists():
            for file in nim_dir.glob("*.h5"):
                print(f"   {file.name}")