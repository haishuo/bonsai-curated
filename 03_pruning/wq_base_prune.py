#!/usr/bin/env python3
"""
Wine Quality Base Pruning Functions - EXACT COPY FROM BONSAI-ARCHIVE
===================================================================

PURPOSE: Core functions for wine quality neural network pruning
PHILOSOPHY: Shared utilities for all pruning experiments
COMPATIBILITY: Designed for wine quality dataset with 11 features → 6 classes

Core Functions:
- Model architecture (WineQualityMLP)
- Data loading and preprocessing  
- Structural pruning operations
- Model evaluation and fine-tuning
- FANIM computation for iterative pruning

EXACT COPY: No modifications from bonsai-archive except file paths for forge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# FORGE-SPECIFIC PATHS (only allowed changes)
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shared.forge_config import get_bonsai_config
forge_config = get_bonsai_config()

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture constants (must match wine_quality_ds.py)
INPUT_SIZE = 11
HIDDEN1_SIZE = 256
HIDDEN2_SIZE = 128
HIDDEN3_SIZE = 64
OUTPUT_SIZE = 6

# File paths - FORGE SPECIFIC
MODEL_FILE = forge_config.models_dir / 'wine_quality_trained_500epochs.pth'
SENSITIVITY_FILE = forge_config.nim_data_dir / 'wine_quality_sensitivity.h5'

class WineQualityMLP(nn.Module):
    """Original Wine Quality MLP architecture for loading trained model."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)  
        self.fc3 = nn.Linear(HIDDEN2_SIZE, HIDDEN3_SIZE)
        self.fc4 = nn.Linear(HIDDEN3_SIZE, OUTPUT_SIZE)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        fc1_out = F.relu(self.fc1(x))
        fc1_out = self.dropout(fc1_out)
        fc2_out = F.relu(self.fc2(fc1_out))
        fc2_out = self.dropout(fc2_out)
        fc3_out = F.relu(self.fc3(fc2_out))
        fc3_out = self.dropout(fc3_out)
        return self.fc4(fc3_out)

class WineQualityMLP_Dynamic(nn.Module):
    """Dynamic MLP that can handle pruned architectures."""
    def __init__(self, hidden_sizes: List[int]):
        super().__init__()
        
        # Build architecture dynamically
        layer_sizes = [INPUT_SIZE] + hidden_sizes + [OUTPUT_SIZE]
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)

def load_wine_quality_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Load and preprocess Wine Quality dataset."""
    print("📂 Loading Wine Quality dataset...")
    
    # Fetch wine quality dataset
    wine = fetch_openml('wine-quality-red', version=1, as_frame=True)
    X, y = wine.data, wine.target
    
    # Convert target to integers and adjust to 0-based indexing
    y = y.astype(int) - 3  # Original quality is 3-8, convert to 0-5
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train.values)
    y_test_tensor = torch.LongTensor(y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    return train_loader, test_loader, scaler

def load_trained_model():
    """Load trained wine quality model."""
    print(f"📂 Loading trained model: {MODEL_FILE}")
    
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}. Run wine_quality_ds.py first.")
    
    # Load model
    model = WineQualityMLP()
    
    # Try to load with metadata, fallback to direct loading
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   📋 Loaded model with metadata")
        else:
            model.load_state_dict(checkpoint)
            print(f"   📋 Loaded model (direct state dict)")
    except Exception as e:
        print(f"   ⚠️  Error loading model: {e}")
        return None
    
    model.to(DEVICE)
    model.eval()
    
    total_params = count_parameters(model)
    print(f"✅ Model loaded: {total_params:,} parameters")
    
    return model

def apply_structural_pruning(model: WineQualityMLP, pruning_decisions: Dict[str, List[str]]) -> WineQualityMLP_Dynamic:
    """Apply structural pruning based on pruning decisions."""
    print(f"✂️  Applying structural pruning...")
    
    # Convert string decisions to indices for each layer
    layer_decisions = {}
    
    for layer_name, decisions in pruning_decisions.items():
        if isinstance(decisions[0], str):
            # Convert 'PRUNE'/'KEEP' to boolean indices
            prune_indices = [i for i, decision in enumerate(decisions) if decision == 'PRUNE']
        else:
            # Already indices
            prune_indices = decisions
        
        layer_decisions[layer_name] = prune_indices
        print(f"   {layer_name}: Pruning {len(prune_indices)} neurons")
    
    # Calculate new layer sizes
    original_sizes = [HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN3_SIZE]
    new_sizes = []
    
    for i, layer_name in enumerate(['fc1', 'fc2', 'fc3']):
        original_size = original_sizes[i]
        prune_indices = layer_decisions.get(layer_name, [])
        new_size = original_size - len(prune_indices)
        new_sizes.append(max(1, new_size))  # Ensure at least 1 neuron remains
        print(f"   {layer_name}: {original_size} → {new_size} neurons")
    
    # Create new model with pruned architecture
    pruned_model = WineQualityMLP_Dynamic(new_sizes)
    pruned_model.to(DEVICE)
    
    # Copy weights from original model (excluding pruned neurons)
    with torch.no_grad():
        # Layer mappings
        layer_mapping = [
            (model.fc1, pruned_model.layers[0], layer_decisions.get('fc1', [])),
            (model.fc2, pruned_model.layers[1], layer_decisions.get('fc2', [])),
            (model.fc3, pruned_model.layers[2], layer_decisions.get('fc3', [])),
            (model.fc4, pruned_model.layers[3], [])  # Output layer not pruned
        ]
        
        for i, (old_layer, new_layer, prune_indices) in enumerate(layer_mapping):
            # Get keep indices (complement of prune indices)
            if i < 3:  # Hidden layers
                all_indices = set(range(original_sizes[i]))
                keep_indices = sorted(list(all_indices - set(prune_indices)))
            else:  # Output layer
                keep_indices = list(range(old_layer.out_features))
            
            # Copy weights and biases for kept neurons
            if i == 0:  # First layer: copy all input connections to kept output neurons
                new_layer.weight.data = old_layer.weight.data[keep_indices, :]
                new_layer.bias.data = old_layer.bias.data[keep_indices]
            else:  # Other layers: copy connections from kept input neurons to kept output neurons
                prev_keep_indices = layer_mapping[i-1][2] if i-1 < 3 else list(range(original_sizes[i-1]))
                if i-1 < 3:
                    prev_all_indices = set(range(original_sizes[i-1]))
                    prev_keep_indices = sorted(list(prev_all_indices - set(prev_keep_indices)))
                
                new_layer.weight.data = old_layer.weight.data[keep_indices, :][:, prev_keep_indices]
                new_layer.bias.data = old_layer.bias.data[keep_indices]
    
    return pruned_model

def fine_tune_model(model: nn.Module, train_loader: DataLoader, epochs: int = 10) -> nn.Module:
    """Fine-tune model after pruning."""
    print(f"🔧 Fine-tuning model for {epochs} epochs...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 2 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    model.eval()
    print("✅ Fine-tuning complete")
    
    return model

def evaluate_model_performance(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_sparsity(original_params: int, current_params: int) -> float:
    """Calculate model sparsity."""
    return (original_params - current_params) / original_params

def measure_inference_time(model: nn.Module, test_loader: DataLoader, num_samples: int = 100) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            data = data.to(DEVICE)
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times)

def load_fanim_scores_for_wilcoxon(filename: Path = None) -> Dict[str, np.ndarray]:
    """
    Load pre-computed FANIM scores for Wilcoxon testing.
    
    Returns only the raw FANIM scores that Wilcoxon tests need.
    
    Returns:
        Dictionary mapping layer names to FANIM score arrays [num_batches, num_neurons]
    """
    if filename is None:
        filename = forge_config.nim_data_dir / 'wine_quality_fanim.h5'
    
    print(f"⚡ Loading FANIM scores for Wilcoxon testing: {filename}")
    
    if not Path(filename).exists():
        raise FileNotFoundError(f"FANIM file not found: {filename}. Run wq_compute_fanim.py first.")
    
    fanim_scores = {}
    
    with h5py.File(filename, 'r') as f:
        # Load metadata for verification
        meta = f['metadata']
        print(f"   📊 FANIM method: {meta['computation_method'][()].decode()}")
        
        # Load raw FANIM scores for Wilcoxon testing
        for layer_name in ['fc1', 'fc2', 'fc3']:
            if layer_name in f:
                layer_group = f[layer_name]
                
                # Only load what Wilcoxon needs: raw FANIM scores
                fanim_scores[layer_name] = layer_group['fanim_scores'][:]  # [num_batches, num_neurons]
                
                num_batches, num_neurons = fanim_scores[layer_name].shape
                print(f"   ✅ {layer_name}: {num_neurons} neurons, {num_batches:,} FANIM records")
            else:
                print(f"   ⚠️  No FANIM data found for {layer_name}")
    
    total_neurons = sum(scores.shape[1] for scores in fanim_scores.values())
    total_records = sum(scores.shape[0] * scores.shape[1] for scores in fanim_scores.values())
    
    print(f"✅ FANIM scores loaded for Wilcoxon: {total_neurons} neurons, {total_records:,} records")
    
    return fanim_scores

class BatchFANIMCollector:
    """
    Collect FANIM scores during training for current model structure.
    This is needed for iterative pruning to get fresh FANIM scores.
    """
    
    def __init__(self, model):
        self.model = model
        self.fanim_data = []
        self.current_batch_data = {}
        self.hooks = []
        
        # Counters
        self.batches_collected = 0
        
        print("🪝 Setting up FANIM collection hooks...")
        self._setup_gradient_hooks()
    
    def _setup_gradient_hooks(self):
        """Register hooks to collect gradients during backpropagation."""
        
        def create_gradient_hook(layer_name):
            def hook_fn(grad):
                """Store raw gradient (NIMs for FANIM computation)"""
                self.current_batch_data[f'{layer_name}_gradient'] = grad.clone().detach()
            return hook_fn
        
        # Hook bias gradients (represent per-neuron sensitivity)
        hook1 = self.model.fc1.bias.register_hook(create_gradient_hook('fc1'))
        hook2 = self.model.fc2.bias.register_hook(create_gradient_hook('fc2'))
        hook3 = self.model.fc3.bias.register_hook(create_gradient_hook('fc3'))
        
        self.hooks = [hook1, hook2, hook3]
        print(f"✅ FANIM hooks registered on fc1.bias, fc2.bias, fc3.bias")
    
    def collect_batch_fanim(self, loss):
        """Collect FANIM data for current batch."""
        # Store activations (mean across batch dimension)
        activations = {
            'fc1': self.model.current_activations.get('fc1', torch.zeros(1)).detach().mean(dim=0) if hasattr(self.model, 'current_activations') else torch.zeros(self.model.fc1.out_features),
            'fc2': self.model.current_activations.get('fc2', torch.zeros(1)).detach().mean(dim=0) if hasattr(self.model, 'current_activations') else torch.zeros(self.model.fc2.out_features),
            'fc3': self.model.current_activations.get('fc3', torch.zeros(1)).detach().mean(dim=0) if hasattr(self.model, 'current_activations') else torch.zeros(self.model.fc3.out_features)
        }
        
        # Create batch record
        batch_record = {
            'batch_loss': loss.item(),
            'fc1_gradient': self.current_batch_data.get('fc1_gradient', torch.zeros(self.model.fc1.out_features)).cpu().numpy(),
            'fc2_gradient': self.current_batch_data.get('fc2_gradient', torch.zeros(self.model.fc2.out_features)).cpu().numpy(),
            'fc3_gradient': self.current_batch_data.get('fc3_gradient', torch.zeros(self.model.fc3.out_features)).cpu().numpy(),
            'fc1_activation': activations['fc1'].cpu().numpy(),
            'fc2_activation': activations['fc2'].cpu().numpy(),
            'fc3_activation': activations['fc3'].cpu().numpy(),
        }
        
        self.fanim_data.append(batch_record)
        self.batches_collected += 1
    
    def compute_fanim_scores(self) -> Dict[str, np.ndarray]:
        """
        Compute FANIM scores from collected data.
        
        Formula: FANIM_i = NIM_i × Δa_i × (1/Loss)
        Where Δa_i = a_i(t+1) - a_i(t) (forward direction)
        
        Returns:
            Dictionary mapping layer names to FANIM score arrays
        """
        print(f"🧮 Computing FANIM scores from {self.batches_collected} collected batches...")
        
        if len(self.fanim_data) < 2:
            raise ValueError("Need at least 2 batches to compute forward FANIM (for deltas)")
        
        fanim_scores = {}
        
        for layer_name in ['fc1', 'fc2', 'fc3']:
            # Extract data arrays
            gradients = np.array([batch[f'{layer_name}_gradient'] for batch in self.fanim_data])
            activations = np.array([batch[f'{layer_name}_activation'] for batch in self.fanim_data])
            losses = np.array([batch['batch_loss'] for batch in self.fanim_data])
            
            num_batches, num_neurons = gradients.shape
            
            # Compute forward deltas: Δa_i = a_i(t+1) - a_i(t)
            forward_deltas = activations[1:] - activations[:-1]  # [num_batches-1, num_neurons]
            current_gradients = gradients[:-1]  # [num_batches-1, num_neurons] 
            current_losses = losses[:-1]  # [num_batches-1]
            
            # Advanced FANIM: NIM × Δa_i × (1/Loss)
            raw_fanim = current_gradients * forward_deltas
            
            # Normalize by loss (avoid division by zero)
            safe_losses = np.maximum(current_losses, 1e-8)
            normalized_fanim = raw_fanim / safe_losses.reshape(-1, 1)
            
            fanim_scores[layer_name] = normalized_fanim
            
            print(f"   ✅ {layer_name}: {normalized_fanim.shape} FANIM scores computed")
        
        return fanim_scores
    
    def cleanup_hooks(self):
        """Remove hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        print("🧹 FANIM collection hooks cleaned up")

def compute_fanim_from_model(model: nn.Module, 
                           train_loader, 
                           num_batches: int = 50,
                           disable_dropout: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute fresh FANIM scores for current model structure.
    
    This is crucial for iterative pruning - we need FANIM scores that match
    the current pruned model, not the original full model.
    
    Args:
        model: Current model (potentially pruned)
        train_loader: Training data for FANIM computation
        num_batches: Number of batches to use for FANIM computation
        disable_dropout: Whether to disable dropout during FANIM collection
        
    Returns:
        Dictionary mapping layer names to FANIM score arrays [num_batches, num_neurons]
    """
    print(f"🧮 Computing fresh FANIM scores for current model structure...")
    print(f"   Target batches: {num_batches}")
    print(f"   Disable dropout: {disable_dropout}")
    
    # Set model mode
    original_training_mode = model.training
    if disable_dropout:
        model.eval()  # Disable dropout for cleaner FANIM signals
        print(f"   📊 Model set to eval mode (dropout disabled)")
    else:
        model.train()
        print(f"   📊 Model set to train mode (dropout active)")
    
    # Setup FANIM collector
    collector = BatchFANIMCollector(model)
    
    # Setup optimizer and criterion for gradient computation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    try:
        batches_processed = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batches_processed >= num_batches:
                break
                
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass to collect gradients
            loss.backward()
            
            # Collect FANIM data
            collector.collect_batch_fanim(loss)
            
            batches_processed += 1
            
            if batches_processed % 10 == 0:
                print(f"   Processed {batches_processed}/{num_batches} batches...")
        
        print(f"✅ FANIM data collection complete: {batches_processed} batches")
        
        # Compute FANIM scores
        fanim_scores = collector.compute_fanim_scores()
        
        return fanim_scores
        
    finally:
        # Clean up hooks and restore model mode
        collector.cleanup_hooks()
        model.train(original_training_mode)
        print(f"🔄 Model training mode restored: {original_training_mode}")