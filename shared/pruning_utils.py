#!/usr/bin/env python3
"""
Shared Pruning Utilities
========================

PURPOSE: Common utilities for all pruning methods
PHILOSOPHY: DRY - Don't Repeat Yourself. Shared infrastructure for all pruning experiments.

Core components:
- Wine Quality model architecture
- Data loading and preprocessing
- Structural pruning operations
- Model evaluation and fine-tuning
- Results management

Used by: All pruning methods (Wilcoxon, random, magnitude, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

# Import forge config
from .forge_config import get_bonsai_config

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize forge configuration
forge_config = get_bonsai_config()

@dataclass
class PruningResults:
    """Standardized results from any pruning method."""
    method_name: str
    baseline_accuracy: float
    pruned_accuracy: float
    final_accuracy: float
    baseline_params: int
    pruned_params: int
    sparsity: float
    neurons_pruned: Dict[str, int]
    processing_time: float
    statistical_info: Optional[Dict] = None
    
    @property
    def accuracy_change(self) -> float:
        """Accuracy change from baseline to final."""
        return self.final_accuracy - self.baseline_accuracy
    
    @property
    def recovery_amount(self) -> float:
        """Recovery from pruning to final (fine-tuning effect)."""
        return self.final_accuracy - self.pruned_accuracy

# ===== MODEL ARCHITECTURES =====

class WineQualityMLP(nn.Module):
    """Standard Wine Quality MLP architecture."""
    
    def __init__(self, input_size=11, hidden1=256, hidden2=128, hidden3=64, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)
    
    def get_layer_sizes(self) -> Dict[str, int]:
        """Get the size of each layer."""
        return {
            'fc1': self.fc1.out_features,
            'fc2': self.fc2.out_features,
            'fc3': self.fc3.out_features
        }

# ===== DATA LOADING =====

def load_wine_quality_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Load and preprocess Wine Quality dataset.
    
    Returns:
        Tuple of (train_loader, test_loader, scaler)
    """
    print("📂 Loading Wine Quality dataset...")
    
    # Load red wine quality dataset
    wine = fetch_openml('wine-quality-red', version=1, as_frame=True)
    X, y = wine.data, wine.target
    
    # Convert quality to class indices (3-8 → 0-5)
    y_mapped = y.astype(int) - 3
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train.values)
    y_test_tensor = torch.LongTensor(y_test.values)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    return train_loader, test_loader, scaler

def load_trained_model() -> nn.Module:
    """Load the trained Wine Quality model using forge config."""
    # Try different possible model names
    possible_names = [
        "wine_quality_model",
        "wine_quality_trained_500epochs", 
        "wine_quality_trained",
        "wine_quality"
    ]
    
    model_file = None
    for name in possible_names:
        candidate = forge_config.model_path(name, "pth")
        if candidate.exists():
            model_file = candidate
            break
    
    if model_file is None:
        print(f"📂 Searching for trained model in: {forge_config.models_dir}")
        # List all .pth files in models directory
        pth_files = list(forge_config.models_dir.glob("*.pth"))
        if pth_files:
            # Use the first .pth file found
            model_file = pth_files[0]
            print(f"🔄 Found model file: {model_file}")
        else:
            raise FileNotFoundError(f"No .pth model files found in {forge_config.models_dir}")
    
    print(f"📂 Loading trained model: {model_file}")
    
    # Load the model file
    model = WineQualityMLP()
    checkpoint = torch.load(model_file, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   📋 Loaded model with metadata")
    else:
        model.load_state_dict(checkpoint)
        print(f"   📋 Loaded raw model state_dict")
    
    model.to(DEVICE)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"✅ Model loaded: {total_params:,} parameters")
    
    return model

# ===== FANIM/BANIM DATA LOADING =====

def load_fanim_data() -> Dict[str, np.ndarray]:
    """Load FANIM scores from HDF5 file using forge config."""
    fanim_file = forge_config.nim_data_path("wine_quality", "fanim")
    
    if not fanim_file.exists():
        raise FileNotFoundError(f"FANIM file not found: {fanim_file}")
    
    fanim_data = {}
    with h5py.File(fanim_file, 'r') as f:
        for layer_name in ['fc1', 'fc2', 'fc3']:
            if layer_name in f:
                layer_group = f[layer_name]
                # Load the actual FANIM scores [num_batches, num_neurons]
                fanim_data[layer_name] = layer_group['fanim_scores'][:]
    
    print(f"✅ FANIM data loaded from {fanim_file}")
    return fanim_data

def load_banim_data() -> Dict[str, np.ndarray]:
    """Load BANIM scores from HDF5 file using forge config."""
    banim_file = forge_config.nim_data_path("wine_quality", "banim")
    
    if not banim_file.exists():
        raise FileNotFoundError(f"BANIM file not found: {banim_file}")
    
    banim_data = {}
    with h5py.File(banim_file, 'r') as f:
        for layer_name in ['fc1', 'fc2', 'fc3']:
            if layer_name in f:
                layer_group = f[layer_name]
                # Load the actual BANIM scores [num_batches, num_neurons]
                banim_data[layer_name] = layer_group['banim_scores'][:]
    
    print(f"✅ BANIM data loaded from {banim_file}")
    return banim_data

# ===== MODEL EVALUATION =====

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
    """Evaluate model accuracy and loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss

def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())

def get_model_sparsity(baseline_params: int, pruned_params: int) -> float:
    """Calculate sparsity ratio."""
    return (baseline_params - pruned_params) / baseline_params

# ===== STRUCTURAL PRUNING =====

def apply_structural_pruning(model: nn.Module, prune_decisions: Dict[str, List[int]]) -> nn.Module:
    """
    Apply structural pruning by removing specified neurons.
    
    Args:
        model: Original model
        prune_decisions: Dict mapping layer names to lists of neuron indices to prune
        
    Returns:
        New pruned model with reduced architecture
    """
    original_sizes = model.get_layer_sizes()
    
    # Calculate new layer sizes
    new_sizes = {}
    for layer_name in ['fc1', 'fc2', 'fc3']:
        prune_indices = prune_decisions.get(layer_name, [])
        new_sizes[layer_name] = original_sizes[layer_name] - len(prune_indices)
    
    print(f"🔧 Structural pruning:")
    for layer_name in ['fc1', 'fc2', 'fc3']:
        pruned_count = len(prune_decisions.get(layer_name, []))
        print(f"   {layer_name}: {original_sizes[layer_name]} → {new_sizes[layer_name]} (-{pruned_count})")
    
    # Create new model with reduced sizes
    pruned_model = WineQualityMLP(
        input_size=11,
        hidden1=new_sizes['fc1'],
        hidden2=new_sizes['fc2'], 
        hidden3=new_sizes['fc3'],
        num_classes=6
    )
    
    # Transfer weights (excluding pruned neurons)
    model_dict = model.state_dict()
    pruned_dict = {}
    
    for layer_name in ['fc1', 'fc2', 'fc3']:
        prune_indices = set(prune_decisions.get(layer_name, []))
        
        # Get weight and bias for this layer
        weight_key = f'{layer_name}.weight'
        bias_key = f'{layer_name}.bias'
        
        old_weight = model_dict[weight_key]
        old_bias = model_dict[bias_key]
        
        # Create masks for keeping neurons
        keep_indices = [i for i in range(old_weight.size(0)) if i not in prune_indices]
        
        # Prune output dimension (rows)
        new_weight = old_weight[keep_indices, :]
        new_bias = old_bias[keep_indices]
        
        pruned_dict[weight_key] = new_weight
        pruned_dict[bias_key] = new_bias
    
    # Handle connections between layers
    # fc2 input needs to match fc1 output
    if 'fc1' in prune_decisions and prune_decisions['fc1']:
        fc1_keep = [i for i in range(original_sizes['fc1']) if i not in prune_decisions['fc1']]
        pruned_dict['fc2.weight'] = pruned_dict['fc2.weight'][:, fc1_keep]
    
    # fc3 input needs to match fc2 output  
    if 'fc2' in prune_decisions and prune_decisions['fc2']:
        fc2_keep = [i for i in range(original_sizes['fc2']) if i not in prune_decisions['fc2']]
        pruned_dict['fc3.weight'] = pruned_dict['fc3.weight'][:, fc2_keep]
    
    # fc4 input needs to match fc3 output
    if 'fc3' in prune_decisions and prune_decisions['fc3']:
        fc3_keep = [i for i in range(original_sizes['fc3']) if i not in prune_decisions['fc3']]
        fc4_weight = model_dict['fc4.weight'][:, fc3_keep]
        pruned_dict['fc4.weight'] = fc4_weight
        pruned_dict['fc4.bias'] = model_dict['fc4.bias']
    else:
        pruned_dict['fc4.weight'] = model_dict['fc4.weight']
        pruned_dict['fc4.bias'] = model_dict['fc4.bias']
    
    pruned_model.load_state_dict(pruned_dict)
    return pruned_model

# ===== FINE-TUNING =====

def fine_tune_model(model: nn.Module, train_loader: DataLoader, epochs: int = 10) -> nn.Module:
    """Fine-tune the pruned model."""
    print(f"🔧 Fine-tuning for {epochs} epochs...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / len(train_loader)
            print(f"   Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
    
    return model

# ===== RESULTS MANAGEMENT =====

def save_pruning_results(results: PruningResults, experiment_name: str) -> Path:
    """Save pruning results using forge config."""
    results_file = forge_config.results_path(experiment_name)
    
    # Convert statistical_info to JSON-serializable format
    statistical_info = results.statistical_info
    if statistical_info is not None:
        statistical_info = make_json_serializable(statistical_info)
    
    results_data = {
        'method': results.method_name,
        'baseline_accuracy': results.baseline_accuracy,
        'pruned_accuracy': results.pruned_accuracy,
        'final_accuracy': results.final_accuracy,
        'accuracy_change': results.accuracy_change,
        'recovery_amount': results.recovery_amount,
        'sparsity': results.sparsity,
        'params_before': results.baseline_params,
        'params_after': results.pruned_params,
        'neurons_pruned': results.neurons_pruned,
        'processing_time': results.processing_time,
        'statistical_info': statistical_info,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Results saved: {results_file}")
    return results_file

def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if obj is None:
        return None
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dictionaries
        return {key: make_json_serializable(value) for key, value in obj.__dict__.items()}
    elif hasattr(obj, '_asdict'):
        # Handle namedtuples
        return make_json_serializable(obj._asdict())
    else:
        # Fallback: convert to string
        return str(obj)

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

# ===== UTILITY CLASSES =====

class BasePruner:
    """Base class for all pruning methods."""
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, train_loader: DataLoader = None):
        self.original_model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        
    def prune(self) -> PruningResults:
        """Execute pruning. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prune() method")
    
    def _create_pruning_results(self, method_name: str, prune_decisions: Dict[str, List[int]], 
                              processing_time: float, statistical_info: Optional[Dict] = None) -> PruningResults:
        """Helper to create standardized results."""
        # Evaluate baseline
        baseline_accuracy, _ = evaluate_model(self.original_model, self.test_loader)
        baseline_params = count_parameters(self.original_model)
        
        # Apply pruning
        pruned_model = apply_structural_pruning(self.original_model, prune_decisions)
        pruned_model.to(DEVICE)
        
        # Evaluate after pruning
        pruned_accuracy, _ = evaluate_model(pruned_model, self.test_loader)
        
        # Fine-tune if training data available
        if self.train_loader is not None:
            fine_tuned_model = fine_tune_model(pruned_model, self.train_loader)
            final_accuracy, _ = evaluate_model(fine_tuned_model, self.test_loader)
        else:
            final_accuracy = pruned_accuracy
        
        # Calculate metrics
        pruned_params = count_parameters(pruned_model)
        sparsity = get_model_sparsity(baseline_params, pruned_params)
        neurons_pruned = {layer: len(indices) for layer, indices in prune_decisions.items()}
        
        return PruningResults(
            method_name=method_name,
            baseline_accuracy=baseline_accuracy,
            pruned_accuracy=pruned_accuracy,
            final_accuracy=final_accuracy,
            baseline_params=baseline_params,
            pruned_params=pruned_params,
            sparsity=sparsity,
            neurons_pruned=neurons_pruned,
            processing_time=processing_time,
            statistical_info=statistical_info
        )