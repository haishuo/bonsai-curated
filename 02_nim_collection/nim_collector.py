#!/usr/bin/env python3
"""
SPEED OPTIMIZED Raw Neuron Impact Data Collector
===============================================

PURPOSE: Collect TRUE activation gradients (∂L/∂a_i) with MAXIMUM SPEED
OPTIMIZATION: Pre-allocated NumPy arrays with direct assignment for lightning-fast collection

DATA COLLECTED (per batch):
- Raw NIM gradients: ∂L/∂a_i for each neuron in each layer  
- Raw activations: a_i values for each neuron in each layer  
- Training loss: Loss value for this specific batch
- Validation loss: Current validation loss

SPEED STRATEGY: 
- Pre-allocate exact-size NumPy arrays upfront
- Use direct array assignment (fastest possible operation)
- Intercept ∂L/∂a_i during normal backprop (no recalculation)
- Write to HDF5 in single vectorized operations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import time

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "01_training"))
from shared.forge_config import get_bonsai_config
from wine_quality_ds import WineQualityDataset, WineQualityMLP, load_wine_quality_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training configuration for raw data collection
RAW_COLLECTION_CONFIG = {
    'epochs': 500,  # Train for 500 epochs + 1 extra for FANIM forward pass
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'collect_every_batch': True,  # Collect data for every single batch
}


class SpeedOptimizedNIMCollector:
    """
    SPEED OPTIMIZED Raw neuron impact data collector.
    
    Uses pre-allocated NumPy arrays and direct assignment for maximum speed.
    Intercepts TRUE activation gradients (∂L/∂a_i) during normal backprop.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or RAW_COLLECTION_CONFIG
        self.forge_config = get_bonsai_config()
        
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # Target layers and sizes
        self.layer_names = ['fc1', 'fc2', 'fc3']
        self.layer_sizes = {'fc1': 256, 'fc2': 128, 'fc3': 64}
        
        # Output file for raw data
        self.output_file = self.forge_config.nim_data_path("wine_quality", "raw_nim_data")
        
        # Storage for current batch data
        self.current_activations = {}
        self.current_activation_grads = {}
        
        logger.info(f"SPEED OPTIMIZED Raw NIM Collector initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Target layers: {self.layer_names}")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"Will collect {self.config['epochs']} epochs of batch-level data")
        
        # Setup pre-allocated storage for maximum speed
        self._setup_preallocated_storage()
        
        logger.info(f"🚀 SPEED OPTIMIZED: Pre-allocated NumPy arrays for MAXIMUM performance!")
    
    def _setup_preallocated_storage(self):
        """
        Setup pre-allocated NumPy arrays for maximum speed data collection.
        
        SPEED OPTIMIZATION: Pre-allocate exact size needed, use direct array assignment.
        """
        # Estimate total batches needed
        max_epochs = self.config['epochs'] + 1  # +1 for FANIM collection
        estimated_batches_per_epoch = 50  # Conservative estimate
        max_batches = max_epochs * estimated_batches_per_epoch
        
        logger.info(f"📊 Pre-allocating storage for up to {max_batches:,} batches")
        
        # Pre-allocate activation gradient arrays (NIMs)
        self.preallocated_nims = {}
        for layer_name in self.layer_names:
            num_neurons = self.layer_sizes[layer_name]
            self.preallocated_nims[layer_name] = np.empty(
                (max_batches, num_neurons), 
                dtype=np.float32
            )
            logger.info(f"   {layer_name} NIMs: {max_batches:,} × {num_neurons} pre-allocated")
        
        # Pre-allocate activation arrays (with extra space for FANIM)
        self.preallocated_activations = {}
        for layer_name in self.layer_names:
            num_neurons = self.layer_sizes[layer_name]
            fanim_extra = estimated_batches_per_epoch
            self.preallocated_activations[layer_name] = np.empty(
                (max_batches + fanim_extra, num_neurons), 
                dtype=np.float32
            )
            logger.info(f"   {layer_name} activations: {max_batches + fanim_extra:,} × {num_neurons} pre-allocated")
        
        # Pre-allocate loss and metadata arrays
        self.preallocated_train_losses = np.empty(max_batches, dtype=np.float32)
        self.preallocated_val_losses = np.empty(max_batches, dtype=np.float32)
        self.preallocated_epochs = np.empty(max_batches, dtype=np.int32)
        self.preallocated_batch_indices = np.empty(max_batches, dtype=np.int32)
        
        # Counters for array indexing
        self.training_batch_count = 0
        self.total_activation_count = 0  # Includes both training and FANIM
        
        logger.info(f"✅ Pre-allocation complete - ready for LIGHTNING FAST data collection!")
    
    def _setup_activation_hooks(self):
        """
        Setup hooks to intercept TRUE activation gradients during normal backprop.
        
        INTERCEPTION: Grab ∂L/∂a_i AS IT'S CALCULATED during backward pass.
        No recalculation, no approximation - pure interception for maximum speed.
        """
        
        def create_forward_hook(layer_name):
            """Hook to capture activations during forward pass."""
            def forward_hook(module, input, output):
                # Store activations (batch mean for storage efficiency)
                activations = output.detach()  # Shape: (batch_size, num_neurons)
                batch_mean_activations = torch.mean(activations, dim=0)  # Shape: (num_neurons,)
                self.current_activations[layer_name] = batch_mean_activations
                return output
            return forward_hook
        
        def create_backward_hook(layer_name):
            """Hook to INTERCEPT TRUE activation gradients ∂L/∂a_i during normal backprop."""
            def backward_hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    # INTERCEPT: grad_output[0] = ∂L/∂a_i (exact value PyTorch computed)
                    activation_grads = grad_output[0].detach().clone()  # Shape: (batch_size, num_neurons)
                    batch_mean_grads = torch.mean(activation_grads, dim=0)  # Shape: (num_neurons,)
                    self.current_activation_grads[layer_name] = batch_mean_grads
                return None
            return backward_hook
        
        # Register hooks for each target layer
        hooks = []
        for layer_name in self.layer_names:
            layer = getattr(self.model, layer_name)
            
            # Forward hook to capture activations
            hooks.append(layer.register_forward_hook(create_forward_hook(layer_name)))
            
            # Backward hook to INTERCEPT ∂L/∂a_i during normal backprop
            hooks.append(layer.register_full_backward_hook(create_backward_hook(layer_name)))
        
        self.hooks = hooks
        logger.info(f"📎 Installed {len(hooks)} hooks for INTERCEPTING ∂L/∂a_i during normal backprop")
        logger.info(f"   Forward hooks: Capture a_i (activations)")
        logger.info(f"   Backward hooks: INTERCEPT ∂L/∂a_i (TRUE activation gradients)")
        logger.info(f"   Method: Grab grad_output[0] during normal backward pass")
        logger.info(f"   Note: PyTorch warning is harmless - the hook works correctly")
    
    def collect_batch_raw_data(self, batch_X: torch.Tensor, batch_y: torch.Tensor,
                              epoch: int, batch_idx: int,
                              criterion: nn.Module, optimizer: optim.Optimizer,
                              val_loader: torch.utils.data.DataLoader) -> Dict:
        """
        Collect raw activation gradients and activations with LIGHTNING SPEED.
        
        SPEED OPTIMIZED: Direct assignment to pre-allocated arrays.
        """
        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
        
        # Clear previous data
        self.current_activations = {}
        self.current_activation_grads = {}
        optimizer.zero_grad()
        
        # Forward pass - triggers forward hooks
        self.model.train()
        outputs = self.model(batch_X)
        
        # Compute training loss
        train_loss = criterion(outputs, batch_y)
        train_loss_value = float(train_loss.item())
        
        # Backward pass - triggers backward hooks to INTERCEPT ∂L/∂a_i
        train_loss.backward()
        
        # Compute validation loss (fast sampling)
        val_loss_value = self._compute_validation_loss_sample(val_loader)
        
        # LIGHTNING FAST: Direct assignment to pre-allocated arrays!
        for layer_name in self.layer_names:
            if layer_name in self.current_activations:
                # Direct assignment - FASTEST possible operation!
                self.preallocated_activations[layer_name][self.total_activation_count] = \
                    self.current_activations[layer_name].cpu().numpy()
            
            if layer_name in self.current_activation_grads:
                # Direct assignment - FASTEST possible operation!
                self.preallocated_nims[layer_name][self.training_batch_count] = \
                    self.current_activation_grads[layer_name].cpu().numpy()
        
        # Store batch metadata - direct assignment
        self.preallocated_train_losses[self.training_batch_count] = train_loss_value
        self.preallocated_val_losses[self.training_batch_count] = val_loss_value
        self.preallocated_epochs[self.training_batch_count] = epoch
        self.preallocated_batch_indices[self.training_batch_count] = batch_idx
        
        # Increment counters
        self.training_batch_count += 1
        self.total_activation_count += 1
        
        # Perform training step
        optimizer.step()
        
        # Return batch info for logging (no storage needed!)
        return {
            'epoch': int(epoch),
            'batch_idx': int(batch_idx),
            'train_loss': float(train_loss_value),
            'val_loss': float(val_loss_value),
            'batch_size': int(len(batch_X)),
            'timestamp': float(time.time())
        }
    
    def _compute_validation_loss_sample(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Compute validation loss on a small sample for speed."""
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for i, (val_X, val_y) in enumerate(val_loader):
                val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                outputs = self.model(val_X)
                loss = criterion(outputs, val_y)
                total_loss += loss.item()
                num_samples += 1
                
                # Only sample first few batches for speed
                if i >= 3:  
                    break
        
        self.model.train()
        return total_loss / max(num_samples, 1)
    
    def train_and_collect_raw_data(self, train_loader: torch.utils.data.DataLoader, 
                                  val_loader: torch.utils.data.DataLoader) -> str:
        """
        Complete training with SPEED OPTIMIZED raw batch-level data collection.
        """
        logger.info("Starting SPEED OPTIMIZED training with raw batch-level data collection...")
        logger.info(f"Training for {self.config['epochs']} epochs + 1 FANIM collection epoch")
        logger.info(f"🚀 SPEED OPTIMIZED: Pre-allocated arrays with direct assignment")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup activation hooks
        self._setup_activation_hooks()
        
        # PHASE 1: Train for 500 epochs with LIGHTNING FAST data collection
        start_time = time.time()
        total_training_batches = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch}/{self.config['epochs']} (TRAINING)")
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # LIGHTNING FAST data collection!
                batch_raw_data = self.collect_batch_raw_data(
                    batch_X, batch_y, epoch, batch_idx, 
                    criterion, optimizer, val_loader
                )
                
                total_training_batches += 1
                
                # Progress logging
                if batch_idx % 50 == 0:
                    logger.info(f"  Batch {batch_idx}: "
                              f"Loss {batch_raw_data['train_loss']:.4f}, "
                              f"Val Loss {batch_raw_data['val_loss']:.4f}")
            
            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch completed: {len(train_loader)} training batches, "
                       f"{epoch_time:.1f}s")
            
            # Save intermediate data every 100 epochs
            if epoch % 100 == 0:
                self._save_intermediate_checkpoint(epoch)
        
        # Save the trained model
        final_model_path = self.forge_config.model_path("wine_quality_trained_500epochs")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': self.config['epochs'],
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'architecture': self.model.get_layer_sizes(),
            'total_training_batches': total_training_batches
        }, final_model_path)
        
        logger.info(f"✅ Trained model saved (epoch 500): {final_model_path}")
        
        # PHASE 2: Epoch 501 - FANIM activation collection
        logger.info(f"Epoch 501/501 (FANIM COLLECTION - forward only)")
        self.model.eval()
        
        total_fanim_batches = 0
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Clear previous activations
                self.current_activations = {}
                
                # Forward pass only
                outputs = self.model(batch_X)
                
                # LIGHTNING FAST: Direct assignment to pre-allocated array
                for layer_name in self.layer_names:
                    if layer_name in self.current_activations:
                        self.preallocated_activations[layer_name][self.total_activation_count] = \
                            self.current_activations[layer_name].cpu().numpy()
                
                self.total_activation_count += 1
                total_fanim_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"  FANIM Batch {batch_idx}: Activations collected")
        
        logger.info(f"✅ FANIM collection complete: {total_fanim_batches} batches")
        
        # Save final dataset
        total_time = time.time() - start_time
        final_output = self._save_complete_raw_dataset(total_time, total_training_batches, total_fanim_batches)
        
        logger.info(f"✅ SPEED OPTIMIZED data collection complete!")
        logger.info(f"📊 Training batches: {total_training_batches}")
        logger.info(f"📊 FANIM batches: {total_fanim_batches}")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Raw data: {final_output}")
        logger.info(f"🚀 MAXIMUM SPEED achieved with pre-allocated arrays!")
        
        return final_output
    
    def _save_intermediate_checkpoint(self, epoch: int):
        """Save intermediate checkpoint."""
        checkpoint_file = self.output_file.parent / f"speed_optimized_checkpoint_epoch_{epoch}.h5"
        logger.info(f"Saving SPEED checkpoint: {checkpoint_file}")
        
        with h5py.File(checkpoint_file, 'w') as f:
            f.attrs['checkpoint_epoch'] = epoch
            f.attrs['training_batch_count'] = self.training_batch_count
            f.attrs['total_activation_count'] = self.total_activation_count
            f.attrs['collection_date'] = datetime.now().isoformat()
    
    def _save_complete_raw_dataset(self, total_time: float, total_training_batches: int, total_fanim_batches: int) -> str:
        """
        Save complete raw dataset with MAXIMUM SPEED.
        
        SPEED OPTIMIZED: Direct array slicing and writing - no loops!
        """
        logger.info(f"Saving SPEED OPTIMIZED raw dataset: {self.output_file}")
        logger.info(f"📊 Training batches: {self.training_batch_count}")
        logger.info(f"📊 Total activations: {self.total_activation_count}")
        
        with h5py.File(self.output_file, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['collection_date'] = datetime.now().isoformat()
            meta.attrs['total_epochs'] = self.config['epochs']
            meta.attrs['total_training_batches'] = self.training_batch_count
            meta.attrs['total_fanim_batches'] = total_fanim_batches
            meta.attrs['total_time_minutes'] = total_time / 60
            meta.attrs['device_used'] = str(self.device)
            meta.attrs['layer_names'] = [name.encode() for name in self.layer_names]
            meta.attrs['data_format'] = 'speed_optimized_preallocated_numpy'
            meta.attrs['speed_optimization'] = 'Pre-allocated NumPy arrays with direct assignment'
            meta.attrs['gradient_method'] = 'Intercepted ∂L/∂a_i during normal backprop'
            meta.attrs['nim_definition'] = 'NIM_i = ∂L/∂a_i (TRUE activation gradients)'
            
            # Raw data by layer - LIGHTNING FAST: Direct array slicing!
            raw_group = f.create_group('raw_data')
            
            for layer_name in self.layer_names:
                layer_group = raw_group.create_group(layer_name)
                
                # SPEED: Direct slice from pre-allocated arrays (no loops!)
                training_gradients = self.preallocated_nims[layer_name][:self.training_batch_count]
                all_activations = self.preallocated_activations[layer_name][:self.total_activation_count]
                
                # Save as compressed datasets
                layer_group.create_dataset(
                    'gradients',  # ∂L/∂a_i (TRUE activation gradients = NIMs)
                    data=training_gradients,
                    compression='gzip',
                    compression_opts=9
                )
                
                layer_group.create_dataset(
                    'activations',
                    data=all_activations, 
                    compression='gzip',
                    compression_opts=9
                )
                
                logger.info(f"  {layer_name}: {training_gradients.shape[0]} gradient batches, {all_activations.shape[0]} activation batches")
            
            # Batch information - SPEED: Direct slice from pre-allocated arrays!
            batch_info = f.create_group('batch_info')
            
            batch_info.create_dataset('epochs', data=self.preallocated_epochs[:self.training_batch_count])
            batch_info.create_dataset('batch_indices', data=self.preallocated_batch_indices[:self.training_batch_count])
            batch_info.create_dataset('train_losses', data=self.preallocated_train_losses[:self.training_batch_count])
            batch_info.create_dataset('val_losses', data=self.preallocated_val_losses[:self.training_batch_count])
            
            logger.info(f"✅ Loss data saved: {self.training_batch_count} training/validation loss values")
            
            # Final statistics
            train_losses = self.preallocated_train_losses[:self.training_batch_count]
            val_losses = self.preallocated_val_losses[:self.training_batch_count]
            meta.attrs['final_train_loss'] = float(train_losses[-1]) if len(train_losses) > 0 else 0.0
            meta.attrs['final_val_loss'] = float(val_losses[-1]) if len(val_losses) > 0 else 0.0
            meta.attrs['avg_train_loss'] = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
            meta.attrs['avg_val_loss'] = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0
        
        logger.info(f"✅ SPEED OPTIMIZED raw dataset saved: {self.output_file}")
        logger.info(f"🚀 Maximum speed achieved with pre-allocated arrays!")
        return str(self.output_file)
    
    def cleanup_hooks(self):
        """Clean up registered hooks."""
        for hook in self.hooks:
            hook.remove()
        logger.info("🧹 Activation hooks cleaned up")


# === CONVENIENCE FUNCTIONS ===

def collect_wine_quality_raw_nim_data_SPEED_OPTIMIZED(config: Optional[Dict] = None) -> str:
    """
    Complete workflow: Train wine quality model with SPEED OPTIMIZED data collection.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Path to saved raw data file
    """
    logger.info("🍷 Starting Wine Quality Raw NIM Data Collection (SPEED OPTIMIZED)")
    
    # Load data
    train_loader, val_loader, test_loader = load_wine_quality_data()
    logger.info(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create fresh model
    model = WineQualityMLP()
    logger.info(f"Model created: {model.get_layer_sizes()}")
    
    # Create SPEED OPTIMIZED collector
    collector = SpeedOptimizedNIMCollector(model, config)
    
    try:
        # Train and collect raw data
        output_file = collector.train_and_collect_raw_data(train_loader, val_loader)
        
        logger.info("✅ SPEED OPTIMIZED Raw NIM data collection complete!")
        return output_file
        
    finally:
        # Always clean up hooks
        collector.cleanup_hooks()


# === DEMO ===

if __name__ == "__main__":
    print("📊 SPEED OPTIMIZED Raw Neuron Impact Data Collector")
    print("=" * 60)
    print("🚀 MAXIMUM SPEED: Pre-allocated NumPy arrays with direct assignment")
    print("🔬 TRUE GRADIENTS: Intercepts ∂L/∂a_i during normal backprop")
    print("")
    
    # Configuration summary
    config = get_bonsai_config()
    print(f"Output directory: {config.nim_data_dir}")
    
    # Device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Estimate data size
    print(f"\n📈 Data Collection Estimate:")
    train_loader, _, _ = load_wine_quality_data()
    batches_per_epoch = len(train_loader)
    total_training_batches = batches_per_epoch * RAW_COLLECTION_CONFIG['epochs']
    total_fanim_batches = batches_per_epoch
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Training epochs: {RAW_COLLECTION_CONFIG['epochs']}")
    print(f"Training batches: {total_training_batches:,}")
    print(f"FANIM batches: {total_fanim_batches:,}")
    print(f"Total batches: {total_training_batches + total_fanim_batches:,}")
    
    print(f"\n🚀 Starting SPEED OPTIMIZED raw data collection...")
    output_file = collect_wine_quality_raw_nim_data_SPEED_OPTIMIZED()
    print(f"✅ Complete! SPEED OPTIMIZED raw data saved to: {output_file}")
    print(f"🚀 Maximum speed achieved with pre-allocated NumPy arrays!")