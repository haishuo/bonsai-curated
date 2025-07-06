#!/usr/bin/env python3
"""
FIXED Raw Neuron Impact Data Collector
======================================

PURPOSE: Collect raw activation and gradient data at maximum granularity (batch-level)
SCOPE: Raw data collection ONLY - no processing, no FANIM/BANIM computation

CRITICAL FIX: Properly handles loss data collection and saves train_losses/val_losses to HDF5

DATA COLLECTED (per batch):
- Raw gradients: ∂L/∂a_i for each neuron in each layer
- Raw activations: a_i values for each neuron in each layer  
- Training loss: Loss value for this specific batch
- Validation loss: Current validation loss

PHILOSOPHY: Maximum granularity collection - batch level for maximum statistical power
FUTURE: This raw data will be processed separately into NIM variants (Raw, FANIM, BANIM)
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


class FixedRawNIMCollector:
    """
    FIXED Raw neuron impact data collector.
    
    Collects raw gradients and activations at batch level for maximum statistical power.
    No processing or analysis - pure data collection only.
    
    CRITICAL FIX: Properly saves loss data to enable correct FANIM/BANIM computation.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or RAW_COLLECTION_CONFIG
        self.forge_config = get_bonsai_config()
        
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # Target layers for data collection
        self.layer_names = ['fc1', 'fc2', 'fc3']
        
        # Output file for raw data
        self.output_file = self.forge_config.nim_data_path("wine_quality", "raw_nim_data")
        
        # Storage for batch-level raw data - SEPARATED for proper loss handling
        self.training_batches = []  # Training batches with loss data
        self.fanim_batches = []     # FANIM batches without loss data
        
        # Gradient collection storage
        self.gradients = {}
        
        logger.info(f"FIXED Raw NIM Collector initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Target layers: {self.layer_names}")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"Will collect {self.config['epochs']} epochs of batch-level data")
        logger.info(f"🔧 FIXED: Proper loss data collection and storage")
    
    def _setup_gradient_hooks(self):
        """Setup hooks to capture gradients for each target layer."""
        
        def create_gradient_hook(layer_name):
            def gradient_hook(grad):
                # Store gradients for this layer
                self.gradients[layer_name] = grad.detach().clone()
                return grad
            return gradient_hook
        
        # Register hooks for each target layer
        for layer_name in self.layer_names:
            layer = getattr(self.model, layer_name)
            layer.weight.register_hook(create_gradient_hook(layer_name))
            
        logger.info(f"Gradient hooks registered for layers: {self.layer_names}")
    
    def collect_batch_raw_data(self, batch_X: torch.Tensor, batch_y: torch.Tensor,
                              epoch: int, batch_idx: int,
                              criterion: nn.Module, optimizer: optim.Optimizer,
                              val_loader: torch.utils.data.DataLoader) -> Dict:
        """
        Collect raw gradients, activations, and losses for a single training batch.
        
        FIXED: Ensures proper loss collection and storage.
        
        Args:
            batch_X: Input features for this batch
            batch_y: Target labels for this batch
            epoch: Current epoch number (1-500)
            batch_idx: Batch index within epoch
            criterion: Loss function
            optimizer: Optimizer (for training step)
            val_loader: Validation data for loss computation
            
        Returns:
            Dictionary with raw gradients, activations, and losses
        """
        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
        
        # Clear previous gradients
        self.gradients = {}
        optimizer.zero_grad()
        
        # Forward pass
        self.model.train()
        outputs = self.model(batch_X)
        
        # Compute training loss
        train_loss = criterion(outputs, batch_y)
        train_loss_value = float(train_loss.item())  # Ensure it's a Python float
        
        # Backward pass to collect gradients
        train_loss.backward()
        
        # Collect raw activations (stored during forward pass by model hooks)
        raw_activations = {}
        for layer_name in self.layer_names:
            if layer_name in self.model.activations:
                raw_activations[layer_name] = self.model.activations[layer_name].detach().clone().cpu().numpy()
        
        # Collect raw gradients
        raw_gradients = {}
        for layer_name in self.layer_names:
            if layer_name in self.gradients:
                raw_gradients[layer_name] = self.gradients[layer_name].cpu().numpy()
        
        # Compute validation loss (sample-based for speed)
        val_loss_value = self._compute_validation_loss_sample(val_loader)
        
        # Perform training step
        optimizer.step()
        
        # Package raw data - FIXED: Ensure loss values are proper floats
        batch_raw_data = {
            'epoch': int(epoch),
            'batch_idx': int(batch_idx),
            'train_loss': float(train_loss_value),  # Guaranteed to be float
            'val_loss': float(val_loss_value),      # Guaranteed to be float
            'raw_gradients': raw_gradients,
            'raw_activations': raw_activations,
            'batch_size': int(len(batch_X)),
            'timestamp': float(time.time())
        }
        
        return batch_raw_data
    
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
                if i >= 5:  
                    break
        
        self.model.train()
        return total_loss / max(num_samples, 1)
    
    def train_and_collect_raw_data(self, train_loader: torch.utils.data.DataLoader, 
                                  val_loader: torch.utils.data.DataLoader) -> str:
        """
        Complete training with raw batch-level data collection.
        
        FIXED: Proper separation of training and FANIM data for correct loss handling.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Path to saved raw data file
        """
        logger.info("Starting FIXED training with raw batch-level data collection...")
        logger.info(f"Training for {self.config['epochs']} epochs + 1 FANIM collection epoch")
        logger.info(f"🔧 FIXED: Proper loss data collection and storage")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup gradient collection hooks
        self._setup_gradient_hooks()
        
        # PHASE 1: Train for 500 epochs with data collection
        start_time = time.time()
        total_training_batches = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch}/{self.config['epochs']} (TRAINING)")
            
            epoch_training_batches = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # Collect raw data for this training batch
                batch_raw_data = self.collect_batch_raw_data(
                    batch_X, batch_y, epoch, batch_idx, 
                    criterion, optimizer, val_loader
                )
                
                epoch_training_batches.append(batch_raw_data)
                total_training_batches += 1
                
                # Progress logging
                if batch_idx % 20 == 0:
                    logger.info(f"  Batch {batch_idx}: "
                              f"Loss {batch_raw_data['train_loss']:.4f}, "
                              f"Val Loss {batch_raw_data['val_loss']:.4f}")
            
            # Store training epoch data
            self.training_batches.extend(epoch_training_batches)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch completed: {len(epoch_training_batches)} training batches, "
                       f"{epoch_time:.1f}s")
            
            # Save intermediate data every 50 epochs
            if epoch % 50 == 0:
                self._save_intermediate_raw_data(epoch)
        
        # Save the trained model from epoch 500 (before FANIM collection)
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
        
        # PHASE 2: Epoch 501 - Forward-only pass for FANIM activation collection
        logger.info(f"Epoch 501/501 (FANIM COLLECTION - forward only, no training)")
        self.model.eval()  # Set to evaluation mode
        
        total_fanim_batches = 0
        with torch.no_grad():  # No gradients needed for FANIM collection
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass only to get activations for FANIM
                outputs = self.model(batch_X)
                
                # Store forward activations for FANIM computation
                fanim_activations = {}
                for layer_name in self.layer_names:
                    if layer_name in self.model.activations:
                        fanim_activations[layer_name] = self.model.activations[layer_name].detach().clone().cpu().numpy()
                
                # Create FANIM collection record (no gradients, no loss, no training)
                fanim_batch_data = {
                    'epoch': 501,
                    'batch_idx': batch_idx,
                    'raw_gradients': {},  # No gradients collected
                    'raw_activations': fanim_activations,  # Only activations
                    'batch_size': len(batch_X),
                    'timestamp': time.time(),
                    'fanim_collection': True  # Flag to identify FANIM data
                }
                
                self.fanim_batches.append(fanim_batch_data)
                total_fanim_batches += 1
                
                if batch_idx % 20 == 0:
                    logger.info(f"  FANIM Batch {batch_idx}: Activations collected")
        
        logger.info(f"✅ FANIM collection complete: {total_fanim_batches} batches")
        
        # Save final raw dataset
        total_time = time.time() - start_time
        final_output = self._save_complete_raw_dataset(total_time, total_training_batches, total_fanim_batches)
        
        logger.info(f"✅ Complete FIXED data collection finished!")
        logger.info(f"📊 Training batches: {total_training_batches}")
        logger.info(f"📊 FANIM batches: {total_fanim_batches}")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Raw data: {final_output}")
        logger.info(f"🧠 Trained model (epoch 500): {final_model_path}")
        logger.info(f"🔬 FANIM activations: Epoch 501 data included")
        logger.info(f"🔧 FIXED: Loss data properly saved for FANIM/BANIM computation")
        
        return final_output
    
    def _save_intermediate_raw_data(self, epoch: int):
        """Save intermediate raw data checkpoint."""
        checkpoint_file = self.output_file.parent / f"raw_nim_checkpoint_epoch_{epoch}.h5"
        logger.info(f"Saving checkpoint: {checkpoint_file}")
        
        with h5py.File(checkpoint_file, 'w') as f:
            f.attrs['checkpoint_epoch'] = epoch
            f.attrs['total_training_batches'] = len(self.training_batches)
            f.attrs['total_fanim_batches'] = len(self.fanim_batches)
            f.attrs['collection_date'] = datetime.now().isoformat()
    
    def _save_complete_raw_dataset(self, total_time: float, total_training_batches: int, total_fanim_batches: int) -> str:
        """
        Save complete raw dataset in HDF5 format.
        
        FIXED: Proper handling of training vs FANIM data for correct loss storage.
        
        Structure:
        /metadata/ - Collection parameters and statistics
        /raw_data/ - Batch-level raw data by layer
            /fc1/ - gradients (training only), activations (training + FANIM)
            /fc2/ - gradients (training only), activations (training + FANIM)
            /fc3/ - gradients (training only), activations (training + FANIM)
        /batch_info/ - Epoch, batch_idx, losses for TRAINING batches only
        """
        logger.info(f"Saving complete FIXED raw dataset: {self.output_file}")
        
        with h5py.File(self.output_file, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['collection_date'] = datetime.now().isoformat()
            meta.attrs['total_epochs'] = self.config['epochs']
            meta.attrs['total_training_batches'] = total_training_batches
            meta.attrs['total_fanim_batches'] = total_fanim_batches
            meta.attrs['total_time_minutes'] = total_time / 60
            meta.attrs['device_used'] = str(self.device)
            meta.attrs['layer_names'] = [name.encode() for name in self.layer_names]
            meta.attrs['data_format'] = 'batch_level_raw_gradients_and_activations_FIXED'
            meta.attrs['fix_applied'] = 'Proper loss data separation and storage'
            
            # Raw data by layer - FIXED: Proper handling of training vs FANIM data
            raw_group = f.create_group('raw_data')
            
            for layer_name in self.layer_names:
                layer_group = raw_group.create_group(layer_name)
                
                # Collect gradients (TRAINING batches only)
                training_gradients = []
                for batch_data in self.training_batches:
                    if layer_name in batch_data['raw_gradients']:
                        training_gradients.append(batch_data['raw_gradients'][layer_name])
                
                # Collect activations (TRAINING + FANIM batches)
                all_activations = []
                # First add training activations
                for batch_data in self.training_batches:
                    if layer_name in batch_data['raw_activations']:
                        all_activations.append(batch_data['raw_activations'][layer_name])
                # Then add FANIM activations
                for batch_data in self.fanim_batches:
                    if layer_name in batch_data['raw_activations']:
                        all_activations.append(batch_data['raw_activations'][layer_name])
                
                # Save as compressed datasets
                if training_gradients:
                    layer_group.create_dataset(
                        'gradients', 
                        data=np.vstack(training_gradients),
                        compression='gzip',
                        compression_opts=9
                    )
                    logger.info(f"  {layer_name}: {len(training_gradients)} gradient batches saved")
                
                if all_activations:
                    layer_group.create_dataset(
                        'activations',
                        data=np.vstack(all_activations), 
                        compression='gzip',
                        compression_opts=9
                    )
                    logger.info(f"  {layer_name}: {len(all_activations)} activation batches saved (training + FANIM)")
            
            # Batch information - FIXED: Only training batches with valid loss data
            batch_info = f.create_group('batch_info')
            
            # Extract data from training batches only (no None values)
            training_epochs = [b['epoch'] for b in self.training_batches]
            training_batch_indices = [b['batch_idx'] for b in self.training_batches]
            training_losses = [b['train_loss'] for b in self.training_batches]
            validation_losses = [b['val_loss'] for b in self.training_batches]
            
            # Verify no None values
            assert all(loss is not None for loss in training_losses), "Found None values in training losses!"
            assert all(loss is not None for loss in validation_losses), "Found None values in validation losses!"
            
            # Create datasets with proper loss data
            batch_info.create_dataset('epochs', data=np.array(training_epochs, dtype=np.int32))
            batch_info.create_dataset('batch_indices', data=np.array(training_batch_indices, dtype=np.int32))
            batch_info.create_dataset('train_losses', data=np.array(training_losses, dtype=np.float32))
            batch_info.create_dataset('val_losses', data=np.array(validation_losses, dtype=np.float32))
            
            logger.info(f"✅ Loss data saved: {len(training_losses)} training loss values")
            logger.info(f"✅ Loss data saved: {len(validation_losses)} validation loss values")
            
            # Final statistics
            meta.attrs['final_train_loss'] = float(training_losses[-1]) if training_losses else 0.0
            meta.attrs['final_val_loss'] = float(validation_losses[-1]) if validation_losses else 0.0
            meta.attrs['avg_train_loss'] = float(np.mean(training_losses)) if training_losses else 0.0
            meta.attrs['avg_val_loss'] = float(np.mean(validation_losses)) if validation_losses else 0.0
        
        logger.info(f"✅ FIXED raw dataset saved: {self.output_file}")
        return str(self.output_file)


# === CONVENIENCE FUNCTIONS ===

def collect_wine_quality_raw_nim_data_FIXED(config: Optional[Dict] = None) -> str:
    """
    Complete workflow: Train wine quality model and collect raw NIM data with FIXED loss handling.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Path to saved raw data file
    """
    logger.info("🍷 Starting Wine Quality Raw NIM Data Collection (FIXED)")
    
    # Load data
    train_loader, val_loader, test_loader = load_wine_quality_data()
    logger.info(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create fresh model
    model = WineQualityMLP()
    logger.info(f"Model created: {model.get_layer_sizes()}")
    
    # Create FIXED collector
    collector = FixedRawNIMCollector(model, config)
    
    # Train and collect raw data
    output_file = collector.train_and_collect_raw_data(train_loader, val_loader)
    
    logger.info("✅ FIXED Raw NIM data collection complete!")
    return output_file


# === DEMO ===

if __name__ == "__main__":
    print("📊 FIXED Raw Neuron Impact Data Collector")
    print("=" * 50)
    print("🔧 CRITICAL FIX: Proper loss data collection and storage")
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
    total_fanim_batches = batches_per_epoch  # One extra epoch
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Training epochs: {RAW_COLLECTION_CONFIG['epochs']}")
    print(f"Training batches: {total_training_batches:,}")
    print(f"FANIM batches: {total_fanim_batches:,}")
    print(f"Total batches: {total_training_batches + total_fanim_batches:,}")
    
    print(f"\n🚀 Starting FIXED raw data collection...")
    output_file = collect_wine_quality_raw_nim_data_FIXED()
    print(f"✅ Complete! FIXED raw data saved to: {output_file}")
    print(f"🔧 Loss data properly collected for correct FANIM/BANIM computation")