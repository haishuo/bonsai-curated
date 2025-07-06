"""
Raw Neuron Impact Data Collector
===============================

PURPOSE: Collect raw activation and gradient data at maximum granularity (batch-level)
SCOPE: Raw data collection ONLY - no processing, no FANIM/BANIM computation

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
from shared.forge_config import get_bonsai_config
from training.wine_quality_ds import WineQualityDataset, WineQualityMLP, load_wine_quality_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training configuration for raw data collection
RAW_COLLECTION_CONFIG = {
    'epochs': 501,  # Collect data for all 501 epochs
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'collect_every_batch': True,  # Collect data for every single batch
}


class RawNIMCollector:
    """
    Raw neuron impact data collector.
    
    Collects raw gradients and activations at batch level for maximum statistical power.
    No processing or analysis - pure data collection only.
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
        
        # Storage for batch-level raw data
        self.raw_data_batches = []
        
        logger.info(f"Raw NIM Collector initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Target layers: {self.layer_names}")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"Will collect {self.config['epochs']} epochs of batch-level data")
    
    def _setup_gradient_hooks(self):
        """Setup hooks to capture gradients for each target layer."""
        self.gradients = {}
        
        def make_hook(layer_name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[layer_name] = grad_output[0].detach().clone()
            return hook
        
        # Register hooks on target layers
        if hasattr(self.model, 'fc1'):
            self.model.fc1.register_backward_hook(make_hook('fc1'))
        if hasattr(self.model, 'fc2'):
            self.model.fc2.register_backward_hook(make_hook('fc2'))
        if hasattr(self.model, 'fc3'):
            self.model.fc3.register_backward_hook(make_hook('fc3'))
    
    def collect_batch_raw_data(self, batch_X: torch.Tensor, batch_y: torch.Tensor, 
                              epoch: int, batch_idx: int, 
                              criterion: nn.Module, optimizer: optim.Optimizer,
                              val_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Collect raw data for a single batch.
        
        Args:
            batch_X: Input features for this batch
            batch_y: Target labels for this batch
            epoch: Current epoch number (1-501)
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
        val_loss = self._compute_validation_loss_sample(val_loader)
        
        # Perform training step
        optimizer.step()
        
        # Package raw data
        batch_raw_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'train_loss': train_loss.item(),
            'val_loss': val_loss,
            'raw_gradients': raw_gradients,
            'raw_activations': raw_activations,
            'batch_size': len(batch_X),
            'timestamp': time.time()
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
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Path to saved raw data file
        """
        logger.info("Starting training with raw batch-level data collection...")
        logger.info(f"Collecting data for {self.config['epochs']} epochs")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup gradient collection hooks
        self._setup_gradient_hooks()
        
        # Collect raw data for all epochs
        start_time = time.time()
        total_batches_processed = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch}/{self.config['epochs']}")
            
            epoch_batches = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                # Collect raw data for this batch
                batch_raw_data = self.collect_batch_raw_data(
                    batch_X, batch_y, epoch, batch_idx, 
                    criterion, optimizer, val_loader
                )
                
                epoch_batches.append(batch_raw_data)
                total_batches_processed += 1
                
                # Progress logging
                if batch_idx % 20 == 0:
                    logger.info(f"  Batch {batch_idx}: "
                              f"Loss {batch_raw_data['train_loss']:.4f}, "
                              f"Val Loss {batch_raw_data['val_loss']:.4f}")
            
            # Store epoch data
            self.raw_data_batches.extend(epoch_batches)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"  Epoch completed: {len(epoch_batches)} batches, "
                       f"{epoch_time:.1f}s")
            
            # Save intermediate data every 50 epochs
            if epoch % 50 == 0:
                self._save_intermediate_raw_data(epoch)
        
        # Save final raw dataset
        total_time = time.time() - start_time
        final_output = self._save_complete_raw_dataset(total_time, total_batches_processed)
        
        logger.info(f"✅ Raw data collection complete!")
        logger.info(f"📊 Total batches: {total_batches_processed}")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Saved to: {final_output}")
        
        return final_output
    
    def _save_intermediate_raw_data(self, epoch: int):
        """Save intermediate raw data checkpoint."""
        checkpoint_file = self.output_file.parent / f"raw_nim_checkpoint_epoch_{epoch}.h5"
        logger.info(f"Saving checkpoint: {checkpoint_file}")
        
        with h5py.File(checkpoint_file, 'w') as f:
            f.attrs['checkpoint_epoch'] = epoch
            f.attrs['total_batches'] = len(self.raw_data_batches)
            f.attrs['collection_date'] = datetime.now().isoformat()
    
    def _save_complete_raw_dataset(self, total_time: float, total_batches: int) -> str:
        """
        Save complete raw dataset in HDF5 format.
        
        Structure:
        /metadata/ - Collection parameters and statistics
        /raw_data/ - Batch-level raw data by layer
            /fc1/ - gradients, activations for FC1 layer
            /fc2/ - gradients, activations for FC2 layer  
            /fc3/ - gradients, activations for FC3 layer
        /batch_info/ - Epoch, batch_idx, losses for each sample
        """
        logger.info(f"Saving complete raw dataset: {self.output_file}")
        
        with h5py.File(self.output_file, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['collection_date'] = datetime.now().isoformat()
            meta.attrs['total_epochs'] = self.config['epochs']
            meta.attrs['total_batches'] = total_batches
            meta.attrs['total_time_minutes'] = total_time / 60
            meta.attrs['device_used'] = str(self.device)
            meta.attrs['layer_names'] = [name.encode() for name in self.layer_names]
            meta.attrs['data_format'] = 'batch_level_raw_gradients_and_activations'
            
            # Raw data by layer
            raw_group = f.create_group('raw_data')
            
            for layer_name in self.layer_names:
                layer_group = raw_group.create_group(layer_name)
                
                # Collect all gradients and activations for this layer
                all_gradients = []
                all_activations = []
                
                for batch_data in self.raw_data_batches:
                    if layer_name in batch_data['raw_gradients']:
                        all_gradients.append(batch_data['raw_gradients'][layer_name])
                    if layer_name in batch_data['raw_activations']:
                        all_activations.append(batch_data['raw_activations'][layer_name])
                
                # Save as compressed datasets
                if all_gradients:
                    layer_group.create_dataset(
                        'gradients', 
                        data=np.vstack(all_gradients),
                        compression='gzip',
                        compression_opts=9
                    )
                if all_activations:
                    layer_group.create_dataset(
                        'activations',
                        data=np.vstack(all_activations), 
                        compression='gzip',
                        compression_opts=9
                    )
            
            # Batch information
            batch_info = f.create_group('batch_info')
            epochs = [b['epoch'] for b in self.raw_data_batches]
            batch_indices = [b['batch_idx'] for b in self.raw_data_batches]
            train_losses = [b['train_loss'] for b in self.raw_data_batches]
            val_losses = [b['val_loss'] for b in self.raw_data_batches]
            
            batch_info.create_dataset('epochs', data=np.array(epochs))
            batch_info.create_dataset('batch_indices', data=np.array(batch_indices))
            batch_info.create_dataset('train_losses', data=np.array(train_losses))
            batch_info.create_dataset('val_losses', data=np.array(val_losses))
            
            # Final statistics
            meta.attrs['final_train_loss'] = train_losses[-1] if train_losses else 0.0
            meta.attrs['final_val_loss'] = val_losses[-1] if val_losses else 0.0
            meta.attrs['avg_train_loss'] = np.mean(train_losses) if train_losses else 0.0
        
        return str(self.output_file)


# === CONVENIENCE FUNCTIONS ===

def collect_wine_quality_raw_nim_data(config: Optional[Dict] = None) -> str:
    """
    Complete workflow: Train wine quality model and collect raw NIM data.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Path to saved raw data file
    """
    logger.info("🍷 Starting Wine Quality Raw NIM Data Collection")
    
    # Load data
    train_loader, val_loader, test_loader = load_wine_quality_data()
    logger.info(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create fresh model
    model = WineQualityMLP()
    logger.info(f"Model created: {model.get_layer_sizes()}")
    
    # Create collector
    collector = RawNIMCollector(model, config)
    
    # Train and collect raw data
    output_file = collector.train_and_collect_raw_data(train_loader, val_loader)
    
    logger.info("✅ Raw NIM data collection complete!")
    return output_file


# === DEMO ===

if __name__ == "__main__":
    print("📊 Raw Neuron Impact Data Collector")
    print("=" * 50)
    
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
    total_batches = batches_per_epoch * RAW_COLLECTION_CONFIG['epochs']
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total epochs: {RAW_COLLECTION_CONFIG['epochs']}")
    print(f"Total batches: {total_batches:,}")
    
    print(f"\n🚀 Starting raw data collection...")
    output_file = collect_wine_quality_raw_nim_data()
    print(f"✅ Complete! Raw data saved to: {output_file}")