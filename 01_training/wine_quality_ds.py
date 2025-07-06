"""
Wine Quality Dataset Module
===========================

PURPOSE: Clean, professional wine quality dataset preparation for Bonsai research
ARCHITECTURE: Uses forge config system and leverages existing common_datasets structure
WORKFLOW: Download → Process → Cache → Load for training

FORGE INTEGRATION:
- Raw data: /mnt/data/common_datasets/wine/ 
- Processed data: /mnt/data/bonsai/datasets/wine_quality/
- Models: /mnt/data/bonsai/models/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pickle
import requests
from typing import Tuple, Optional, Dict, Any
import logging

# Import our forge configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shared.forge_config import get_bonsai_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wine Quality Dataset Configuration
WINE_QUALITY_CONFIG = {
    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    'target_column': 'quality',
    'features_to_drop': [],  # Keep all features for now
    'test_size': 0.2,
    'val_size': 0.2,  # 20% of training set becomes validation
    'random_state': 42,
    'batch_size': 32,
    'num_classes': 6,  # Quality ratings 3-8 → classes 0-5
}

# Neural Network Architecture
ARCHITECTURE_CONFIG = {
    'input_size': 11,      # Wine features
    'hidden_layers': [256, 128, 64],
    'output_size': 6,      # Quality classes 0-5
    'dropout_rate': 0.3,
}


class WineQualityMLP(nn.Module):
    """
    Clean MLP architecture for wine quality classification.
    
    Architecture: 11 → 256 → 128 → 64 → 6
    Features: ReLU activation, dropout for regularization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        if config is None:
            config = ARCHITECTURE_CONFIG
            
        self.config = config
        
        # Define layers
        self.fc1 = nn.Linear(config['input_size'], config['hidden_layers'][0])
        self.fc2 = nn.Linear(config['hidden_layers'][0], config['hidden_layers'][1])
        self.fc3 = nn.Linear(config['hidden_layers'][1], config['hidden_layers'][2])
        self.fc4 = nn.Linear(config['hidden_layers'][2], config['output_size'])
        
        self.dropout = nn.Dropout(config['dropout_rate'])
        
        # Storage for activations (needed for sensitivity analysis)
        self.activations = {}
        
        # Register forward hooks for sensitivity collection
        self.fc1.register_forward_hook(self._save_activation('fc1'))
        self.fc2.register_forward_hook(self._save_activation('fc2'))
        self.fc3.register_forward_hook(self._save_activation('fc3'))
    
    def _save_activation(self, layer_name: str):
        """Create hook to save layer activations."""
        def hook(module, input, output):
            # Store only the pre-dropout activations
            self.activations[layer_name] = output.detach()
        return hook
    
    def forward(self, x):
        # Layer 1: Input → 256
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Layer 2: 256 → 128  
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Layer 3: 128 → 64
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Output layer: 64 → 6
        x = self.fc4(x)
        
        return x
    
    def get_layer_sizes(self) -> Dict[str, int]:
        """Get neuron count for each layer (needed for pruning)."""
        return {
            'fc1': self.config['hidden_layers'][0],
            'fc2': self.config['hidden_layers'][1], 
            'fc3': self.config['hidden_layers'][2],
        }


class WineQualityDataset:
    """
    Professional wine quality dataset manager.
    
    Handles downloading, preprocessing, caching, and loading of wine quality data
    using the forge configuration system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or WINE_QUALITY_CONFIG
        self.forge_config = get_bonsai_config()
        
        # Set up paths using forge config structure
        # Raw data: Use symlink to common_datasets
        self.datasets_symlink_dir = self.forge_config.datasets_dir
        self.raw_data_dir = self.datasets_symlink_dir / "wine"  # Via symlink
        
        # Processed data: Store in scratch for this specific dataset
        self.processed_data_dir = self.forge_config.scratch_dir / "wine_quality_processed"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.raw_csv_path = self.raw_data_dir / "winequality-red.csv"
        self.processed_cache = self.processed_data_dir / "processed_data.pkl"
        self.scaler_cache = self.processed_data_dir / "scaler.pkl"
        
        logger.info(f"Wine Quality Dataset initialized")
        logger.info(f"Raw data: {self.raw_data_dir}")
        logger.info(f"Processed data: {self.processed_data_dir}")
    
    def download_raw_data(self, force_download: bool = False) -> Path:
        """
        Download raw wine quality data if not already present.
        
        Args:
            force_download: Re-download even if file exists
            
        Returns:
            Path to downloaded CSV file
        """
        if self.raw_csv_path.exists() and not force_download:
            logger.info(f"Raw data already exists: {self.raw_csv_path}")
            return self.raw_csv_path
        
        # Ensure raw data directory exists
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading wine quality data from UCI repository...")
        
        try:
            response = requests.get(self.config['url'], timeout=30)
            response.raise_for_status()
            
            with open(self.raw_csv_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Downloaded: {self.raw_csv_path}")
            return self.raw_csv_path
            
        except Exception as e:
            logger.error(f"Failed to download wine quality data: {e}")
            raise
    
    def load_and_preprocess(self, force_reprocess: bool = False) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Load and preprocess wine quality data.
        
        Args:
            force_reprocess: Reprocess even if cache exists
            
        Returns:
            Tuple of (features, labels, scaler)
        """
        # Check cache first
        if (self.processed_cache.exists() and self.scaler_cache.exists() 
            and not force_reprocess):
            logger.info("Loading preprocessed data from cache...")
            
            with open(self.processed_cache, 'rb') as f:
                features, labels = pickle.load(f)
            
            with open(self.scaler_cache, 'rb') as f:
                scaler = pickle.load(f)
            
            logger.info(f"✅ Loaded cached data: {features.shape[0]} samples, {features.shape[1]} features")
            return features, labels, scaler
        
        # Download raw data if needed
        self.download_raw_data()
        
        logger.info("Processing raw wine quality data...")
        
        # Load CSV with semicolon separator (UCI format)
        df = pd.read_csv(self.raw_csv_path, sep=';')
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        logger.info(f"Features: {[col for col in df.columns if col != self.config['target_column']]}")
        
        # Separate features and target
        X = df.drop(columns=[self.config['target_column']])
        y = df[self.config['target_column']]
        
        # Drop any specified features
        if self.config['features_to_drop']:
            X = X.drop(columns=self.config['features_to_drop'])
            logger.info(f"Dropped features: {self.config['features_to_drop']}")
        
        # Convert quality scores (3-8) to class indices (0-5)
        y_min = y.min()  # Should be 3
        y_max = y.max()  # Should be 8
        y_classes = y - y_min  # 3-8 → 0-5
        
        logger.info(f"Quality range: {y_min}-{y_max} → classes 0-{y_max-y_min}")
        logger.info(f"Class distribution: {y_classes.value_counts().sort_index().to_dict()}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
        
        logger.info(f"Features standardized: mean≈0, std≈1")
        
        # Cache processed data
        with open(self.processed_cache, 'wb') as f:
            pickle.dump((X_scaled, y_classes.values), f)
        
        with open(self.scaler_cache, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info(f"✅ Cached processed data to {self.processed_data_dir}")
        
        return X_scaled, y_classes.values, scaler
    
    def create_data_loaders(self, force_reprocess: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for training, validation, and testing.
        
        Args:
            force_reprocess: Reprocess data even if cache exists
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load preprocessed data
        X, y, scaler = self.load_and_preprocess(force_reprocess)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Split training into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config['val_size'],
            random_state=self.config['random_state'],
            stratify=y_train
        )
        
        logger.info(f"Data splits:")
        logger.info(f"  Training: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples") 
        logger.info(f"  Testing: {len(X_test)} samples")
        
        # Convert to PyTorch tensors
        def create_loader(X_data, y_data, shuffle=True):
            X_tensor = torch.FloatTensor(X_data)
            y_tensor = torch.LongTensor(y_data)
            dataset = TensorDataset(X_tensor, y_tensor)
            return DataLoader(
                dataset, 
                batch_size=self.config['batch_size'],
                shuffle=shuffle,
                num_workers=4,  # Leverage forge's multi-core CPU
                pin_memory=True  # Optimize for GPU transfer
            )
        
        train_loader = create_loader(X_train, y_train, shuffle=True)
        val_loader = create_loader(X_val, y_val, shuffle=False)
        test_loader = create_loader(X_test, y_test, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        X, y, scaler = self.load_and_preprocess()
        
        return {
            'num_samples': len(X),
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'class_distribution': {int(cls): int(count) for cls, count in 
                                 zip(*np.unique(y, return_counts=True))},
            'feature_names': ['fixed acidity', 'volatile acidity', 'citric acid',
                            'residual sugar', 'chlorides', 'free sulfur dioxide',
                            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
            'target_range': f"{y.min()}-{y.max()}",
            'config': self.config,
            'paths': {
                'raw_data': str(self.raw_data_dir),
                'processed_data': str(self.processed_data_dir),
            }
        }


# === CONVENIENCE FUNCTIONS ===

def get_wine_quality_model(device: str = 'cuda') -> WineQualityMLP:
    """Get a fresh wine quality model on specified device."""
    model = WineQualityMLP()
    return model.to(device)


def load_wine_quality_data(force_reprocess: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Quick function to load wine quality data loaders."""
    dataset = WineQualityDataset()
    return dataset.create_data_loaders(force_reprocess=force_reprocess)


def print_dataset_summary():
    """Print comprehensive dataset summary."""
    dataset = WineQualityDataset()
    info = dataset.get_dataset_info()
    
    print("🍷 Wine Quality Dataset Summary")
    print("=" * 50)
    print(f"Samples: {info['num_samples']}")
    print(f"Features: {info['num_features']}")
    print(f"Classes: {info['num_classes']} (quality {info['target_range']})")
    print(f"\nClass Distribution:")
    for cls, count in info['class_distribution'].items():
        print(f"  Class {cls}: {count} samples ({count/info['num_samples']*100:.1f}%)")
    print(f"\nData Paths:")
    print(f"  Raw: {info['paths']['raw_data']}")
    print(f"  Processed: {info['paths']['processed_data']}")


# === DEMO ===

if __name__ == "__main__":
    print("🍷 Wine Quality Dataset Module Demo")
    print("=" * 50)
    
    # Print dataset summary
    print_dataset_summary()
    
    print(f"\n🔧 Testing data loaders...")
    train_loader, val_loader, test_loader = load_wine_quality_data()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    batch_X, batch_y = next(iter(train_loader))
    print(f"\nSample batch: {batch_X.shape} features, {batch_y.shape} labels")
    print(f"Feature range: [{batch_X.min():.3f}, {batch_X.max():.3f}]")
    print(f"Label range: [{batch_y.min()}, {batch_y.max()}]")
    
    # Test model
    print(f"\n🧠 Testing model architecture...")
    model = get_wine_quality_model('cpu')  # Use CPU for demo
    
    with torch.no_grad():
        output = model(batch_X)
        print(f"Model output shape: {output.shape}")
        print(f"Layer sizes: {model.get_layer_sizes()}")
        print(f"Activations captured: {list(model.activations.keys())}")
    
    print(f"\n✅ Wine Quality Dataset module ready for training!")