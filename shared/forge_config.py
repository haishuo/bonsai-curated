"""
Forge Configuration Management
=============================

PURPOSE: Centralized configuration for the forge ML machine
ARCHITECTURE: Leverages forge's dual-drive setup optimized for ML workloads

Drive 1 (Projects): /mnt/projects/ - Source code, configs, environments  
Drive 2 (Data): /mnt/data/ - Datasets, databases, scratch files
Drive 2 (Artifacts): /mnt/artifacts/ - Model outputs, logs, trained weights

PHILOSOPHY: One config, consistent paths, optimized for RTX 5070Ti workflows
"""

from pathlib import Path
from typing import Dict, Optional
import os
import json
from datetime import datetime


class ForgeConfig:
    """
    Centralized configuration for forge machine ML workflows.
    
    Manages paths across forge's optimized dual-drive architecture:
    - Projects drive: Source code and configurations
    - Data drive: Datasets and scratch files  
    - Artifacts drive: Model outputs and results
    """
    
    def __init__(self, project_name: str = "bonsai"):
        self.project_name = project_name
        
        # Base mount points (forge-specific)
        self.projects_base = Path("/mnt/projects")
        self.data_base = Path("/mnt/data") 
        self.artifacts_base = Path("/mnt/artifacts")
        
        # Project-specific paths
        self.project_root = self.projects_base / f"project_{project_name}"
        self.data_root = self.data_base / project_name
        self.artifacts_root = self.artifacts_base / project_name
        
        # Ensure critical directories exist
        self._ensure_directory_structure()
    
    # === PROJECT PATHS (Drive 1: /mnt/projects) ===
    
    @property
    def quick_dir(self) -> Path:
        """Active development directory."""
        return self.project_root / "quick"
    
    @property
    def archive_dir(self) -> Path:
        """Archived research code."""
        return self.project_root / "archive"
    
    @property
    def configs_dir(self) -> Path:
        """Configuration files and experiment specs."""
        return self.quick_dir / "configs"
    
    @property
    def notebooks_dir(self) -> Path:
        """Jupyter notebooks for analysis."""
        return self.quick_dir / "notebooks"
    
    # === DATA PATHS (Drive 2: /mnt/data) ===
    
    @property
    def datasets_dir(self) -> Path:
        """Datasets for this specific project (bonsai-curated)."""
        return self.data_base / "bonsai" / "quick" / "datasets"
    
    @property
    def models_dir(self) -> Path:
        """Trained model files (.pth, .pkl)."""
        return self.data_base / "bonsai" / "quick" / "models"
    
    @property
    def nim_data_dir(self) -> Path:
        """Neuron Impact Metrics and FANIM files (.h5)."""
        return self.data_base / "bonsai" / "quick" / "nim_data"
    
    @property
    def scratch_dir(self) -> Path:
        """Temporary files and intermediate data."""
        return self.data_root / "scratch"
    
    # === ARTIFACTS PATHS (Drive 2: /mnt/artifacts) ===
    
    @property  
    def results_dir(self) -> Path:
        """Experimental results (JSON, CSV)."""
        return self.artifacts_root / "results"
    
    @property
    def plots_dir(self) -> Path:
        """Generated visualizations and figures."""
        return self.artifacts_root / "plots"
    
    @property
    def reports_dir(self) -> Path:
        """Analysis reports and documentation."""
        return self.artifacts_root / "reports"
    
    @property
    def logs_dir(self) -> Path:
        """Training logs and debug output."""
        return self.artifacts_root / "logs"
    
    # === DATASET-SPECIFIC PATHS ===
    
    def dataset_path(self, dataset_name: str) -> Path:
        """Get path for specific dataset."""
        return self.datasets_dir / dataset_name
    
    def model_path(self, model_name: str, extension: str = ".pth") -> Path:
        """Get path for specific model file."""
        if not extension.startswith('.'):
            extension = f'.{extension}'
        return self.models_dir / f"{model_name}{extension}"
    
    def nim_data_path(self, experiment_name: str, suffix: str = "nim") -> Path:
        """Get path for NIM/FANIM data."""
        return self.nim_data_dir / f"{experiment_name}_{suffix}.h5"
    
    def results_path(self, experiment_name: str, suffix: str = "results") -> Path:
        """Get path for experimental results."""
        return self.results_dir / f"{experiment_name}_{suffix}.json"
    
    def plot_path(self, plot_name: str, extension: str = ".png") -> Path:
        """Get path for plot files."""
        if not extension.startswith('.'):
            extension = f'.{extension}'
        return self.plots_dir / f"{plot_name}{extension}"
    
    # === EXPERIMENT MANAGEMENT ===
    
    def create_experiment_paths(self, experiment_name: str) -> Dict[str, Path]:
        """
        Create all paths for a new experiment.
        
        Returns:
            Dictionary with all relevant paths for the experiment
        """
        paths = {
            'model': self.model_path(experiment_name),
            'nim_data': self.nim_data_path(experiment_name, "fanim"),
            'results': self.results_path(experiment_name),
            'plots_dir': self.plots_dir / experiment_name,
            'logs_dir': self.logs_dir / experiment_name,
        }
        
        # Ensure experiment directories exist
        paths['plots_dir'].mkdir(parents=True, exist_ok=True)
        paths['logs_dir'].mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def get_timestamp_suffix(self) -> str:
        """Get timestamp for experiment versioning."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # === SYSTEM UTILITIES ===
    
    def _ensure_directory_structure(self):
        """Ensure all critical directories exist."""
        directories = [
            # Data directories for bonsai-curated
            self.datasets_dir,
            self.models_dir, 
            self.nim_data_dir,
            self.scratch_dir,
            # Artifact directories  
            self.results_dir,
            self.plots_dir,
            self.reports_dir,
            self.logs_dir,
            # Project directories
            self.configs_dir,
            self.notebooks_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def check_disk_space(self) -> Dict[str, Dict[str, float]]:
        """Check available disk space on both drives."""
        def get_disk_usage(path: Path) -> Dict[str, float]:
            """Get disk usage stats in GB."""
            if path.exists():
                stat = os.statvfs(path)
                total = (stat.f_blocks * stat.f_frsize) / (1024**3)
                free = (stat.f_bavail * stat.f_frsize) / (1024**3)
                used = total - free
                return {
                    'total_gb': round(total, 2),
                    'used_gb': round(used, 2),
                    'free_gb': round(free, 2),
                    'used_percent': round((used / total) * 100, 1)
                }
            return {'error': 'Path not accessible'}
        
        return {
            'projects_drive': get_disk_usage(self.projects_base),
            'data_drive': get_disk_usage(self.data_base),
        }
    
    def save_config_snapshot(self, experiment_name: str, config_data: Dict) -> Path:
        """Save configuration snapshot for experiment reproducibility."""
        config_file = self.configs_dir / f"{experiment_name}_config.json"
        
        snapshot = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'forge_paths': {
                'datasets': str(self.datasets_dir),
                'models': str(self.models_dir),
                'nim_data': str(self.nim_data_dir),
                'results': str(self.results_dir),
            },
            'config': config_data
        }
        
        with open(config_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        return config_file
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""ForgeConfig(project='{self.project_name}')
Projects: {self.project_root}
Data: {self.data_root}  
Artifacts: {self.artifacts_root}"""
    
    def __repr__(self) -> str:
        return f"ForgeConfig(project_name='{self.project_name}')"


# === CONVENIENCE FUNCTIONS ===

def get_bonsai_config() -> ForgeConfig:
    """Get standard Bonsai project configuration."""
    return ForgeConfig("bonsai")


def setup_experiment(experiment_name: str, config_data: Optional[Dict] = None) -> Dict[str, Path]:
    """
    Quick setup for new experiment with all paths.
    
    Args:
        experiment_name: Name of the experiment
        config_data: Optional configuration to save
        
    Returns:
        Dictionary of all experiment paths
    """
    forge_config = get_bonsai_config()
    paths = forge_config.create_experiment_paths(experiment_name)
    
    if config_data:
        forge_config.save_config_snapshot(experiment_name, config_data)
    
    return paths


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    # Demo the configuration system
    print("🔧 Forge Configuration System Demo")
    print("=" * 50)
    
    config = get_bonsai_config()
    print(config)
    
    print(f"\n📁 Key Directories:")
    print(f"Datasets: {config.datasets_dir}")
    print(f"Models: {config.models_dir}")
    print(f"NIM Data: {config.nim_data_dir}")
    print(f"Results: {config.results_dir}")
    print(f"Plots: {config.plots_dir}")
    
    print(f"\n💾 Disk Usage:")
    disk_usage = config.check_disk_space()
    for drive, stats in disk_usage.items():
        if 'error' not in stats:
            print(f"{drive}: {stats['used_gb']:.1f}GB used / {stats['total_gb']:.1f}GB total ({stats['used_percent']:.1f}%)")
    
    print(f"\n🧪 Example Experiment Setup:")
    paths = setup_experiment("wine_quality_test")
    for name, path in paths.items():
        print(f"{name}: {path}")