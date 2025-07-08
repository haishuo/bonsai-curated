#!/usr/bin/env python3
"""
Bonsai Pruning Methods - Production Implementation
=================================================

PURPOSE: Core Bonsai neural pruning methods only
PHILOSOPHY: Statistically rigorous, mathematically grounded pruning

Core Bonsai Methods:
1. Wilcoxon FANIM - Mathematically rigorous forward temporal differences
2. Wilcoxon BANIM - Production-efficient backward temporal differences

Uses:
- Shared infrastructure from shared/pruning_utils.py
- Modular statistical testing from 03_pruning/statistical_tests.py
- Forge configuration system for paths
"""

import numpy as np
import time
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Ensure we can import from the same directory and shared
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from shared.pruning_utils import (
    BasePruner, PruningResults, load_fanim_data, load_banim_data,
    DEVICE
)
from statistical_tests import (
    analyze_neural_impact_statistics, get_pruning_decisions,
    RiskProfile, PruningStatistics
)

# ===== STATISTICAL BONSAI METHODS =====

class WilcoxonFANIMPruner(BasePruner):
    """
    Wilcoxon statistical pruning using FANIM scores.
    
    Mathematical Framework:
    - Uses forward temporal differences: Δa_i = a_i(t+1) - a_i(t)
    - Theoretically rigorous Taylor series approximation
    - H₀: median(FANIM_i) = 0 [no net impact]
    - Decision: significant AND median > 0 → PRUNE
    """
    
    def __init__(self, model, test_loader, train_loader=None, 
                 risk_profile: RiskProfile = RiskProfile.PRODUCTION_ML,
                 alpha: float = 0.05, min_sample_size: int = 10):
        super().__init__(model, test_loader, train_loader)
        self.risk_profile = risk_profile
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        
    def prune(self) -> PruningResults:
        """Execute Wilcoxon FANIM pruning."""
        start_time = time.time()
        
        print(f"🌳 BONSAI WILCOXON FANIM PRUNING")
        print(f"📊 Risk Profile: {self.risk_profile.name}")
        print(f"🔬 Temporal Direction: Forward (mathematically rigorous)")
        
        # Load FANIM data
        fanim_data = load_fanim_data()
        
        # Perform statistical analysis
        statistics = analyze_neural_impact_statistics(
            impact_data=fanim_data,
            temporal_direction="FANIM", 
            risk_profile=self.risk_profile,
            alpha=self.alpha,
            min_sample_size=self.min_sample_size
        )
        
        # Extract pruning decisions
        prune_decisions = get_pruning_decisions(statistics)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create results using shared infrastructure
        return self._create_pruning_results(
            method_name=f"Wilcoxon_FANIM_{self.risk_profile.name}",
            prune_decisions=prune_decisions,
            processing_time=processing_time,
            statistical_info=statistics.__dict__
        )

class WilcoxonBANIMPruner(BasePruner):
    """
    Wilcoxon statistical pruning using BANIM scores.
    
    Engineering Framework:
    - Uses backward temporal differences: Δa_i = a_i(t) - a_i(t-1)
    - Production-efficient (no extra epoch required)
    - Mathematical compromise for computational efficiency
    - Same statistical testing as FANIM
    """
    
    def __init__(self, model, test_loader, train_loader=None,
                 risk_profile: RiskProfile = RiskProfile.PRODUCTION_ML,
                 alpha: float = 0.05, min_sample_size: int = 10):
        super().__init__(model, test_loader, train_loader)
        self.risk_profile = risk_profile
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        
    def prune(self) -> PruningResults:
        """Execute Wilcoxon BANIM pruning."""
        start_time = time.time()
        
        print(f"🌳 BONSAI WILCOXON BANIM PRUNING")
        print(f"📊 Risk Profile: {self.risk_profile.name}")
        print(f"🔧 Temporal Direction: Backward (production efficient)")
        
        # Load BANIM data
        banim_data = load_banim_data()
        
        # Perform statistical analysis
        statistics = analyze_neural_impact_statistics(
            impact_data=banim_data,
            temporal_direction="BANIM",
            risk_profile=self.risk_profile,
            alpha=self.alpha,
            min_sample_size=self.min_sample_size
        )
        
        # Extract pruning decisions
        prune_decisions = get_pruning_decisions(statistics)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create results using shared infrastructure
        return self._create_pruning_results(
            method_name=f"Wilcoxon_BANIM_{self.risk_profile.name}",
            prune_decisions=prune_decisions,
            processing_time=processing_time,
            statistical_info=statistics.__dict__
        )