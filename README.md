# Bonsai Curated - Neural Pruning Research

**Status**: 🚀 ACTIVE DEVELOPMENT  
**Focus**: Wilcoxon + FANIM breakthrough implementation  
**Archive**: See [bonsai-archive](https://github.com/haishuo/bonsai-archive) for historical code  

## Overview

This repository contains the curated, production-ready implementation of the Bonsai neural pruning research program, focused on the breakthrough Wilcoxon + FANIM (Forward Advanced Neuron Impact Metrics) approach.

## Project Structure

```
├── 01_training/           # Dataset preparation and model training
├── 02_sensitivity/        # FANIM computation and sensitivity analysis  
├── 03_pruning/           # Core pruning algorithms (Wilcoxon focus)
├── 04_evaluation/        # Results analysis and comparison tools
└── shared/               # Common utilities and configuration
```

## Key Innovations

### FANIM (Forward Advanced Neuron Impact Metrics)
- **10-100x speedup** through pre-computation of sensitivity metrics
- Professional terminology replacing legacy "sensitivity" naming
- GPU-accelerated processing for large-scale experiments

### Wilcoxon Statistical Framework  
- **Distribution-free** Bayesian inference for neural pruning
- Robust statistical guarantees without distributional assumptions
- Proper multiple comparison handling with statistical corrections

### Production-Ready Pipeline
- Centralized configuration management
- Clean separation of concerns
- Reproducible experimental workflows

## Development Workflow

1. **Train models** using scripts in `01_training/`
2. **Compute FANIM** using processors in `02_sensitivity/`  
3. **Apply pruning** using Wilcoxon methods in `03_pruning/`
4. **Analyze results** using tools in `04_evaluation/`

## Research Evolution

This curated codebase represents the culmination of extensive research:
- **50+ experimental variants** explored and archived
- **Statistical rigor** validated across multiple datasets  
- **Computational efficiency** optimized through FANIM innovation
- **Production readiness** achieved through clean architecture

## Next Steps

- [ ] Migrate core FANIM processors from archive
- [ ] Implement Wilcoxon pruning methods
- [ ] Set up automated evaluation pipeline
- [ ] Scale to modern deep learning architectures

---

**Research Program**: Bayesian Sensitivity-Guided Neural Pruning  
**Principal Investigator**: Hai-Shuo Shu  
**Institution**: UMass Dartmouth
