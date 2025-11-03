# Quick Summary

## What This Repository Is

**Correct experimental design** for evolving unified transformer frameworks using LLM-guided evolution.

This fixes the fundamental misconfiguration in `../beyond-transformer/` where:
- ❌ Wrong: Evolved peripheral architecture (FFN, norms, residuals)
- ✅ Correct: Evolve attention mechanisms within unified frameworks

---

## Three Key Documents

### 1. [README.md](README.md) - Start Here
- Research motivation and background
- Overview of unified frameworks (3 papers)
- High-level experimental design
- Repository structure and getting started

### 2. [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) - Full Protocol
- Detailed methodology (36 pages)
- Exact specifications for all phases
- Statistical analysis protocols
- Success criteria and reporting standards

### 3. [LESSONS_FROM_FAILED_EXPERIMENT.md](LESSONS_FROM_FAILED_EXPERIMENT.md) - What Went Wrong
- Complete post-mortem of failed experiment
- Critical errors identified
- Why 50% fitness → -2.93% final performance
- Checklist to avoid same mistakes

---

## The Core Problem (Solved)

### Failed Experiment
```python
# WRONG: Evolved everything except attention
evolution.forbid("attention frameworks")
evolution.allow("FFN, normalization, residuals, block structure")

result = SwiGLU + RMSNorm + Parallel + LayerScale
# Generic architecture search, not unified frameworks research
```

### This Experiment
```python
# CORRECT: Evolve only attention mechanisms
evolution.allow("attention mechanisms within framework notation")
evolution.forbid("FFN, normalization, residuals, block structure")

result = Novel coefficient dynamics / regression / mixing
# Actually tests unified frameworks research question
```

---

## What Gets Evolved

### ✅ CAN Evolve
- Attention coefficient computation: `α = f(Q, K, V, params)`
- Mixing functions within framework notation
- Framework-specific parameters and mechanisms
- Novel dynamics/regression/mixing patterns

### ❌ CANNOT Evolve
- FFN activation (stays GELU, no SwiGLU)
- Normalization type (stays LayerNorm, no RMSNorm)
- Block structure (stays sequential, no parallel)
- Residual connections (stays standard, no LayerScale)

---

## Fitness Function (Fixed)

### ❌ Failed Experiment
```python
fitness = -scaling_slope at 2,000 steps  # Short-horizon proxy
# Result: "Fast learners" that plateau early
```

### ✅ This Experiment
```python
fitness = -scaling_slope at 100,000 steps  # Converged performance
# Result: "Good convergers" with better final performance
```

---

## Expected Workflow

### Phase 1: Evolution (per framework)
```bash
# Takes ~48-72 hours per framework
python evolution/evolve_coefficient_dynamics.py  # Framework 1
python evolution/evolve_test_time_regression.py  # Framework 2
python evolution/evolve_matrix_mixer.py          # Framework 3
```

### Phase 2: Validation
```bash
# Takes ~11 hours per framework × 2 variants
python training/validate_best.py --framework coefficient_dynamics
# Repeat for test_time_regression and matrix_mixer
```

### Phase 3: Analysis
```bash
# Cross-framework comparison
python analysis/compare_frameworks.py
```

---

## Key Differences from Failed Experiment

| Aspect | Failed | Correct |
|--------|--------|---------|
| **Evolution target** | Peripheral components | Attention mechanisms |
| **Fitness horizon** | 2k steps (proxy) | 100k steps (converged) |
| **Hyperparameters** | Baseline's only | Joint optimization |
| **Framework usage** | Fixed (coefficient_dynamics only) | All three evolved separately |
| **Research question** | Generic architecture search | Unified frameworks innovation |

---

## Success Criteria

**Minimum:**
- p < 0.05 (statistical significance)
- Improvement > 1% (practical significance)
- Consistent across ≥7/10 seeds

**Strong:**
- Cohen's |d| > 0.5 (medium effect)
- Better scaling law slope
- Interpretable innovation within framework

---

## Repository Status

**Current:** 📄 Documentation complete, no implementation yet

**Next Steps:**
1. Implement fixed transformer with evolvable attention
2. Create three framework implementations (coefficient_dynamics, test_time_regression, matrix_mixer)
3. Set up OpenEvolve configs with correct constraints
4. Run evolution experiments
5. Validate and analyze results

---

## Quick Navigation

- **Theory:** [README.md](README.md) - Unified frameworks background
- **Methods:** [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) - Full protocol
- **Lessons:** [LESSONS_FROM_FAILED_EXPERIMENT.md](LESSONS_FROM_FAILED_EXPERIMENT.md) - What went wrong
- **Failed experiment:** `../beyond-transformer/FINAL_EXPERIMENT_REPORT.md` - Complete analysis

---

## Citation

If you find this experimental design useful:

```bibtex
@misc{evolving_unified_transformers_2025,
  title={Evolving Unified Transformer Frameworks: Corrected Experimental Design},
  author={Research Project},
  year={2025},
  month={November},
  note={Corrects misconfiguration in previous experiment. 
        Evolution target: attention mechanisms within unified framework notation.
        Fitness: 100k-step converged performance (not 2k-step proxy).}
}
```

---

**Repository:** https://github.com/hz1ulqu01gmnZH4/evolving-unified-transformers  
**Created:** 2025-11-03  
**Status:** Pre-implementation (design phase)  
**Purpose:** LLM-guided evolution of attention mechanisms within unified frameworks
