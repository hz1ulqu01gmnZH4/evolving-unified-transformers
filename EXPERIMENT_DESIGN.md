# Detailed Experimental Design

## Overview

This document specifies the complete experimental protocol for evolving unified transformer frameworks using LLM-guided evolution.

---

## 1. Research Hypotheses

### Primary Hypothesis
**H1:** LLM-guided evolution can discover attention mechanism variants within unified framework notation that achieve better scaling laws than standard implementations.

### Secondary Hypotheses
**H2:** Different frameworks (coefficient dynamics vs regression vs matrix mixing) have different evolvability

**H3:** Evolved mechanisms will show interpretable innovations consistent with framework principles

**H4:** Improvements will generalize across model sizes (true scaling law improvement, not size-specific optimization)

---

## 2. Independent Variables

### 2.1 Framework Type (3 levels)
1. **Coefficient Dynamics** - Sieber et al. (2025)
2. **Test-Time Regression** - Wang et al. (2025)
3. **Matrix Mixer** - Hwang et al. (2024)

### 2.2 Implementation Variant (2 levels per framework)
1. **Standard** - Reference implementation from paper
2. **Evolved** - Best variant from 35 iterations of GPT-5 evolution

---

## 3. Dependent Variables

### Primary DV: Scaling Law Slope
```python
# Fit across 4 model sizes: 0.5M, 1M, 2M, 4M parameters
loss = intercept + slope × log(n_params)

# Lower slope (more negative) = better scaling
fitness = -slope
```

**Measurement:**
- Train each size for 100k steps (full convergence)
- Record final validation loss
- Fit linear regression in log-parameter space
- Report slope, R², confidence intervals

### Secondary DVs

1. **Final Loss** - Absolute performance at fixed size (3M params)
2. **Training Stability** - Gradient norm variance, loss smoothness
3. **Computational Cost** - FLOPs per forward pass
4. **Convergence Speed** - Steps to reach 90% of final performance

---

## 4. Control Variables

### 4.1 Fixed Architecture Components

**Must remain identical across all conditions:**

```python
# Embedding & Position
vocab_size = 16384  # BPE tokens
d_model = 128
pos_encoding = 'sinusoidal'  # Standard sin/cos

# FFN (FIXED - NOT evolvable)
d_ff = 512  # 4× expansion
activation = 'GELU'  # Standard, no SwiGLU
dropout_ffn = 0.1

# Normalization (FIXED - NOT evolvable)
norm_type = 'LayerNorm'  # NOT RMSNorm
norm_placement = 'pre'  # Pre-norm

# Block Structure (FIXED - NOT evolvable)
attention_ffn_order = 'sequential'  # NOT parallel
residual_type = 'standard'  # x + layer(norm(x)), NO LayerScale
num_layers = 6

# Attention Parameters (framework-specific but matched)
num_heads = 4
dropout_attn = 0.1
```

### 4.2 Training Hyperparameters

**Option A: Joint Optimization (Recommended)**
```python
# Find optimal hyperparameters for each variant
for variant in [standard, evolved]:
    hparams[variant] = optuna_search(
        variant,
        search_space={
            'learning_rate': [1e-4, 1e-3],
            'warmup_steps': [500, 2000],
            'weight_decay': [0, 0.1],
            'batch_size': [32, 64, 128]
        },
        n_trials=50,
        objective='validation_loss'
    )
```

**Option B: Shared Hyperparameters (Faster)**
```python
# Find hyperparameters for standard, use for both
shared_hparams = optuna_search(standard_implementation)
# Apply to both standard and evolved
```

**If using Option B, report both:**
- Standard with shared_hparams
- Evolved with shared_hparams
- Note: Evolved may be handicapped by non-optimal hyperparameters

### 4.3 Dataset & Tokenization

```python
dataset = 'TinyStories'  # Eldan & Li, 2023
tokenization = 'BPE'
vocab_size = 16384
train_sequences = 3_561_832
val_sequences = 35_886
seq_length = 128
```

**Dataset caching:**
- Pre-tokenize full dataset
- Cache to disk for reproducibility
- Use same cache across all experiments

---

## 5. Experimental Procedure

### Phase 1: Evolution (Per Framework)

**For each framework (3 runs):**

**1.1 Initial Implementation**
```python
# Create standard implementation from paper
standard_impl = StandardAttention(framework='coefficient_dynamics')

# Verify correctness
assert test_forward_pass(standard_impl)
assert test_backward_pass(standard_impl)
assert test_causality(standard_impl)
```

**1.2 OpenEvolve Configuration**
```yaml
# config_coefficient_dynamics.yaml
max_iterations: 35
checkpoint_interval: 5

llm:
  primary_model: "gpt-5"
  temperature: 0.8
  max_tokens: 16000

prompt:
  system_message: |
    You are evolving the ATTENTION MECHANISM within coefficient dynamics framework.

    **Can evolve:**
    - How coefficients α_t are computed from Q, K, V
    - Dynamics of coefficient evolution (if stateful)
    - Temperature, scaling, mixing functions
    - Learned parameters within attention

    **Cannot evolve:**
    - FFN (must stay standard GELU)
    - Normalization (must stay LayerNorm)
    - Block structure (must stay sequential)
    - Anything outside EVOLVE-BLOCK markers

    **Constraints:**
    - Must be expressible as α_t = f(inputs, params)
    - Maintain causality (no future information)
    - Output shape: [batch, seq_len, d_model]
    - Numerical stability (no NaN/Inf)

evaluator:
  # CRITICAL: Use 100k steps, not 2k!
  training_steps: 100000
  model_sizes: [0.5M, 1M, 2M, 4M]
  timeout: 7200  # 2 hours per evaluation
```

**1.3 Evolution Run**
```bash
python evolution/evolve_coefficient_dynamics.py \
  --config evolution/config_coefficient_dynamics.yaml \
  --output experiments/01_evolve_coefficient_dynamics/ \
  --gpu 0
```

**Duration:** ~48-72 hours (35 iterations × 4 sizes × ~30min each)

**1.4 Select Best Variant**
```python
# Load evolution history
history = load_evolution_history('experiments/01_evolve_coefficient_dynamics/')

# Select by fitness at iteration with best validation
best_iteration = max(history, key=lambda x: x['fitness'])
best_impl = load_program(best_iteration['program_id'])

# Save for validation phase
save_checkpoint(best_impl, 'evolved_coefficient_dynamics.pt')
```

### Phase 2: Validation (Per Framework)

**For each framework (3 runs):**

**2.1 Hyperparameter Optimization**

If using Option A (joint optimization):
```python
# Search for standard implementation
standard_hparams = optuna_study(
    model=standard_coefficient_dynamics,
    n_trials=100,
    training_steps=50000,  # Use 50k for HPO (faster)
    pruner=MedianPruner()
)

# Search for evolved implementation
evolved_hparams = optuna_study(
    model=evolved_coefficient_dynamics,
    n_trials=100,
    training_steps=50000,
    pruner=MedianPruner()
)
```

If using Option B (shared):
```python
shared_hparams = optuna_study(
    model=standard_coefficient_dynamics,
    n_trials=100,
    training_steps=50000
)
# Use for both
```

**2.2 Full Validation Run**

**Configuration:**
```python
validation_config = {
    'training_steps': 100000,
    'seeds': list(range(42, 52)),  # 10 seeds
    'model_size': '3M',  # Fixed size for direct comparison
    'dataset': 'TinyStories_BPE',
    'batch_size': 64,
    'eval_frequency': 1000
}
```

**Run:**
```bash
# Standard implementation (10 seeds)
for seed in 42..51; do
    python training/train.py \
        --model standard_coefficient_dynamics \
        --hparams standard_hparams.json \
        --seed $seed \
        --output experiments/02_validate_coefficient_dynamics/standard/seed_$seed/
done

# Evolved implementation (10 seeds)
for seed in 42..51; do
    python training/train.py \
        --model evolved_coefficient_dynamics \
        --hparams evolved_hparams.json \
        --seed $seed \
        --output experiments/02_validate_coefficient_dynamics/evolved/seed_$seed/
done
```

**Duration:** ~11 hours (20 runs × ~33 minutes each)

**2.3 Scaling Law Validation**

```python
# Train 4 model sizes for standard and evolved
model_sizes = [0.5M, 1M, 2M, 4M]

for size in model_sizes:
    # Standard
    train(standard, size, hparams=standard_hparams, steps=100000, seed=42)

    # Evolved
    train(evolved, size, hparams=evolved_hparams, steps=100000, seed=42)

# Fit scaling laws
standard_slope, standard_r2 = fit_scaling_law(standard_results)
evolved_slope, evolved_r2 = fit_scaling_law(evolved_results)

# Compare
improvement = ((standard_slope - evolved_slope) / standard_slope) * 100
```

### Phase 3: Statistical Analysis

**3.1 Paired t-test**
```python
# Compare losses across 10 seeds
from scipy import stats

standard_losses = [load_result(f'standard/seed_{s}') for s in range(42, 52)]
evolved_losses = [load_result(f'evolved/seed_{s}') for s in range(42, 52)]

t_stat, p_value = stats.ttest_rel(standard_losses, evolved_losses)
```

**3.2 Effect Size**
```python
# Cohen's d for paired samples
differences = np.array(standard_losses) - np.array(evolved_losses)
cohens_d = np.mean(differences) / np.std(differences, ddof=1)

# Interpretation
if abs(cohens_d) < 0.2:
    effect = 'negligible'
elif abs(cohens_d) < 0.5:
    effect = 'small'
elif abs(cohens_d) < 0.8:
    effect = 'medium'
else:
    effect = 'large'
```

**3.3 Confidence Intervals**
```python
from scipy import stats

# 95% CI for mean difference
mean_diff = np.mean(differences)
sem = stats.sem(differences)
ci = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=sem)

print(f"Mean improvement: {mean_diff:.4f}")
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

### Phase 4: Cross-Framework Comparison

**4.1 Compare Best Variants**
```python
frameworks = [
    'coefficient_dynamics',
    'test_time_regression',
    'matrix_mixer'
]

results = {}
for framework in frameworks:
    # Load best evolved variant
    model = load_best(f'experiments/{framework}/evolved/')

    # Evaluate on common test set
    results[framework] = evaluate(model, test_set, seeds=range(42, 52))

# Statistical comparison
anova_result = stats.f_oneway(*[results[f] for f in frameworks])
```

**4.2 Computational Cost Analysis**
```python
from torch.utils.flop_counter import FlopCounterMode

for framework in frameworks:
    model = load_best(framework)

    with FlopCounterMode(model) as counter:
        output = model(sample_input)
        flops = counter.get_total_flops()

    print(f"{framework}: {flops/1e9:.2f} GFLOPs")
```

---

## 6. Success Criteria

### Minimum Viable Success

**For evolution to be considered successful:**
1. **Statistical significance:** p < 0.05 (two-tailed)
2. **Practical significance:** Improvement > 1% on final loss
3. **Reproducibility:** Consistent across majority of seeds (≥7/10)
4. **Scaling law improvement:** Better slope across model sizes

### Strong Success

**Additional criteria for strong success:**
1. **Effect size:** Cohen's |d| > 0.5 (medium effect)
2. **Generalization:** Improvement on held-out test set
3. **Efficiency:** No increase in FLOPs (or <10% increase)
4. **Interpretability:** Evolved mechanism has clear interpretation

---

## 7. Reporting Standards

### Required Reporting

**For each framework experiment, report:**

1. **Evolution Results**
   - Number of iterations run
   - Best fitness achieved
   - Fitness progression plot
   - Best model iteration and code diff

2. **Validation Results**
   - Mean ± std loss for standard (10 seeds)
   - Mean ± std loss for evolved (10 seeds)
   - Paired t-test: t-statistic, p-value, df
   - Effect size: Cohen's d with interpretation
   - 95% confidence intervals
   - Per-seed results table

3. **Scaling Law Results**
   - Losses at each model size (0.5M, 1M, 2M, 4M)
   - Fitted slopes with R² values
   - Slope comparison with statistical test
   - Scaling law plots

4. **Computational Cost**
   - FLOPs per forward pass (standard vs evolved)
   - Training time per epoch
   - Memory usage

5. **Qualitative Analysis**
   - Code diff showing evolved changes
   - Interpretation of evolved mechanism
   - Connection to framework principles
   - Potential failure modes or limitations

### Visualization Requirements

**Required plots:**
1. Evolution fitness progression (with best trajectory)
2. Box plots comparing standard vs evolved (10 seeds)
3. Per-seed bar chart
4. Scaling law curves (4 model sizes)
5. Training curves (loss over steps, standard vs evolved)
6. Attention pattern visualizations (if interpretable)

---

## 8. Negative Result Protocols

### If Evolution Fails to Improve

**Possible reasons to investigate:**

1. **Framework constraints too restrictive**
   - Analysis: Try manually designed variants
   - Report: What constraints limited search space?

2. **Standard implementation already optimal**
   - Analysis: Literature review of attention variants
   - Report: Why might standard be optimal for this task/scale?

3. **Fitness function issues**
   - Analysis: Check if 100k steps is sufficient convergence
   - Analysis: Try different model sizes
   - Report: Sensitivity to training duration and scale

4. **LLM limitations**
   - Analysis: Examine proposed mutations (quality, diversity)
   - Report: What types of mutations did GPT-5 propose?
   - Compare: Try different LLMs (Claude, Gemini)

5. **Task mismatch**
   - Analysis: Try different dataset (WikiText, C4)
   - Report: Is TinyStories too simple to show differences?

### Reporting Negative Results

**If evolved ≤ standard (not significant):**
1. Report full methodology and results
2. Analyze failure modes
3. Discuss implications (standard formulations validated)
4. Suggest future work to overcome limitations

**Do NOT:**
- Cherry-pick successful seeds
- Hide negative results
- Modify success criteria post-hoc
- Over-interpret marginal differences

---

## 9. Reproducibility Checklist

**Before claiming results, ensure:**

- [ ] All random seeds documented and fixed
- [ ] Full hyperparameter configurations saved
- [ ] Dataset version and preprocessing script provided
- [ ] Evolution config files version controlled
- [ ] Checkpoint files saved for best models
- [ ] Training logs preserved
- [ ] GPU type and CUDA version documented
- [ ] Python package versions frozen (requirements.txt)
- [ ] Code for all analysis scripts provided
- [ ] Statistical test implementations documented

---

## 10. Ethical Considerations

### Compute Cost

**Estimated total compute:**
- Evolution: 3 frameworks × 35 iterations × 4 sizes × 30min = ~210 GPU-hours
- Validation: 3 frameworks × 2 variants × 10 seeds × 33min = ~33 GPU-hours
- Scaling laws: 3 frameworks × 2 variants × 4 sizes × 33min = ~13 GPU-hours
- **Total: ~256 GPU-hours** (~10 GPU-days)

**Cost estimate (A100 at $2/hour):** ~$512

**Carbon footprint:** Document GPU type and estimate CO₂ emissions

### Publication Ethics

- Pre-register experiment plan before running
- Report all results (including negative)
- Share code and data
- Acknowledge computational resources
- Disclose any conflicts of interest

---

## 11. Timeline

**Estimated timeline for complete experiment:**

| Phase | Duration | Parallelization |
|-------|----------|-----------------|
| Setup & Testing | 1 week | N/A |
| Evolution (3 frameworks) | 1 week | 3× parallel |
| Validation (3 frameworks) | 2 days | 3× parallel |
| Scaling laws | 1 day | 6× parallel |
| Analysis & Writing | 1 week | N/A |
| **Total** | **3 weeks** | |

**With single GPU:** 4-6 weeks

---

## 12. Stopping Criteria

**Stop evolution early if:**
1. No improvement for 10 consecutive iterations
2. LLM produces invalid code repeatedly (>5 failures in a row)
3. Fitness degrades below baseline by >50%
4. Numerical instabilities (NaN/Inf) in majority of iterations

**Stop validation early if:**
1. Clear divergence in training (loss → ∞)
2. Results after 5 seeds show huge effect (|d| > 2.0) with p < 0.001
3. Computational resource limits reached

---

**Document version:** 1.0
**Last updated:** 2025-11-03
**Status:** Pre-experiment specification
