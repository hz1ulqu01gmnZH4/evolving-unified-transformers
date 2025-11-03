# Lessons from the Failed Experiment

## Executive Summary

The previous experiment in `../beyond-transformer/` was **fundamentally misconfigured** from the start. While it produced rigorous validation results, it tested the **wrong hypothesis** due to a mismatch between intended research goals and actual experimental setup.

**Intended:** Evolve attention mechanisms within unified framework notation

**Actually tested:** Generic architecture search with fixed attention mechanisms

This document analyzes what went wrong and how to avoid similar mistakes.

---

## 1. The Fundamental Misconfiguration

### What Was Supposed to Happen

**Original intention:**
```python
# Evolve the ATTENTION MECHANISM within unified frameworks
class EvolvableAttention(nn.Module):
    # EVOLVE-BLOCK: Modify coefficient dynamics, regression, or mixing
    def forward(self, x):
        # Standard: α = softmax(QK^T/√d)
        # Evolved: α = novel_function(Q, K, learned_params)
        pass
```

**Keep everything else fixed:**
- FFN: Standard GELU (no SwiGLU evolution)
- Normalization: Standard LayerNorm (no RMSNorm evolution)
- Block structure: Sequential (no parallel evolution)

### What Actually Happened

**OpenEvolve configuration said:**
```yaml
**Important constraints:**
- Do NOT modify the attention frameworks themselves (they're imported from causal_wrappers)
```

**This directly contradicted the research goal!**

Evolution was **forbidden** from modifying attention mechanisms and instead evolved peripheral components:
- FFN: GELU → SwiGLU (gated activation)
- Normalization: LayerNorm → RMSNorm  
- Block structure: Sequential → Parallel branches
- Added: LayerScale, learnable branch weights

**Result:** A generic architecture search completely disconnected from unified frameworks research.

---

## 2. Critical Errors in Experimental Design

### Error #1: Short-Horizon Fitness Function

**What was used:**
```python
fitness = -slope at 2,000 training steps
```

**Why this failed:**
- Optimized for "fast learners" (quick early improvement)
- Not for "good convergers" (better final performance)
- Created selection pressure for architectures that plateau early

**Evidence of failure:**
- At 2k steps: +50% fitness improvement ✓
- At 50k steps: +0.84-3.13% (marginal)
- At 100k steps: -2.93% (significantly worse!) ✗

**Lesson:** **Never use proxy metrics for fitness.** Early training dynamics ≠ final performance.

**Correct approach:**
```python
fitness = -slope at 100,000 training steps (converged)
```

### Error #2: Hyperparameter Mismatch

**What was used:**
```python
# Find hyperparameters for baseline
baseline_hparams = optuna_search(baseline_architecture)

# Use baseline's hyperparameters for evolved (!!)
evolved_result = train(evolved_architecture, baseline_hparams)
```

**Why this failed:**
- Modern components (SwiGLU, RMSNorm) may need different learning rates
- Evolved architecture never got a fair chance with tuned hyperparameters
- Like comparing a marathoner in sprinter's shoes to a sprinter in their own shoes

**Lesson:** Either jointly optimize hyperparameters or use shared hyperparameters found for both.

**Correct approach:**
```python
# Option A: Joint optimization (better)
standard_hparams = optuna_search(standard)
evolved_hparams = optuna_search(evolved)

# Option B: Shared hyperparameters (faster, still fair)
shared_hparams = optuna_search(standard)
# Use same for both
```

### Error #3: Unclear Evolution Scope

**What the config said:**
```yaml
**What you can evolve:**
1. TransformerFFN: activation functions, gating...
2. UnifiedTransformerBlock: normalization, residuals...

**What you cannot evolve:**
- Do NOT modify the attention frameworks
```

**The contradiction:**
- Allowed evolving FFN/normalization (not related to unified frameworks)
- Forbidden evolving attention (the actual unified frameworks research topic)

**Lesson:** Be extremely explicit about what is and isn't evolvable.

**Correct approach:**
```yaml
**ONLY attention mechanisms can evolve:**
- EVOLVE-BLOCK markers only around attention classes
- All other components (FFN, norm, residuals) explicitly fixed
- Clear constraints on what must be preserved (causality, shape, etc.)
```

### Error #4: No Framework Comparison

**What was missing:**
The three unified frameworks were never compared:
- Coefficient Dynamics
- Test-Time Regression
- Matrix Mixer

All experiments used `coefficient_dynamics` only. The research motivation (comparing unified frameworks) was never tested.

**Lesson:** If your research is about comparing frameworks, actually compare them!

**Correct approach:**
Run separate evolution for each framework, then compare best variants.

---

## 3. Statistical Rigor (This Part Was Actually Good!)

### What Was Done Correctly

**Initial validation (insufficient):**
- 3 seeds → marginal results (+0.84-3.13%)
- Statistical uncertainty (p=0.33)

**Expert consultation:**
- Identified short-horizon fitness as root cause
- Recommended BPE tokenization, longer training, more seeds

**Rigorous final validation:**
- 10 seeds (proper statistical power)
- 100k training steps (full convergence)
- BPE tokenization (realistic task)
- Full dataset (3.6M sequences)

**Results:**
- -2.93% degradation (p < 0.0001, Cohen's d = -6.87)
- Consistent across all 10 seeds
- Clear, definitive conclusion

**Lesson:** When initial results are ambiguous, increase rigor rather than accepting marginal differences.

---

## 4. Misleading Initial Success

### The "50% Improvement" Trap

**Evolution reported:**
```
Best model: Iteration 15
Fitness improvement: +50.4%
Components: SwiGLU, RMSNorm, parallel branches, LayerScale
```

**Why this was misleading:**
1. Measured at 2k steps (short horizon)
2. Fitness function optimized for wrong thing
3. Looked like huge success, was actually misleading signal

**Reality check:**
```
2k steps: +50% fitness ✓ (what fitness function saw)
50k steps: +0.84-3.13% (diminishing)
100k steps: -2.93% ✗ (final truth)
```

**Lesson:** Be skeptical of impressive results on proxy metrics. Always validate with final objective.

---

## 5. Architectural Analysis

### Modern Components ≠ Automatic Improvement

**Evolved architecture used:**
- SwiGLU (used in PaLM, LLaMA 2)
- RMSNorm (used in T5, LLaMA)
- Parallel branches (used in PaLM)
- LayerScale (used in Vision Transformers)

**All proven techniques at large scale (>100M params)**

**But at small scale (3M params):**
- Combination performed 2.93% worse
- Possible reasons:
  - Scale mismatch (components designed for larger models)
  - Hyperparameter mismatch (not tuned for evolved architecture)
  - Task mismatch (TinyStories too simple)
  - Overengineering (too complex for problem)

**Lesson:** Context matters. Techniques that work at one scale/task/setting may not transfer.

---

## 6. The "Sprinter vs Marathoner" Problem

### Visualizing the Failure Mode

```
Learning Dynamics Over Training:

Steps:        0     2k    10k   50k   100k
            ─────────────────────────────────
Baseline:   High  Mid   Mid   Low   Lower  (Marathoner)
Evolved:    High  Low!  Low   Mid   Higher (Sprinter)
                  ↑
              Fitness evaluation
              (misleading signal!)
```

**What fitness function saw:**
- Evolved better at 2k steps ✓
- Selection pressure for "fast learners"

**What actually matters:**
- Baseline better at 100k steps ✓
- Final convergence is what counts

**Lesson:** Short-horizon optimization creates fundamentally wrong selection pressure.

---

## 7. Validity Threats We Identified

### Internal Validity

**Good:**
- ✓ FLOPs perfectly matched (1.000x ratio)
- ✓ Parameter count matched
- ✓ Statistical power (10 seeds)

**Questionable:**
- ⚠ Hyperparameters optimized for baseline only
- ⚠ Evolved may have been handicapped

### External Validity

**Limitations:**
- Single dataset (TinyStories)
- Single task (language modeling)
- Single model size (~3M params)
- Results may not generalize

### Construct Validity

**Questions:**
- Does TinyStories adequately test architectural differences?
- Are 3M params sufficient to show benefits?
- Would results differ on more complex tasks?

**Lesson:** Acknowledge limitations honestly. Don't overclaim generalization.

---

## 8. What Should Have Been Done

### Correct Experimental Design

**Phase 1: Setup**
```python
# 1. Create three fixed architectures (ONLY attention evolvable)
class FixedTransformer:
    def __init__(self, attention_framework):
        self.attention = EvolvableAttention(framework)  # EVOLVABLE
        self.ffn = StandardFFN()  # FIXED: GELU
        self.norm1 = LayerNorm()  # FIXED: LayerNorm
        self.norm2 = LayerNorm()  # FIXED: LayerNorm
```

**Phase 2: Evolution (per framework)**
```python
# Evolve ONLY attention mechanisms
for framework in ['coefficient_dynamics', 'test_time_regression', 'matrix_mixer']:
    best = evolve(
        framework=framework,
        fitness=lambda model: -scaling_slope_at_100k_steps(model),
        iterations=35
    )
```

**Phase 3: Validation**
```python
# Compare standard vs evolved with fair hyperparameters
for framework in frameworks:
    standard_hparams = optuna_search(standard[framework])
    evolved_hparams = optuna_search(evolved[framework])

    results[framework] = compare(
        standard[framework] + standard_hparams,
        evolved[framework] + evolved_hparams,
        seeds=10,
        steps=100k
    )
```

---

## 9. Key Takeaways

### For Fitness Function Design

1. **Use final objective, not proxies**
   - Early training ≠ final performance
   - Validate fitness function before evolution

2. **Consider time horizon carefully**
   - Short horizons can mislead (sprinter vs marathoner)
   - Use sufficient convergence time

3. **Check correlation**
   - Does proxy correlate with final objective?
   - Measure correlation on pilot data

### For Experimental Setup

1. **Be explicit about evolution scope**
   - Use EVOLVE-BLOCK markers clearly
   - Document what can and cannot change
   - Verify LLM respects constraints

2. **Control for confounds**
   - Match hyperparameters fairly
   - Match computational cost (FLOPs)
   - Match model capacity (parameters)

3. **Plan for negative results**
   - Pre-register hypotheses
   - Define success criteria in advance
   - Report failures honestly

### For Statistical Rigor

1. **Power analysis**
   - Calculate required sample size
   - Use enough seeds (≥10 for t-tests)

2. **Multiple testing correction**
   - If testing multiple frameworks, correct p-values
   - Bonferroni or Holm-Bonferroni

3. **Effect size matters**
   - Don't rely only on p-values
   - Report Cohen's d, confidence intervals
   - Discuss practical significance

### For Research Communication

1. **Be clear about what was tested**
   - Intended research question
   - Actual experimental implementation
   - Any mismatches between them

2. **Acknowledge limitations**
   - Validity threats
   - Generalization concerns
   - Alternative explanations

3. **Learn from failures**
   - Negative results are valuable
   - Document what went wrong
   - Help others avoid same mistakes

---

## 10. Positive Outcomes from Failed Experiment

### What We Learned

**1. Methodological lessons:**
- Short-horizon fitness functions are dangerous
- Hyperparameter matching is critical
- Statistical rigor reveals ground truth

**2. Empirical findings:**
- Modern components (SwiGLU, RMSNorm) don't guarantee improvement
- Context and scale matter for architectural choices
- Evolution can optimize wrong objectives efficiently

**3. Meta-lessons:**
- LLMs can generate syntactically correct, runnable code
- But evolution still needs correct fitness function
- Human expertise essential for experimental design

### Value of Rigorous Validation

**The experiment sequence showed:**
1. Initial success (+50% fitness at 2k steps)
2. Ambiguous signals (+0.84-3.13% at 50k steps)
3. Expert consultation (identified root cause)
4. Rigorous validation (-2.93% at 100k steps)

**This process is valuable:**
- Shows importance of skepticism about early results
- Demonstrates value of expert consultation
- Validates proper experimental controls

**Lesson:** The journey from misleading success to definitive failure is scientifically valuable.

---

## 11. Checklist for Future Experiments

### Before Running Evolution

- [ ] Clearly define research question
- [ ] Specify exactly what can/cannot evolve
- [ ] Validate fitness function on pilot data
- [ ] Check fitness correlates with final objective
- [ ] Document evolution constraints explicitly
- [ ] Set up monitoring for constraint violations

### During Evolution

- [ ] Check that LLM respects constraints
- [ ] Monitor fitness progression
- [ ] Save all generated code for analysis
- [ ] Track which components actually changed
- [ ] Verify evolved code is runnable

### Before Claiming Results

- [ ] Validate with sufficient seeds (≥10)
- [ ] Train to full convergence (not proxy)
- [ ] Match or optimize hyperparameters fairly
- [ ] Verify computational fairness (FLOPs)
- [ ] Run statistical tests properly
- [ ] Check for alternative explanations

### When Reporting

- [ ] Report intended vs actual experiment
- [ ] Document all design choices
- [ ] Include negative results
- [ ] Discuss limitations honestly
- [ ] Provide code and data for reproducibility
- [ ] Acknowledge what went wrong (if applicable)

---

## 12. Conclusion

The failed experiment in `../beyond-transformer/` provides a valuable **cautionary tale** about:

1. **Fitness function design** - Short-horizon proxies can catastrophically mislead
2. **Experimental constraints** - Be explicit about what evolves
3. **Hyperparameter fairness** - Match or jointly optimize
4. **Statistical rigor** - Use sufficient power and proper controls
5. **Research clarity** - Ensure experiment tests intended hypothesis

**The new experiment** (`evolving-unified-transformers`) corrects all these issues:
- ✓ Fitness at 100k steps (not 2k)
- ✓ Only attention mechanisms evolve (not peripheral components)
- ✓ Hyperparameters optimized fairly
- ✓ Tests actual research question (unified frameworks)
- ✓ Pre-registered design and stopping criteria

---

**Document version:** 1.0  
**Created:** 2025-11-03  
**Purpose:** Learn from mistakes, improve future experiments  
**Status:** Post-mortem analysis of completed failed experiment
