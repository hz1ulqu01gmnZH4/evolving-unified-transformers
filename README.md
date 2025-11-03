# Evolving Unified Transformer Frameworks

**LLM-guided evolution of attention mechanisms within unified framework notation using OpenEvolve + GPT-5**

---

## 🎯 Research Question

**Can LLM-guided evolution discover novel attention mechanism variants within unified framework formalisms that outperform standard implementations?**

This project explores using Large Language Models (GPT-5) to evolve the **mathematical formulations** of attention mechanisms while constraining them to remain within established unified framework notations.

---

## 📚 Background: Unified Frameworks

Three recent papers provide complementary unified views of sequence models:

### 1. Coefficient Dynamics (Sieber et al., 2025)
**Framework:** All sequence models compute outputs as linear combinations where coefficients evolve via linear dynamics

```
output_t = Σ α_t,i · V_i  where α evolves via autonomous linear dynamics
```

**Standard implementation:** Softmax attention with fixed dynamics
**Evolution goal:** Discover novel coefficient evolution rules

### 2. Test-Time Regression (Wang et al., 2025)
**Framework:** Sequence models perform associative recall via regression at test time

```
memorization: store (K, V) pairs
retrieval: y = regression(K, V, query)
```

**Standard implementation:** Softmax kernel regression
**Evolution goal:** Discover novel regression kernels/methods

### 3. Matrix Mixers (Hwang et al., 2024)
**Framework:** Sequence mixing as structured matrix operations

```
output = M · X  where M has specific structure (dense, low-rank, etc.)
```

**Standard implementation:** Dense O(n²) or low-rank O(nr) mixing
**Evolution goal:** Discover novel mixing matrix structures

---

## 🔬 Experimental Design

### What Gets Evolved

**ONLY the attention mechanism implementations** within unified framework notation:

```python
# EVOLVE-BLOCK-START: CoefficientDynamics
class EvolvableCoefficientDynamics(nn.Module):
    """
    Evolve how attention coefficients evolve over sequence positions.

    Constraints:
    - Must be expressible as α_t = f(α_{t-1}, inputs)
    - Must maintain causality
    - Output shape: [batch, seq_len, hidden_dim]
    """
    def __init__(self, hidden_dim, num_heads):
        # Evolution can modify initialization
        pass

    def forward(self, x, mask=None):
        # Evolution can modify coefficient dynamics
        # Standard: α = softmax(QK^T/sqrt(d))
        # Evolved: α = novel_mixing_function(...)
        pass
# EVOLVE-BLOCK-END
```

### What Stays FIXED

**All peripheral components remain standard:**
- ✓ FFN: Standard GELU activation (NO SwiGLU, NO gating evolution)
- ✓ Normalization: Standard LayerNorm (NO RMSNorm evolution)
- ✓ Block structure: Sequential attention → FFN (NO parallel branches)
- ✓ Residual connections: Standard addition (NO LayerScale, NO gating)
- ✓ Hyperparameters: Fixed across all variants for fair comparison

### Fitness Function

**Scaling law slope at final convergence (100k steps):**
```
fitness = -slope  where  loss = intercept + slope × log(n_params)
```

**Key difference from failed experiment:**
- Use 100k-step converged loss (NOT 2k-step proxy)
- Compare architectures with same hyperparameters (tuned jointly)
- Measure final performance, not early learning speed

---

## 🏗️ Architecture

### Fixed Baseline Transformer

```python
class StandardTransformer(nn.Module):
    """
    Standard transformer with fixed peripheral components.
    ONLY attention mechanism is evolvable.
    """
    def __init__(self, framework='coefficient_dynamics'):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Evolvable attention within framework
        self.attention = EvolvableAttention(framework, d_model, n_heads)

        # FIXED components
        self.norm1 = nn.LayerNorm(d_model)  # Standard LayerNorm
        self.ffn = StandardFFN(d_model)      # Standard GELU FFN
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Standard sequential architecture
        x = self.embedding(x) + self.pos_encoding(x)
        x = x + self.attention(self.norm1(x))  # Pre-norm
        x = x + self.ffn(self.norm2(x))
        return x

class StandardFFN(nn.Module):
    """Fixed FFN - NOT evolvable."""
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()  # Fixed activation

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
```

### Evolvable Attention Frameworks

**Three separate evolution runs:**

1. **Coefficient Dynamics Evolution**
   ```python
   # Baseline: Standard softmax attention
   α = softmax(QK^T / sqrt(d_k))
   output = α @ V

   # Evolved examples:
   # - Modified temperature schedules
   # - Learnable dynamics parameters
   # - Novel coefficient update rules
   # - Hybrid local/global mixing
   ```

2. **Test-Time Regression Evolution**
   ```python
   # Baseline: Softmax kernel regression
   K_reg = exp(QK^T / τ)
   output = K_reg @ V

   # Evolved examples:
   # - Different kernels (polynomial, RBF, learned)
   # - Adaptive regularization
   # - Multi-resolution regression
   # - Residual regression paths
   ```

3. **Matrix Mixer Evolution**
   ```python
   # Baseline: Dense or low-rank mixing
   M = dense_matrix or low_rank_approximation
   output = M @ X

   # Evolved examples:
   # - Novel structured matrices
   # - Learnable sparsity patterns
   # - Hierarchical mixing
   # - Adaptive rank allocation
   ```

---

## 📊 Experimental Protocol

### Phase 1: Evolution (Per Framework)

**For each framework (coefficient_dynamics, test_time_regression, matrix_mixer):**

1. **Initial implementation:** Standard formulation from paper
2. **Evolution:** 35 iterations with GPT-5
3. **Fitness:** Scaling law slope at **100k training steps** (4 model sizes)
4. **Selection:** Best variant by fitness

**Evolution constraints:**
- Maintain framework notation (e.g., coefficient dynamics must express as α evolution)
- Keep causality for autoregressive modeling
- Numerical stability (no NaN/Inf)
- Output shape compatibility: [batch, seq_len, hidden_dim]

### Phase 2: Validation (Per Framework)

**Compare standard vs evolved implementations:**

1. **Configuration:**
   - Tokenization: BPE (16k vocab)
   - Training: 100k steps
   - Seeds: 10 (for statistical power)
   - Dataset: TinyStories (3.6M sequences)

2. **Hyperparameters:**
   - **Joint optimization:** Run Optuna for BOTH standard and evolved
   - OR use same hyperparameters found for standard
   - Document which approach is used

3. **Statistical analysis:**
   - Paired t-test across 10 seeds
   - Cohen's d effect size
   - 95% confidence intervals

### Phase 3: Cross-Framework Comparison

**Compare best variants across frameworks:**
```
Best coefficient_dynamics vs Best test_time_regression vs Best matrix_mixer
```

**Additional analysis:**
- Computational cost (FLOPs)
- Training stability
- Generalization to other tasks
- Component ablation studies

---

## 🔑 Key Differences from Failed Experiment

### What Went Wrong Before

| Aspect | Failed Experiment | Correct Approach |
|--------|-------------------|------------------|
| **Evolution target** | Peripheral architecture (FFN, norms, residuals) | Attention mechanisms within frameworks |
| **Framework usage** | Fixed at coefficient_dynamics, never compared | Evolve each framework separately |
| **Fitness horizon** | 2k steps (short-horizon proxy) | 100k steps (final convergence) |
| **Hyperparameters** | Evolved used baseline's hyperparams | Joint optimization or same fixed params |
| **Research question** | Generic architecture search | Framework-specific mechanism discovery |

### Critical Fixes

1. **Constrain evolution scope:**
   ```yaml
   **What CAN evolve:**
   - Attention coefficient computation
   - Mixing functions within framework notation
   - Framework-specific parameters

   **What CANNOT evolve:**
   - FFN activation functions
   - Normalization type
   - Block structure (must stay sequential)
   - Residual connection patterns
   ```

2. **Use correct fitness:**
   ```python
   # WRONG (failed experiment)
   fitness = -slope at 2k training steps

   # CORRECT
   fitness = -slope at 100k training steps (converged)
   ```

3. **Fair hyperparameter comparison:**
   ```python
   # Option A: Joint optimization (better but expensive)
   standard_hparams = optuna_search(standard_attention)
   evolved_hparams = optuna_search(evolved_attention)

   # Option B: Fixed hyperparameters (faster, still fair)
   shared_hparams = optuna_search(standard_attention)
   compare(standard, evolved, both using shared_hparams)
   ```

---

## 📁 Repository Structure

```
evolving-unified-transformers/
├── README.md                          # This file
├── EXPERIMENT_DESIGN.md               # Detailed methodology
├── LESSONS_FROM_FAILED_EXPERIMENT.md  # What went wrong before
│
├── frameworks/
│   ├── __init__.py
│   ├── coefficient_dynamics.py       # Evolvable coefficient dynamics
│   ├── test_time_regression.py       # Evolvable regression
│   ├── matrix_mixer.py                # Evolvable matrix mixing
│   └── base.py                        # Base classes and interfaces
│
├── models/
│   ├── __init__.py
│   ├── transformer.py                 # Standard transformer (FIXED components)
│   └── language_model.py              # Full LM with embeddings
│
├── evolution/
│   ├── __init__.py
│   ├── config_coefficient_dynamics.yaml
│   ├── config_test_time_regression.yaml
│   ├── config_matrix_mixer.yaml
│   └── evaluator.py                   # Scaling law evaluation at 100k steps
│
├── training/
│   ├── __init__.py
│   ├── train.py                       # Training script
│   ├── hyperparameter_search.py      # Optuna search
│   └── validation.py                  # Full validation pipeline
│
├── data/
│   ├── __init__.py
│   ├── tinystories_loader.py          # BPE tokenized dataloader
│   └── create_bpe_tokenizer.py        # Tokenizer creation
│
├── analysis/
│   ├── __init__.py
│   ├── scaling_laws.py                # Scaling law fitting
│   ├── statistical_tests.py           # t-tests, effect sizes
│   └── visualizations.py              # Result visualization
│
└── experiments/
    ├── 01_evolve_coefficient_dynamics/
    ├── 02_evolve_test_time_regression/
    ├── 03_evolve_matrix_mixer/
    └── 04_cross_framework_comparison/
```

---

## 🎯 Expected Outcomes

### Success Criteria

**Successful evolution would show:**
1. Evolved attention mechanism outperforms standard implementation within framework
2. Improvement is statistically significant (p < 0.05, reasonable effect size)
3. Improvement generalizes across different model sizes (scaling law)
4. Mechanism is interpretable within framework notation

**Example successful result:**
```
Framework: Coefficient Dynamics
Standard: α = softmax(QK^T/√d), loss = 2.137
Evolved: α = learned_dynamics(Q, K, θ), loss = 2.089
Improvement: +2.25%, p=0.003, Cohen's d=0.82
Interpretation: Evolved mechanism learns adaptive temperature per head
```

### Null Results

**If evolution fails to improve:**
1. Standard formulations may already be optimal for this task/scale
2. Framework constraints may be too restrictive
3. Fitness landscape may be too flat for gradient-free search
4. LLMs may lack domain knowledge to propose improvements

**Still valuable:** Negative results validate existing formulations

---

## 🚀 Getting Started

### Installation

```bash
cd evolving-unified-transformers
uv venv
source .venv/bin/activate
uv pip install torch transformers datasets tokenizers openevolve optuna matplotlib scipy
```

### Quick Test

```bash
# Test that frameworks work
python frameworks/coefficient_dynamics.py

# Test full transformer
python models/transformer.py

# Run mini evolution (5 iterations)
python evolution/run_mini_experiment.py
```

### Full Experiment

```bash
# Phase 1: Evolve coefficient dynamics (35 iterations, ~24 hours)
python evolution/evolve_coefficient_dynamics.py

# Phase 2: Validate best variant (10 seeds, ~11 hours)
python training/validate_best.py --framework coefficient_dynamics

# Phase 3: Analysis
python analysis/compare_frameworks.py
```

---

## 📖 References

### Unified Frameworks Papers

1. **Sieber et al. (2025)** - "Design Principles for Sequence Models via Coefficient Dynamics"
   - arXiv:2510.09389
   - Framework: Coefficient evolution view

2. **Wang et al. (2025)** - "Test-time regression: a unifying framework for designing sequence models with associative memory"
   - arXiv:2501.12352
   - Framework: Statistical learning view

3. **Hwang et al. (2024)** - "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers"
   - arXiv:2407.09941
   - Framework: Linear algebra view

### Evolution Framework

4. **Huang et al. (2024)** - "OpenEvolve: Open-Ended Evolution with LLMs"
   - Framework for LLM-guided code evolution

### Related Work

5. **Vaswani et al. (2017)** - "Attention Is All You Need" (Transformer baseline)
6. **Kaplan et al. (2020)** - "Scaling Laws for Neural Language Models"
7. **Hoffmann et al. (2022)** - "Training Compute-Optimal Large Language Models" (Chinchilla)

---

## 🤝 Contributing

This is a research experiment repository. Contributions welcome for:
- Additional unified framework implementations
- Better fitness functions
- Improved statistical analysis
- Visualization improvements

---

## 📝 License

MIT License - See LICENSE file

---

## 🔗 Links

- **Failed experiment analysis:** See `../beyond-transformer/FINAL_EXPERIMENT_REPORT.md`
- **Unified frameworks code:** See `../beyond-transformer/unified_frameworks/`
- **OpenEvolve:** https://github.com/CarperAI/OpenEvolve

---

**Status:** 🚧 Experimental setup - Not yet run

**Created:** 2025-11-03

**Goal:** Discover novel attention mechanisms within unified framework notation through LLM-guided evolution
