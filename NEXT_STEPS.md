# Next Steps for Implementation

**Status:** Documentation complete, ready for implementation phase

This document outlines the concrete steps needed to implement the corrected experimental design.

---

## Phase 0: Environment Setup (1-2 days)

### 0.1 Create Project Structure

```bash
cd /home/ak/evolving-unified-transformers

# Create directory structure
mkdir -p frameworks/{coefficient_dynamics,test_time_regression,matrix_mixer}
mkdir -p models
mkdir -p evolution/{configs,evaluators}
mkdir -p training
mkdir -p data
mkdir -p analysis
mkdir -p experiments/{01_evolve_coefficient_dynamics,02_evolve_test_time_regression,03_evolve_matrix_mixer,04_cross_framework}
mkdir -p tests
```

### 0.2 Setup Python Environment

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers datasets tokenizers
uv pip install optuna scipy matplotlib seaborn pandas
uv pip install pytest pytest-cov black isort mypy

# Install OpenEvolve
# Note: May need to install from source or specific repo
# pip install openevolve  # Or clone and install locally
```

### 0.3 Copy Unified Frameworks Code

```bash
# Copy from existing implementation
cp -r /home/ak/beyond-transformer/unified_frameworks/src/* frameworks/

# These files provide the base implementations:
# - coefficient_dynamics.py
# - test_time_regression.py
# - matrix_mixer.py
```

### 0.4 Create Causal Wrappers

```bash
# Copy causal wrappers (or create new ones)
cp /home/ak/beyond-transformer/causal_wrappers.py frameworks/
```

---

## Phase 1: Core Implementation (1 week)

### 1.1 Implement Fixed Transformer Base (Day 1-2)

**File:** `models/transformer.py`

**Requirements:**
- All peripheral components FIXED (not evolvable)
- Only attention mechanism is a parameter
- Standard architecture: embedding → positional encoding → blocks → LM head

**Key components to implement:**

```python
class StandardFFN(nn.Module):
    """FIXED FFN - NOT evolvable."""
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        # Standard GELU activation
        # NO SwiGLU, NO gating
        pass

class FixedTransformerBlock(nn.Module):
    """Transformer block with fixed components, evolvable attention."""
    def __init__(self, d_model, attention_module):
        self.attention = attention_module  # EVOLVABLE
        self.norm1 = nn.LayerNorm(d_model)  # FIXED: LayerNorm
        self.ffn = StandardFFN(d_model)     # FIXED: GELU FFN
        self.norm2 = nn.LayerNorm(d_model)  # FIXED: LayerNorm
        # NO parallel branches, NO LayerScale

    def forward(self, x, mask=None):
        # Standard sequential: attn → norm → ffn → norm
        # NO modifications to this structure
        pass

class FixedTransformer(nn.Module):
    """Complete transformer with only attention evolvable."""
    pass
```

**Test:**
```bash
python tests/test_transformer_base.py
# Verify: forward pass, backward pass, output shapes
```

---

### 1.2 Implement Evolvable Attention Frameworks (Day 3-4)

Create three evolvable attention implementations:

#### File: `frameworks/evolvable_coefficient_dynamics.py`

```python
# EVOLVE-BLOCK-START: CoefficientDynamics
class EvolvableCoefficientDynamics(nn.Module):
    """
    Evolvable attention using coefficient dynamics framework.

    Standard implementation: α = softmax(QK^T / sqrt(d_k))

    Evolution can modify:
    - How coefficients are computed
    - Dynamics parameters
    - Mixing functions

    Constraints:
    - Must maintain causality (causal mask)
    - Output shape: [batch, seq_len, d_model]
    - Numerical stability (no NaN/Inf)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Standard Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Standard softmax attention (baseline)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention coefficients
        attn_weights = F.softmax(scores, dim=-1)

        # Apply to values
        output = torch.matmul(attn_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output
# EVOLVE-BLOCK-END: CoefficientDynamics
```

#### File: `frameworks/evolvable_test_time_regression.py`

Similar structure but with regression-based attention formulation.

#### File: `frameworks/evolvable_matrix_mixer.py`

Similar structure but with matrix mixing formulation.

**Test:**
```bash
python tests/test_evolvable_frameworks.py
# Verify: all three frameworks work, causality preserved, shapes correct
```

---

### 1.3 Create Language Model Wrapper (Day 4)

**File:** `models/language_model.py`

```python
class LanguageModel(nn.Module):
    """
    Complete language model for TinyStories.

    Uses FixedTransformer with specified attention framework.
    """
    def __init__(self, vocab_size=16384, d_model=128, num_layers=6,
                 num_heads=4, attention_framework='coefficient_dynamics'):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer with evolvable attention
        if attention_framework == 'coefficient_dynamics':
            attention_cls = EvolvableCoefficientDynamics
        elif attention_framework == 'test_time_regression':
            attention_cls = EvolvableTestTimeRegression
        elif attention_framework == 'matrix_mixer':
            attention_cls = EvolvableMatrixMixer
        else:
            raise ValueError(f"Unknown framework: {attention_framework}")

        self.transformer = FixedTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_cls=attention_cls
        )

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Embed + positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x, attention_mask)

        # LM head
        logits = self.lm_head(x)

        return logits
```

---

### 1.4 Setup Data Pipeline (Day 5)

#### Copy BPE tokenizer creation

```bash
cp /home/ak/beyond-transformer/create_bpe_tokenizer.py data/
cp /home/ak/beyond-transformer/tinystories_bpe_loader.py data/
```

#### Create or verify cached dataset

```bash
# Check if BPE cache exists
ls /home/ak/beyond-transformer/tinystories_bpe_cache/

# If exists, create symlink or copy
ln -s /home/ak/beyond-transformer/tinystories_bpe_cache data/tinystories_bpe_cache

# Or run cache creation
cd data
python create_bpe_tokenizer.py  # Creates tokenizer
python cache_dataset.py          # Caches full dataset
```

**File:** `data/dataloader.py`

```python
def get_tinystories_dataloader(split='train', batch_size=64, seq_len=128):
    """
    Returns dataloader for TinyStories with BPE tokenization.
    Uses cached dataset for reproducibility.
    """
    pass
```

---

### 1.5 Create Training Script (Day 6-7)

**File:** `training/train.py`

```python
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_steps=100000,
    eval_frequency=1000,
    device='cuda',
    output_dir='checkpoints/'
):
    """
    Train language model for specified number of steps.

    Args:
        model: LanguageModel instance
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer (AdamW)
        num_steps: Total training steps (100k for final evaluation)
        eval_frequency: How often to evaluate on val set
        device: 'cuda' or 'cpu'
        output_dir: Where to save checkpoints

    Returns:
        final_val_loss: Loss on validation set after training
    """
    model.to(device)
    model.train()

    step = 0
    train_losses = []
    val_losses = []

    while step < num_steps:
        for batch in train_loader:
            # Forward pass
            # Compute loss
            # Backward pass
            # Optimizer step

            step += 1
            if step >= num_steps:
                break

            # Periodic evaluation
            if step % eval_frequency == 0:
                val_loss = evaluate(model, val_loader, device)
                val_losses.append((step, val_loss))
                # Save checkpoint

    return val_losses[-1][1]  # Return final validation loss
```

**File:** `training/hyperparameter_search.py`

```python
import optuna

def hyperparameter_search(
    framework='coefficient_dynamics',
    n_trials=100,
    training_steps=50000,  # Use 50k for HPO (faster)
    output_dir='hparams/'
):
    """
    Run Optuna hyperparameter search for given framework.

    Search space:
    - learning_rate: [1e-4, 1e-3]
    - warmup_steps: [500, 2000]
    - weight_decay: [0, 0.1]
    - batch_size: [32, 64, 128]

    Returns:
        best_hparams: Dict with best hyperparameters
    """

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        warmup = trial.suggest_int('warmup_steps', 500, 2000)
        weight_decay = trial.suggest_float('weight_decay', 0, 0.1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        # Train model
        model = create_model(framework)
        train_loader = get_dataloader('train', batch_size)
        val_loader = get_dataloader('val', batch_size)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, training_steps)

        final_loss = train_model(model, train_loader, val_loader, optimizer,
                                training_steps, device='cuda')

        return final_loss

    study = optuna.create_study(direction='minimize', pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
```

---

## Phase 2: Evolution Setup (3-4 days)

### 2.1 Create OpenEvolve Configurations (Day 8)

#### File: `evolution/configs/coefficient_dynamics.yaml`

```yaml
max_iterations: 35
checkpoint_interval: 5

llm:
  primary_model: "gpt-5"
  temperature: 0.8
  max_tokens: 16000
  timeout: 300

prompt:
  system_message: |
    You are evolving the attention mechanism within the COEFFICIENT DYNAMICS framework.

    **Framework principle:**
    Attention computes outputs as linear combinations where coefficients α evolve.
    Standard: α = softmax(QK^T / sqrt(d_k))

    **What you CAN evolve:**
    - How coefficients α are computed from Q, K, V
    - Temperature, scaling, mixing functions
    - Dynamics parameters (if making α stateful)
    - Learned parameters within attention computation

    **What you CANNOT evolve:**
    - FFN (it's fixed with GELU - DO NOT TOUCH)
    - Normalization (it's fixed with LayerNorm - DO NOT TOUCH)
    - Block structure (it's fixed as sequential - DO NOT TOUCH)
    - Anything outside EVOLVE-BLOCK-START/END markers

    **Critical constraints:**
    - MUST maintain causality (use causal mask, no future information)
    - MUST output shape [batch, seq_len, d_model]
    - MUST be numerically stable (no NaN/Inf)
    - MUST be expressible within coefficient dynamics notation

    **Fitness function:**
    - Models are trained for 100,000 steps to full convergence
    - Evaluated across 4 model sizes: 0.5M, 1M, 2M, 4M parameters
    - Fitness = -slope of (loss vs log(n_params))
    - Better scaling (steeper negative slope) = higher fitness

    Focus on discovering novel coefficient computation methods that improve
    how the model scales with parameter count.

database:
  population_size: 20
  archive_size: 10
  num_islands: 2
  elite_selection_ratio: 0.3

evaluator:
  training_steps: 100000  # CRITICAL: Full convergence, not 2k!
  model_sizes: [524288, 1048576, 2097152, 4194304]  # 0.5M, 1M, 2M, 4M
  timeout: 7200  # 2 hours per evaluation
  parallel_evaluations: 1
```

**Create similar configs for:**
- `evolution/configs/test_time_regression.yaml`
- `evolution/configs/matrix_mixer.yaml`

### 2.2 Create Evaluator (Day 9-10)

**File:** `evolution/evaluators/scaling_law_evaluator.py`

```python
class ScalingLawEvaluator:
    """
    Evaluates evolved attention mechanisms by fitting scaling laws.

    Process:
    1. Train 4 model sizes (0.5M, 1M, 2M, 4M params) for 100k steps each
    2. Record final validation loss for each size
    3. Fit: loss = intercept + slope × log(n_params)
    4. Return fitness = -slope
    """

    def __init__(self, config):
        self.training_steps = config['training_steps']  # Must be 100k!
        self.model_sizes = config['model_sizes']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate_program(self, program_code, framework='coefficient_dynamics'):
        """
        Evaluate an evolved attention mechanism.

        Args:
            program_code: Python code for evolved attention class
            framework: Which framework is being evolved

        Returns:
            fitness: -slope (higher is better)
            metadata: Dict with losses, R², etc.
        """

        # 1. Load evolved attention class from code
        attention_cls = self._load_attention_class(program_code)

        # 2. Train each model size
        losses = []
        for n_params in self.model_sizes:
            # Create model of appropriate size
            model = self._create_model(attention_cls, n_params, framework)

            # Train for 100k steps
            final_loss = self._train_to_convergence(model, self.training_steps)

            losses.append(final_loss)

        # 3. Fit scaling law
        slope, intercept, r_squared = self._fit_scaling_law(
            self.model_sizes, losses
        )

        # 4. Return fitness
        fitness = -slope  # More negative slope = better scaling = higher fitness

        metadata = {
            'losses': losses,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'model_sizes': self.model_sizes
        }

        return fitness, metadata

    def _fit_scaling_law(self, sizes, losses):
        """Fit loss = intercept + slope × log(n_params)"""
        import numpy as np
        from scipy.stats import linregress

        log_params = np.log(np.array(sizes))
        losses = np.array(losses)

        slope, intercept, r_value, p_value, std_err = linregress(log_params, losses)
        r_squared = r_value ** 2

        return slope, intercept, r_squared
```

### 2.3 Create Evolution Runner Scripts (Day 11)

**File:** `evolution/evolve_coefficient_dynamics.py`

```python
#!/usr/bin/env python3
"""
Evolve attention mechanisms within coefficient dynamics framework.

Usage:
    python evolution/evolve_coefficient_dynamics.py \
        --config evolution/configs/coefficient_dynamics.yaml \
        --output experiments/01_evolve_coefficient_dynamics/ \
        --gpu 0
"""

import argparse
from openevolve import OpenEvolveRunner
from evaluators.scaling_law_evaluator import ScalingLawEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)

    # Create evaluator
    evaluator = ScalingLawEvaluator(config['evaluator'])

    # Set initial program (standard implementation)
    initial_program = load_file('frameworks/evolvable_coefficient_dynamics.py')

    # Run evolution
    runner = OpenEvolveRunner(
        config=config,
        evaluator=evaluator,
        initial_program=initial_program,
        output_dir=args.output
    )

    runner.run()

if __name__ == '__main__':
    main()
```

**Create similar scripts for:**
- `evolution/evolve_test_time_regression.py`
- `evolution/evolve_matrix_mixer.py`

---

## Phase 3: Validation Implementation (2-3 days)

### 3.1 Create Validation Pipeline (Day 12-13)

**File:** `training/validate_best.py`

```python
#!/usr/bin/env python3
"""
Validate best evolved attention mechanism with proper statistical rigor.

Runs:
- 10 seeds for standard implementation
- 10 seeds for evolved implementation
- 100k training steps each
- Fixed model size (3M params)

Usage:
    python training/validate_best.py \
        --framework coefficient_dynamics \
        --evolved-code experiments/01_evolve_coefficient_dynamics/best_program.py \
        --output experiments/02_validate_coefficient_dynamics/
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', required=True,
                       choices=['coefficient_dynamics', 'test_time_regression', 'matrix_mixer'])
    parser.add_argument('--evolved-code', required=True, help='Path to best evolved program')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 52)))
    parser.add_argument('--training-steps', type=int, default=100000)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Load hyperparameters (from HPO or use shared)
    standard_hparams = load_hparams(f'hparams/{args.framework}_standard.json')
    evolved_hparams = load_hparams(f'hparams/{args.framework}_evolved.json')

    results = {'standard': [], 'evolved': []}

    # Train standard implementation (10 seeds)
    print("Training standard implementation...")
    for seed in args.seeds:
        model = create_standard_model(args.framework, seed)
        final_loss = train_model(model, standard_hparams, args.training_steps, seed)
        results['standard'].append(final_loss)

        # Save checkpoint
        save_checkpoint(model, output / 'standard' / f'seed_{seed}.pt')

    # Train evolved implementation (10 seeds)
    print("Training evolved implementation...")
    evolved_attention = load_evolved_attention(args.evolved_code)
    for seed in args.seeds:
        model = create_model_with_attention(evolved_attention, args.framework, seed)
        final_loss = train_model(model, evolved_hparams, args.training_steps, seed)
        results['evolved'].append(final_loss)

        # Save checkpoint
        save_checkpoint(model, output / 'evolved' / f'seed_{seed}.pt')

    # Statistical analysis
    from scipy import stats
    import numpy as np

    standard_losses = np.array(results['standard'])
    evolved_losses = np.array(results['evolved'])

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(standard_losses, evolved_losses)

    # Effect size (Cohen's d)
    differences = standard_losses - evolved_losses
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)

    # Confidence intervals
    mean_diff = np.mean(differences)
    sem = stats.sem(differences)
    ci = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=sem)

    # Save results
    results_summary = {
        'framework': args.framework,
        'standard': {
            'mean': float(np.mean(standard_losses)),
            'std': float(np.std(standard_losses, ddof=1)),
            'losses': standard_losses.tolist()
        },
        'evolved': {
            'mean': float(np.mean(evolved_losses)),
            'std': float(np.std(evolved_losses, ddof=1)),
            'losses': evolved_losses.tolist()
        },
        'statistics': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'mean_difference': float(mean_diff),
            'ci_95': [float(ci[0]), float(ci[1])],
            'significant': bool(p_value < 0.05)
        }
    }

    save_json(results_summary, output / 'validation_results.json')

    # Print summary
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"Framework: {args.framework}")
    print(f"Standard: {results_summary['standard']['mean']:.4f} ± {results_summary['standard']['std']:.4f}")
    print(f"Evolved:  {results_summary['evolved']['mean']:.4f} ± {results_summary['evolved']['std']:.4f}")
    print(f"Improvement: {(mean_diff/np.mean(standard_losses))*100:+.2f}%")
    print(f"p-value: {p_value:.4e}")
    print(f"Cohen's d: {cohens_d:.2f}")
    print(f"Significant: {'YES' if p_value < 0.05 else 'NO'}")
    print("="*70)

if __name__ == '__main__':
    main()
```

---

### 3.2 Create Analysis Scripts (Day 14)

**File:** `analysis/compare_frameworks.py`

```python
#!/usr/bin/env python3
"""
Compare best variants across all three frameworks.

Usage:
    python analysis/compare_frameworks.py \
        --results experiments/02_validate_*/validation_results.json \
        --output experiments/04_cross_framework/
"""

def main():
    # Load results from all three frameworks
    # Perform ANOVA or Kruskal-Wallis test
    # Generate comparison plots
    # Write final report
    pass
```

**File:** `analysis/visualizations.py`

```python
def plot_validation_results(results, output_path):
    """
    Create comprehensive visualization:
    - Box plots (standard vs evolved)
    - Per-seed bar charts
    - Confidence interval plots
    - Training curves
    """
    pass

def plot_scaling_laws(evolution_history, output_path):
    """
    Visualize scaling law fits during evolution
    """
    pass

def plot_evolution_progress(evolution_history, output_path):
    """
    Plot fitness over iterations
    """
    pass
```

---

## Phase 4: Testing & Validation (2-3 days)

### 4.1 Unit Tests (Day 15)

**File:** `tests/test_transformer_base.py`

```python
def test_fixed_ffn():
    """Verify FFN is standard GELU with no gating."""
    ffn = StandardFFN(d_model=128, d_ff=512)
    # Check no SwiGLU components
    # Check GELU activation
    pass

def test_fixed_transformer_block():
    """Verify block structure is sequential with LayerNorm."""
    # Check no parallel branches
    # Check LayerNorm (not RMSNorm)
    # Check no LayerScale
    pass

def test_causality():
    """Verify causal masking works correctly."""
    # Check future positions are masked
    pass
```

### 4.2 Integration Tests (Day 16)

**File:** `tests/test_end_to_end.py`

```python
def test_mini_training_run():
    """Run a mini training (1000 steps) to verify pipeline works."""
    pass

def test_evolution_evaluator():
    """Test scaling law evaluator on known architectures."""
    pass
```

### 4.3 Validation Checks (Day 17)

Create checklist script to verify experimental setup:

**File:** `scripts/verify_setup.py`

```python
#!/usr/bin/env python3
"""
Verify experimental setup is correct before running full evolution.

Checks:
- [ ] FFN is standard GELU (no SwiGLU)
- [ ] Normalization is LayerNorm (no RMSNorm)
- [ ] Block structure is sequential (no parallel)
- [ ] No LayerScale in residuals
- [ ] EVOLVE-BLOCK markers only around attention
- [ ] Fitness uses 100k steps (not 2k)
- [ ] Hyperparameters properly configured
- [ ] Dataset cache exists
- [ ] All three frameworks implemented
"""

def main():
    checks = []

    # Check 1: FFN implementation
    print("Checking FFN implementation...")
    ffn_code = open('models/transformer.py').read()
    assert 'SwiGLU' not in ffn_code, "ERROR: Found SwiGLU in FFN!"
    assert 'GELU' in ffn_code, "ERROR: GELU not found in FFN!"
    checks.append(("FFN is standard GELU", True))

    # Check 2: Normalization
    print("Checking normalization...")
    assert 'RMSNorm' not in ffn_code, "ERROR: Found RMSNorm!"
    assert 'LayerNorm' in ffn_code, "ERROR: LayerNorm not found!"
    checks.append(("Normalization is LayerNorm", True))

    # Check 3: Fitness function
    print("Checking evaluator...")
    eval_config = yaml.load(open('evolution/configs/coefficient_dynamics.yaml'))
    training_steps = eval_config['evaluator']['training_steps']
    assert training_steps == 100000, f"ERROR: training_steps = {training_steps}, should be 100000!"
    checks.append(("Fitness uses 100k steps", True))

    # ... more checks ...

    # Print summary
    print("\n" + "="*70)
    print("SETUP VERIFICATION RESULTS")
    print("="*70)
    for check, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    print("="*70)

    if all(passed for _, passed in checks):
        print("\n✅ All checks passed! Ready to run evolution.")
    else:
        print("\n❌ Some checks failed! Fix issues before running.")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

---

## Phase 5: Execution (3-4 weeks)

### Week 1: Hyperparameter Optimization

```bash
# Run HPO for standard implementations (3 frameworks)
python training/hyperparameter_search.py --framework coefficient_dynamics --n-trials 100
python training/hyperparameter_search.py --framework test_time_regression --n-trials 100
python training/hyperparameter_search.py --framework matrix_mixer --n-trials 100

# Estimated time: ~2-3 days per framework with pruning
```

### Week 2-3: Evolution

```bash
# Run evolution for each framework (can parallelize with 3 GPUs)
# Each takes ~48-72 hours

# GPU 0: Coefficient Dynamics
CUDA_VISIBLE_DEVICES=0 python evolution/evolve_coefficient_dynamics.py \
    --config evolution/configs/coefficient_dynamics.yaml \
    --output experiments/01_evolve_coefficient_dynamics/ &

# GPU 1: Test-Time Regression
CUDA_VISIBLE_DEVICES=1 python evolution/evolve_test_time_regression.py \
    --config evolution/configs/test_time_regression.yaml \
    --output experiments/02_evolve_test_time_regression/ &

# GPU 2: Matrix Mixer
CUDA_VISIBLE_DEVICES=2 python evolution/evolve_matrix_mixer.py \
    --config evolution/configs/matrix_mixer.yaml \
    --output experiments/03_evolve_matrix_mixer/ &

# Wait for all to complete
wait

# Estimated time: 2-3 days if parallelized, 1 week if sequential
```

### Week 4: Validation

```bash
# Validate each framework's best variant
python training/validate_best.py \
    --framework coefficient_dynamics \
    --evolved-code experiments/01_evolve_coefficient_dynamics/best_program.py \
    --output experiments/02_validate_coefficient_dynamics/

# Repeat for other frameworks
# Estimated time: ~11 hours per framework = ~1.5 days total
```

### Week 4-5: Analysis

```bash
# Cross-framework comparison
python analysis/compare_frameworks.py \
    --results experiments/02_validate_*/validation_results.json \
    --output experiments/04_cross_framework/

# Generate all visualizations
python analysis/visualizations.py

# Write final report
python analysis/generate_report.py
```

---

## Phase 6: Documentation & Reporting (1 week)

### 6.1 Document Results

Create results document with:
- Evolution progression for each framework
- Validation statistics
- Cross-framework comparison
- Code diffs showing what evolved
- Interpretation of evolved mechanisms

### 6.2 Create Visualizations

Generate all required plots:
- Evolution fitness curves (3 frameworks)
- Validation box plots (3 frameworks × 2 variants)
- Scaling law fits
- Cross-framework comparison
- Attention pattern visualizations (if interpretable)

### 6.3 Write Paper/Report

Follow reporting standards from EXPERIMENT_DESIGN.md:
- Complete methodology
- All results (including negative)
- Statistical analysis
- Discussion and interpretation
- Limitations and future work

---

## Timeline Summary

| Phase | Duration | Parallel? |
|-------|----------|-----------|
| Phase 0: Setup | 1-2 days | No |
| Phase 1: Core Implementation | 1 week | Partially |
| Phase 2: Evolution Setup | 3-4 days | No |
| Phase 3: Validation Implementation | 2-3 days | No |
| Phase 4: Testing | 2-3 days | Partially |
| Phase 5: Execution | 3-4 weeks | Yes (3 GPUs) |
| Phase 6: Documentation | 1 week | No |
| **Total** | **6-8 weeks** | |

**With 3 GPUs (parallelized evolution):** ~6 weeks
**With 1 GPU (sequential evolution):** ~8 weeks

---

## Milestones & Checkpoints

### Milestone 1: Core Implementation Complete
- [ ] Fixed transformer with evolvable attention implemented
- [ ] All three framework attention classes created
- [ ] Unit tests passing
- [ ] Mini training run successful

### Milestone 2: Evolution Ready
- [ ] OpenEvolve configs created
- [ ] Scaling law evaluator implemented
- [ ] Verification script passing
- [ ] HPO complete for all frameworks

### Milestone 3: Evolution Complete
- [ ] All three frameworks evolved (35 iterations each)
- [ ] Best variants identified
- [ ] Evolution history saved
- [ ] Code diffs extracted

### Milestone 4: Validation Complete
- [ ] 10-seed validation for all frameworks
- [ ] Statistical tests computed
- [ ] Results JSON files saved
- [ ] Initial analysis complete

### Milestone 5: Ready for Publication
- [ ] All visualizations generated
- [ ] Complete results document written
- [ ] Code and data publicly available
- [ ] Reproducibility verified

---

## Critical Path Items

**Blockers that must be completed sequentially:**

1. Core implementation → Testing → Verification
2. Verification passing → HPO
3. HPO complete → Evolution
4. Evolution complete → Validation
5. Validation complete → Analysis

**Can be parallelized:**

- Implementing three frameworks (Day 3-4)
- Running HPO for three frameworks (Week 1)
- Running evolution for three frameworks (Week 2-3)
- Creating analysis scripts while validation runs

---

## Resource Requirements

### Compute
- **Minimum:** 1 GPU (A100 or V100) for ~8 weeks
- **Recommended:** 3 GPUs for ~6 weeks (parallel evolution)
- **Ideal:** 6 GPUs for ~4 weeks (parallel evolution + validation)

### Storage
- Dataset cache: ~5 GB
- Evolution checkpoints: ~10 GB per framework
- Validation checkpoints: ~2 GB per framework
- Total: ~50-60 GB

### Memory
- 16+ GB RAM
- 40+ GB GPU memory (for model sizes up to 4M params)

---

## Risk Mitigation

### Risk 1: Evolution Takes Too Long
**Mitigation:**
- Start with shorter pilot run (10 iterations)
- Use smaller model sizes for fitness evaluation
- Consider reducing to 2 frameworks if time-constrained

### Risk 2: LLM Generates Invalid Code
**Mitigation:**
- Strict EVOLVE-BLOCK constraints
- Syntax checking before evaluation
- Fallback to previous valid code
- Manual inspection of first few iterations

### Risk 3: No Improvement Found
**Mitigation:**
- This is scientifically valuable (negative result)
- Validate standard implementations
- Analyze why evolution didn't help
- Document in paper as limitation of approach

### Risk 4: Computational Budget Exceeded
**Mitigation:**
- Reduce number of evolution iterations (35 → 20)
- Reduce HPO trials (100 → 50)
- Use smaller validation (10 seeds → 5 seeds)
- Focus on one framework initially

---

## Next Immediate Actions

**This week:**
1. Set up Python environment
2. Copy unified frameworks code
3. Implement FixedTransformer base class
4. Create first evolvable attention (coefficient_dynamics)
5. Write and run unit tests

**Next week:**
1. Complete other two frameworks
2. Create training pipeline
3. Run mini training test (1000 steps)
4. Set up OpenEvolve config
5. Run verification script

**Week 3:**
1. Start HPO for standard implementations
2. Create evaluator
3. Test evaluator on known architectures
4. Prepare for evolution launch

---

## Success Criteria Checklist

Before claiming experiment complete:

- [ ] All three frameworks evolved
- [ ] Validation with 10 seeds each
- [ ] Statistical tests computed
- [ ] Results match expected format
- [ ] Code and data available
- [ ] Verification script passes
- [ ] Documentation complete
- [ ] Visualizations generated
- [ ] Reproducibility verified
- [ ] Limitations documented

---

**Document Version:** 1.0
**Last Updated:** 2025-11-03
**Status:** Ready for implementation
**Estimated Total Time:** 6-8 weeks
**Estimated Compute Cost:** $500-1000 (3 GPUs × 3 weeks × $2/GPU-hour)
