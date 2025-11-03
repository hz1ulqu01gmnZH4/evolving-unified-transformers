"""
Scaling Law Evaluator for OpenEvolve Fitness Function

This evaluator computes fitness by:
1. Training 4 model sizes (0.5M, 1M, 2M, 4M params) for 100k steps each
2. Fitting scaling law: loss = intercept + slope × log(n_params)
3. Returning -slope as fitness (better scaling = higher fitness)

CRITICAL: Uses 100k-step converged performance, NOT 2k-step proxy!
"""

import sys
sys.path.insert(0, '/home/ak/evolving-unified-transformers')

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import torch
import torch.nn as nn
import numpy as np
from scipy import stats

from models.language_model import create_language_model
from data.tinystories_loader import create_dataloaders
from training.train import Trainer


class ScalingLawEvaluator:
    """
    Evaluates evolved code by measuring scaling law slope.

    This is the FITNESS FUNCTION for OpenEvolve evolution.
    """

    def __init__(
        self,
        framework: str,
        tokenizer_path: str = '/home/ak/evolving-unified-transformers/data/tinystories_bpe_16k.json',
        training_steps: int = 100000,
        eval_frequency: int = 5000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        warmup_steps: int = 1000,
        device: str = 'cuda',
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            framework: Framework name ('coefficient_dynamics', 'test_time_regression', 'matrix_mixer')
            tokenizer_path: Path to BPE tokenizer
            training_steps: Number of training steps (MUST be 100000 for correct fitness!)
            eval_frequency: Evaluate every N steps
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            device: Device to train on
            cache_dir: Directory to cache results
        """
        self.framework = framework
        self.tokenizer_path = tokenizer_path
        self.training_steps = training_steps
        self.eval_frequency = eval_frequency
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Model sizes to evaluate (parameters)
        # These are configured to give approximately 0.5M, 1M, 2M, 4M params
        self.model_configs = [
            {'d_model': 96, 'num_layers': 4, 'num_heads': 4, 'target_params': 0.5e6},
            {'d_model': 128, 'num_layers': 4, 'num_heads': 4, 'target_params': 1.0e6},
            {'d_model': 128, 'num_layers': 6, 'num_heads': 4, 'target_params': 2.0e6},
            {'d_model': 192, 'num_layers': 6, 'num_heads': 6, 'target_params': 4.0e6},
        ]

        print(f"ScalingLawEvaluator initialized:")
        print(f"  Framework: {framework}")
        print(f"  Training steps: {training_steps:,}")
        print(f"  Model sizes: {len(self.model_configs)}")
        print(f"  Device: {self.device}")

    def train_single_model(
        self,
        model_config: Dict,
        variant: str,
        output_dir: str,
        seed: int = 42
    ) -> float:
        """
        Train a single model and return final validation loss.

        Args:
            model_config: Model configuration (d_model, num_layers, num_heads)
            variant: 'standard' or 'evolved'
            output_dir: Directory to save checkpoints
            seed: Random seed

        Returns:
            Final validation loss
        """
        torch.manual_seed(seed)

        # Create model
        model = create_language_model(
            framework=self.framework,
            variant=variant,
            vocab_size=16384,
            d_model=model_config['d_model'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=0.1,
            max_seq_len=128
        )

        param_count = model.count_parameters()['total']
        print(f"  Training {variant} model: {param_count:,} params "
              f"(d_model={model_config['d_model']}, layers={model_config['num_layers']})")

        # Create dataloaders
        dataloaders = create_dataloaders(
            tokenizer_path=self.tokenizer_path,
            batch_size=self.batch_size,
            max_length=128,
            num_workers=4
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=0.0,
            warmup_steps=self.warmup_steps,
            max_steps=self.training_steps,
            eval_frequency=self.eval_frequency,
            save_frequency=self.training_steps,  # Only save at end
            output_dir=output_dir
        )

        # Train
        final_loss = trainer.train()

        return final_loss

    def fit_scaling_law(
        self,
        param_counts: List[int],
        losses: List[float]
    ) -> Tuple[float, float, float]:
        """
        Fit scaling law: loss = intercept + slope × log(params)

        Args:
            param_counts: List of parameter counts
            losses: List of corresponding losses

        Returns:
            (slope, intercept, r_squared)
        """
        # Convert to log space
        log_params = np.log(param_counts)
        losses_array = np.array(losses)

        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, losses_array)

        r_squared = r_value ** 2

        print(f"\nScaling law fit:")
        print(f"  loss = {intercept:.4f} + {slope:.4f} × log(params)")
        print(f"  R² = {r_squared:.4f}")
        print(f"  Better scaling → more negative slope")

        return slope, intercept, r_squared

    def evaluate_variant(
        self,
        variant: str,
        output_base_dir: str,
        seed: int = 42
    ) -> Dict:
        """
        Evaluate a variant across all model sizes.

        Args:
            variant: 'standard' or 'evolved'
            output_base_dir: Base directory for outputs
            seed: Random seed

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*70}")
        print(f"Evaluating {variant} variant")
        print(f"{'='*70}")

        param_counts = []
        losses = []

        for i, config in enumerate(self.model_configs):
            print(f"\nModel {i+1}/{len(self.model_configs)}")

            output_dir = os.path.join(output_base_dir, f"{variant}_size_{i}")
            os.makedirs(output_dir, exist_ok=True)

            # Train model
            loss = self.train_single_model(
                model_config=config,
                variant=variant,
                output_dir=output_dir,
                seed=seed
            )

            # Get actual parameter count
            model = create_language_model(
                framework=self.framework,
                variant=variant,
                vocab_size=16384,
                d_model=config['d_model'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads']
            )
            param_count = model.count_parameters()['total']

            param_counts.append(param_count)
            losses.append(loss)

            print(f"  → Final loss: {loss:.4f} (params: {param_count:,})")

        # Fit scaling law
        slope, intercept, r_squared = self.fit_scaling_law(param_counts, losses)

        # Compute fitness
        fitness = -slope  # Better scaling (more negative slope) = higher fitness

        results = {
            'variant': variant,
            'param_counts': param_counts,
            'losses': losses,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'fitness': fitness
        }

        print(f"\n{variant} results:")
        print(f"  Slope: {slope:.6f}")
        print(f"  Fitness: {fitness:.6f}")

        return results

    def evaluate_program(
        self,
        program_code: str,
        output_dir: str,
        iteration: int = 0
    ) -> float:
        """
        Evaluate evolved program code.

        This is the main fitness function called by OpenEvolve.

        Args:
            program_code: Python code for evolved attention mechanism
            output_dir: Directory to save results
            iteration: Evolution iteration number

        Returns:
            Fitness value (higher is better)
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING ITERATION {iteration}")
        print(f"{'='*70}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save evolved code
            evolved_code_path = output_dir / f'evolved_iteration_{iteration}.py'
            with open(evolved_code_path, 'w') as f:
                f.write(program_code)

            # TODO: Load evolved code and replace evolvable attention class
            # For now, evaluate standard variant as placeholder
            # In full implementation, this would:
            # 1. Execute program_code to get evolved attention class
            # 2. Replace EvolvableCoefficientDynamics/etc with evolved version
            # 3. Evaluate evolved variant

            print("\n[WARNING] Using standard variant for now")
            print("[TODO] Implement evolved code loading in production version")

            results = self.evaluate_variant(
                variant='standard',  # Will be 'evolved' once code loading is implemented
                output_base_dir=str(output_dir),
                seed=42 + iteration  # Different seed per iteration
            )

            # Save results
            results_path = output_dir / 'results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            fitness = results['fitness']

            print(f"\n{'='*70}")
            print(f"ITERATION {iteration} COMPLETE")
            print(f"Fitness: {fitness:.6f} (slope: {results['slope']:.6f})")
            print(f"{'='*70}\n")

            return fitness

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"ERROR in iteration {iteration}:")
            print(f"{'='*70}")
            print(traceback.format_exc())

            # Return very poor fitness on error
            return -1000.0


def quick_evaluator_test():
    """Quick test with reduced training steps."""
    print("Quick Evaluator Test (100 steps per model)")
    print("="*70)

    evaluator = ScalingLawEvaluator(
        framework='coefficient_dynamics',
        training_steps=100,  # Quick test
        eval_frequency=50,
        batch_size=16,
        device='cuda'
    )

    # Test with just 2 model sizes for speed
    evaluator.model_configs = evaluator.model_configs[:2]

    output_dir = '/home/ak/evolving-unified-transformers/tests/evaluator_test'

    results = evaluator.evaluate_variant(
        variant='standard',
        output_base_dir=output_dir,
        seed=42
    )

    print("\n" + "="*70)
    print("Quick test complete!")
    print(f"Slope: {results['slope']:.6f}")
    print(f"Fitness: {results['fitness']:.6f}")
    print(f"R²: {results['r_squared']:.4f}")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run quick test')
    args = parser.parse_args()

    if args.test:
        quick_evaluator_test()
    else:
        print("Use --test for quick evaluator test")
        print("This module is meant to be imported by OpenEvolve")
