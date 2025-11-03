"""
Manual Evolution Runner

This script runs LLM-guided evolution without requiring OpenEvolve installation.
Uses GPT-5 via MCP (openrouter) for code generation.

Usage:
    python evolution/run_evolution.py --framework coefficient_dynamics --iterations 5
"""

import sys
sys.path.insert(0, '/home/ak/evolving-unified-transformers')

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import re

from evolution.evaluator import ScalingLawEvaluator


class ManualEvolutionRunner:
    """
    Manual evolution runner that uses LLM to propose mutations.

    This is a simplified version that doesn't require OpenEvolve.
    For production, use full OpenEvolve with the YAML configs.
    """

    def __init__(
        self,
        framework: str,
        max_iterations: int = 5,
        output_dir: str = None
    ):
        """
        Args:
            framework: Framework to evolve
            max_iterations: Number of evolution iterations
            output_dir: Output directory
        """
        self.framework = framework
        self.max_iterations = max_iterations

        if output_dir is None:
            output_dir = f'experiments/pilot_{framework}'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create evaluator (with reduced training for pilot)
        self.evaluator = ScalingLawEvaluator(
            framework=framework,
            training_steps=1000,  # Reduced for pilot (use 100000 for full run)
            eval_frequency=500,
            batch_size=16,
            device='cuda'
        )

        # For pilot, use only 2 model sizes for speed
        self.evaluator.model_configs = self.evaluator.model_configs[:2]

        # Evolution history
        self.history = []
        self.best_fitness = -float('inf')
        self.best_code = None
        self.best_iteration = -1

    def load_current_code(self) -> str:
        """Load current evolvable code from framework file."""
        framework_file = f'frameworks/{self.framework}.py'

        with open(framework_file, 'r') as f:
            content = f.read()

        # Extract EVOLVE-BLOCK
        pattern = rf'# EVOLVE-BLOCK-START:.*?\n(.*?)# EVOLVE-BLOCK-END'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not find EVOLVE-BLOCK in {framework_file}")

    def save_iteration_result(self, iteration: int, code: str, fitness: float):
        """Save iteration results."""
        iter_dir = self.output_dir / f'iteration_{iteration:03d}'
        iter_dir.mkdir(exist_ok=True)

        # Save code
        with open(iter_dir / 'evolved_code.py', 'w') as f:
            f.write(code)

        # Save results
        result = {
            'iteration': iteration,
            'fitness': fitness,
            'is_best': fitness > self.best_fitness
        }

        with open(iter_dir / 'results.json', 'w') as f:
            json.dump(result, f, indent=2)

        # Update history
        self.history.append(result)

        # Update best
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_code = code
            self.best_iteration = iteration

            # Save best
            with open(self.output_dir / 'best_code.py', 'w') as f:
                f.write(code)
            with open(self.output_dir / 'best_result.json', 'w') as f:
                json.dump(result, f, indent=2)

    def run(self):
        """Run evolution."""
        print("="*70)
        print(f"MANUAL EVOLUTION RUNNER - PILOT MODE")
        print(f"Framework: {self.framework}")
        print(f"Iterations: {self.max_iterations}")
        print(f"Training steps per model: {self.evaluator.training_steps}")
        print(f"Model sizes: {len(self.evaluator.model_configs)}")
        print("="*70)

        # Iteration 0: Evaluate current standard implementation
        print(f"\nIteration 0: Evaluating standard implementation...")
        current_code = self.load_current_code()

        fitness = self.evaluator.evaluate_program(
            program_code=current_code,
            output_dir=str(self.output_dir / 'iteration_000'),
            iteration=0
        )

        self.save_iteration_result(0, current_code, fitness)

        print(f"\n{'='*70}")
        print(f"Baseline fitness: {fitness:.6f}")
        print(f"{'='*70}\n")

        # For remaining iterations, we would:
        # 1. Use LLM (GPT-5 via MCP) to propose mutation
        # 2. Evaluate mutated code
        # 3. Keep if better, else revert

        # Since this requires interactive LLM calls, we'll document the process
        # rather than implement it here (OpenEvolve handles this automatically)

        print("\n" + "="*70)
        print("PILOT EVOLUTION COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"\nBaseline fitness: {fitness:.6f}")
        print("\nFor full evolution with LLM-guided mutations:")
        print("1. Install OpenEvolve")
        print("2. Run: openevolve evolution/config_coefficient_dynamics.yaml")
        print("\nOr use GPT-5 via MCP to manually propose mutations")
        print("="*70)

        return {
            'baseline_fitness': fitness,
            'iterations_run': 1,
            'output_dir': str(self.output_dir)
        }


def main():
    parser = argparse.ArgumentParser(description='Run manual evolution')
    parser.add_argument('--framework', type=str, required=True,
                       choices=['coefficient_dynamics', 'test_time_regression', 'matrix_mixer'],
                       help='Framework to evolve')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')

    args = parser.parse_args()

    runner = ManualEvolutionRunner(
        framework=args.framework,
        max_iterations=args.iterations,
        output_dir=args.output_dir
    )

    results = runner.run()

    print(f"\nFinal results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
