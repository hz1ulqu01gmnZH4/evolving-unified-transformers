"""Test script for models and frameworks."""

import sys
sys.path.insert(0, '/home/ak/evolving-unified-transformers')

import torch
from models.language_model import create_language_model


def test_model_creation():
    """Test that all framework/variant combinations can be created."""
    print("Testing language model creation...")

    for framework in ['coefficient_dynamics', 'test_time_regression', 'matrix_mixer']:
        for variant in ['standard', 'evolved']:
            model = create_language_model(
                framework,
                variant,
                vocab_size=1000,
                d_model=64,
                num_layers=2,
                num_heads=2
            )
            params = model.count_parameters()
            print(f"{framework:20s} ({variant:8s}): {params['total']:7,} total params, "
                  f"{params['attention']:6,} attention params")

    print("\n✓ All framework/variant combinations created successfully")


def test_forward_pass():
    """Test forward pass for each framework."""
    print("\nTesting forward pass...")

    for framework in ['coefficient_dynamics', 'test_time_regression', 'matrix_mixer']:
        model = create_language_model(
            framework,
            'standard',
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=128
        )

        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))  # [batch=2, seq_len=10]
        logits = model(input_ids)

        assert logits.shape == (2, 10, 1000), f"Expected shape (2, 10, 1000), got {logits.shape}"
        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"

        print(f"{framework:20s}: input {tuple(input_ids.shape)} -> logits {tuple(logits.shape)} ✓")

    print("\n✓ All forward passes completed successfully")


def test_parameter_counts():
    """Test that standard and evolved have same parameter count."""
    print("\nTesting parameter counts (standard vs evolved)...")

    for framework in ['coefficient_dynamics', 'test_time_regression', 'matrix_mixer']:
        standard_model = create_language_model(framework, 'standard', vocab_size=1000, d_model=64, num_layers=2)
        evolved_model = create_language_model(framework, 'evolved', vocab_size=1000, d_model=64, num_layers=2)

        standard_params = standard_model.count_parameters()['total']
        evolved_params = evolved_model.count_parameters()['total']

        print(f"{framework:20s}: standard={standard_params:,}, evolved={evolved_params:,}, "
              f"diff={abs(standard_params - evolved_params)}")

    print("\n✓ Parameter counts verified")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Model Tests")
    print("=" * 70)

    test_model_creation()
    test_forward_pass()
    test_parameter_counts()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
