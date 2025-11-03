"""
Quick integration test to verify the entire training pipeline works.

This runs a mini training loop (100 steps) to ensure:
1. Models can be created
2. Data can be loaded
3. Training loop runs without errors
4. Gradients flow correctly
5. Evaluation works
"""

import sys
sys.path.insert(0, '/home/ak/evolving-unified-transformers')

import torch
from models.language_model import create_language_model
from data.tinystories_loader import create_dataloaders
from training.train import Trainer


def quick_test():
    """Run a quick 100-step training test."""
    print("=" * 70)
    print("Quick Integration Test (100 steps)")
    print("=" * 70)

    # Configuration
    config = {
        'framework': 'coefficient_dynamics',
        'variant': 'standard',
        'vocab_size': 16384,
        'd_model': 64,  # Smaller for faster testing
        'num_layers': 2,  # Fewer layers for faster testing
        'num_heads': 2,
        'dropout': 0.1,
        'max_seq_len': 128,
        'batch_size': 16,  # Smaller batch for faster testing
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'warmup_steps': 50,
        'max_steps': 100,
        'eval_frequency': 50,
        'save_frequency': 100,
        'num_workers': 0  # No multiprocessing for testing
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    print(f"\n1. Creating model ({config['framework']}/{config['variant']})...")
    model = create_language_model(
        framework=config['framework'],
        variant=config['variant'],
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    )

    params = model.count_parameters()
    print(f"   ✓ Model created: {params['total']:,} parameters")

    # Create dataloaders
    print("\n2. Creating dataloaders...")
    tokenizer_path = '/home/ak/evolving-unified-transformers/data/tinystories_bpe_16k.json'

    dataloaders = create_dataloaders(
        tokenizer_path=tokenizer_path,
        batch_size=config['batch_size'],
        max_length=config['max_seq_len'],
        num_workers=config['num_workers']
    )

    print(f"   ✓ Dataloaders created")

    # Test a batch
    print("\n3. Testing data loading...")
    batch = next(iter(dataloaders['train']))
    print(f"   ✓ Batch loaded: {batch['input_ids'].shape}")

    # Create trainer
    print("\n4. Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        eval_frequency=config['eval_frequency'],
        save_frequency=config['save_frequency'],
        output_dir='./tests/test_checkpoints'
    )

    print(f"   ✓ Trainer created")

    # Run training
    print(f"\n5. Running {config['max_steps']} training steps...")
    final_loss = trainer.train()

    print(f"\n   ✓ Training completed!")
    print(f"   Final validation loss: {final_loss:.4f}")

    # Verify checkpoints
    print("\n6. Verifying checkpoints...")
    import os
    checkpoint_path = './tests/test_checkpoints/checkpoint_latest.pt'
    assert os.path.exists(checkpoint_path), "Checkpoint not saved!"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert checkpoint['step'] == config['max_steps'], "Checkpoint step mismatch!"
    print(f"   ✓ Checkpoint verified (step {checkpoint['step']})")

    # Test all frameworks
    print("\n7. Testing all frameworks...")
    for framework in ['coefficient_dynamics', 'test_time_regression', 'matrix_mixer']:
        for variant in ['standard', 'evolved']:
            model_test = create_language_model(
                framework, variant,
                vocab_size=1000, d_model=32, num_layers=1, num_heads=2
            )
            # Test forward pass
            test_input = torch.randint(0, 1000, (2, 10))
            logits = model_test(test_input)
            assert logits.shape == (2, 10, 1000), f"Shape mismatch for {framework}/{variant}"
            print(f"   ✓ {framework:20s} ({variant:8s}): forward pass OK")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nThe implementation is ready for:")
    print("1. Creating OpenEvolve configuration files")
    print("2. Running full evolution experiments")
    print("3. Validation with 10 seeds × 100k steps")


if __name__ == "__main__":
    quick_test()
