"""
Training script for language models.

Trains a fixed transformer with evolvable attention on TinyStories.
"""

import sys
sys.path.insert(0, '/home/ak/evolving-unified-transformers')

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.language_model import create_language_model
from data.tinystories_loader import create_dataloaders


class Trainer:
    """Trainer for language models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        eval_frequency: int = 1000,
        save_frequency: int = 5000,
        output_dir: str = './checkpoints'
    ):
        """
        Args:
            model: Language model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            eval_frequency: Evaluate every N steps
            save_frequency: Save checkpoint every N steps
            output_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.max_steps = max_steps
        self.eval_frequency = eval_frequency
        self.save_frequency = save_frequency
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler with warmup
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

    def get_lr(self, step: int) -> float:
        """Get learning rate with linear warmup."""
        if step < self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps)
        return self.base_lr

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)  # [batch, seq_len]
        labels = batch['labels'].to(self.device)

        # Forward pass
        logits = self.model(input_ids)  # [batch, seq_len, vocab_size]

        # Compute loss (shift targets by 1 for next-token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.global_step)

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits = self.model(input_ids)

            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f'checkpoint_step_{step}.pt'
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, checkpoint_path)

        # Also save latest
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, latest_path)

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Training step
            loss = self.train_step(batch)
            self.train_losses.append((self.global_step, loss))

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{self.get_lr(self.global_step):.6f}'})

            self.global_step += 1

            # Evaluate
            if self.global_step % self.eval_frequency == 0:
                val_loss = self.evaluate()
                self.val_losses.append((self.global_step, val_loss))
                print(f"\nStep {self.global_step}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")

            # Save checkpoint
            if self.global_step % self.save_frequency == 0:
                self.save_checkpoint(self.global_step)

        pbar.close()

        # Final evaluation
        final_val_loss = self.evaluate()
        self.val_losses.append((self.global_step, final_val_loss))
        print(f"\nFinal validation loss: {final_val_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint(self.global_step)

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_val_loss': final_val_loss
        }
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        return final_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train language model')
    parser.add_argument('--framework', type=str, required=True,
                       choices=['coefficient_dynamics', 'test_time_regression', 'matrix_mixer'],
                       help='Framework to use')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['standard', 'evolved'],
                       help='Model variant')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--tokenizer_path', type=str,
                       default='/home/ak/evolving-unified-transformers/data/tinystories_bpe_16k.json',
                       help='Path to BPE tokenizer')

    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=16384)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int, default=128)

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--eval_frequency', type=int, default=1000)
    parser.add_argument('--save_frequency', type=int, default=5000)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    print(f"Creating {args.framework} ({args.variant}) model...")
    model = create_language_model(
        framework=args.framework,
        variant=args.variant,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    )

    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} (attention: {params['attention']:,})")

    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        num_workers=args.num_workers
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
        output_dir=args.output_dir
    )

    # Train
    final_loss = trainer.train()

    print(f"\nTraining complete! Final validation loss: {final_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
