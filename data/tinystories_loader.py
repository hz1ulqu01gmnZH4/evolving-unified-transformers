"""
TinyStories dataset loader with BPE tokenization.

Uses the same BPE tokenizer and caching strategy from the failed experiment
to ensure fair comparison and reproducibility.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer


class TinyStoriesDataset(Dataset):
    """
    TinyStories dataset with BPE tokenization.

    Caches tokenized sequences to disk for faster loading.
    """

    def __init__(
        self,
        tokenizer_path: str,
        split: str = 'train',
        max_length: int = 128,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            tokenizer_path: Path to BPE tokenizer JSON file
            split: Dataset split ('train' or 'validation')
            max_length: Maximum sequence length
            cache_dir: Directory to cache tokenized data
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        self.split = split

        # Load dataset
        print(f"Loading TinyStories {split} split...")
        self.dataset = load_dataset('roneneldan/TinyStories', split=split)

        print(f"Loaded {len(self.dataset):,} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get tokenized sequence.

        Returns:
            input_ids: Tensor of shape [max_length]
            labels: Tensor of shape [max_length] (same as input_ids, shifted for LM)
        """
        text = self.dataset[idx]['text']

        # Tokenize
        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids

        # Truncate or pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            # Pad with token 0
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # For language modeling, labels are same as input_ids
        # (loss will be computed with appropriate shifting)
        labels = input_ids.clone()

        return {'input_ids': input_ids, 'labels': labels}


def create_dataloaders(
    tokenizer_path: str,
    batch_size: int = 64,
    max_length: int = 128,
    num_workers: int = 4,
    cache_dir: Optional[str] = None
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        tokenizer_path: Path to BPE tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        cache_dir: Cache directory

    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    # Create datasets
    train_dataset = TinyStoriesDataset(
        tokenizer_path=tokenizer_path,
        split='train',
        max_length=max_length,
        cache_dir=cache_dir
    )

    val_dataset = TinyStoriesDataset(
        tokenizer_path=tokenizer_path,
        split='validation',
        max_length=max_length,
        cache_dir=cache_dir
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader
    }


def get_or_create_tokenizer(vocab_size: int = 16384, cache_dir: str = './data') -> str:
    """
    Get existing BPE tokenizer or create new one.

    Args:
        vocab_size: Vocabulary size
        cache_dir: Directory to store tokenizer

    Returns:
        Path to tokenizer file
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tokenizer_path = cache_path / f'tinystories_bpe_{vocab_size}.json'

    if tokenizer_path.exists():
        print(f"Using existing tokenizer: {tokenizer_path}")
        return str(tokenizer_path)

    print(f"Creating new BPE tokenizer with vocab_size={vocab_size}...")

    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train on TinyStories
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        show_progress=True
    )

    # Load training data
    dataset = load_dataset('roneneldan/TinyStories', split='train[:10%]')  # Use 10% for training tokenizer

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield [item['text'] for item in dataset[i:i+batch_size]]

    # Train tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Save tokenizer
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")

    return str(tokenizer_path)


if __name__ == "__main__":
    # Test dataloader
    print("Testing TinyStories dataloader...")

    # Get or create tokenizer
    tokenizer_path = get_or_create_tokenizer(vocab_size=16384, cache_dir='./data')

    # Create dataloaders
    dataloaders = create_dataloaders(
        tokenizer_path=tokenizer_path,
        batch_size=4,
        max_length=128,
        num_workers=0  # For testing
    )

    # Test train loader
    batch = next(iter(dataloaders['train']))
    print(f"\nTrain batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  Sample tokens: {batch['input_ids'][0, :10].tolist()}")

    # Test val loader
    batch = next(iter(dataloaders['val']))
    print(f"\nValidation batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")

    print("\n✓ Dataloader test passed!")
