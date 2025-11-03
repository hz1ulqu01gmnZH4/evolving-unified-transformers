"""
Language Model with Fixed Architecture

Complete language model for training and evaluation.
Adds embeddings and output projection to the fixed transformer.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable

from .transformer import FixedTransformer, create_causal_mask


class LanguageModel(nn.Module):
    """
    Autoregressive language model with fixed architecture and evolvable attention.

    Architecture:
    1. Token embedding
    2. Fixed transformer with evolvable attention
    3. Output projection to vocabulary

    All components except attention mechanism are fixed.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        attention_module_factory: Callable,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        tie_weights: bool = True
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            attention_module_factory: Function to create attention module
            d_ff: FFN hidden dimension (default: 4 * d_model)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            tie_weights: Whether to tie input/output embeddings
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Dropout after embedding
        self.embedding_dropout = nn.Dropout(dropout)

        # Fixed transformer with evolvable attention
        self.transformer = FixedTransformer(
            num_layers=num_layers,
            d_model=d_model,
            attention_module_factory=attention_module_factory,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Optionally tie input and output embeddings
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using standard scheme."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Initialize output projection (if not tied)
        if self.output_projection.weight is not self.token_embedding.weight:
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for language modeling.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            mask: Attention mask [batch, seq_len, seq_len] or None (will create causal mask)
            return_logits: If True, return logits; if False, return probabilities

        Returns:
            Logits [batch, seq_len, vocab_size] or probabilities
        """
        batch_size, seq_len = input_ids.shape

        # Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len, input_ids.device)

        # Embed tokens
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.embedding_dropout(x)

        # Pass through transformer
        x = self.transformer(x, mask=mask, add_positional_encoding=True)

        # Project to vocabulary
        logits = self.output_projection(x)  # [batch, seq_len, vocab_size]

        if return_logits:
            return logits
        else:
            return torch.softmax(logits, dim=-1)

    def count_parameters(self) -> dict:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count attention parameters specifically
        attention_params = 0
        for name, module in self.named_modules():
            if 'attention' in name.lower():
                attention_params += sum(p.numel() for p in module.parameters())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'attention': attention_params,
            'non_attention': total_params - attention_params
        }


def create_language_model(
    framework: str,
    variant: str,
    vocab_size: int = 16384,
    d_model: int = 128,
    num_layers: int = 6,
    num_heads: int = 4,
    d_ff: Optional[int] = None,
    dropout: float = 0.1,
    max_seq_len: int = 512
) -> LanguageModel:
    """
    Factory function to create language model with specific framework and variant.

    Args:
        framework: One of ['coefficient_dynamics', 'test_time_regression', 'matrix_mixer']
        variant: One of ['standard', 'evolved']
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: FFN hidden dimension (default: 4 * d_model)
        dropout: Dropout rate
        max_seq_len: Maximum sequence length

    Returns:
        LanguageModel instance
    """
    # Import framework modules
    if framework == 'coefficient_dynamics':
        from frameworks.coefficient_dynamics import StandardCoefficientDynamics, EvolvableCoefficientDynamics
        AttentionClass = StandardCoefficientDynamics if variant == 'standard' else EvolvableCoefficientDynamics
    elif framework == 'test_time_regression':
        from frameworks.test_time_regression import StandardTestTimeRegression, EvolvableTestTimeRegression
        AttentionClass = StandardTestTimeRegression if variant == 'standard' else EvolvableTestTimeRegression
    elif framework == 'matrix_mixer':
        from frameworks.matrix_mixer import StandardMatrixMixer, EvolvableMatrixMixer
        AttentionClass = StandardMatrixMixer if variant == 'standard' else EvolvableMatrixMixer
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # Create attention module factory
    def attention_factory(d_model, num_heads, dropout):
        return AttentionClass(d_model, num_heads, dropout)

    # Create and return model
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        attention_module_factory=attention_factory,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len,
        tie_weights=True
    )

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing language model creation...")

    for framework in ['coefficient_dynamics', 'test_time_regression', 'matrix_mixer']:
        for variant in ['standard', 'evolved']:
            model = create_language_model(framework, variant, vocab_size=1000, d_model=64, num_layers=2, num_heads=2)
            params = model.count_parameters()
            print(f"{framework} ({variant}): {params['total']:,} total params, {params['attention']:,} attention params")

    # Test forward pass
    model = create_language_model('coefficient_dynamics', 'standard', vocab_size=1000, d_model=64)
    input_ids = torch.randint(0, 1000, (2, 10))  # [batch=2, seq_len=10]
    logits = model(input_ids)
    print(f"\nForward pass test: input {input_ids.shape} -> logits {logits.shape}")
    print("All tests passed!")
