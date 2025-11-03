"""
Base classes and interfaces for evolvable attention frameworks.

This module defines the abstract base class that all framework implementations must inherit from.
The key principle: ONLY attention mechanisms can evolve, everything else stays fixed.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class EvolvableAttentionBase(nn.Module, ABC):
    """
    Abstract base class for evolvable attention mechanisms.

    All framework implementations (coefficient_dynamics, test_time_regression, matrix_mixer)
    must inherit from this class and implement the forward method.

    Key constraints:
    - Must maintain causality (no future information)
    - Output shape must match input: [batch, seq_len, hidden_dim]
    - Must be numerically stable (no NaN/Inf)
    - Must be differentiable
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for attention mechanism.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim]
            mask: Optional attention mask [batch, seq_len, seq_len] or [batch, 1, seq_len, seq_len]
                  True/1 for positions to attend to, False/0 for masked positions

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim]
        """
        pass

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask of shape [seq_len, seq_len] where mask[i,j] = True if i >= j
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask


class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network with GELU activation.

    IMPORTANT: This component is FIXED and should NEVER be evolved.
    - Activation: GELU (NOT SwiGLU, NOT any gated variant)
    - Structure: Linear -> GELU -> Dropout -> Linear
    - No parallel branches, no gating, no modifications allowed
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard 4x expansion

        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()  # FIXED: Standard GELU
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def validate_attention_output(output: torch.Tensor, expected_shape: tuple) -> bool:
    """
    Validate that attention output meets requirements.

    Args:
        output: Attention output tensor
        expected_shape: Expected shape (batch, seq_len, hidden_dim)

    Returns:
        True if valid, raises exception otherwise
    """
    if output.shape != expected_shape:
        raise ValueError(f"Output shape {output.shape} doesn't match expected {expected_shape}")

    if torch.isnan(output).any():
        raise ValueError("Output contains NaN values")

    if torch.isinf(output).any():
        raise ValueError("Output contains Inf values")

    return True
