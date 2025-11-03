"""
Fixed Transformer Architecture

This module implements the transformer with FIXED peripheral components.
ONLY the attention mechanism can be evolved - everything else stays standard.

Key design principle:
- FFN: Standard GELU (NOT SwiGLU, NOT any gated variant)
- Normalization: Standard LayerNorm (NOT RMSNorm)
- Block structure: Sequential attention → FFN (NOT parallel)
- Residuals: Standard addition (NOT LayerScale, NOT gating)

The attention mechanism is provided as a module and can be any of the three frameworks:
- CoefficientDynamics (standard or evolved)
- TestTimeRegression (standard or evolved)
- MatrixMixer (standard or evolved)
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from frameworks.base import StandardFFN


class FixedTransformerBlock(nn.Module):
    """
    Transformer block with fixed peripheral components and evolvable attention.

    Architecture (FIXED - cannot be evolved):
    1. x = x + attention(norm1(x))  # Pre-norm with standard residual
    2. x = x + ffn(norm2(x))         # Pre-norm with standard residual

    Components that are FIXED:
    - norm1, norm2: Standard LayerNorm
    - ffn: Standard GELU FFN
    - Residual: Standard addition (no scaling, no gating)
    - Block structure: Sequential (attention → FFN)

    Components that are EVOLVABLE:
    - attention: Can be any framework (coefficient_dynamics, test_time_regression, matrix_mixer)
    """

    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        d_ff: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            attention_module: Evolvable attention mechanism (framework-specific)
            d_ff: FFN hidden dimension (default: 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()

        # FIXED: Standard LayerNorm (NOT RMSNorm, NOT any variant)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # EVOLVABLE: Attention mechanism (can be any framework)
        self.attention = attention_module

        # FIXED: Standard GELU FFN (NOT SwiGLU, NOT gated)
        self.ffn = StandardFFN(d_model, d_ff or (4 * d_model), dropout)

        # FIXED: Standard dropout (used in residuals)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with FIXED sequential architecture.

        FIXED structure (cannot be modified):
        1. Attention block: x = x + dropout(attention(norm1(x)))
        2. FFN block: x = x + dropout(ffn(norm2(x)))

        No parallel branches, no layer scaling, no gating - just standard residuals.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len] or None

        Returns:
            Output [batch, seq_len, d_model]
        """
        # FIXED: Pre-norm attention with standard residual
        attn_output = self.attention(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_output)

        # FIXED: Pre-norm FFN with standard residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x


class FixedTransformer(nn.Module):
    """
    Complete transformer with fixed architecture and evolvable attention.

    This is the transformer that will be used for all experiments.
    The ONLY difference between standard and evolved variants is the attention mechanism.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        attention_module_factory,  # Function that creates attention module
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        """
        Args:
            num_layers: Number of transformer blocks
            d_model: Model dimension
            attention_module_factory: Function (d_model, num_heads, dropout) -> attention_module
            num_heads: Number of attention heads
            d_ff: FFN hidden dimension (default: 4 * d_model)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length (for positional encoding)
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Create transformer blocks with evolvable attention
        self.blocks = nn.ModuleList([
            FixedTransformerBlock(
                d_model=d_model,
                attention_module=attention_module_factory(d_model, num_heads, dropout),
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # FIXED: Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Positional encoding (sinusoidal, standard)
        self.register_buffer(
            'positional_encoding',
            self._create_positional_encoding(max_seq_len, d_model)
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create standard sinusoidal positional encoding.

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding [max_len, d_model]
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        add_positional_encoding: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through transformer.

        Args:
            x: Input embeddings [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len] or None
            add_positional_encoding: Whether to add positional encoding

        Returns:
            Output [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        if add_positional_encoding:
            x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final layer norm
        x = self.final_norm(x)

        return x


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal mask for autoregressive modeling.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask [seq_len, seq_len] where mask[i,j] = True if i >= j
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask
