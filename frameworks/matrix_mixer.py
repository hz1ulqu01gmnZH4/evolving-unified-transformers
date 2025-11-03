"""
Matrix Mixer Framework (Hwang et al., 2024 - Hydra)

Framework: Sequence mixing as structured matrix operations

Mathematical formulation:
    output = M · X  where M has specific structure (dense, low-rank, sparse, etc.)

This module provides both:
1. StandardMatrixMixer: Reference implementation (dense or low-rank mixing)
2. EvolvableMatrixMixer: Template for LLM-guided evolution

Evolution goal: Discover novel mixing matrix structures that outperform standard forms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .base import EvolvableAttentionBase, validate_attention_output


class StandardMatrixMixer(EvolvableAttentionBase):
    """
    Standard implementation: Attention as structured matrix mixing.

    This is the BASELINE implementation.
    Interprets attention as applying a mixing matrix M to sequence X.
    M is data-dependent (computed from Q, K) and has low-rank structure.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(hidden_dim, num_heads, dropout)

        # Projections for constructing mixing matrix
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Matrix mixer with attention-based mixing matrix.

        Matrix interpretation:
        - M = softmax(QK^T / √d)  # Data-dependent mixing matrix
        - V = value transformation of X
        - output = M @ V  # Apply mixing

        M is:
        - Dense O(n²) or low-rank O(nr)
        - Data-dependent (computed from input)
        - Stochastic rows (due to softmax)

        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len, seq_len] or None

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head mixing
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Construct mixing matrix M = softmax(QK^T / √d)
        mixing_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask (for causal mixing)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mixing_scores = mixing_scores.masked_fill(~mask, float('-inf'))

        # Mixing matrix (stochastic rows)
        mixing_matrix = F.softmax(mixing_scores, dim=-1)  # [batch, heads, seq_len, seq_len]
        mixing_matrix = self.dropout_layer(mixing_matrix)

        # Apply mixing: output = M @ V
        output = torch.matmul(mixing_matrix, V)  # [batch, heads, seq_len, head_dim]

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        validate_attention_output(output, (batch_size, seq_len, self.hidden_dim))

        return output


# EVOLVE-BLOCK-START: MatrixMixer
class EvolvableMatrixMixer(EvolvableAttentionBase):
    """
    Evolvable matrix mixer mechanism.

    EVOLUTION INSTRUCTIONS FOR LLM:

    You may evolve the structure, parameterization, and computation of mixing matrix M.

    **What you CAN modify:**
    1. Matrix structure
       - Dense, low-rank, sparse, block-diagonal, Toeplitz, circulant
       - Learned sparsity patterns
       - Hierarchical structure (coarse-to-fine)
    2. Matrix parameterization
       - How M is computed from input
       - Learned vs data-dependent components
       - Factorizations (M = ABC, SVD-like, etc.)
    3. Rank allocation
       - Adaptive rank per head or position
       - Learnable rank scheduling
    4. Mixing strategies
       - Sequential mixing (M1 @ M2 @ ... @ X)
       - Parallel mixing paths
       - Residual mixing (X + M @ X)

    **What you MUST preserve:**
    1. Function signature: forward(x, mask=None) -> output
    2. Input/output shapes: [batch, seq_len, hidden_dim]
    3. Causality: If mask provided, M must be lower-triangular (or respect mask)
    4. Numerical stability: No NaN/Inf
    5. Expressibility as matrix operation: output = f(M, X)

    **Constraints:**
    - Must be expressible as structured matrix operations
    - Must be differentiable
    - Mixing matrix M can depend on data (like attention) or be learned (like MLP-Mixer)

    **Current implementation:** Identical to StandardMatrixMixer
    **Evolution goal:** Discover better mixing matrix structures
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(hidden_dim, num_heads, dropout)

        # Standard projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)

        # Evolution can add learned parameters for matrix structure
        # Example: self.mixing_pattern = nn.Parameter(torch.randn(seq_len, seq_len))
        # Example: self.rank_scheduler = nn.Parameter(torch.ones(num_heads))
        # Example: self.sparsity_mask = nn.Parameter(torch.ones(seq_len, seq_len))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolvable matrix mixer.

        Standard implementation:
        - M = softmax(QK^T / √d)  # Data-dependent dense matrix
        - output = M @ V

        Evolution can modify:
        - Matrix structure (sparse, low-rank, block, learned patterns)
        - Parameterization (how M is computed)
        - Rank allocation (adaptive rank)
        - Mixing strategy (sequential, parallel, residual)

        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len, seq_len] or None

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Construct mixing matrix (EVOLVABLE: change structure/parameterization)
        mixing_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask (MUST preserve for causality)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mixing_scores = mixing_scores.masked_fill(~mask, float('-inf'))

        # Compute mixing matrix (EVOLVABLE: change normalization/structure)
        mixing_matrix = F.softmax(mixing_scores, dim=-1)
        mixing_matrix = self.dropout_layer(mixing_matrix)

        # Apply mixing (EVOLVABLE: change mixing strategy)
        output = torch.matmul(mixing_matrix, V)

        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        # Validate (MUST pass)
        validate_attention_output(output, (batch_size, seq_len, self.hidden_dim))

        return output
# EVOLVE-BLOCK-END: MatrixMixer
