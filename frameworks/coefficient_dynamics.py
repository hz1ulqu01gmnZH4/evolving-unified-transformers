"""
Coefficient Dynamics Framework (Sieber et al., 2025)

Framework: All sequence models compute outputs as linear combinations where
           coefficients evolve via linear dynamics.

Mathematical formulation:
    output_t = Σ α_t,i · V_i
    where α evolves via autonomous linear dynamics

This module provides both:
1. StandardCoefficientDynamics: Reference implementation (softmax attention)
2. EvolvableCoefficientDynamics: Template for LLM-guided evolution

Evolution goal: Discover novel coefficient evolution rules that outperform softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .base import EvolvableAttentionBase, validate_attention_output


class StandardCoefficientDynamics(EvolvableAttentionBase):
    """
    Standard implementation: Softmax attention as coefficient dynamics.

    This is the BASELINE implementation that evolution will compete against.
    It implements standard multi-head attention with softmax coefficient computation.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(hidden_dim, num_heads, dropout)

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard softmax attention.

        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len, seq_len] or [batch, 1, seq_len, seq_len]

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, hidden_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (coefficient dynamics: α = softmax(QK^T/√d))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, seq_len, seq_len]

        # Apply mask if provided (for causality)
        if mask is not None:
            if mask.dim() == 2:  # [seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention coefficients α_t
        attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention coefficients to values (output_t = Σ α_t,i · V_i)
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, head_dim]

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        # Validate output
        validate_attention_output(output, (batch_size, seq_len, self.hidden_dim))

        return output


# EVOLVE-BLOCK-START: CoefficientDynamics
class EvolvableCoefficientDynamics(EvolvableAttentionBase):
    """
    Evolvable coefficient dynamics attention mechanism.

    EVOLUTION INSTRUCTIONS FOR LLM:

    You may evolve how attention coefficients (α) are computed and how they evolve.

    **What you CAN modify:**
    1. How scores are computed from Q, K, V
       - Example: Use different kernels, learned temperature, adaptive scaling
    2. How coefficients evolve across positions
       - Example: Add recurrence, momentum, learned dynamics
    3. Mixing functions and aggregation strategies
       - Example: Different ways to combine α with V
    4. Learned parameters within attention
       - Example: Position-dependent biases, learnable gates

    **What you MUST preserve:**
    1. Function signature: forward(x, mask=None) -> output
    2. Input shape: [batch, seq_len, hidden_dim]
    3. Output shape: [batch, seq_len, hidden_dim]
    4. Causality: If mask is provided, respect it (no future information)
    5. Numerical stability: No NaN, no Inf

    **Constraints:**
    - Must be expressible within coefficient dynamics framework (output = Σ α · V)
    - Must be differentiable (for gradient-based training)
    - Should maintain multi-head structure (but can modify how heads interact)

    **Current implementation:** Identical to StandardCoefficientDynamics
    **Evolution goal:** Discover better coefficient evolution rules
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(hidden_dim, num_heads, dropout)

        # Standard projections (can be modified by evolution)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)

        # Evolution can add learned parameters here
        # Example: self.temperature = nn.Parameter(torch.ones(1))
        # Example: self.dynamics_matrix = nn.Parameter(torch.randn(seq_len, seq_len))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolvable coefficient dynamics attention.

        Standard implementation (identical to baseline):
        - Compute Q, K, V
        - scores = QK^T / √d
        - α = softmax(scores)
        - output = α @ V

        Evolution can modify:
        - Score computation (different kernels, scaling, biases)
        - Coefficient evolution (recurrence, dynamics, gating)
        - Value aggregation (different mixing strategies)

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

        # Compute scores (EVOLVABLE: modify this computation)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask (MUST preserve this for causality)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Compute attention coefficients (EVOLVABLE: modify coefficient dynamics)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply coefficients to values (EVOLVABLE: modify mixing strategy)
        output = torch.matmul(attn_weights, V)

        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        # Validate (MUST pass validation)
        validate_attention_output(output, (batch_size, seq_len, self.hidden_dim))

        return output
# EVOLVE-BLOCK-END: CoefficientDynamics
