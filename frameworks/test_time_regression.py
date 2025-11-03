"""
Test-Time Regression Framework (Wang et al., 2025)

Framework: Sequence models perform associative recall via regression at test time

Mathematical formulation:
    Memorization: store (K, V) pairs during forward pass
    Retrieval: y = regression(K, V, query) at each position

This module provides both:
1. StandardTestTimeRegression: Reference implementation (softmax kernel regression)
2. EvolvableTestTimeRegression: Template for LLM-guided evolution

Evolution goal: Discover novel regression kernels/methods that outperform softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .base import EvolvableAttentionBase, validate_attention_output


class StandardTestTimeRegression(EvolvableAttentionBase):
    """
    Standard implementation: Softmax kernel regression (equivalent to attention).

    This is the BASELINE implementation.
    Interprets attention as kernel regression with exp(·) kernel.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(hidden_dim, num_heads, dropout)

        # Query, Key, Value projections (for regression)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)
        self.temperature = 1.0 / math.sqrt(self.head_dim)  # Standard temperature

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Test-time regression with softmax kernel.

        Regression interpretation:
        - K, V = "memorized" key-value pairs
        - Q = query at test time
        - Kernel: K(q, k) = exp(q·k / τ)
        - Output: weighted average of V based on kernel similarities

        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len, seq_len] or None

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q (queries), K (keys), V (values)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head regression
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute kernel similarities (K(q, k) = exp(q·k / τ))
        similarities = torch.matmul(Q, K.transpose(-2, -1)) * self.temperature

        # Apply mask (causal masking for autoregressive)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            similarities = similarities.masked_fill(~mask, float('-inf'))

        # Kernel regression weights
        kernel_weights = F.softmax(similarities, dim=-1)
        kernel_weights = self.dropout_layer(kernel_weights)

        # Regression: output = Σ K(q, k_i) · v_i
        output = torch.matmul(kernel_weights, V)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        validate_attention_output(output, (batch_size, seq_len, self.hidden_dim))

        return output


# EVOLVE-BLOCK-START: TestTimeRegression
class EvolvableTestTimeRegression(EvolvableAttentionBase):
    """
    Evolvable test-time regression mechanism.

    EVOLUTION INSTRUCTIONS FOR LLM:

    You may evolve the regression kernel, aggregation method, and learned parameters.

    **What you CAN modify:**
    1. Regression kernel function
       - Standard: K(q,k) = exp(q·k / τ)
       - Alternatives: Polynomial, RBF, learned kernels, mixture of kernels
    2. Temperature/bandwidth parameter
       - Can be learned, adaptive, position-dependent
    3. Regularization strategy
       - Ridge regression, Lasso, elastic net, learned regularization
    4. Multi-resolution regression
       - Different kernels at different scales
       - Hierarchical regression paths
    5. Value transformation before regression
       - Learned basis functions, nonlinear transforms

    **What you MUST preserve:**
    1. Function signature: forward(x, mask=None) -> output
    2. Input/output shapes: [batch, seq_len, hidden_dim]
    3. Causality: Respect mask (no future K,V pairs)
    4. Numerical stability: No NaN/Inf
    5. Interpretability as regression (retrieve based on memorized K,V)

    **Constraints:**
    - Must be expressible as regression framework (query → retrieve from stored K,V)
    - Must be differentiable
    - Should maintain multi-head structure

    **Current implementation:** Identical to StandardTestTimeRegression
    **Evolution goal:** Discover better regression kernels/methods
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(hidden_dim, num_heads, dropout)

        # Standard projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)
        self.temperature = 1.0 / math.sqrt(self.head_dim)

        # Evolution can add learned parameters here
        # Example: self.kernel_type = 'rbf'
        # Example: self.bandwidth = nn.Parameter(torch.ones(num_heads, 1, 1))
        # Example: self.regularization = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolvable test-time regression.

        Standard implementation:
        - K(q, k) = exp(q·k / τ)
        - output = Σ K(q, k_i) · v_i / Σ K(q, k_i)

        Evolution can modify:
        - Kernel function (polynomial, RBF, learned, etc.)
        - Temperature/bandwidth (learned, adaptive)
        - Regularization (ridge, lasso, learned)
        - Multi-scale or hierarchical regression

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

        # Compute kernel similarities (EVOLVABLE: change kernel function)
        similarities = torch.matmul(Q, K.transpose(-2, -1)) * self.temperature

        # Apply mask (MUST preserve for causality)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            similarities = similarities.masked_fill(~mask, float('-inf'))

        # Compute regression weights (EVOLVABLE: modify regression method)
        kernel_weights = F.softmax(similarities, dim=-1)
        kernel_weights = self.dropout_layer(kernel_weights)

        # Perform regression (EVOLVABLE: modify aggregation)
        output = torch.matmul(kernel_weights, V)

        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        # Validate (MUST pass)
        validate_attention_output(output, (batch_size, seq_len, self.hidden_dim))

        return output
# EVOLVE-BLOCK-END: TestTimeRegression
