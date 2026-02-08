import torch
import torch.nn as nn
from configs import SEED, BLOCK_SIZE, HEAD_SIZE, NUM_EMBED_DIMS
import torch.nn.functional as F

torch.manual_seed(SEED)


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(NUM_EMBED_DIMS, head_size, bias=False)
        self.query = nn.Linear(NUM_EMBED_DIMS, head_size, bias=False)
        self.value = nn.Linear(NUM_EMBED_DIMS, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)
        # Compute affinities between keys and queries
        weights = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        # Ensure does not interract with the past
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        # Normalize the weights to sum 1
        weights = F.softmax(weights, dim=-1)
        # Apply to the values v
        return weights @ v  # (B, T, C)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Concatenate outputs from all heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out  # Output shape: (B, T, num_heads * head_size) = (B, T, NUM_EMBED_DIMS)
