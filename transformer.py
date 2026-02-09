import torch.nn as nn
from head import MultiHeadAttention
from layers import FeedForward


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, num_embeddings, num_heads, dropout=0.1):
        super().__init__()
        head_size = num_embeddings // num_heads
        self.self_att_heads = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(num_embeddings)
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.self_att_heads(self.ln1(x)))
        x = x + self.dropout(self.feed_forward(self.ln2(x)))
        return x
