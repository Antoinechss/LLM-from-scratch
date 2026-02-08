import torch.nn as nn
from head import MultiHeadAttention
from layers import FeedForward


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, num_embeddings, num_heads):
        super().__init__()
        head_size = num_embeddings // num_heads
        self.self_att_heads = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(num_embeddings)

    def forward(self, x):
        x = self.self_att_heads(x)
        x = self.feed_forward(x)
        return x
