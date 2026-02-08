import torch
import torch.nn as nn
from configs import SEED, BATCH_SIZE, BLOCK_SIZE, CHANNELS, HEAD_SIZE
import torch.nn.functional as F

torch.manual_seed(SEED)

B, T, C = BATCH_SIZE, BLOCK_SIZE, CHANNELS
x = torch.randn(B, T, C)

# --------------
# Self Attention = How to get information from the past in a data dependent way ?
# Goal = to have weights as the dot product of keys and queries
# --------------
key = nn.Linear(C, HEAD_SIZE, bias=False)
query = nn.Linear(C, HEAD_SIZE, bias=False)
value = nn.Linear(C, HEAD_SIZE, bias=False)
k = key(x)    # (B, T, HEAD_SIZE)
q = query(x)  # (B, T, HEAD_SIZE)
v = value(x)  # v reveals characteristics about x for this particular head of attention
# Weights matrix reflect the affinity between the key and the query
# Tells how much to take from the past info to predict next tokens
weights = q @ k.transpose(-2, -1)  # (B, T, HEAD_SIZE) x (B, HEAD_SIZE, T) = (B, T, T)

# Simple average computation of all past tokens and current token
tril = torch.tril(torch.ones(T, T))
weights = weights.masked_fill(tril == 0, float("-inf"))
weights = F.softmax(weights, dim=-1)
out = weights @ v
print(weights[0])
print(out[0])


