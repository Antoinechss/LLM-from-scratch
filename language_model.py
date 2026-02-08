# A character level based language model

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset import vocab_size
from data_loader import get_batch

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    """
    This class implements a simple character-level language model
    nn.Embedding(vocab_size, vocab_size) : creates an embedding table where each token (character)
    in the vocabulary is mapped to a vector of length vocab_size.
    Forward Pass
        Inputs:
        idx: Tensor of token indices (shape: batch_size × block_size)
        targets: Tensor of target token indices (same shape)

        self.token_embedding_table(idx): Looks up the embedding for each token in idx.

        Output shape: (batch_size, block_size, vocab_size)
        Each "channel" (the last dimension) represents the scores (logits) for each possible next character in the vocabulary.
        For each character in the input, the model predicts which character comes next (bigram modeling).
        F.cross_entropy(logits, targets): Computes the loss between the predicted logits and the actual next character.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        # Reshape for cross_entropy format expectations
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Creates a new text sequence by sampling characters from the model one at a time
        Input idx (B,T) indixes of current context
        Output idx (B, T+1) new seq with additional generated token
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # Get predictions for next characater 
            logits = logits[:, -1, :]  # Focus on last time step (most recent character)
            probs = F.softmax(logits, dim=-1)  # Get probabilities with softmax over all possible next characters
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # (B, T+1) Append to running sequence
        return idx


model = BigramLanguageModel(vocab_size)
x, y = get_batch("train")
logits, loss = model(x, y)
print(loss)
