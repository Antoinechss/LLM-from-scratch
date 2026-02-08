# A character level based language model

import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import SEED, NUM_EMBED_DIMS, BLOCK_SIZE, device  
from dataset import VOCAB_SIZE

torch.manual_seed(SEED)


class BigramLanguageModel(nn.Module):
    """
    This class implements a simple character-level language model
    nn.Embedding(vocab_size, vocab_size) : creates an embedding table where each token (character)
    in the vocabulary is mapped to a vector of length vocab_size.
    """

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, NUM_EMBED_DIMS)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, NUM_EMBED_DIMS)
        self.lm_head = nn.linear(NUM_EMBED_DIMS, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        """
        Inputs:
            idx: Tensor of token indices (shape: batch_size × block_size)
            targets: Tensor of target token indices (same shape)

            self.token_embedding_table(idx): Looks up the embedding for each token in idx.

        Output shape: (batch_size, block_size, vocab_size)
            Each "channel" (the last dimension) represents the scores (logits) for each possible next character in the vocabulary.
            For each character in the input, the model predicts which character comes next (bigram modeling).
            F.cross_entropy(logits, targets): Computes the loss between the predicted logits and the actual next character.
        """
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # (B, T, NUM_EMBED_DIMS)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        logits = self.lm_head(token_emb + pos_emb)  # (B, T, VOCAB_SIZE)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # Reshape for cross_entropy format expectations
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
            logits, loss = self(
                idx
            )  # Get predictions for next characater : forward function
            logits = logits[:, -1, :]  # Focus on last time step (most recent character)
            probs = F.softmax(
                logits, dim=-1
            )  # Get probabilities with softmax over all possible next characters
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # (B, T+1) Append to running sequence
        return idx  # Returns extended sequence with newly generated tokens
