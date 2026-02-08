# Prepare the batch of input that is sent into transformer

import torch
from dataset import train_text, val_text
from tokenizer import encode
from configs import BATCH_SIZE, BLOCK_SIZE, SEED, device

torch.manual_seed(SEED)

# Encode the data
train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids = torch.tensor(encode(val_text), dtype=torch.long)


def get_batch(split):
    """Generate a small batch of data of inputs x and outputs y"""
    data = train_ids if split == "train" else val_ids
    # Generate random starting points for the blocks
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
