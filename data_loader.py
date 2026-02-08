# Prepare the batch of input that is sent into transformer

import torch
from dataset import train_text, val_text
from tokenizer import encode 

torch.manual_seed(1337)

# Encode the data
train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids = torch.tensor(encode(val_text), dtype=torch.long)

# Configs
BATCH_SIZE = 4  # How many independent sequences will be processed in parallel
BLOCK_SIZE = 8  # Max context length for prediction


def get_batch(split):
    """Generate a small batch of data of inputs x and outputs y"""
    data = train_ids if split == 'train' else val_ids
    # Generate random starting points for the blocks
    ix = torch.randint(len(data)-BLOCK_SIZE, (BATCH_SIZE,)) 
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


x, y = get_batch('train')
print(x)
print(y)