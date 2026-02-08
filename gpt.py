from language_model import BigramLanguageModel
from train import train
from tokenizer import decode
import torch
from configs import MAX_TOKEN_GENERATION, device
from dataset import VOCAB_SIZE

# Instanciate model:
model = BigramLanguageModel(VOCAB_SIZE)
# Train model:
train(model)
# Test text generation:
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=MAX_TOKEN_GENERATION)[0].tolist()))


