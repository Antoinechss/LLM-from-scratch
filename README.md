# LLM from Scratch

Building a Large Language Model (GPT-like architecture) from scratch in Python and training it on the Tiny Shakespeare dataset.

## Project Overview

This project implements a character-level transformer language model inspired by GPT architecture. The model learns to generate text by training on Shakespeare's complete works.

## Architecture

- **Token & Position Embeddings**: Convert characters to learnable vectors
- **Multi-Head Self-Attention**: Parallel attention mechanisms for capturing patterns
- **Feed-Forward Networks**: Non-linear transformations for feature learning
- **Layer Normalization**: Stabilize training and improve convergence
- **Transformer Blocks**: Stacked attention + feed-forward layers with residual connections

## Project Structure

```
LLM-from-scratch/
├── configs.py          # Hyperparameters and configuration
├── dataset.py          # Dataset loading and preprocessing
├── tokenizer.py        # Character-level tokenization (encode/decode)
├── data_loader.py      # Batch generation
├── head.py             # Self-attention heads
├── layers.py           # Feed-forward networks
├── transformer.py      # Transformer block
├── language_model.py   # Main model architecture
├── train.py            # Training loop and evaluation
├── gpt.py              # Main script to run training
└── README.md
```

## Installation

```bash
# Clone the repository
git clone [your-repo-url](https://github.com/Antoinechss/LLM-from-scratch.git)
cd LLM-from-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch datasets matplotlib
```

## Usage

### Train the Model

```bash
python gpt.py
```

### Run

```bash
python gpt.py
```

File will train the model and print a couple of lines of Shakespeare Style text generation 

## Configuration

Edit `configs.py` to adjust hyperparameters:

```python
BATCH_SIZE = 64          # Number of sequences processed in parallel
BLOCK_SIZE = 256         # Context length
NUM_EMBED_DIMS = 384     # Embedding dimension
NUM_HEADS = 6            # Number of attention heads
NUM_BLOCKS = 6           # Number of transformer blocks
learning_rate = 3e-4     # Learning rate
MAX_ITERS = 5000         # Training iterations
```

## Model Details

- **Dataset**: Tiny Shakespeare (1MB of text)
- **Tokenization**: Character-level (65 unique characters)
- **Context Window**: 256 characters
- **Architecture**: 6-layer transformer with 6 attention heads
- **Parameters**: 10M (depending on configuration)

## Training Progress

The model tracks training and validation losses. A plot is automatically generated and saved as `training_progress.png`.

## Generated Text Samples

### Result 

```
[Generated text will be added here after training completes]
```


## Features

- Multi-head self-attention mechanism
- Positional embeddings
- Layer normalization
- Residual connections
- Dropout regularization
- Training/validation split
- Loss plotting and monitoring
- Text generation

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) 
