import torch
from data_loader import get_batch
from configs import learning_rate, MAX_ITERS, device, EVAL_INTERVAL
import matplotlib.pyplot as plt


@torch.no_grad()  # All operations inside the function run with gradient tracking disabled for inference
def estimate_loss(model):
    """
    More stable estimate of model performance by averaging losses over multiple batches
    Reduce noise from individual batch variations
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_INTERVAL)
        for k in range(EVAL_INTERVAL):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(model):
    """Training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model = model.to(device)

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    iterations = []

    for iter in range(MAX_ITERS):
        # Periodically evaluate loss on train and val sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            # Store losses for plotting
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            iterations.append(iter)

        # Sample a batch of data
        xbatch, ybatch = get_batch("train")
        # Evaluate the loss
        logits, loss = model(xbatch, ybatch)
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Plot the results
    plot_losses(iterations, train_losses, val_losses)

    return train_losses, val_losses, iterations


def plot_losses(iterations, train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label='Train Loss', marker='o')
    plt.plot(iterations, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()
    print("Plot saved as 'training_progress.png'")
