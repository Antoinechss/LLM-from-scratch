import torch.nn as nn

# ------- Feed Forward Layer ---------
# Takes the attention-weighted representations and further processes them.
# Helps the model learn higher-level features and patterns.


class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity"""

    def __init__(self, num_embedings):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_embedings, 4*num_embedings),  # Expand
            nn.ReLU(),
            nn.Linear(4*num_embedings, num_embedings),  # Project back
        )

    def forward(self, x):
        return self.network(x)
