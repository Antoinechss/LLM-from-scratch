import torch.nn as nn


class FeedFoward(nn.Module):
    """Simple linear layer followed by non-linearity"""

    def __init__(self, num_embedings):
        super().__init()
        self.network = nn.Sequential(
            nn.Linear(num_embedings, num_embedings),
            nn.Relu(),
        )

    def forward(self, x):
        return self.network(x)
