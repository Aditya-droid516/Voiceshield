import torch
import torch.nn as nn


class Head(nn.Module):
    """Learned softmax-weighted sum over WavLM layers -> small MLP -> fake logit."""

    def __init__(self, n_layers=13, dim=768, hidden=256, dropout=0.3):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.fc1 = nn.Linear(dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def layer_weights(self):
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x):
        # x: [B, n_layers, dim]
        w = self.layer_weights()
        x = (x * w[None, :, None]).sum(dim=1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)  # logits; higher = more fake
