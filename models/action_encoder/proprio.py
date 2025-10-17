import torch
import torch.nn as nn


class ProprioEncoder(nn.Module):
    def __init__(self, in_chans, emb_dim):
        """
        Simple MLP encoder for proprioceptive/action data.

        Args:
            in_chans (int): Input dimension (e.g., action dimension)
            emb_dim (int): Output embedding dimension
        """
        super().__init__()
        self.name = "proprio"
        self.emb_dim = emb_dim
        self.in_chans = in_chans

        # Simple MLP with one hidden layer
        hidden_dim = max(emb_dim * 2, 64)
        self.encoder = nn.Sequential(
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        """
        Forward pass through the proprioceptive encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, in_chans)

        Returns:
            torch.Tensor: Encoded features of shape (B, T, emb_dim)
        """
        # Flatten batch and time dimensions
        b, t = x.shape[:2]
        x_flat = x.reshape(b * t, -1)

        # Encode
        encoded = self.encoder(x_flat)

        # Reshape back
        encoded = encoded.reshape(b, t, -1)

        return encoded


class DummyActionEncoder(nn.Module):
    def __init__(self, in_chans, emb_dim):
        """
        Dummy action encoder that just passes through the input.
        Used when actions are not encoded.
        """
        super().__init__()
        self.name = "dummy"
        self.emb_dim = in_chans
        self.in_chans = in_chans

    def forward(self, x):
        return x
