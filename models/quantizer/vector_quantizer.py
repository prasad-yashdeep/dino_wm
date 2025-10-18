import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(n_embed, embedding_dim)
        # Initialize codebook with reasonable scale to match typical encoder outputs
        # Encoder outputs typically have std ~ 0.5-1.0, so initialize uniformly in [-1, 1]
        self.embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, z_e):
        """Vector quantization step.

        Args:
            z_e (torch.Tensor): Input tensor of shape (B, T, H, W, C) where
                                B = batch size, T = time steps, H = height,
                                W = width, C = embedding dimension.

        Returns:
            z_q (torch.Tensor): Quantized tensor of the same shape as z_e.
            loss (torch.Tensor): Loss value for the quantization step.
            encoding_indices (torch.Tensor): Indices of the nearest embeddings.
        """
        # Flatten input
        C = z_e.shape[-1]
        z_e_flat = z_e.view(-1, C)
        distances = torch.sum((z_e_flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2, dim=2)

        # Nearest embedding index
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_embed).type(z_e.dtype)

        # Quantize
        z_q = torch.matmul(encodings, self.embedding.weight).view(z_e.shape)

        # Losses
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        loss = codebook_loss + commitment_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, loss, encoding_indices.view(z_e.shape[:-1])

    def compute_codebook_utilization(self, encoding_indices):
        """Compute what percentage of the codebook is actually being used.

        Args:
            encoding_indices (torch.Tensor): Indices from quantization step

        Returns:
            utilization (float): Percentage of codebook entries used (0.0 to 1.0)
            n_unique (int): Number of unique codes used
        """
        unique_codes = torch.unique(encoding_indices)
        n_unique = len(unique_codes)
        utilization = n_unique / self.n_embed

        return utilization, n_unique
