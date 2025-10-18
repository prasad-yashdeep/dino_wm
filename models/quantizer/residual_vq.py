import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ): Multi-layer quantization with residuals.

    Layer 1: Quantize main signal (coarse representation)
    Layer 2: Quantize residual from layer 1 (fine details)
    Layer 3+: Further refinement

    Effective vocabulary = n_embed_layer1 × n_embed_layer2 × ...

    Example: 3 layers of 32 codes each = 32^3 = 32,768 effective codes!
    """

    def __init__(self, n_embed=32, embedding_dim=64, n_layers=3, commitment_cost=0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.commitment_cost = commitment_cost

        # Create codebooks for each layer
        self.codebooks = nn.ModuleList([
            nn.Embedding(n_embed, embedding_dim)
            for _ in range(n_layers)
        ])

        # Initialize codebooks
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1 / n_embed, 1 / n_embed)

        self.effective_vocab_size = n_embed ** n_layers
        print(f"Residual VQ initialized:")
        print(f"  - Layers: {n_layers}")
        print(f"  - Codes per layer: {n_embed}")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - Effective vocabulary: {self.effective_vocab_size:,} codes")

    def forward(self, z_e):
        """
        Args:
            z_e: Input tensor of shape (B, T, H, W, C)

        Returns:
            z_q_st: Quantized tensor with straight-through gradients
            loss: Total quantization loss
            encoding_indices: Indices for all layers (B, T, H, W, n_layers)
        """
        original_shape = z_e.shape
        C = z_e.shape[-1]

        # Flatten
        z_e_flat = z_e.view(-1, C)  # (N, C)

        # Residual quantization
        residual = z_e_flat
        quantized_sum = torch.zeros_like(z_e_flat)
        encoding_indices_list = []
        total_loss = 0.0

        for layer_idx, codebook in enumerate(self.codebooks):
            # Compute distances
            distances = torch.sum(
                (residual.unsqueeze(1) - codebook.weight.unsqueeze(0)) ** 2,
                dim=2
            )

            # Get nearest code
            indices = torch.argmin(distances, dim=1)
            encoding_indices_list.append(indices)

            # Quantize
            encodings = F.one_hot(indices, self.n_embed).type(z_e.dtype)
            z_q = torch.matmul(encodings, codebook.weight)

            # Accumulate quantized values
            quantized_sum = quantized_sum + z_q

            # Compute loss for this layer
            commitment_loss = self.commitment_cost * F.mse_loss(residual, z_q.detach())
            codebook_loss = F.mse_loss(residual.detach(), z_q)
            total_loss += (codebook_loss + commitment_loss)

            # Update residual for next layer (straight-through gradient)
            residual = residual - z_q.detach()

        # Straight-through estimator on final sum
        z_q_st = z_e_flat + (quantized_sum - z_e_flat).detach()
        z_q_st = z_q_st.view(original_shape)

        # Stack indices
        encoding_indices = torch.stack(encoding_indices_list, dim=-1)
        encoding_indices = encoding_indices.view(original_shape[:-1] + (self.n_layers,))

        # Average loss
        loss = total_loss / self.n_layers

        return z_q_st, loss, encoding_indices

    def decode_indices(self, indices):
        """
        Decode multi-layer indices back to embeddings.

        Args:
            indices: Tensor of shape (..., n_layers)

        Returns:
            embeddings: Tensor of shape (..., embedding_dim)
        """
        original_shape = indices.shape[:-1]
        indices_flat = indices.view(-1, self.n_layers)

        # Sum quantized values from all layers
        decoded = torch.zeros(indices_flat.shape[0], self.embedding_dim, device=indices.device)
        for layer_idx, codebook in enumerate(self.codebooks):
            layer_indices = indices_flat[:, layer_idx]
            decoded += codebook(layer_indices)

        decoded = decoded.view(original_shape + (self.embedding_dim,))
        return decoded
