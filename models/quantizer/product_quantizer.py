import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductQuantizer(nn.Module):
    """
    Product Quantization: Split embedding into groups and quantize each separately.

    Instead of:
        64-dim → 1 codebook of 128 codes

    Use:
        64-dim → 4 codebooks of 16 codes each (4 groups of 16-dim)
        Effective vocabulary: 16^4 = 65,536 unique combinations!

    Benefits:
    - Exponentially larger effective vocabulary
    - Same memory footprint (4 × 16 × 16 = 1024 params vs 128 × 64 = 8192)
    - Better gradient flow (smaller quantization groups)
    """

    def __init__(self, n_embed_per_group=16, embedding_dim=64, n_groups=4, commitment_cost=0.25):
        super().__init__()
        self.n_embed_per_group = n_embed_per_group
        self.embedding_dim = embedding_dim
        self.n_groups = n_groups
        self.commitment_cost = commitment_cost

        # Check that embedding_dim is divisible by n_groups
        assert embedding_dim % n_groups == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by n_groups ({n_groups})"

        self.group_dim = embedding_dim // n_groups

        # Create separate codebooks for each group
        self.codebooks = nn.ModuleList([
            nn.Embedding(n_embed_per_group, self.group_dim)
            for _ in range(n_groups)
        ])

        # Initialize each codebook
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1 / n_embed_per_group, 1 / n_embed_per_group)

        # Effective vocabulary size
        self.effective_vocab_size = n_embed_per_group ** n_groups
        print(f"Product Quantizer initialized:")
        print(f"  - Groups: {n_groups}")
        print(f"  - Codes per group: {n_embed_per_group}")
        print(f"  - Group dimension: {self.group_dim}")
        print(f"  - Effective vocabulary: {self.effective_vocab_size:,} codes")
        print(f"  - Memory: {n_groups * n_embed_per_group * self.group_dim:,} parameters")

    def forward(self, z_e):
        """
        Args:
            z_e: Input tensor of shape (B, T, H, W, C)

        Returns:
            z_q_st: Quantized tensor (same shape as z_e) with straight-through gradients
            loss: Quantization loss (commitment + codebook)
            encoding_indices: Indices tensor of shape (B, T, H, W, n_groups)
        """
        original_shape = z_e.shape
        C = z_e.shape[-1]

        # Flatten spatial dimensions
        z_e_flat = z_e.view(-1, C)  # (N, C) where N = B*T*H*W

        # Split into groups
        z_e_groups = z_e_flat.chunk(self.n_groups, dim=-1)  # List of (N, group_dim) tensors

        # Quantize each group separately
        z_q_groups = []
        encoding_indices_list = []
        total_loss = 0.0

        for i, (z_e_group, codebook) in enumerate(zip(z_e_groups, self.codebooks)):
            # Compute distances to codebook
            distances = torch.sum(
                (z_e_group.unsqueeze(1) - codebook.weight.unsqueeze(0)) ** 2,
                dim=2
            )  # (N, n_embed_per_group)

            # Get nearest code indices
            indices = torch.argmin(distances, dim=1)  # (N,)
            encoding_indices_list.append(indices)

            # Get quantized values
            encodings = F.one_hot(indices, self.n_embed_per_group).type(z_e.dtype)
            z_q_group = torch.matmul(encodings, codebook.weight)  # (N, group_dim)

            # Compute losses for this group
            commitment_loss = self.commitment_cost * F.mse_loss(z_e_group, z_q_group.detach())
            codebook_loss = F.mse_loss(z_e_group.detach(), z_q_group)
            total_loss += (codebook_loss + commitment_loss)

            # Straight-through estimator
            z_q_group_st = z_e_group + (z_q_group - z_e_group).detach()
            z_q_groups.append(z_q_group_st)

        # Concatenate quantized groups
        z_q_flat = torch.cat(z_q_groups, dim=-1)  # (N, C)
        z_q_st = z_q_flat.view(original_shape)

        # Stack indices: (N, n_groups) -> (B, T, H, W, n_groups)
        encoding_indices = torch.stack(encoding_indices_list, dim=-1)
        encoding_indices = encoding_indices.view(original_shape[:-1] + (self.n_groups,))

        # Average loss across groups
        loss = total_loss / self.n_groups

        return z_q_st, loss, encoding_indices

    def decode_indices(self, indices):
        """
        Decode indices back to embeddings.

        Args:
            indices: Tensor of shape (..., n_groups) with code indices

        Returns:
            embeddings: Tensor of shape (..., embedding_dim)
        """
        original_shape = indices.shape[:-1]
        indices_flat = indices.view(-1, self.n_groups)  # (N, n_groups)

        # Decode each group
        decoded_groups = []
        for i, codebook in enumerate(self.codebooks):
            group_indices = indices_flat[:, i]  # (N,)
            decoded = codebook(group_indices)  # (N, group_dim)
            decoded_groups.append(decoded)

        # Concatenate
        decoded_flat = torch.cat(decoded_groups, dim=-1)  # (N, embedding_dim)
        decoded = decoded_flat.view(original_shape + (self.embedding_dim,))

        return decoded
