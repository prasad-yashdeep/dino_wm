"""
EMA Vector Quantizer with Code Reset

Based on:
- "Robust Training of Vector Quantized Bottleneck Models" (https://arxiv.org/pdf/2005.08520)
- Karpathy's deep-vector-quantization implementation

Key improvements over standard VQ-VAE:
1. Exponential Moving Average (EMA) updates for codebook (more stable than gradient descent)
2. Code reset/revival mechanism to prevent dead codes
3. Laplace smoothing for numerical stability
4. Optional codebook loss or EMA-only mode

Usage:
    quantizer = EMAVectorQuantizer(
        n_embed=128,
        embedding_dim=384,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        reset_unused_codes=True
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embed,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        reset_unused_codes=True,
        reset_threshold=1.0,  # Reset if code usage < threshold after N steps
        reset_interval=100,   # Check for dead codes every N forward passes
        kld_scale=10.0,  # #editedbyB: Added KL divergence loss scaling factor
    ):
        """
        EMA Vector Quantizer with automatic code reset.

        Args:
            n_embed: Number of codebook entries
            embedding_dim: Dimension of each embedding
            commitment_cost: Weight for commitment loss (encourages encoder to commit to codes)
            decay: EMA decay rate (higher = slower updates, typical: 0.99)
            epsilon: Small constant for Laplace smoothing
            reset_unused_codes: Whether to reset codes with low usage
            reset_threshold: Usage threshold for resetting (in counts, not percentage)
            reset_interval: How often to check for dead codes (in forward passes)
            kld_scale: Scale factor for full commitment loss (default: 10.0) #editedbyB
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.reset_unused_codes = reset_unused_codes
        self.reset_threshold = reset_threshold
        self.reset_interval = reset_interval
        self.kld_scale = kld_scale  # #editedbyB: Store KL divergence scaling factor

        # Codebook embeddings (learnable, but updated via EMA)
        self.embedding = nn.Embedding(n_embed, embedding_dim)

        # Initialize to match encoder output distribution
        # Conv2D encoder uses BatchNorm + Tanh → outputs follow truncated normal in [-1, +1]
        # Tanh output characteristics:
        #   - Range: [-1, +1] (bounded)
        #   - Mean: ≈ 0 (symmetric)
        #   - Std: ≈ 0.5-0.7 (most values near 0, tails at ±1)
        # We use uniform [-1, 1] for simplicity - matches Tanh range exactly
        self.embedding.weight.data.uniform_(0, 2)

        # EMA cluster size (N_i in paper) - how many embeddings assigned to each code
        self.register_buffer("ema_cluster_size", torch.zeros(n_embed))
        # EMA sum of embeddings (m_i in paper) - sum of all embeddings assigned to each code
        self.register_buffer("ema_embed_avg", self.embedding.weight.data.clone())

        # Track initialization and step count for code reset
        self.register_buffer('_initialized_from_data', torch.tensor(False))
        self.register_buffer('_step_count', torch.tensor(0))

        # Track last reset step for each code (for monitoring)
        self.register_buffer('_code_reset_steps', torch.zeros(n_embed, dtype=torch.long))

    def forward(self, z_e):
        """
        Vector quantization with EMA updates.

        Args:
            z_e: Input embeddings of shape (B, T, H, W, C) or (B, C, H, W)

        Returns:
            z_q_st: Quantized embeddings (with straight-through gradients)
            loss: Commitment loss (codebook updates via EMA, not gradients)
            encoding_indices: Index of nearest codebook entry for each position
        """
        # Flatten input
        input_shape = z_e.shape
        C = z_e.shape[-1]
        z_e_flat = z_e.reshape(-1, C)

        # Compute distances to codebook entries
        # distances[i, j] = ||z_e[i] - codebook[j]||^2
        distances = torch.sum((z_e_flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2, dim=2)

        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_embed).type(z_e.dtype)

        # Quantize by selecting nearest codebook entry
        z_q_flat = torch.matmul(encodings, self.embedding.weight)
        z_q = z_q_flat.view(input_shape)

        # Update EMA statistics (only during training)
        if self.training:
            self._update_ema(z_e_flat, encodings)

            # Periodically check for dead codes and reset them
            self._step_count += 1
            if self.reset_unused_codes and self._step_count % self.reset_interval == 0:
                self._reset_dead_codes(z_e_flat)

        # Commitment loss: encourages encoder outputs to be close to codebook entries #editedbyB
        # Full commitment loss (Karpathy style): bidirectional matching #editedbyB
        # 1. Encoder commitment: encoder outputs should match codebook entries #editedbyB
        # 2. Codebook regularization: codebook should respond to encoder outputs #editedbyB
        commitment_loss = (
            self.commitment_cost * F.mse_loss(z_q.detach(), z_e) +  # Encoder matches codebook #editedbyB
            F.mse_loss(z_q, z_e.detach())  # Codebook matches encoder #editedbyB
        )
        commitment_loss = commitment_loss * self.kld_scale  # Scale by KLD factor #editedbyB

        # Straight-through estimator: gradient flows through z_e
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, commitment_loss, encoding_indices.view(input_shape[:-1])

    def _update_ema(self, z_e_flat, encodings):
        """
        Update codebook using Exponential Moving Average.

        Args:
            z_e_flat: Flattened input embeddings (N, embedding_dim)
            encodings: One-hot encodings (N, n_embed)
        """
        # Update cluster sizes (how many embeddings assigned to each code)
        # N_i = decay * N_i + (1 - decay) * sum(encodings[:, i])
        cluster_size = torch.sum(encodings, dim=0)
        self.ema_cluster_size.data.mul_(self.decay).add_(
            cluster_size, alpha=1 - self.decay
        )

        # Update sum of embeddings assigned to each code
        # m_i = decay * m_i + (1 - decay) * sum(z_e where assigned to i)
        embed_sum = torch.matmul(encodings.t(), z_e_flat)
        self.ema_embed_avg.data.mul_(self.decay).add_(
            embed_sum, alpha=1 - self.decay
        )

        # Update codebook: codebook[i] = m_i / (N_i + epsilon)
        # Laplace smoothing (epsilon) prevents division by zero
        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.n_embed * self.epsilon) * n
        )
        embed_normalized = self.ema_embed_avg / cluster_size_smoothed.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    def _reset_dead_codes(self, z_e_flat):
        """
        Reset codes that have low usage (dead codes).

        Strategy: Replace dead codes with random encoder outputs that are
        far from existing codebook entries (maximize coverage).

        Args:
            z_e_flat: Current batch of encoder outputs (N, embedding_dim)
        """
        # Find codes with usage below threshold
        dead_mask = self.ema_cluster_size < self.reset_threshold
        n_dead = dead_mask.sum().item()

        if n_dead > 0:
            # Get random encoder outputs from current batch
            random_indices = torch.randint(0, z_e_flat.shape[0], (n_dead,), device=z_e_flat.device)
            random_embeddings = z_e_flat[random_indices]

            # Add small noise to avoid exact duplicates
            random_embeddings = random_embeddings + torch.randn_like(random_embeddings) * 0.01

            # Reset dead codes
            dead_indices = torch.where(dead_mask)[0]
            self.embedding.weight.data[dead_indices] = random_embeddings

            # Reset EMA statistics for these codes
            self.ema_cluster_size.data[dead_indices] = self.epsilon
            self.ema_embed_avg.data[dead_indices] = random_embeddings

            # Track when codes were reset
            self._code_reset_steps[dead_indices] = self._step_count

            # Log reset event
            print(f"  🔄 Reset {n_dead} dead codes at step {self._step_count.item()}")

    def compute_codebook_utilization(self, encoding_indices):
        """
        Compute what percentage of the codebook is actively being used.

        Args:
            encoding_indices: Indices from quantization step

        Returns:
            utilization: Percentage of codebook entries used (0.0 to 1.0)
            n_unique: Number of unique codes used
        """
        unique_codes = torch.unique(encoding_indices)
        n_unique = len(unique_codes)
        utilization = n_unique / self.n_embed
        return utilization, n_unique

    def get_ema_statistics(self):
        """
        Get EMA statistics for monitoring.

        Returns:
            dict with:
                - cluster_sizes: How many embeddings assigned to each code
                - dead_codes: Indices of codes with zero usage
                - low_usage_codes: Indices of codes with usage below threshold
        """
        cluster_sizes = self.ema_cluster_size.cpu().numpy()
        dead_codes = torch.where(self.ema_cluster_size == 0)[0].cpu().numpy()
        low_usage_codes = torch.where(self.ema_cluster_size < self.reset_threshold)[0].cpu().numpy()

        return {
            "cluster_sizes": cluster_sizes,
            "dead_codes": dead_codes,
            "low_usage_codes": low_usage_codes,
            "mean_usage": cluster_sizes.mean(),
            "std_usage": cluster_sizes.std(),
        }

    def initialize_from_data(self, data_embeddings, use_kmeans=True):
        """
        Initialize codebook from data using k-means.

        Args:
            data_embeddings: Embeddings from training data, shape (N, embedding_dim)
            use_kmeans: If True, use k-means. If False, random sampling.
        """
        if self._initialized_from_data.item():
            print("Warning: Codebook already initialized from data. Skipping.")
            return

        device = self.embedding.weight.device
        data_embeddings = data_embeddings.to(device)

        if use_kmeans:
            print(f"Initializing {self.n_embed} codebook entries using k-means...")

            n_samples = data_embeddings.shape[0]
            if n_samples < self.n_embed:
                raise ValueError(f"Not enough samples ({n_samples}) for {self.n_embed} codes")

            # Randomly sample initial centroids
            indices = torch.randperm(n_samples)[:self.n_embed]
            centroids = data_embeddings[indices].clone()

            # Run k-means for 10 iterations
            for iteration in range(10):
                distances = torch.cdist(data_embeddings, centroids)
                assignments = torch.argmin(distances, dim=1)

                for k in range(self.n_embed):
                    mask = (assignments == k)
                    if mask.sum() > 0:
                        centroids[k] = data_embeddings[mask].mean(dim=0)

            # Set codebook and EMA statistics
            self.embedding.weight.data.copy_(centroids)
            self.ema_embed_avg.copy_(centroids)

            # Initialize cluster sizes based on k-means assignments
            for k in range(self.n_embed):
                mask = (assignments == k)
                self.ema_cluster_size[k] = mask.sum().float()

            print(f"✅ K-means initialization complete")
        else:
            # Random sampling
            indices = torch.randperm(data_embeddings.shape[0])[:self.n_embed]
            sampled_embeddings = data_embeddings[indices]
            self.embedding.weight.data.copy_(sampled_embeddings)
            self.ema_embed_avg.copy_(sampled_embeddings)
            # Initialize with small cluster sizes
            self.ema_cluster_size.fill_(1.0)
            print(f"✅ Random sampling initialization complete")

        self._initialized_from_data.fill_(True)
