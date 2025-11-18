"""
Gumbel-Softmax Vector Quantizer

Based on:
- "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2017)
- "Robust Training of Vector Quantized Bottleneck Models" (https://arxiv.org/pdf/2005.08520)

Key difference from standard VQ-VAE:
- Uses DIFFERENTIABLE soft quantization during training (gradients flow!)
- Uses hard quantization during inference (discrete codes)
- Temperature annealing: soft � hard over training

Advantages:
- Gradients flow through quantization operation (no straight-through estimator needed)
- Better codebook learning in early training
- Can combine with EMA updates

Disadvantages:
- Requires temperature schedule tuning
- More complex than standard VQ-VAE

Usage:
    quantizer = GumbelQuantizer(
        n_embed=128,
        embedding_dim=384,
        temperature_start=1.0,
        temperature_end=0.1,
        temperature_decay=0.999
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelQuantizer(nn.Module):
    def __init__(
        self,
        n_embed,
        embedding_dim,
        temperature_start=1.0,
        temperature_end=0.1,
        temperature_decay=0.999,
        hard=False,  # Whether to use hard quantization even during training
        straight_through=False,  # Use straight-through in hard mode
        use_ema=False,  # Whether to use EMA updates for codebook
        ema_decay=0.99,
        ema_epsilon=1e-5,
    ):
        """
        Gumbel-Softmax Vector Quantizer.

        Args:
            n_embed: Number of codebook entries
            embedding_dim: Dimension of each embedding
            temperature_start: Initial temperature (higher = softer, more exploration)
            temperature_end: Final temperature (lower = harder, more exploitation)
            temperature_decay: Temperature decay per step (multiplicative)
            hard: If True, use hard quantization (argmax) even during training
            straight_through: If True, use straight-through estimator with hard quantization
            use_ema: If True, update codebook via EMA instead of gradients
            ema_decay: EMA decay rate (only used if use_ema=True)
            ema_epsilon: Laplace smoothing constant (only used if use_ema=True)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.hard = hard
        self.straight_through = straight_through
        self.use_ema = use_ema

        # Codebook embeddings
        self.embedding = nn.Embedding(n_embed, embedding_dim)

        # Initialize to match encoder output distribution
        # Conv2D encoder uses BatchNorm + Tanh - outputs bounded in [-1, +1]
        # Tanh characteristics: mean ~ 0, std ~ 0.5-0.7, range [-1, +1]
        # We use uniform [-1, 1] for simplicity - matches Tanh range exactly
        self.embedding.weight.data.uniform_(-1, 1)

        # Temperature (annealed during training)
        self.register_buffer("temperature", torch.tensor(temperature_start))

        # EMA buffers (only used if use_ema=True)
        if use_ema:
            self.ema_decay = ema_decay
            self.ema_epsilon = ema_epsilon
            self.register_buffer("ema_cluster_size", torch.zeros(n_embed))
            self.register_buffer("ema_embed_avg", self.embedding.weight.data.clone())

        # Track initialization
        self.register_buffer('_initialized_from_data', torch.tensor(False))

    def forward(self, z_e):
        """
        Gumbel-Softmax quantization.

        Args:
            z_e: Input embeddings of shape (B, T, H, W, C) or (B, C, H, W)

        Returns:
            z_q: Quantized embeddings (soft during training, hard during eval)
            loss: Commitment loss (if not using EMA) or zero
            encoding_indices: Index of selected codebook entry (hard assignment)
        """
        # Flatten input
        input_shape = z_e.shape
        C = z_e.shape[-1]
        z_e_flat = z_e.reshape(-1, C)

        # Compute logits (negative distances or cosine similarities)
        # Using negative squared distance as logits
        # logits[i, j] = -||z_e[i] - codebook[j]||^2
        distances = torch.sum((z_e_flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2, dim=2)
        logits = -distances

        # Hard quantization (for eval or if hard=True)
        encoding_indices = torch.argmax(logits, dim=1)

        if not self.training or self.hard:
            # Hard quantization (discrete)
            encodings = F.one_hot(encoding_indices, self.n_embed).type(z_e.dtype)
            z_q_flat = torch.matmul(encodings, self.embedding.weight)

            if self.straight_through and self.training:
                # Straight-through estimator: gradient flows through z_e
                z_q_flat = z_e_flat + (z_q_flat - z_e_flat).detach()
        else:
            # Soft quantization with Gumbel-Softmax (training mode)
            # Sample from Gumbel distribution
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            logits_with_gumbel = (logits + gumbel_noise) / self.temperature

            # Softmax to get soft assignment probabilities
            soft_weights = F.softmax(logits_with_gumbel, dim=1)

            # Soft quantization: weighted sum of codebook entries
            z_q_flat = torch.matmul(soft_weights, self.embedding.weight)

            # Optionally: straight-through to make it "hard" for downstream
            # but keep soft gradients
            if self.straight_through:
                encodings = F.one_hot(encoding_indices, self.n_embed).type(z_e.dtype)
                z_q_hard = torch.matmul(encodings, self.embedding.weight)
                z_q_flat = z_q_hard + (z_q_flat - z_q_hard).detach()

        z_q = z_q_flat.view(input_shape)

        # Update EMA statistics (if enabled)
        if self.training and self.use_ema:
            encodings = F.one_hot(encoding_indices, self.n_embed).type(z_e.dtype)
            self._update_ema(z_e_flat, encodings)
            loss = torch.tensor(0.0, device=z_e.device)  # No gradient-based loss
        else:
            # Commitment loss (encourages encoder to match codebook)
            # Only used if not using EMA
            commitment_cost = 0.25
            loss = commitment_cost * F.mse_loss(z_e, z_q.detach())

        # Anneal temperature (exponential decay)
        if self.training:
            self.temperature.data = torch.clamp(
                self.temperature * self.temperature_decay,
                min=self.temperature_end
            )

        return z_q, loss, encoding_indices.view(input_shape[:-1])

    def _update_ema(self, z_e_flat, encodings):
        """
        Update codebook using EMA (same as EMAVectorQuantizer).

        Args:
            z_e_flat: Flattened input embeddings (N, embedding_dim)
            encodings: One-hot encodings (N, n_embed)
        """
        cluster_size = torch.sum(encodings, dim=0)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )

        embed_sum = torch.matmul(encodings.t(), z_e_flat)
        self.ema_embed_avg.data.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.ema_epsilon)
            / (n + self.n_embed * self.ema_epsilon) * n
        )
        embed_normalized = self.ema_embed_avg / cluster_size_smoothed.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    def compute_codebook_utilization(self, encoding_indices):
        """Compute codebook utilization."""
        unique_codes = torch.unique(encoding_indices)
        n_unique = len(unique_codes)
        utilization = n_unique / self.n_embed
        return utilization, n_unique

    def get_temperature(self):
        """Get current temperature value."""
        return self.temperature.item()

    def set_temperature(self, temperature):
        """Manually set temperature (useful for testing)."""
        self.temperature.data = torch.tensor(temperature)

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

            indices = torch.randperm(n_samples)[:self.n_embed]
            centroids = data_embeddings[indices].clone()

            for iteration in range(10):
                distances = torch.cdist(data_embeddings, centroids)
                assignments = torch.argmin(distances, dim=1)

                for k in range(self.n_embed):
                    mask = (assignments == k)
                    if mask.sum() > 0:
                        centroids[k] = data_embeddings[mask].mean(dim=0)

            self.embedding.weight.data.copy_(centroids)

            if self.use_ema:
                self.ema_embed_avg.copy_(centroids)
                for k in range(self.n_embed):
                    mask = (assignments == k)
                    self.ema_cluster_size[k] = mask.sum().float()

            print(f" K-means initialization complete")
        else:
            indices = torch.randperm(data_embeddings.shape[0])[:self.n_embed]
            sampled_embeddings = data_embeddings[indices]
            self.embedding.weight.data.copy_(sampled_embeddings)

            if self.use_ema:
                self.ema_embed_avg.copy_(sampled_embeddings)
                self.ema_cluster_size.fill_(1.0)

            print(f" Random sampling initialization complete")

        self._initialized_from_data.fill_(True)
