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
        # Initialize codebook with reasonable scale to match typical encoder outputs #edited by B
        # Encoder outputs (with Tanh) have mean~0, std~0.5, so use normal distribution #edited by B
        # This provides better coverage than uniform initialization #edited by B
        # Can be replaced with k-means initialization using initialize_from_data() #edited by B
        # self.embedding.weight.data.normal_(0, 1)

        # Register _initialized_from_data as a buffer so it persists across checkpoint save/load #edited by B
        # This prevents k-means from re-running in Stage 3 when loading Stage 2 checkpoint #edited by B
        self.register_buffer('_initialized_from_data', torch.tensor(False))  #edited by B

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

    def initialize_from_data(self, data_embeddings, use_kmeans=True):  #edited by B
        """Initialize codebook from actual data embeddings using k-means clustering. #edited by B

        This is NOT cheating - it only uses training data that the encoder already saw. #edited by B
        Similar to how VQ-VAE paper initializes codebooks. #edited by B

        Args: #edited by B
            data_embeddings (torch.Tensor): Embeddings from training data, shape (N, embedding_dim) #edited by B
            use_kmeans (bool): If True, use k-means clustering. If False, use random sampling. #edited by B
        """ #edited by B
        if self._initialized_from_data.item():  #edited by B - check tensor value
            print("Warning: Codebook already initialized from data. Skipping.")  #edited by B
            return  #edited by B

        device = self.embedding.weight.device  #edited by B
        data_embeddings = data_embeddings.to(device)  #edited by B

        if use_kmeans:  #edited by B
            # Simple k-means implementation (no sklearn dependency) #edited by B
            print(f"Initializing {self.n_embed} codebook entries using k-means...")  #edited by B

            # Randomly sample initial centroids from data #edited by B
            n_samples = data_embeddings.shape[0]  #edited by B
            if n_samples < self.n_embed:  #edited by B
                raise ValueError(f"Not enough samples ({n_samples}) for {self.n_embed} codes")  #edited by B

            indices = torch.randperm(n_samples)[:self.n_embed]  #edited by B
            centroids = data_embeddings[indices].clone()  #edited by B

            # Run k-means for a few iterations #edited by B
            for iteration in range(10):  #edited by B
                # Assign points to nearest centroid #edited by B
                distances = torch.cdist(data_embeddings, centroids)  #edited by B
                assignments = torch.argmin(distances, dim=1)  #edited by B

                # Update centroids #edited by B
                for k in range(self.n_embed):  #edited by B
                    mask = (assignments == k)  #edited by B
                    if mask.sum() > 0:  #edited by B
                        centroids[k] = data_embeddings[mask].mean(dim=0)  #edited by B
                    # If no points assigned, keep current centroid #edited by B

            self.embedding.weight.data.copy_(centroids)  #edited by B
            print(f"✅ K-means initialization complete")  #edited by B
        else:  #edited by B
            # Random sampling from data #edited by B
            indices = torch.randperm(data_embeddings.shape[0])[:self.n_embed]  #edited by B
            self.embedding.weight.data.copy_(data_embeddings[indices])  #edited by B
            print(f"✅ Random sampling initialization complete")  #edited by B

        self._initialized_from_data.fill_(True)  #edited by B - set tensor value to True
