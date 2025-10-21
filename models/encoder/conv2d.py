import torch
import torch.nn as nn
from einops import rearrange


class Conv2DEncoder(nn.Module):
    def __init__(self, input_channels=3, emb_dim=16, image_size=224, depth=4, hidden_dim=64):
        """
        2D Convolutional Encoder that preserves spatial dimensions.
        
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB)
            emb_dim (int): Number of output channels in latent space
            image_size (int): Input image size (assumes square images)
            depth (int): Number of downsampling layers
            hidden_dim (int): Base number of hidden channels (doubles each layer)
        """
        super().__init__()
        self.name = "conv2d"
        self.input_channels = input_channels
        self.emb_dim = emb_dim
        self.image_size = image_size
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.latent_ndim = 3  # 3D latent representation (C', H', W')
        
        # Calculate output spatial dimensions
        # Each conv layer with stride 2 reduces spatial dimensions by half
        self.output_h = image_size // (2 ** depth)
        self.output_w = image_size // (2 ** depth)
        self.patch_size = 2 ** depth  # Effective patch size due to downsampling
        
        # Build encoder layers dynamically
        layers = []
        in_channels = input_channels
        
        for i in range(depth):
            out_channels = hidden_dim * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # Final projection to latent space
        # BatchNorm before Tanh for better normalization and training stability
        # Tanh bounds output to [-1, +1] to prevent extreme values
        layers.extend([
            nn.Conv2d(in_channels, emb_dim, kernel_size=1),
            nn.BatchNorm2d(emb_dim),  # Added for better normalization before activation
            # nn.Tanh()  # Critical: prevents runaway activations (max was 130+) #edited by B
        ])
        
        self.encoder = nn.Sequential(*layers)
        
    def get_init_args(self):
        """Return the arguments needed to reconstruct this encoder instance."""
        return {
            'input_channels': self.input_channels,
            'emb_dim': self.emb_dim,
            'image_size': self.image_size,
            'depth': self.depth,
            'hidden_dim': self.hidden_dim
        }

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded features of shape (B, C', H', W') - keeping spatial dimensions
        """
        # Apply encoder and return spatial features directly
        encoded = self.encoder(x)  # Shape: (B, C', H', W')
        encoded = rearrange(encoded, 'b c h w -> b h w c')  # Ensure correct shape
        return encoded
