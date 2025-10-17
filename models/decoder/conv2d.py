import torch
import torch.nn as nn
from einops import rearrange


class Conv2DDecoder(nn.Module):
    def __init__(self, emb_dim=16, output_channels=3, depth=4, hidden_dim=64):
        """
        2D Convolutional Decoder that reconstructs images from encoded features.
        Mirrors the encoder operations in reverse.
        
        Args:
            emb_dim (int): Number of input channels from latent space
            output_channels (int): Number of output channels (e.g., 3 for RGB)
            depth (int): Number of upsampling layers (should match encoder depth)
            hidden_dim (int): Base number of hidden channels (matches encoder hidden_dim)
        """
        super().__init__()
        self.name = "conv2d_decoder"
        self.emb_dim = emb_dim
        self.output_channels = output_channels
        self.depth = depth
        self.hidden_dim = hidden_dim
        
        # Build decoder layers dynamically to reverse encoder operations
        layers = []
        
        # First projection: expand from latent space to highest hidden dimension
        final_hidden_channels = hidden_dim * (2 ** (depth - 1))
        layers.extend([
            nn.Conv2d(emb_dim, final_hidden_channels, kernel_size=1),
            nn.BatchNorm2d(final_hidden_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Build upsampling layers in reverse order
        for i in range(depth - 1, 0, -1):
            in_channels = hidden_dim * (2 ** i)
            out_channels = hidden_dim * (2 ** (i - 1))
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        # Final upsampling layer: hidden_dim -> output_channels
        layers.extend([
            nn.ConvTranspose2d(hidden_dim, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Output activation to constrain values to [-1, 1]
        ])
        
        self.decoder = nn.Sequential(*layers)
        
    def get_init_args(self):
        """Return the arguments needed to reconstruct this decoder instance."""
        return {
            'emb_dim': self.emb_dim,
            'output_channels': self.output_channels,
            'depth': self.depth,
            'hidden_dim': self.hidden_dim
        }

    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Encoded features of shape (B, T, H', W', C)
            
        Returns:
            torch.Tensor: Reconstructed image of shape (B, output_channels, H, W)
        """
        # Apply decoder to reconstruct the image
        x = rearrange(x, 'b t h w c -> (b t) c h w')  # Ensure input shape is (B, C, H', W')
        reconstructed = self.decoder(x)  # Shape: (B, output_channels, H, W)
        return reconstructed
