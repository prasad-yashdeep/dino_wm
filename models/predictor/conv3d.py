import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Conv3DPredictor(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512, depth=4, dropout=0.1):
        """
        3D Convolutional Predictor for Conv2D encoder output.
        
        Designed to work directly with Conv2DEncoder output format.
        Takes input of shape (B, T, C, H, W) and returns same shape.
        
        Args:
            emb_dim (int): Input channel dimension (encoder emb_dim + action_dim)
            hidden_dim (int): Hidden dimension for intermediate layers
            depth (int): Number of 3D conv layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout_rate = dropout
        
        # Build the 3D convolutional layers
        layers = []
        
        # First layer: emb_dim -> hidden_dim
        layers.append(nn.Conv3d(
            emb_dim, hidden_dim, 
            kernel_size=(3, 3, 3), 
            padding=(1, 1, 1),
            bias=False
        ))
        layers.append(nn.BatchNorm3d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(dropout))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(depth - 2):
            layers.append(nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False
            ))
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout3d(dropout))
        
        # Final layer: hidden_dim -> emb_dim
        layers.append(nn.Conv3d(
            hidden_dim, emb_dim,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=True
        ))
        
        self.conv_layers = nn.Sequential(*layers)
        
    def get_init_args(self):
        """Return the arguments needed to reconstruct this predictor instance."""
        return {
            'emb_dim': self.emb_dim,
            'hidden_dim': self.hidden_dim,
            'depth': self.depth,
            'dropout': self.dropout_rate
        }
    
    def forward(self, x):
        """
        Forward pass through the 3D convolutional predictor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C)
            
        Returns:
            torch.Tensor: Predicted features of shape (B, T, H, W, C)
        """
        # Rearrange for 3D convolution: (B, C, T, H, W)
        x = rearrange(x, 'b t h w c -> b c t h w')
        
        # Apply 3D convolution layers
        features = self.conv_layers(x)
        
        # Apply residual connection
        output = features + x

        # Rearrange back to (B, T, H, W, C)
        output = rearrange(output, 'b c t h w -> b t h w c')

        return output


class Conv3DPredictorCausal(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512, depth=4, num_frames=1, dropout=0.1):
        """
        3D Convolutional Predictor with causal masking.
        
        Uses causal convolutions to ensure predictions only depend on past frames.
        
        Args:
            emb_dim (int): Input channel dimension (encoder emb_dim + action_dim)
            hidden_dim (int): Hidden dimension for intermediate layers
            depth (int): Number of 3D conv layers
            num_hist (int): Number of historical frames to consider
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout_rate = dropout
        
        # Build the 3D convolutional layers with causal masking
        layers = []

        # Set the kernel size depending on num_frames
        temporal_kernel_size = min(3, num_frames)
        kernel_size = (temporal_kernel_size, 3, 3)
        
        # First layer: emb_dim -> hidden_dim
        layers.append(CausalConv3d(
            emb_dim, hidden_dim, 
            kernel_size=kernel_size, 
            padding=(1, 1)
        ))
        layers.append(nn.BatchNorm3d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(dropout))
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(depth - 2):
            layers.append(CausalConv3d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                padding=(1, 1)
            ))
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout3d(dropout))
        
        # Final layer: hidden_dim -> emb_dim
        layers.append(CausalConv3d(
            hidden_dim, emb_dim,
            kernel_size=kernel_size,
            padding=(1, 1)
        ))
        
        self.conv_layers = nn.Sequential(*layers)
        
    def get_init_args(self):
        """Return the arguments needed to reconstruct this predictor instance."""
        return {
            'emb_dim': self.emb_dim,
            'hidden_dim': self.hidden_dim,
            'depth': self.depth,
            'num_frames': self.conv_layers[0].kernel_size[0],  # Use the kernel size of the first layer
            'dropout': self.dropout_rate
        }
    
    def forward(self, x):
        """
        Forward pass through the causal 3D convolutional predictor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C)
            
        Returns:
            torch.Tensor: Predicted features of shape (B, T, H, W, C)
        """
        x = rearrange(x, 'b t h w c -> b c t h w')  # Rearrange to (B, C, T, H, W)
        
        # Apply 3D convolution layers with causal masking
        features = self.conv_layers(x)
        
        # Apply residual connection
        output = features + x
        output = rearrange(output, 'b c t h w -> b t h w c')  # Rearrange back to (B, T, H, W, C)
        
        return output


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        """
        3D Convolution with causal masking in the temporal dimension.
        """
        super().__init__()
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # For causal convolution, we need to pad only the past in temporal dimension
        self.temporal_padding = self.kernel_size[0] - 1
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=self.kernel_size,
            padding=(0, self.padding[0], self.padding[1]),  # No temporal padding here
            bias=bias
        )
        
    def forward(self, x): 
        """
        Forward pass with causal masking.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
            Returns:
                torch.Tensor: Output tensor of shape (B, C', T, H, W)
            """
        # Pad only the past in temporal dimension
        x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0)) # (leftW, rightW, topH, bottomH, frontT, backT)
        # Apply convolution to predict the residual
        output = self.conv(x)
        return output
